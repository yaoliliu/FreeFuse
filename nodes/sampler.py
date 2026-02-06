"""
FreeFuse Phase 1 Sampler

Runs partial denoising to collect attention patterns and generate masks.
After this node, use standard KSampler for Phase 2 generation.

Uses ComfyUI's replace patch mechanism to intercept attention internals.
"""

import torch
import torch.nn.functional as F
import comfy.samplers
import comfy.sample

from ..freefuse_core.attention_replace import (
    FreeFuseState,
    FreeFuseFluxBlockReplace,
    FreeFuseSDXLAttnReplace,
    apply_freefuse_replace_patches,
    compute_flux_similarity_maps_from_outputs,
)
from ..freefuse_core.mask_utils import generate_masks

# SDXL UNet region → ComfyUI (block_name, block_num) mapping
# Each region contains N transformer blocks with cross-attention (attn2).
# The user picks a region + transformer index to select which attn2 to collect from.
SDXL_COLLECT_REGION_MAP = {
    "output_early ★ (recommended)": ("output", 0, 10),   # output_blocks.0, 10 transformers
    "output_mid":                    ("output", 1, 10),   # output_blocks.1, 10 transformers
    "output_late":                   ("output", 2, 10),   # output_blocks.2, 10 transformers
    "middle":                        ("middle", 0, 10),   # middle_block,    10 transformers
    "input_deep":                    ("input",  7, 10),   # input_blocks.7,  10 transformers
    "input_deep_2":                  ("input",  8, 10),   # input_blocks.8,  10 transformers
    "output_shallow":                ("output", 3,  2),   # output_blocks.3,  2 transformers
    "input_shallow":                 ("input",  4,  2),   # input_blocks.4,   2 transformers
}


class EarlyStopException(Exception):
    """Exception to signal early termination of sampling after mask collection."""
    pass


class FreeFusePhase1Sampler:
    """
    Phase 1: Collect similarity maps and generate spatial masks.
    
    Runs a few denoising steps (default 5) to collect attention patterns,
    then generates masks for each LoRA concept.
    
    Output the patched model to standard KSampler for Phase 2.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "neg_conditioning": ("CONDITIONING",),
                "latent": ("LATENT",),
                "freefuse_data": ("FREEFUSE_DATA",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 150, 
                    "tooltip": "Total steps for sigma schedule (should match Phase 2)"}),
                "collect_step": ("INT", {"default": 5, "min": 1, "max": 20,
                    "tooltip": "Step at which to collect attention and stop Phase 1"}),
                "cfg": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
            },
            "optional": {
                # Block selection — Flux
                "collect_block": ("INT", {
                    "default": 18, "min": 0, "max": 56,
                    "tooltip": "[Flux only] Which double_block to collect attention from (0-56). Ignored for SDXL."
                }),
                # Block selection — SDXL
                "collect_region": (list(SDXL_COLLECT_REGION_MAP.keys()), {
                    "default": "output_early ★ (recommended)",
                    "tooltip": "[SDXL only] UNet region to collect cross-attention from. Ignored for Flux."
                }),
                "collect_tf_index": ("INT", {
                    "default": 3, "min": 0, "max": 9,
                    "tooltip": "[SDXL only] Transformer block index within the selected region (0-9). Ignored for Flux."
                }),
                # Similarity map computation parameters
                "temperature": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 10000.0, "step": 100.0,
                    "tooltip": "Temperature for softmax in similarity computation. 0=auto (Flux:4000, SDXL:300)"
                }),
                "top_k_ratio": ("FLOAT", {
                    "default": 0.3, "min": 0.01, "max": 1.0, "step": 0.05,
                    "tooltip": "Ratio of top-k tokens to use for similarity computation"
                }),
                # LoRA control
                "disable_lora_phase1": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Disable LoRA during Phase 1 for cleaner base model attention (recommended)"
                }),
                # Mask post-processing parameters
                "bg_scale": ("FLOAT", {
                    "default": 0.95, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Scale factor for background similarity (higher = more background)"
                }),
                "use_morphological_cleaning": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply morphological operations to clean up masks"
                }),
                "balance_iterations": ("INT", {
                    "default": 15, "min": 0, "max": 50,
                    "tooltip": "Number of iterations for balanced argmax algorithm"
                }),
                # stabilized_balanced_argmax fine-tuning parameters
                "balance_lr": ("FLOAT", {
                    "default": 0.01, "min": 0.001, "max": 0.1, "step": 0.001,
                    "tooltip": "Learning rate for bias updates in mask balancing (higher = faster convergence but may overshoot)"
                }),
                "gravity_weight": ("FLOAT", {
                    "default": 0.00004, "min": 0.0, "max": 1.0, "step": 0.00001,
                    "tooltip": "Centroid attraction weight - pulls pixels toward their concept's center for spatial coherence"
                }),
                "spatial_weight": ("FLOAT", {
                    "default": 0.00004, "min": 0.0, "max": 1.0, "step": 0.00001,
                    "tooltip": "Neighbor voting weight - encourages spatially smooth masks by considering neighboring pixels"
                }),
                "momentum": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 0.9, "step": 0.05,
                    "tooltip": "Probability smoothing momentum between iterations (higher = more stable but slower adaptation)"
                }),
                "centroid_margin": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.5, "step": 0.05,
                    "tooltip": "Margin to clamp centroids away from image borders (0 = no clamping)"
                }),
                "border_penalty": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Penalty for assigning concept pixels near image borders (0 = no penalty)"
                }),
                "anisotropy": ("FLOAT", {
                    "default": 1.3, "min": 0.5, "max": 3.0, "step": 0.1,
                    "tooltip": "Horizontal stretch factor for spatial coordinates (>1 = prefer wider masks, <1 = prefer taller masks)"
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL", "FREEFUSE_MASKS", "IMAGE")
    RETURN_NAMES = ("model", "masks", "mask_preview")
    FUNCTION = "collect_masks"
    CATEGORY = "FreeFuse"
    
    DESCRIPTION = """Phase 1 of FreeFuse: collect attention and generate masks.
    
Uses full sigma schedule (e.g., 28 steps) but stops early after collecting
attention at collect_step. This ensures correct noise levels while saving time.

Block Selection (model-type aware):
- Flux: Use 'collect_block' (INT 0-56) to pick a double_block.
- SDXL: Use 'collect_region' + 'collect_tf_index' to pick a UNet cross-attention block.
  The recommended default is output_early (region) + 3 (tf_index).
  Other parameters for the non-active model type are simply ignored.

Other Parameters:
- temperature: Softmax temperature for similarity (0=auto-detect by model type)
- top_k_ratio: Ratio of tokens used for similarity computation
- disable_lora_phase1: Whether to disable LoRA during collection (recommended)
- bg_scale: Background scaling factor for mask generation
- use_morphological_cleaning: Clean up masks with morphological operations

Mask Balancing Parameters (Advanced):
- balance_iterations: Number of iterations for the balanced argmax algorithm
- balance_lr: Learning rate for bias updates (higher = faster but less stable)
- gravity_weight: Centroid attraction for spatial coherence
- spatial_weight: Neighbor voting for spatially smooth masks
- momentum: Probability smoothing between iterations
- centroid_margin: Clamp centroids away from borders
- border_penalty: Penalize concept assignment near borders
- anisotropy: Horizontal stretch factor (>1 prefers wider masks)

After this node, connect the model output to a standard KSampler
for Phase 2 generation with the same seed and steps."""
    
    def collect_masks(self, model, conditioning, neg_conditioning, latent, 
                      freefuse_data, seed, steps, collect_step, cfg, sampler_name, scheduler,
                      collect_block=18, 
                      collect_region="output_early ★ (recommended)", collect_tf_index=3,
                      temperature=0.0, top_k_ratio=0.3,
                      disable_lora_phase1=True, bg_scale=0.95, 
                      use_morphological_cleaning=True, balance_iterations=15,
                      balance_lr=0.01, gravity_weight=0.00004, spatial_weight=0.00004,
                      momentum=0.2, centroid_margin=0.0, border_penalty=0.0,
                      anisotropy=1.3):
        
        concepts = freefuse_data.get("concepts", {})
        settings = freefuse_data.get("settings", {})
        token_pos_maps = freefuse_data.get("token_pos_maps", {})
        include_background = settings.get("enable_background", True)
        
        if not concepts:
            print("[FreeFuse] Warning: No concepts defined, returning empty masks")
            empty_masks = {}
            preview = torch.zeros(1, 64, 64, 3)
            return (model, {"masks": empty_masks}, preview)
        
        # Check if token positions are available
        if not token_pos_maps:
            print("[FreeFuse] Warning: No token positions computed. "
                  "Connect FreeFuseTokenPositions node before this sampler.")
            # Create fallback uniform masks
            latent_image = latent["samples"]
            latent_h, latent_w = latent_image.shape[2], latent_image.shape[3]
            img_h, img_w = latent_h * 8, latent_w * 8
            
            masks = {name: torch.ones(latent_h, latent_w) for name in concepts}
            preview = self._create_preview(masks, img_w, img_h)
            return (model, {"masks": masks}, preview)
        
        # Clone model to add attention hooks
        model_clone = model.clone()
        
        # Sanity check: collect_step must be within total steps
        if collect_step > steps:
            print(f"[FreeFuse] Warning: collect_step ({collect_step}) > steps ({steps}); "
                  f"clamping to {steps}.")
            collect_step = steps
        if collect_step < 1:
            print(f"[FreeFuse] Warning: collect_step ({collect_step}) < 1; "
                  "clamping to 1.")
            collect_step = 1

        # Set up FreeFuse state with the new system
        freefuse_state = FreeFuseState()
        freefuse_state.phase = "collect"
        # Collect at the specified step (0-indexed internally)
        freefuse_state.collect_step = collect_step - 1  # Convert to 0-indexed
        freefuse_state.collect_block = collect_block  # Used by Flux
        freefuse_state.token_pos_maps = token_pos_maps
        freefuse_state.include_background = include_background
        
        # Detect model type first (needed for default temperature and block routing)
        model_type = "auto"
        model_name = model_clone.model.__class__.__name__.lower()
        if "flux" in model_name:
            model_type = "flux"
        else:
            model_type = "sdxl"
        
        # Use user-provided parameters, with auto-detection for temperature=0
        # Default temperature differs by model type: Flux=4000, SDXL=300
        if temperature == 0.0:
            auto_temperature = 4000.0 if model_type == "flux" else 300.0
        else:
            auto_temperature = temperature
        
        freefuse_state.top_k_ratio = top_k_ratio
        freefuse_state.temperature = auto_temperature
        
        # Build SDXL collect_blocks from region + tf_index (only used for SDXL)
        sdxl_collect_blocks = None
        if model_type == "sdxl":
            region_info = SDXL_COLLECT_REGION_MAP.get(collect_region)
            if region_info is None:
                # Fallback to recommended default
                print(f"[FreeFuse] Warning: Unknown region '{collect_region}', using default")
                region_info = ("output", 0, 10)
            block_name, block_num, max_tf = region_info
            # Clamp tf_index to valid range for this region
            tf_idx = min(collect_tf_index, max_tf - 1)
            if tf_idx != collect_tf_index:
                print(f"[FreeFuse] Warning: collect_tf_index={collect_tf_index} clamped to {tf_idx} "
                      f"(region '{collect_region}' has {max_tf} transformers)")
            sdxl_collect_blocks = [(block_name, block_num, tf_idx)]
            print(f"[FreeFuse] SDXL collect block: ({block_name}, {block_num}, {tf_idx}) "
                  f"from region='{collect_region}'")
            
        apply_freefuse_replace_patches(
            model_clone, freefuse_state, 
            model_type=model_type,
            sdxl_collect_blocks=sdxl_collect_blocks,
        )
        
        # Get latent info
        latent_image = latent["samples"]
        batch_size = latent_image.shape[0]
        
        # Calculate image dimensions
        latent_h, latent_w = latent_image.shape[2], latent_image.shape[3]
        img_h, img_w = latent_h * 8, latent_w * 8
        
        block_info = f"double_block {collect_block}" if model_type == "flux" else f"{collect_region} tf={collect_tf_index}"
        print(f"[FreeFuse] Phase 1: {steps} total steps, collecting at step {collect_step}, {block_info}")
        print(f"[FreeFuse] Concepts: {list(concepts.keys())}")
        print(f"[FreeFuse] Token positions: {token_pos_maps}")
        print(f"[FreeFuse] Model type: {model_type}")
        print(f"[FreeFuse] Settings: temperature={auto_temperature}, top_k_ratio={top_k_ratio}, disable_lora={disable_lora_phase1}")
        
        # Optionally disable LoRA during Phase 1 to get clean base model attention patterns
        # This matches the diffusers implementation behavior
        bypass_manager = model_clone.model_options.get("transformer_options", {}).get("freefuse_bypass_manager")
        if bypass_manager is not None and disable_lora_phase1:
            bypass_manager.disable_lora()
            print("[FreeFuse] Phase 1: LoRA disabled for clean attention collection")
        elif bypass_manager is not None:
            print("[FreeFuse] Phase 1: LoRA enabled (user override)")
        
        # Create noise for Phase 1
        noise = comfy.sample.prepare_noise(latent_image, seed, None)
        
        # Configure step callback to update current step and enable early stopping
        def step_callback(step, x0, x, total_steps):
            freefuse_state.current_step = step
            # Check if we've collected similarity maps and can stop early
            if freefuse_state.similarity_maps and step > freefuse_state.collect_step:
                print(f"[FreeFuse] Early stopping at step {step + 1} (collected at step {collect_step})")
                raise EarlyStopException("Similarity maps collected, stopping early")
        
        # Run Phase 1 sampling (may terminate early via EarlyStopException)
        early_stopped = False
        try:
            samples = comfy.sample.sample(
                model_clone,
                noise,
                steps,
                cfg,
                sampler_name,
                scheduler,
                conditioning,
                neg_conditioning,
                latent_image,
                denoise=1.0,
                disable_noise=False,
                start_step=0,
                last_step=steps,
                force_full_denoise=False,
                noise_mask=None,
                callback=step_callback,
                seed=seed,
            )
        except EarlyStopException as e:
            # Expected early termination - similarity maps already collected
            early_stopped = True
            print(f"[FreeFuse] Phase 1 completed early: {e}")
        except Exception as e:
            print(f"[FreeFuse] Phase 1 sampling error: {e}")
            import traceback
            traceback.print_exc()
            # Re-enable LoRA before returning on error (if it was disabled)
            if bypass_manager is not None and disable_lora_phase1:
                bypass_manager.enable_lora()
            # Return empty masks on error
            empty_masks = {name: torch.ones(latent_h, latent_w) for name in concepts}
            preview = self._create_preview(empty_masks, img_w, img_h)
            return (model_clone, {"masks": empty_masks}, preview)
        
        # Get similarity maps directly from freefuse_state
        similarity_maps = freefuse_state.similarity_maps
        
        # Debug: Show raw similarity maps
        print(f"[FreeFuse] Raw similarity maps: {list(similarity_maps.keys())}")
        for name, sim_map in similarity_maps.items():
            if sim_map is not None:
                print(f"   {name}: shape={sim_map.shape}, min={sim_map.min():.6f}, max={sim_map.max():.6f}, mean={sim_map.mean():.6f}")
        
        # Generate masks directly from raw similarity maps
        # The new generate_masks can handle (B, N, 1) format and includes:
        # - stabilized_balanced_argmax from diffusers
        # - morphological cleaning
        # - proper background handling
        if similarity_maps:
            masks = generate_masks(
                similarity_maps, 
                include_background=include_background,  # Use setting from concept map
                method="stabilized",  # Use the sophisticated algorithm from diffusers
                bg_scale=bg_scale,  # User-configurable background scale
                use_morphological_cleaning=use_morphological_cleaning,  # User-configurable
                debug=True,  # Enable debug output
                max_iter=balance_iterations,  # User-configurable iterations
                latent_h=latent_h,  # CRITICAL: Pass latent dimensions for correct reshape!
                latent_w=latent_w,  # Without this, non-square images produce corrupted masks!
                # stabilized_balanced_argmax fine-tuning parameters
                balance_lr=balance_lr,
                gravity_weight=gravity_weight,
                spatial_weight=spatial_weight,
                momentum=momentum,
                centroid_margin=centroid_margin,
                border_penalty=border_penalty,
                anisotropy=anisotropy,
            )
        else:
            # Fallback: create uniform masks
            print("[FreeFuse] Warning: No similarity maps, using uniform masks")
            masks = {}
            for name in concepts.keys():
                masks[name] = torch.ones(latent_h, latent_w, device=latent_image.device)
            if include_background:
                masks["_background_"] = torch.zeros(latent_h, latent_w, device=latent_image.device)
        
        # Create preview image
        preview = self._create_preview(masks, img_w, img_h)
        
        # Update state for Phase 2
        freefuse_state.phase = "generate"
        freefuse_state.masks = masks
        
        # Re-enable LoRA for Phase 2 generation (if it was disabled)
        if bypass_manager is not None and disable_lora_phase1:
            bypass_manager.enable_lora()
            print("[FreeFuse] Phase 1 complete: LoRA re-enabled for Phase 2")
        
        # Store masks in model options for Phase 2
        if "transformer_options" not in model_clone.model_options:
            model_clone.model_options["transformer_options"] = {}
        model_clone.model_options["transformer_options"]["freefuse_masks"] = masks
        model_clone.model_options["freefuse_state"] = freefuse_state
        
        print(f"[FreeFuse] Phase 1 complete. Generated {len(masks)} masks.")
        print(f"[FreeFuse] Mask keys: {list(masks.keys())}")
        for name, mask in masks.items():
            coverage = mask.sum() / mask.numel() * 100
            print(f"   {name}: coverage={coverage:.1f}%")
        
        return (
            model_clone, 
            {"masks": masks, "similarity_maps": similarity_maps}, 
            preview
        )
    
    def _process_similarity_maps(self, similarity_maps, latent_size):
        """Convert similarity maps to spatial format."""
        if not similarity_maps:
            return {}
            
        result = {}
        h, w = latent_size
        img_len = h * w
        
        for name, sim_map in similarity_maps.items():
            if sim_map is None:
                continue
                
            # sim_map shape: (B, img_len, 1) or (B, img_len)
            if sim_map.dim() == 3:
                sim_map = sim_map.squeeze(-1)
            
            if sim_map.shape[-1] == img_len:
                # Take first batch, reshape to spatial
                spatial = sim_map[0].view(h, w)
                result[name] = spatial.detach()
            else:
                # Need to resize
                current_len = sim_map.shape[-1]
                current_h = int((current_len) ** 0.5)
                current_w = current_len // current_h
                
                if current_h * current_w == current_len:
                    spatial = sim_map[0].view(current_h, current_w)
                    spatial = F.interpolate(
                        spatial.unsqueeze(0).unsqueeze(0),
                        size=(h, w),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()
                    result[name] = spatial.detach()
                else:
                    print(f"[FreeFuse] Warning: Cannot reshape sim_map for {name}, len={current_len}")
                    result[name] = torch.ones(h, w, device=sim_map.device)
        
        print(f"[FreeFuse] Processed {len(result)} similarity maps to spatial format")
        return result
    
    def _create_preview(self, masks, width, height):
        """Create color-coded mask preview."""
        colors = [
            (1.0, 0.0, 0.0),   # Red
            (0.0, 1.0, 0.0),   # Green
            (0.0, 0.0, 1.0),   # Blue
            (1.0, 1.0, 0.0),   # Yellow
            (1.0, 0.0, 1.0),   # Magenta
            (0.0, 1.0, 1.0),   # Cyan
            (0.25, 0.25, 0.25),  # Gray (background)
        ]
        
        device = next(iter(masks.values())).device if masks else "cpu"
        preview = torch.zeros(3, height, width, device=device)
        
        for i, (name, mask) in enumerate(masks.items()):
            color = colors[i % len(colors)]
            
            # Handle different mask dimensions
            if mask.dim() == 2:
                m = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                m = mask.unsqueeze(0)
            else:
                m = mask
            
            # Resize mask
            mask_resized = F.interpolate(
                m.float(),
                size=(height, width),
                mode='nearest'
            ).squeeze()
            
            for c in range(3):
                preview[c] += mask_resized * color[c]
        
        # Convert to ComfyUI format: (B, H, W, C)
        preview = preview.clamp(0, 1).permute(1, 2, 0).unsqueeze(0)
        
        return preview.cpu()
