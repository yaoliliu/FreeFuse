"""
FreeFuse Phase 1 Sampler

Runs partial denoising to collect attention patterns and generate masks.
After this node, use standard KSampler for Phase 2 generation.

Uses ComfyUI's replace patch mechanism to intercept attention internals.
"""

import torch
import torch.nn.functional as F
import os
import json
import comfy.samplers
import comfy.sample

from ..freefuse_core.attention_replace import (
    FreeFuseState,
    FreeFuseFluxBlockReplace,
    FreeFuseSDXLAttnReplace,
    FreeFuseZImageBlockReplace,
    apply_freefuse_replace_patches,
    compute_flux_similarity_maps_from_outputs,
    compute_z_image_similarity_maps,
)
from ..freefuse_core.mask_utils import generate_masks
from ..freefuse_core.voting import create_consensus_similarity_maps

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
                "collect_block_end": ("INT", {
                    "default": 18, "min": 0, "max": 56,
                    "tooltip": "[Flux only] End block for range collection (inclusive). If > collect_block, collects from all blocks in range and uses pixel voting. Ignored for SDXL."
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
                # Preview parameters
                "preview_sensitivity": ("FLOAT", {
                    "default": 5.0, "min": 1.0, "max": 20.0, "step": 0.5,
                    "tooltip": "Contrast for mask preview visualization (higher = meer contrast)"
                }),
            }
        }
    
    # 👇 4 outputs
    RETURN_TYPES = ("MODEL", "FREEFUSE_MASKS", "IMAGE", "FREEFUSE_MASKS")
    RETURN_NAMES = ("model", "masks", "mask_preview", "raw_similarity")
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

Preview Parameters:
- preview_sensitivity: Contrast enhancement for mask preview visualization

After this node, connect the model output to a standard KSampler
for Phase 2 generation with the same seed and steps."""
    
    def collect_masks(self, model, conditioning, neg_conditioning, latent,
                      freefuse_data, seed, steps, collect_step, cfg, sampler_name, scheduler,
                      collect_block=18, collect_block_end=18,
                      collect_region="output_early ★ (recommended)", collect_tf_index=3,
                      temperature=0.0, top_k_ratio=0.3,
                      disable_lora_phase1=True, bg_scale=0.95,
                      use_morphological_cleaning=True, balance_iterations=15,
                      balance_lr=0.01, gravity_weight=0.00004, spatial_weight=0.00004,
                      momentum=0.2, centroid_margin=0.0, border_penalty=0.0,
                      anisotropy=1.3, preview_sensitivity=5.0):
        
        concepts = freefuse_data.get("concepts", {})
        settings = freefuse_data.get("settings", {})
        token_pos_maps = freefuse_data.get("token_pos_maps", {})
        include_background = settings.get("enable_background", True)
        
        if not concepts:
            print("[FreeFuse] Warning: No concepts defined, returning empty masks")
            empty_masks = {}
            preview = torch.zeros(1, 64, 64, 3)
            empty_raw = {"masks": {}, "similarity_maps": {}}
            return (model, {"masks": empty_masks}, preview, empty_raw)
        
        # Check if token positions are available
        if not token_pos_maps:
            print("[FreeFuse] Warning: No token positions computed. "
                  "Connect FreeFuseTokenPositions node before this sampler.")
            # Create fallback uniform masks
            latent_image = latent["samples"]
            latent_h, latent_w = latent_image.shape[2], latent_image.shape[3]
            img_h, img_w = latent_h * 8, latent_w * 8
            
            masks = {name: torch.ones(latent_h, latent_w) for name in concepts}
            preview = self._create_argmax_preview(masks, img_w, img_h, sensitivity=preview_sensitivity)
            empty_raw = {"masks": {}, "similarity_maps": {}}
            return (model, {"masks": masks}, preview, empty_raw)
        
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
        # Block range for Flux: if collect_block_end > collect_block, collect from all blocks in range
        freefuse_state.collect_block = collect_block  # Start block (used by Flux)
        freefuse_state.collect_block_end = collect_block_end  # End block (used by Flux)
        freefuse_state.block_voting_maps = {}  # Store per-block winner maps for voting
        freefuse_state.collect_region = collect_region  # Used by SDXL
        freefuse_state.collect_tf_index = collect_tf_index  # Used by SDXL
        freefuse_state.token_pos_maps = token_pos_maps
        freefuse_state.include_background = include_background
        # Track which blocks have been collected (for range mode)
        freefuse_state.collected_blocks = set()
        
        # Detect model type first (needed for default temperature and block routing)
        model_type = "auto"
        model_name = model_clone.model.__class__.__name__.lower()
        # Check Qwen-Image FIRST (before Flux) as Qwen-Image may have flux-like class names
        if "qwenimage" in model_name or "qwen_image" in model_name or "qwen" in model_name:
            model_type = "qwen_image"
        elif "nextdit" in model_name or "lumina" in model_name:
            model_type = "z_image"
        elif "flux2" in model_name:
            model_type = "flux2"
        elif "flux" in model_name:
            model_type = "flux"
        else:
            model_type = "sdxl"

        # Use user-provided parameters, with auto-detection for temperature=0
        # Default temperature differs by model type: Flux=4000, SDXL=300, Z-Image=4000, Qwen-Image=4000
        if temperature == 0.0:
            if model_type == "z_image" or model_type == "qwen_image":
                auto_temperature = 4000.0   # matches reference FreeFuseZImageAttnProcessor
            elif model_type == "flux":
                auto_temperature = 4000.0
            else:
                auto_temperature = 300.0
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

        # For Flux: determine block range
        flux_collect_blocks = None
        if model_type == "flux" and collect_block_end > collect_block:
            flux_collect_blocks = list(range(collect_block, collect_block_end + 1))
            print(f"[FreeFuse] Flux range mode: collecting from blocks {collect_block} to {collect_block_end} "
                  f"({len(flux_collect_blocks)} blocks)")

        # Qwen-Image: Phase 1 mask collection is NOT compatible due to 5D tensor format
        # Return fallback masks and let Phase 2 handle mask application
        if model_type == "qwen_image":
            print("[FreeFuse] Qwen-Image: Phase 1 mask collection is not compatible.")
            print("[FreeFuse] Qwen-Image: Use manual masks with FreeFuseMaskApplicator instead.")
            print("[FreeFuse] Qwen-Image: Returning split masks (left/right) for Phase 2.")

            # Get latent info for fallback masks
            latent_image = latent["samples"]
            latent_h, latent_w = latent_image.shape[2], latent_image.shape[3]
            img_h, img_w = latent_h * 8, latent_w * 8

            # Create split masks: first concept on left, others on right
            masks = {}
            concept_list = list(concepts.keys())

            for i, name in enumerate(concept_list):
                mask = torch.zeros(latent_h, latent_w, device=latent_image.device)
                if i == 0:
                    # First concept: left half
                    mask[:, :latent_w // 2] = 1.0
                    print(f"[FreeFuse] Qwen-Image: '{name}' → LEFT side")
                else:
                    # Other concepts: right half
                    mask[:, latent_w // 2:] = 1.0
                    print(f"[FreeFuse] Qwen-Image: '{name}' → RIGHT side")
                masks[name] = mask

            # Background: full image at low weight
            if include_background:
                masks["__background__"] = torch.ones(latent_h, latent_w, device=latent_image.device) * 0.1
                print(f"[FreeFuse] Qwen-Image: '__background__' → full image (10%)")

            # Create colored preview showing left/right split
            preview = self._create_argmax_preview(masks, img_w, img_h, sensitivity=preview_sensitivity)
            empty_raw = {"masks": {}, "similarity_maps": {}}

            return (model_clone, {"masks": masks}, preview, empty_raw)

        # For Qwen-Image: determine block range (not used, kept for compatibility)
        qwen_collect_blocks = None

        apply_freefuse_replace_patches(
            model_clone, freefuse_state,
            model_type=model_type,
            sdxl_collect_blocks=sdxl_collect_blocks,
            flux_collect_blocks=flux_collect_blocks,
            qwen_collect_blocks=qwen_collect_blocks,
        )

        # Get latent info
        latent_image = latent["samples"]
        batch_size = latent_image.shape[0]
        
        # Calculate image dimensions
        latent_h, latent_w = latent_image.shape[2], latent_image.shape[3]
        img_h, img_w = latent_h * 8, latent_w * 8
        
        # For Z-Image: store sequence lengths in state so block_replace can access them
        if model_type == "z_image":
            z_img_seq_len = latent_h * latent_w
            # Estimate cap_seq_len from token positions map
            z_cap_seq_len = 256  # Default
            for positions_list in token_pos_maps.values():
                if positions_list and positions_list[0]:
                    max_pos = max(positions_list[0])
                    z_cap_seq_len = max(z_cap_seq_len, max_pos + 10)
            freefuse_state.collected_outputs["img_seq_len"] = z_img_seq_len
            freefuse_state.collected_outputs["cap_seq_len"] = z_cap_seq_len
            freefuse_state.collected_outputs["latent_h"] = latent_h
            freefuse_state.collected_outputs["latent_w"] = latent_w
            print(f"[FreeFuse] Z-Image sequence info: img_seq_len={z_img_seq_len}, cap_seq_len={z_cap_seq_len}")

        # Block info for logging
        if model_type == "flux":
            block_info = f"double_block {collect_block}"
        elif model_type == "z_image":
            block_info = f"layer {collect_block}"
        else:
            block_info = f"{collect_region} tf={collect_tf_index}"

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
            # In range mode (Flux), wait until all blocks have been collected
            if flux_collect_blocks and len(freefuse_state.collected_blocks) < len(flux_collect_blocks):
                return  # Continue sampling until all blocks collected
            # In single-block mode, check if we collected at the target step
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
            preview = self._create_argmax_preview(empty_masks, img_w, img_h, sensitivity=preview_sensitivity)
            empty_raw = {"masks": {}, "similarity_maps": {}}
            return (model_clone, {"masks": empty_masks}, preview, empty_raw)
        
        # Get similarity maps directly from freefuse_state
        similarity_maps = freefuse_state.similarity_maps

        # === RANGE MODE: Perform majority voting across blocks ===
        # Uses the proven consensus algorithm from blocks_analysis.py
        # Only Flux supports range mode (Qwen-Image returns early)
        if flux_collect_blocks and len(flux_collect_blocks) > 1:
            print(f"[FreeFuse] Range mode: aggregating {len(flux_collect_blocks)} blocks via majority voting")

            # Get all block-specific similarity maps
            block_sim_maps = freefuse_state.collected_outputs.get("block_similarity_maps", {})

            if block_sim_maps:
                # Get ALL concept names (including background)
                concept_names = [c for c in freefuse_state.token_pos_maps.keys() if not c.startswith("__")]
                if freefuse_state.include_background:
                    concept_names.append("__background__")

                # Use shared voting utility (same as blocks_analysis.py)
                latent_h, latent_w = latent_image.shape[2], latent_image.shape[3]
                similarity_maps = create_consensus_similarity_maps(
                    all_blocks_masks=block_sim_maps,
                    concept_names=concept_names,
                    latent_h=latent_h,
                    latent_w=latent_w,
                    device=latent_image.device,
                )
                
                # Generate masks ONCE from the voting result (no second stabilization needed)
                # The voting already produced clean hard masks
                masks = {}
                for name, sim_map in similarity_maps.items():
                    # Convert (B, img_len, 1) to (H, W) directly
                    mask = sim_map[0].squeeze(-1).view(latent_h, latent_w)
                    masks[name] = mask
                
                print(f"[FreeFuse] Majority voting complete. Aggregated {len(block_sim_maps)} blocks.")
                
                # Skip the normal generate_masks() call - we already have clean masks from voting
                # Jump directly to preview generation
                preview = self._create_argmax_preview(masks, img_w, img_h, sensitivity=preview_sensitivity)
                
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

                # Return early - range mode complete
                raw_similarity_data = {
                    "masks": similarity_maps,
                    "similarity_maps": similarity_maps,
                    "is_raw": True
                }
                return (
                    model_clone,
                    {"masks": masks, "similarity_maps": similarity_maps},
                    preview,
                    raw_similarity_data
                )
            else:
                print(f"[FreeFuse] Warning: Range mode enabled but no block_similarity_maps found")

        # Single-block mode (non-range): continue with normal processing
        # Debug: Show raw similarity maps
        print(f"[FreeFuse] Raw similarity maps: {list(similarity_maps.keys())}")
        for name, sim_map in similarity_maps.items():
            if sim_map is not None:
                print(f"   {name}: shape={sim_map.shape}, min={sim_map.min():.6f}, max={sim_map.max():.6f}, mean={sim_map.mean():.6f}")
        
        # 👇 BEWAAR DE RAAUWE SIMILARITY MAPS VOORDAT ZE WORDEN VERWERKT!
        raw_similarity_data = {
            "masks": similarity_maps,  # Hergebruik dezelfde structuur
            "similarity_maps": similarity_maps,
            "is_raw": True  # Optionele flag
        }
        
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

        # Optional debug dump for masks
        if os.environ.get("FREEFUSE_DEBUG_ZIMAGE") == "1" and model_type == "z_image":
            try:
                debug_dir = os.path.join(os.getcwd(), "debug_z_image_comfyui")
                os.makedirs(debug_dir, exist_ok=True)
                mask_cpu = {k: v.detach().cpu() for k, v in masks.items()}
                torch.save(mask_cpu, os.path.join(debug_dir, "masks.pt"))
            except Exception as e:
                print(f"[FreeFuse Z-Image Debug] Failed to save masks: {e}")

        if os.environ.get("FREEFUSE_DEBUG_QWEN_IMAGE") == "1" and model_type == "qwen_image":
            try:
                debug_dir = os.path.join(os.getcwd(), "debug_qwen_image_comfyui")
                os.makedirs(debug_dir, exist_ok=True)
                mask_cpu = {k: v.detach().cpu() for k, v in masks.items()}
                torch.save(mask_cpu, os.path.join(debug_dir, "masks.pt"))
            except Exception as e:
                print(f"[FreeFuse Qwen-Image Debug] Failed to save masks: {e}")
        
        # 🔥 ARGMAX WINNER PREVIEW (zelfde als overlay node)
        preview = self._create_argmax_preview(masks, img_w, img_h, sensitivity=preview_sensitivity)
        
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
            preview,
            raw_similarity_data  # 👈 EXTRA UITGANG!
        )
    
    @staticmethod
    def _infer_spatial_size_from_latent(
        seq_len: int, lat_h: int, lat_w: int
    ) -> tuple:
        """Infer (h, w) from a flattened sequence length using latent aspect ratio.

        Tries power-of-2 downscale factors first, then aspect-ratio-preserving
        factorisation, then square, and finally a generic closest-to-square search.
        """
        # Strategy 1: power-of-2 downscale factors of latent_size
        for d in (1, 2, 4, 8, 16):
            h, w = lat_h // d, lat_w // d
            if h > 0 and w > 0 and h * w == seq_len:
                return (h, w)

        # Strategy 2: aspect-ratio-preserving decomposition
        if lat_h > 0 and lat_w > 0:
            ratio = lat_w / lat_h
            h = max(int(round((seq_len / max(ratio, 1e-8)) ** 0.5)), 1)
            w = seq_len // h
            if h * w == seq_len:
                return (h, w)

        # Strategy 3: square
        side = int(round(seq_len ** 0.5))
        if side * side == seq_len:
            return (side, side)

        # Strategy 4: closest-to-square factor search
        import math as _math
        best_h, best_w = 1, seq_len
        for h in range(1, int(_math.isqrt(seq_len)) + 1):
            if seq_len % h == 0:
                w = seq_len // h
                if abs(w - h) < abs(best_w - best_h):
                    best_h, best_w = h, w
        return (best_h, best_w)

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
                # Need to resize – infer (current_h, current_w) from latent aspect ratio
                current_len = sim_map.shape[-1]
                current_h, current_w = self._infer_spatial_size_from_latent(
                    current_len, h, w
                )
                
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
    
    def _create_argmax_preview(self, masks, width, height, sensitivity=5.0):
        """Create argmax winner preview (zelfde als overlay node).
        
        Toont per pixel welke concept wint (harde segmentatie).
        Inclusief background als aparte kleur.
        
        Args:
            masks: Dict of concept name -> binary mask tensor (H, W)
            width, height: Target preview size
            sensitivity: Contrast enhancement (niet gebruikt voor binaire masks, maar voor compatibiliteit)
        """
        colors = [
            (1.0, 0.0, 0.0),   # Rood
            (0.0, 1.0, 0.0),   # Groen
            (0.0, 0.0, 1.0),   # Blauw
            (1.0, 1.0, 0.0),   # Geel
            (1.0, 0.0, 1.0),   # Magenta
            (0.0, 1.0, 1.0),   # Cyaan
            (1.0, 0.5, 0.0),   # Oranje
            (0.5, 0.0, 1.0),   # Paars
            (1.0, 0.0, 0.5),   # Roze
            (0.5, 1.0, 0.0),   # Lichtgroen
            (0.3, 0.3, 0.3),   # Donkergrijs (background)
        ]
        
        if not masks:
            return torch.zeros(1, height, width, 3)
        
        device = next(iter(masks.values())).device
        preview = torch.zeros(3, height, width, device=device)
        
        # Prepare all masks (inclusief background!)
        concept_masks = []
        concept_names = []
        
        for name, mask in masks.items():
            concept_names.append(name)
            
            # Ensure mask is 2D
            if mask.dim() == 3:
                mask = mask[0]
            
            # Resize mask to target size (nearest for binary masks)
            mask_resized = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=(height, width),
                mode='nearest'
            ).squeeze(0).squeeze(0)
            
            concept_masks.append(mask_resized)
        
        # Create winner image
        if concept_masks:
            # Stack masks: (C, H, W)
            stacked = torch.stack(concept_masks, dim=0)
            
            # Find winner (concept with highest value at each pixel)
            # Voor binaire masks: elk pixel hoort bij exact één concept
            winner_indices = torch.argmax(stacked, dim=0)  # (H, W)
            
            # Color each pixel according to winner
            for idx, color in enumerate(colors[:len(concept_names)]):
                # Create mask for pixels where this concept wins
                mask = (winner_indices == idx).float()
                
                # Add color to preview
                for c in range(3):
                    preview[c] += mask * color[c]
        
        # Clamp en converteer naar ComfyUI format
        preview = preview.clamp(0, 1).permute(1, 2, 0).unsqueeze(0)
        
        return preview.cpu()
