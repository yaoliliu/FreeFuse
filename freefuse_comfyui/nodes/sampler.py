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
                "collect_block": ("INT", {
                    "default": 18, "min": 0, "max": 56,
                    "tooltip": "Which transformer block to collect attention from"
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

After this node, connect the model output to a standard KSampler
for Phase 2 generation with the same seed and steps."""
    
    def collect_masks(self, model, conditioning, neg_conditioning, latent, 
                      freefuse_data, seed, steps, collect_step, cfg, sampler_name, scheduler,
                      collect_block=18):
        
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
        
        # Set up FreeFuse state with the new system
        freefuse_state = FreeFuseState()
        freefuse_state.phase = "collect"
        # Collect at the specified step (0-indexed internally)
        freefuse_state.collect_step = collect_step - 1  # Convert to 0-indexed
        freefuse_state.collect_block = collect_block
        freefuse_state.token_pos_maps = token_pos_maps
        freefuse_state.include_background = include_background
        
        # Copy settings from freefuse_data
        freefuse_state.top_k_ratio = settings.get("top_k_ratio", 0.3)
        freefuse_state.temperature = settings.get("temperature", 1000.0)
        
        # Detect model type and apply replace patches
        model_type = "auto"
        model_name = model_clone.model.__class__.__name__.lower()
        if "flux" in model_name:
            model_type = "flux"
        else:
            model_type = "sdxl"
            
        apply_freefuse_replace_patches(model_clone, freefuse_state, model_type=model_type)
        
        # Get latent info
        latent_image = latent["samples"]
        batch_size = latent_image.shape[0]
        
        # Calculate image dimensions
        latent_h, latent_w = latent_image.shape[2], latent_image.shape[3]
        img_h, img_w = latent_h * 8, latent_w * 8
        
        print(f"[FreeFuse] Phase 1: {steps} total steps, collecting at step {collect_step}, block {collect_block}")
        print(f"[FreeFuse] Concepts: {list(concepts.keys())}")
        print(f"[FreeFuse] Token positions: {token_pos_maps}")
        print(f"[FreeFuse] Model type: {model_type}")
        
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
                include_background=False,
                method="stabilized",  # Use the sophisticated algorithm from diffusers
                bg_scale=0.95,  # Same as diffusers default
                use_morphological_cleaning=False,
                debug=True,  # Enable debug output
                max_iter=5,  # Test with iterations
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
