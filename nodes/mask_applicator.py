"""
FreeFuse Mask Applicator Node

Applies Phase 1 masks to LoRA-loaded model for Phase 2 generation.
This is the bridge between Phase 1 (mask collection) and Phase 2 (generation).

The masks are applied to the bypass LoRA hooks created by FreeFuseLoRALoader.
Each adapter's h(x) output is multiplied by its corresponding spatial mask.

NEW in v2: Attention Bias Support
- Constructs soft attention bias to guide text-image cross-attention
- Supports both Flux (joint attention) and SDXL (cross-attention)
"""

import torch
import torch.nn.functional as F
import logging
from typing import Dict, Optional, List, Tuple

import comfy.model_patcher
import comfy.weight_adapter
from comfy.weight_adapter.base import WeightAdapterBase, WeightAdapterTrainBase

from ..freefuse_core.bypass_lora_loader import (
    OffsetBypassInjectionManager,
    MultiAdapterBypassForwardHook,
)
from ..freefuse_core.attention_bias import (
    construct_attention_bias,
    construct_attention_bias_sdxl,
    AttentionBiasConfig,
)
from ..freefuse_core.attention_bias_patch import (
    apply_attention_bias_patches,
)


class FreeFuseMaskApplicator:
    """
    Apply FreeFuse masks to model with loaded LoRAs.
    
    This node takes:
    - Model with LoRAs loaded in bypass mode
    - Masks from Phase 1 sampler
    - FreeFuse data with adapter info
    
    And outputs a model with masked LoRA application.
    
    NEW: Attention Bias Support
    - When enabled, constructs soft attention bias matrix
    - Guides text-image attention to respect spatial masks
    - Supports both Flux and SDXL architectures
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "masks": ("FREEFUSE_MASKS",),
                "freefuse_data": ("FREEFUSE_DATA",),
            },
            "optional": {
                "enable_token_masking": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Zero out LoRA at other concepts' token positions"
                }),
                "latent": ("LATENT", {
                    "tooltip": "Optional latent for size reference"
                }),
                # Attention Bias parameters
                "enable_attention_bias": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply attention bias to guide text-image attention based on masks"
                }),
                "bias_scale": ("FLOAT", {
                    "default": 5.0, "min": 0.0, "max": 20.0, "step": 0.5,
                    "tooltip": "Strength of negative bias (suppress cross-LoRA attention)"
                }),
                "positive_bias_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Strength of positive bias (enhance same-LoRA attention)"
                }),
                "bidirectional": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply bias for both image→text and text→image directions (Flux only)"
                }),
                "use_positive_bias": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Add positive bias for same-LoRA pairs in addition to negative bias"
                }),
                "bias_blocks": (["all", "double_stream_only", "single_stream_only", "last_half_double", "none"], {
                    "default": "double_stream_only",
                    "tooltip": "Which transformer blocks to apply attention bias"
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_masks"
    CATEGORY = "FreeFuse"
    
    DESCRIPTION = """Apply Phase 1 masks to LoRAs for Phase 2 generation.
    
Connect this between your LoRA loaders and KSampler.
The masks from FreeFusePhase1Sampler control where each LoRA affects the image.

NEW: Attention Bias
When enabled, constructs soft attention bias to guide cross-attention:
- Negative bias: Suppress image regions attending to wrong LoRA's text
- Positive bias: Encourage image regions attending to correct LoRA's text
- Bidirectional: Also guide text tokens to their spatial regions"""
    
    def apply_masks(
        self,
        model,
        masks,
        freefuse_data,
        enable_token_masking=True,
        latent=None,
        # Attention bias parameters
        enable_attention_bias=True,
        bias_scale=5.0,
        positive_bias_scale=1.0,
        bidirectional=True,
        use_positive_bias=True,
        bias_blocks="double_stream_only",
    ):
        # Extract data
        mask_dict = masks.get("masks", {})
        adapters = freefuse_data.get("adapters", [])
        token_pos_maps = freefuse_data.get("token_pos_maps", {})
        
        if not mask_dict:
            print("[FreeFuse] Warning: No masks provided, returning model unchanged")
            return (model,)
        
        if not adapters:
            print("[FreeFuse] Warning: No adapters registered, returning model unchanged")
            return (model,)
        
        # Determine latent size from masks or latent input
        latent_size = self._get_latent_size(mask_dict, latent)
        if latent_size is None:
            print("[FreeFuse] Warning: Could not determine latent size")
            return (model,)
        
        # Clone the model
        model_clone = model.clone()
        
        # Detect model type
        model_type = self._detect_model_type(model_clone)
        print(f"[FreeFuse] Detected model type: {model_type}")
        
        # Get adapter name to mask mapping
        adapter_mask_map = {}
        for adapter_info in adapters:
            name = adapter_info.get("name")
            if name and name in mask_dict:
                adapter_mask_map[name] = {
                    "mask": mask_dict[name],
                    "info": adapter_info,
                }
            elif name:
                print(f"[FreeFuse] Warning: No mask found for adapter '{name}'")
        
        if not adapter_mask_map:
            print("[FreeFuse] Warning: No adapter-mask mappings, returning model unchanged")
            return (model,)
        
        # Apply masks using the hook system
        self._apply_masks_to_model(
            model_clone,
            adapter_mask_map,
            latent_size,
            token_pos_maps if enable_token_masking else None,
        )
        
        print(f"[FreeFuse] Applied masks to {len(adapter_mask_map)} adapters")
        print(f"[FreeFuse] Latent size: {latent_size}")
        
        # Apply attention bias if enabled
        if enable_attention_bias and bias_blocks != "none":
            self._apply_attention_bias(
                model_clone,
                mask_dict,
                token_pos_maps,
                latent_size,
                model_type,
                bias_scale=bias_scale,
                positive_bias_scale=positive_bias_scale,
                bidirectional=bidirectional,
                use_positive_bias=use_positive_bias,
                bias_blocks=bias_blocks,
            )
        
        return (model_clone,)
    
    def _detect_model_type(self, model_patcher) -> str:
        """Detect whether model is Flux or SDXL."""
        model = model_patcher.model
        model_class_name = model.__class__.__name__.lower()
        
        if "flux" in model_class_name:
            return "flux"
        
        # Check for Flux model structure
        if hasattr(model, 'diffusion_model'):
            dm = model.diffusion_model
            if hasattr(dm, 'double_blocks') and hasattr(dm, 'single_blocks'):
                return "flux"
        
        return "sdxl"
    
    def _apply_attention_bias(
        self,
        model_patcher,
        mask_dict: Dict[str, torch.Tensor],
        token_pos_maps: Dict[str, List[List[int]]],
        latent_size: Tuple[int, int],
        model_type: str,
        bias_scale: float,
        positive_bias_scale: float,
        bidirectional: bool,
        use_positive_bias: bool,
        bias_blocks: str,
    ):
        """Apply attention bias patches to the model."""
        # Create config
        config = AttentionBiasConfig(
            enabled=True,
            bias_scale=bias_scale,
            positive_bias_scale=positive_bias_scale,
            bidirectional=bidirectional,
            use_positive_bias=use_positive_bias,
            apply_to_blocks=bias_blocks if bias_blocks != "all" else None,
        )
        
        # Get sequence lengths
        latent_h, latent_w = latent_size
        
        if model_type == "flux":
            # For Flux, image sequence length is packed: (H/2) * (W/2)
            img_seq_len = (latent_h // 2) * (latent_w // 2)
            
            # Estimate txt_seq_len from token_pos_maps
            txt_seq_len = 512  # Default for T5
            for positions_list in token_pos_maps.values():
                if positions_list and positions_list[0]:
                    max_pos = max(positions_list[0])
                    txt_seq_len = max(txt_seq_len, max_pos + 10)
            
            # Flatten masks to (B, img_seq_len)
            lora_masks_flat = {}
            for name, mask in mask_dict.items():
                if name.startswith("_"):
                    continue
                # Ensure mask is (H, W)
                if mask.dim() == 3:
                    mask = mask[0]
                # Pack like Flux does: (H, W) -> (H/2, W/2, 4) -> (H*W/4,)
                h, w = mask.shape
                mask_packed = mask.view(h // 2, 2, w // 2, 2).permute(0, 2, 1, 3)
                mask_packed = mask_packed.reshape(-1)  # Average the 4 values or take max
                lora_masks_flat[name] = mask_packed.unsqueeze(0)  # Add batch dim
            
            # Apply attention bias patches with dynamic bias construction
            # Bias will be built at runtime based on actual txt/img sequence lengths
            apply_attention_bias_patches(
                model_patcher=model_patcher,
                attention_bias=None,  # Not used - bias built dynamically
                config=config,
                txt_seq_len=txt_seq_len,  # Estimate - actual determined at runtime
                model_type="flux",
                lora_masks=lora_masks_flat,
                token_pos_maps=token_pos_maps,
            )
            print(f"[FreeFuse] Applied attention bias for Flux "
                  f"(bias_scale={bias_scale}, positive_scale={positive_bias_scale}, "
                  f"bidirectional={bidirectional}, blocks={bias_blocks})")
        
        else:  # SDXL
            # For SDXL, use the direct SDXL bias patches
            apply_attention_bias_patches(
                model_patcher=model_patcher,
                attention_bias=None,  # SDXL computes per-layer
                config=config,
                txt_seq_len=77,  # CLIP
                model_type="sdxl",
                lora_masks=mask_dict,
                token_pos_maps=token_pos_maps,
                latent_size=latent_size,
            )
            print(f"[FreeFuse] Applied attention bias for SDXL "
                  f"(bias_scale={bias_scale}, positive_scale={positive_bias_scale})")
    
    def _get_latent_size(
        self,
        mask_dict: Dict[str, torch.Tensor],
        latent: Optional[Dict],
    ) -> Optional[Tuple[int, int]]:
        """Determine latent size from masks or latent input."""
        # Try from latent input
        if latent is not None and "samples" in latent:
            samples = latent["samples"]
            return (samples.shape[2], samples.shape[3])
        
        # Try from mask dimensions
        for mask in mask_dict.values():
            if mask.dim() == 2:
                return (mask.shape[0], mask.shape[1])
            elif mask.dim() == 3:
                return (mask.shape[1], mask.shape[2])
        
        return None
    
    def _apply_masks_to_model(
        self,
        model_patcher: comfy.model_patcher.ModelPatcher,
        adapter_mask_map: Dict[str, Dict],
        latent_size: Tuple[int, int],
        token_pos_maps: Optional[Dict[str, List[int]]],
    ):
        """Apply masks to model's LoRA layers using FreeFuse bypass hooks.
        
        The masks are applied by setting them on the OffsetBypassInjectionManagers
        stored in model_options by load_bypass_lora_for_models_fixed().
        
        Formula: output = g(f(x) + mask * h(x))
        """
        # Collect all masks
        masks = {}
        for name, data in adapter_mask_map.items():
            masks[name] = data["mask"]
        
        # Get transformer_options
        transformer_options = model_patcher.model_options.get("transformer_options", {})
        
        # Get txt_len from freefuse_data or use default
        # For Flux: txt_len is typically CLIP (77) + T5 (~256 max) 
        # In practice, the actual length depends on the prompt
        freefuse_data = transformer_options.get("freefuse_data", {})
        txt_len = freefuse_data.get("txt_len", 512)  # Default for Flux CLIP+T5
        
        # Look for multiple bypass managers (one per LoRA)
        managers_list = transformer_options.get("freefuse_bypass_managers", [])
        
        hooks_updated = 0
        if managers_list:
            for manager_info in managers_list:
                manager = manager_info.get("manager")
                if manager is not None and isinstance(manager, OffsetBypassInjectionManager):
                    manager.set_masks(masks, latent_size, txt_len)
                    hooks_updated += manager.get_hook_count()
            
            if hooks_updated > 0:
                logging.info(f"[FreeFuse] Applied masks via {len(managers_list)} bypass managers ({hooks_updated} hooks), txt_len={txt_len}")
                return
        
        # Fallback: Single manager
        manager = transformer_options.get("freefuse_bypass_manager")
        if manager is not None and isinstance(manager, OffsetBypassInjectionManager):
            manager.set_masks(masks, latent_size, txt_len)
            logging.info(f"[FreeFuse] Applied masks via single bypass manager ({manager.get_hook_count()} hooks), txt_len={txt_len}")
            return
        
        # Fallback: Look for hooks in model traversal
        hooks_found = self._find_hooks_in_model(model_patcher, masks, latent_size, txt_len)
        if hooks_found > 0:
            logging.info(f"[FreeFuse] Applied masks to {hooks_found} hooks via model traversal")
            return
        
        # Last resort: store masks for wrapper-based application
        logging.warning("[FreeFuse] No bypass hooks found, using wrapper fallback")
        self._fallback_mask_storage(model_patcher, masks, latent_size, token_pos_maps)
    
    def _find_hooks_in_model(
        self,
        model_patcher: comfy.model_patcher.ModelPatcher,
        masks: Dict[str, torch.Tensor],
        latent_size: Tuple[int, int],
        txt_len: int = 512,
    ) -> int:
        """Find MultiAdapterBypassForwardHook instances in the model and set masks."""
        hooks_found = 0
        
        # Get the diffusion model
        diffusion_model = self._get_diffusion_model(model_patcher)
        if diffusion_model is None:
            return 0
        
        # Traverse all modules looking for our hooks
        for name, module in diffusion_model.named_modules():
            # Check if this module's forward has been replaced by our hook
            forward = getattr(module, 'forward', None)
            if forward is not None:
                # Check if it's a bound method from MultiAdapterBypassForwardHook
                if hasattr(forward, '__self__'):
                    hook_self = forward.__self__
                    if isinstance(hook_self, MultiAdapterBypassForwardHook):
                        hook_self.set_masks(masks, latent_size, txt_len)
                        hooks_found += 1
        
        return hooks_found
    
    def _get_diffusion_model(
        self,
        model_patcher: comfy.model_patcher.ModelPatcher,
    ):
        """Get the actual diffusion model from the patcher."""
        model = model_patcher.model
        if hasattr(model, 'diffusion_model'):
            return model.diffusion_model
        elif hasattr(model, 'model'):
            if hasattr(model.model, 'diffusion_model'):
                return model.model.diffusion_model
            return model.model
        return model
    
    def _get_txt_len(
        self,
        model_patcher: comfy.model_patcher.ModelPatcher,
    ) -> Optional[int]:
        """Get text sequence length for Flux models."""
        # Check if already stored
        transformer_options = model_patcher.model_options.get("transformer_options", {})
        if "txt_len" in transformer_options:
            return transformer_options["txt_len"]
        
        # Check freefuse_data
        freefuse_data = transformer_options.get("freefuse_data", {})
        if "txt_len" in freefuse_data:
            return freefuse_data["txt_len"]
        
        # Default for Flux T5 + CLIP
        return None
    
    def _fallback_mask_storage(
        self,
        model_patcher: comfy.model_patcher.ModelPatcher,
        masks: Dict[str, torch.Tensor],
        latent_size: Tuple[int, int],
        token_pos_maps: Optional[Dict[str, List[int]]],
    ):
        """Fallback: Store masks in model_options for use by wrapper."""
        if "transformer_options" not in model_patcher.model_options:
            model_patcher.model_options["transformer_options"] = {}
        
        model_patcher.model_options["transformer_options"]["freefuse_masks"] = masks
        model_patcher.model_options["transformer_options"]["freefuse_latent_size"] = latent_size
        
        if token_pos_maps:
            model_patcher.model_options["transformer_options"]["freefuse_token_pos_maps"] = token_pos_maps
        
        # Set up the model wrapper for fallback mask application
        self._wrap_model_forward(model_patcher, masks, latent_size, token_pos_maps)
    
    def _wrap_model_forward(
        self,
        model_patcher: comfy.model_patcher.ModelPatcher,
        masks: Dict[str, torch.Tensor],
        latent_size: Tuple[int, int],
        token_pos_maps: Optional[Dict[str, List[int]]],
    ):
        """Wrap the model's forward pass to apply masks to LoRA outputs.
        
        ComfyUI's bypass mode applies LoRA as: output = base + lora_path
        We modify this to: output = base + mask * lora_path
        
        This is done via the sampler_cfg_function callback.
        """
        # Store configuration for the wrapper
        wrapper_config = {
            "masks": masks,
            "latent_size": latent_size,
            "token_pos_maps": token_pos_maps,
            "enabled": True,
        }
        
        model_patcher.model_options["transformer_options"]["freefuse_wrapper_config"] = wrapper_config
        
        # Use set_model_unet_function_wrapper for deeper control
        original_wrapper = model_patcher.model_options.get("model_function_wrapper")
        
        def freefuse_model_wrapper(model_function, params):
            """Wrapper that enables mask application during forward."""
            # Set up mask context
            x = params.get("input", params.get("x"))
            
            if x is not None and wrapper_config["enabled"]:
                # Store current batch info for mask application
                batch_size = x.shape[0]
                spatial_size = x.shape[2:]  # (H, W) or (T, H, W)
                
                wrapper_config["current_batch"] = batch_size
                wrapper_config["current_spatial"] = spatial_size
            
            # Call original wrapper if exists
            if original_wrapper is not None:
                return original_wrapper(model_function, params)
            else:
                return model_function(params["input"], params["timestep"], **params.get("c", {}))
        
        model_patcher.set_model_unet_function_wrapper(freefuse_model_wrapper)


class FreeFuseMaskDebug:
    """
    Debug node to inspect FreeFuse masks.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("FREEFUSE_MASKS",),
            },
            "optional": {
                "target_size": ("INT", {
                    "default": 512, "min": 64, "max": 2048,
                    "tooltip": "Target size for visualization"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview", "info")
    FUNCTION = "debug_masks"
    CATEGORY = "FreeFuse"
    
    DESCRIPTION = """Visualize FreeFuse masks for debugging."""
    
    def debug_masks(self, masks, target_size=512):
        mask_dict = masks.get("masks", {})
        sim_maps = masks.get("similarity_maps", {})
        
        # Build info string
        info_lines = ["FreeFuse Mask Debug:"]
        info_lines.append(f"  Masks: {len(mask_dict)}")
        for name, mask in mask_dict.items():
            info_lines.append(f"    {name}: shape={tuple(mask.shape)}, "
                            f"min={mask.min():.3f}, max={mask.max():.3f}")
        
        info_lines.append(f"  Similarity maps: {len(sim_maps)}")
        for name, sim in sim_maps.items():
            info_lines.append(f"    {name}: shape={tuple(sim.shape)}")
        
        info = "\n".join(info_lines)
        
        # Create visualization
        preview = self._create_debug_preview(mask_dict, target_size)
        
        return (preview, info)
    
    def _create_debug_preview(self, masks: Dict[str, torch.Tensor], target_size: int) -> torch.Tensor:
        """Create a multi-panel debug visualization."""
        if not masks:
            return torch.zeros(1, target_size, target_size, 3)
        
        colors = [
            (1.0, 0.3, 0.3),   # Red
            (0.3, 1.0, 0.3),   # Green
            (0.3, 0.3, 1.0),   # Blue
            (1.0, 1.0, 0.3),   # Yellow
            (1.0, 0.3, 1.0),   # Magenta
            (0.3, 1.0, 1.0),   # Cyan
            (0.5, 0.5, 0.5),   # Gray
        ]
        
        # Create combined visualization
        combined = torch.zeros(3, target_size, target_size)
        
        for i, (name, mask) in enumerate(masks.items()):
            color = colors[i % len(colors)]
            
            # Ensure mask is 2D
            if mask.dim() == 3:
                mask = mask[0]
            
            # Resize to target
            mask_resized = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=False
            ).squeeze()
            
            # Add colored mask
            for c in range(3):
                combined[c] += mask_resized * color[c]
        
        # Normalize
        combined = combined.clamp(0, 1)
        
        # Convert to (B, H, W, C)
        preview = combined.permute(1, 2, 0).unsqueeze(0)
        
        return preview


# Export node mappings
NODE_CLASS_MAPPINGS = {
    "FreeFuseMaskApplicator": FreeFuseMaskApplicator,
    "FreeFuseMaskDebug": FreeFuseMaskDebug,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeFuseMaskApplicator": "FreeFuse Mask Applicator",
    "FreeFuseMaskDebug": "FreeFuse Mask Debug",
}
