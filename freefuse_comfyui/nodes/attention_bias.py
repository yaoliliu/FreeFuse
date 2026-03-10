"""
FreeFuse Attention Bias Node

Standalone node for applying attention bias to models.
This provides more granular control over attention bias application.

Can be used:
1. With FreeFuse mask pipeline (automatic mask integration)
2. Standalone with manually provided masks
"""

import torch
import torch.nn.functional as F
import logging
from typing import Dict, Optional, List, Tuple

from ..freefuse_core.attention_bias import (
    construct_attention_bias,
    construct_attention_bias_sdxl,
    AttentionBiasConfig,
)
from ..freefuse_core.attention_bias_patch import (
    apply_attention_bias_patches,
)


class FreeFuseAttentionBias:
    """
    Apply attention bias to guide text-image cross-attention based on spatial masks.
    
    This node is separate from FreeFuseMaskApplicator for users who want:
    - To use attention bias without LoRA masking
    - More control over bias parameters
    - To visualize/debug attention bias separately
    
    The attention bias works by modifying the attention computation:
        Standard: softmax(QK^T / sqrt(d))
        With bias: softmax(QK^T / sqrt(d) + bias_matrix)
    
    The bias matrix contains:
    - Negative values: Suppress attention between mismatched regions/tokens
    - Positive values: Enhance attention between matched regions/tokens
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
                "latent": ("LATENT", {
                    "tooltip": "Latent for size reference"
                }),
                "bias_scale": ("FLOAT", {
                    "default": 5.0, "min": 0.0, "max": 20.0, "step": 0.5,
                    "tooltip": "Strength of negative bias (suppress cross-LoRA attention). "
                              "Higher values = stronger separation between concepts."
                }),
                "positive_bias_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Strength of positive bias (enhance same-LoRA attention). "
                              "Should typically be lower than negative scale."
                }),
                "bidirectional": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply bias for both image→text and text→image directions. "
                              "Only applies to Flux (SDXL cross-attention is inherently unidirectional)."
                }),
                "use_positive_bias": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Add positive bias for same-LoRA pairs in addition to negative bias. "
                              "Disable if you only want to suppress cross-LoRA attention."
                }),
                "bias_blocks": (["all", "double_stream_only", "single_stream_only", "last_half_double", "last_half", "none"], {
                    "default": "double_stream_only",
                    "tooltip": "Which transformer blocks to apply attention bias:\n"
                              "- all: Apply to all blocks\n"
                              "- double_stream_only: Only double-stream blocks (Flux)\n"
                              "- single_stream_only: Only single-stream blocks (Flux)\n"
                              "- last_half_double: Last half of double blocks\n"
                              "- last_half: Last half of layers (Z-Image)\n"
                              "- none: Disable attention bias"
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL", "ATTENTION_BIAS")
    RETURN_NAMES = ("model", "attention_bias")
    FUNCTION = "apply_bias"
    CATEGORY = "FreeFuse"
    
    DESCRIPTION = """Apply attention bias to guide text-image cross-attention.

This modifies the attention mechanism to:
1. SUPPRESS attention between mismatched regions/tokens (negative bias)
2. ENHANCE attention between matched regions/tokens (positive bias)

Example: If "Harry Potter" concept occupies the left side of the image,
the attention bias will:
- Discourage left-image tokens from attending to "Daiyu" text tokens
- Encourage left-image tokens to attend to "Harry Potter" text tokens

Parameters:
- bias_scale: Higher = stronger separation (5-10 is typical)
- positive_bias_scale: Usually 1-2, keeps attention focused
- bidirectional: For Flux, also constrains text→image attention
- bias_blocks: Control which layers get the bias"""
    
    def apply_bias(
        self,
        model,
        masks,
        freefuse_data,
        latent=None,
        bias_scale=5.0,
        positive_bias_scale=1.0,
        bidirectional=True,
        use_positive_bias=True,
        bias_blocks="double_stream_only",
    ):
        if bias_blocks == "none":
            print("[FreeFuse] Attention bias disabled (bias_blocks='none')")
            return (model, None)
        
        # Extract data
        mask_dict = masks.get("masks", {})
        token_pos_maps = freefuse_data.get("token_pos_maps", {})
        
        if not mask_dict:
            print("[FreeFuse] Warning: No masks provided for attention bias")
            return (model, None)
        
        if not token_pos_maps:
            print("[FreeFuse] Warning: No token positions provided for attention bias")
            return (model, None)
        
        # Get latent size
        latent_size = self._get_latent_size(mask_dict, latent)
        if latent_size is None:
            print("[FreeFuse] Warning: Could not determine latent size")
            return (model, None)
        
        # Clone model
        model_clone = model.clone()
        
        # Detect model type
        model_type = self._detect_model_type(model_clone)
        print(f"[FreeFuse] Attention bias for {model_type} model")
        
        # Create config
        config = AttentionBiasConfig(
            enabled=True,
            bias_scale=bias_scale,
            positive_bias_scale=positive_bias_scale,
            bidirectional=bidirectional,
            use_positive_bias=use_positive_bias,
            apply_to_blocks=bias_blocks if bias_blocks != "all" else None,
        )
        
        latent_h, latent_w = latent_size
        attention_bias = None
        
        if model_type == "flux":
            # For Flux, masks from generate_masks are ALREADY in packed space (H/16, W/16)
            # So img_seq_len is simply latent_h * latent_w (no additional packing needed)
            img_seq_len = latent_h * latent_w
            txt_seq_len = self._estimate_txt_seq_len(token_pos_maps)
            
            # Flatten masks - they are ALREADY in packed resolution
            # No need to re-pack them!
            lora_masks_flat = {}
            for name, mask in mask_dict.items():
                if name.startswith("_"):
                    continue
                if mask.dim() == 3:
                    mask = mask[0]
                # Masks are already in packed space (H/16, W/16), just flatten
                mask_flat = mask.reshape(-1)
                lora_masks_flat[name] = mask_flat.unsqueeze(0)
            
            attention_bias = construct_attention_bias(
                lora_masks=lora_masks_flat,
                token_pos_maps=token_pos_maps,
                txt_seq_len=txt_seq_len,
                img_seq_len=img_seq_len,
                bias_scale=bias_scale,
                positive_bias_scale=positive_bias_scale,
                bidirectional=bidirectional,
                use_positive_bias=use_positive_bias,
            )
            
            # Apply attention bias patches with dynamic bias construction
            # Pass lora_masks and token_pos_maps so bias can be built at runtime
            # with actual txt/img sequence lengths
            apply_attention_bias_patches(
                model_patcher=model_clone,
                attention_bias=attention_bias,  # For debugging/preview
                config=config,
                txt_seq_len=txt_seq_len,  # Estimate - actual determined at runtime
                model_type="flux",
                lora_masks=lora_masks_flat,
                token_pos_maps=token_pos_maps,
            )
        else:  # SDXL
            # SDXL uses per-layer bias computation
            apply_attention_bias_patches(
                model_patcher=model_clone,
                attention_bias=None,
                config=config,
                txt_seq_len=77,
                model_type="sdxl",
                lora_masks=mask_dict,
                token_pos_maps=token_pos_maps,
                latent_size=latent_size,
            )
        
        print(f"[FreeFuse] Applied attention bias: "
              f"bias_scale={bias_scale}, positive_scale={positive_bias_scale}, "
              f"bidirectional={bidirectional}, blocks={bias_blocks}")
        
        # Return attention bias for visualization/debugging
        return (model_clone, {"bias": attention_bias, "config": config.to_dict()})
    
    def _get_latent_size(self, mask_dict, latent) -> Optional[Tuple[int, int]]:
        """Get latent size from masks or latent.
        
        IMPORTANT: For Flux, masks are in packed space (H/16, W/16), while latent
        input is in original space (H/8, W/8). We should prioritize mask dimensions
        as they represent the actual spatial resolution used for attention bias.
        """
        # First try to get from masks - they are the authoritative source
        # for the spatial resolution (especially for Flux with packed dimensions)
        for mask in mask_dict.values():
            if mask.dim() == 2:
                return (mask.shape[0], mask.shape[1])
            elif mask.dim() == 3:
                return (mask.shape[1], mask.shape[2])
        
        # Fallback to latent if no masks available
        if latent is not None and "samples" in latent:
            samples = latent["samples"]
            return (samples.shape[2], samples.shape[3])
        
        return None
    
    def _detect_model_type(self, model_patcher) -> str:
        """Detect model type."""
        model = model_patcher.model
        if "flux" in model.__class__.__name__.lower():
            return "flux"
        if hasattr(model, 'diffusion_model'):
            dm = model.diffusion_model
            if hasattr(dm, 'double_blocks'):
                return "flux"
        return "sdxl"
    
    def _estimate_txt_seq_len(self, token_pos_maps) -> int:
        """Estimate text sequence length from token positions."""
        txt_seq_len = 512
        for positions_list in token_pos_maps.values():
            if positions_list and positions_list[0]:
                max_pos = max(positions_list[0])
                txt_seq_len = max(txt_seq_len, max_pos + 10)
        return txt_seq_len


class FreeFuseAttentionBiasVisualize:
    """
    Visualize attention bias matrix for debugging.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "attention_bias": ("ATTENTION_BIAS",),
            },
            "optional": {
                "target_size": ("INT", {
                    "default": 512, "min": 128, "max": 1024,
                    "tooltip": "Output image size"
                }),
                "show_region": (["full", "img_to_txt", "txt_to_img"], {
                    "default": "img_to_txt",
                    "tooltip": "Which region of the bias matrix to visualize"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview", "info")
    FUNCTION = "visualize"
    CATEGORY = "FreeFuse"
    
    DESCRIPTION = """Visualize the attention bias matrix.

Shows the bias values as a heatmap:
- Blue: Negative bias (attention suppressed)
- Red: Positive bias (attention enhanced)
- White: No bias (neutral)

Regions:
- full: Entire (txt+img) x (txt+img) matrix
- img_to_txt: Image queries → text keys (main bias region)
- txt_to_img: Text queries → image keys (bidirectional)"""
    
    def visualize(self, attention_bias, target_size=512, show_region="img_to_txt"):
        bias_tensor = attention_bias.get("bias")
        config = attention_bias.get("config", {})
        
        if bias_tensor is None:
            info = "No attention bias tensor available"
            preview = torch.ones(1, target_size, target_size, 3) * 0.5
            return (preview, info)
        
        # Get dimensions
        B, total_len, _ = bias_tensor.shape
        
        # Build info string
        info_lines = [
            "Attention Bias Info:",
            f"  Shape: {tuple(bias_tensor.shape)}",
            f"  Min: {bias_tensor.min():.3f}",
            f"  Max: {bias_tensor.max():.3f}",
            f"  Config: {config}",
        ]
        info = "\n".join(info_lines)
        
        # Extract region to visualize
        bias_2d = bias_tensor[0].float()  # Take first batch
        
        if show_region == "img_to_txt":
            # Assume txt is first half (rough estimate)
            txt_len = total_len // 2
            bias_2d = bias_2d[txt_len:, :txt_len]
        elif show_region == "txt_to_img":
            txt_len = total_len // 2
            bias_2d = bias_2d[:txt_len, txt_len:]
        # else: full
        
        # Normalize for visualization
        max_abs = max(abs(bias_2d.min()), abs(bias_2d.max()), 1e-6)
        bias_normalized = bias_2d / max_abs  # Now in [-1, 1]
        
        # Create RGB image: blue for negative, red for positive
        h, w = bias_normalized.shape
        preview = torch.zeros(h, w, 3)
        
        # Red channel: positive values
        preview[:, :, 0] = torch.clamp(bias_normalized, 0, 1)
        # Blue channel: negative values (inverted)
        preview[:, :, 2] = torch.clamp(-bias_normalized, 0, 1)
        # Green channel: zero regions
        preview[:, :, 1] = 1 - torch.abs(bias_normalized)
        
        # Resize to target
        preview = preview.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        preview = F.interpolate(preview, size=(target_size, target_size), mode='nearest')
        preview = preview.squeeze(0).permute(1, 2, 0).unsqueeze(0)  # (1, H, W, 3)
        
        return (preview, info)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "FreeFuseAttentionBias": FreeFuseAttentionBias,
    "FreeFuseAttentionBiasVisualize": FreeFuseAttentionBiasVisualize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeFuseAttentionBias": "FreeFuse Attention Bias",
    "FreeFuseAttentionBiasVisualize": "FreeFuse Attention Bias Visualize",
}
