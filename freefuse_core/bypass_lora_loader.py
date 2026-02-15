"""
FreeFuse Bypass LoRA Loader - Fixed version for Flux fused weights

This module provides a fixed implementation of ComfyUI's load_bypass_lora_for_models()
that correctly handles Flux's fused QKV weights (tuple keys).

Problem in ComfyUI's original implementation:
    - Flux LoRA keys come as tuples: (key_string, (dim, offset, size))
    - Original code checks `if key in model_sd_keys` where key is a tuple
    - This always fails since model_sd_keys contains strings

Our fix:
    1. Parse tuple keys to extract actual key string and offset
    2. Support multiple adapters per module (Q/K/V share same qkv module)
    3. Apply adapter output to correct slice using offset information

Future: This fix should be submitted as a PR to ComfyUI core.
"""

import logging
import copy
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.lora
import comfy.lora_convert
import comfy.model_management
import comfy.weight_adapter
from comfy.weight_adapter.base import WeightAdapterBase, WeightAdapterTrainBase
from comfy.weight_adapter.bypass import (
    BypassForwardHook,
    get_module_type_info,
)
from comfy.patcher_extension import PatcherInjection


# Type alias
BypassAdapter = Union[WeightAdapterBase, WeightAdapterTrainBase]


class OffsetBypassForwardHook(BypassForwardHook):
    """
    Extended BypassForwardHook that supports offset for fused weights.
    
    For Flux models, Q/K/V weights are fused into a single qkv Linear layer.
    Each LoRA adapter (q_lora, k_lora, v_lora) targets a different slice of
    the output:
        - Q: output[:, :, 0:hidden_dim]
        - K: output[:, :, hidden_dim:2*hidden_dim] 
        - V: output[:, :, 2*hidden_dim:3*hidden_dim]
    
    The offset parameter (dim, start, size) specifies which slice to modify.
    """
    
    def __init__(
        self,
        module: nn.Module,
        adapter: BypassAdapter,
        multiplier: float = 1.0,
        offset: Optional[Tuple[int, int, int]] = None,
        adapter_name: str = None,
    ):
        super().__init__(module, adapter, multiplier)
        self.offset = offset  # (dim, start, size) or None
        self.adapter_name = adapter_name
    
    def _bypass_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Bypass forward with offset support for fused weights.
        
        If offset is set, the adapter output is applied only to the specified
        slice of the base output, not the entire output.
        """
        # Check for custom bypass_forward on adapter
        adapter_bypass = getattr(self.adapter, "bypass_forward", None)
        if adapter_bypass is not None:
            adapter_type = type(self.adapter)
            is_default_bypass = (
                adapter_type.bypass_forward is WeightAdapterBase.bypass_forward
                or adapter_type.bypass_forward is WeightAdapterTrainBase.bypass_forward
            )
            if not is_default_bypass:
                # Custom bypass - can't apply offset easily
                return adapter_bypass(self.original_forward, x, *args, **kwargs)
        
        # Default bypass: g(f(x) + h(x))
        base_out = self.original_forward(x, *args, **kwargs)

        # Runtime device can differ from inject-time device under low-vram offload.
        # Keep adapter tensors on the same device as activations.
        if self._adapter_needs_device_move(self.adapter, x.device):
            self._move_adapter_weights_to_device(x.device, dtype=None)

        h_out = self.adapter.h(x, base_out)
        
        # Apply offset if specified (for fused weights like Flux QKV)
        if self.offset is not None:
            dim, start, size = self.offset
            
            # Create a zero tensor matching base_out shape
            h_out_full = torch.zeros_like(base_out)
            
            # Place h_out into the correct slice
            # h_out shape should already match the slice size
            slices = [slice(None)] * base_out.dim()
            slices[dim] = slice(start, start + size)
            h_out_full[tuple(slices)] = h_out
            
            h_out = h_out_full
        
        return self.adapter.g(base_out + h_out)

    def _iter_adapter_tensors(self):
        adapter = self.adapter
        if isinstance(adapter, nn.Module):
            for p in adapter.parameters():
                yield p
            for b in adapter.buffers():
                yield b
            return

        if not hasattr(adapter, "weights") or adapter.weights is None:
            return

        weights = adapter.weights
        if isinstance(weights, (list, tuple)):
            for w in weights:
                if isinstance(w, torch.Tensor):
                    yield w
        elif isinstance(weights, torch.Tensor):
            yield weights

    def _adapter_needs_device_move(self, adapter, device: torch.device) -> bool:
        for t in self._iter_adapter_tensors():
            if t.device != device:
                return True
        return False


class MultiAdapterBypassForwardHook:
    """
    Hook that manages multiple adapters for the same module.
    
    For fused weights (Flux QKV), multiple LoRA adapters target the same
    Linear layer but different output slices. This hook collects all
    adapters and applies them together.
    
    Also supports FreeFuse spatial masks for multi-concept LoRA composition.
    
    Formula without mask: output = g(f(x) + h1(x)[offset1] + h2(x)[offset2] + ...)
    Formula with mask: output = g(f(x) + mask1 * h1(x)[offset1] + mask2 * h2(x)[offset2] + ...)
    
    Note: Spatial masks are only applied to specific layers to match diffusers implementation:
    - Flux: single_transformer_blocks (to_q/k/v, proj_mlp, proj_out), 
            transformer_blocks (to_q/k/v, to_out, ff but NOT ff_context or add_*_proj)
    - SDXL: attn1 (all), attn2 (to_q, to_out only), ff.net
            NOT attn2.to_k or attn2.to_v (they process text tokens)
    """
    
    def __init__(self, module: nn.Module, module_key: str = None):
        self.module = module
        self.module_key = module_key  # Full path like "diffusion_model.double_blocks.0.img_attn.qkv"
        self.adapters: List[Dict[str, Any]] = []  # List of {'adapter', 'strength', 'offset', 'adapter_name'}
        self.original_forward = None
        
        # Get module info once
        self.module_info = get_module_type_info(module)
        
        # FreeFuse mask support
        self.masks: Optional[Dict[str, torch.Tensor]] = None  # adapter_name -> mask
        self.latent_size: Optional[Tuple[int, int]] = None  # (H, W) of latent
        self.mask_enabled: bool = False
        self.txt_len: int = 256  # Default Flux txt length (CLIP + T5)
        
        # LoRA enable/disable support (for Phase 1 mask collection)
        self.lora_enabled: bool = True
        
        # Cache mask application decision
        self._should_apply_mask_cached: Optional[bool] = None
        self._mask_type_cached: Optional[str] = None  # 'img_only', 'img_with_text', or None
    
    def __deepcopy__(self, memo):
        # Prevent deepcopy of self.module - we want the new hook to reference 
        # the exact same module object, otherwise we patch a copy that isn't used.
        cls = self.__class__
        new_obj = cls.__new__(cls)
        memo[id(self)] = new_obj
        
        # Reference copy for the module we want to patch
        new_obj.module = self.module
        new_obj.module_key = self.module_key
        new_obj.module_info = self.module_info # dict, copy is fine
        
        # Deep copy other state
        new_obj.adapters = copy.deepcopy(self.adapters, memo)
        new_obj.original_forward = None # Reset hook state
        
        # FreeFuse state
        new_obj.masks = copy.deepcopy(self.masks, memo)
        new_obj.latent_size = copy.deepcopy(self.latent_size, memo)
        new_obj.mask_enabled = self.mask_enabled
        new_obj.txt_len = self.txt_len
        new_obj.lora_enabled = self.lora_enabled
        new_obj._should_apply_mask_cached = self._should_apply_mask_cached
        new_obj._mask_type_cached = self._mask_type_cached
        
        return new_obj

    def add_adapter(
        self,
        adapter: BypassAdapter,
        multiplier: float = 1.0,
        offset: Optional[Tuple[int, int, int]] = None,
        adapter_name: str = None,
    ):
        """Add an adapter to this module."""
        # Set module info on adapter
        adapter.multiplier = multiplier
        adapter.is_conv = self.module_info["is_conv"]
        adapter.conv_dim = self.module_info["conv_dim"]
        adapter.kernel_size = self.module_info["kernel_size"]
        adapter.in_channels = self.module_info["in_channels"]
        adapter.out_channels = self.module_info["out_channels"]
        if self.module_info["is_conv"]:
            adapter.kw_dict = {
                "stride": self.module_info["stride"],
                "padding": self.module_info["padding"],
                "dilation": self.module_info["dilation"],
                "groups": self.module_info["groups"],
            }
        else:
            adapter.kw_dict = {}
        
        self.adapters.append({
            'adapter': adapter,
            'strength': multiplier,
            'offset': offset,
            'adapter_name': adapter_name,
        })
        logging.debug(f"[OffsetBypass] Added adapter to module {type(self.module).__name__} "
                     f"(offset={offset}, name={adapter_name})")
    
    def set_masks(
        self,
        masks: Dict[str, torch.Tensor],
        latent_size: Tuple[int, int],
        txt_len: int = 256,
    ):
        """
        Set FreeFuse masks for spatial LoRA application.
        
        Args:
            masks: Dict mapping adapter_name -> spatial mask (H, W)
            latent_size: (H, W) of the latent space
            txt_len: Length of text tokens (for Flux: CLIP + T5 = 256)
        """
        self.masks = masks
        self.latent_size = latent_size
        self.mask_enabled = True
        self.txt_len = txt_len
        logging.debug(f"[OffsetBypass] Set masks for {len(masks)} adapters, latent_size={latent_size}, txt_len={txt_len}")
    
    def clear_masks(self):
        """Disable mask application."""
        self.masks = None
        self.latent_size = None
        self.mask_enabled = False
    
    def disable_lora(self):
        """Disable LoRA output (for Phase 1 mask collection)."""
        self.lora_enabled = False
    
    def enable_lora(self):
        """Enable LoRA output (for Phase 2 generation)."""
        self.lora_enabled = True
    
    def _should_apply_spatial_mask(self) -> bool:
        """
        Determine if spatial mask should be applied to this layer.
        
        This matches the diffusers implementation:
        
        For Flux:
        - single_transformer_blocks: to_q, to_k, to_v, proj_mlp, proj_out -> YES (with text)
        - transformer_blocks: to_q, to_k, to_v, to_out, ff -> YES (img only)
        - transformer_blocks: ff_context, add_q_proj, add_k_proj, add_v_proj, to_add_out -> NO
        
        For SDXL:
        - attn1: all layers -> YES
        - attn2: to_q, to_out -> YES
        - attn2: to_k, to_v -> NO (process text tokens, not image features)
        - ff.net: YES
        
        For Z-Image (Lumina2/NextDiT):
        - layers.N.attention.qkv, out_proj -> YES (z_image_unified, [txt, img] sequence)
        - layers.N.feed_forward -> YES (z_image_unified)
        - layers.N.adaLN_modulation -> NO (global, not per-token)
        
        Returns:
            True if spatial mask should be applied, False otherwise
        """
        if self._should_apply_mask_cached is not None:
            return self._should_apply_mask_cached
        
        if self.module_key is None:
            self._should_apply_mask_cached = True  # Default: apply mask
            self._mask_type_cached = 'img_with_text'
            return True
        
        key = self.module_key.lower()
        
        # Detect model type from key structure
        # IMPORTANT: Check SDXL first because SDXL keys contain "transformer_blocks" 
        # which would also match Flux patterns. SDXL has down_blocks/mid_block/up_blocks
        # as the outermost structure, while Flux has double_blocks/single_blocks.
        is_sdxl = 'down_blocks' in key or 'mid_block' in key or 'up_blocks' in key
        is_flux = ('single_blocks' in key or 'double_blocks' in key) and not is_sdxl
        # Also check for Flux-style transformer_blocks that are NOT inside SDXL structure
        if not is_sdxl and 'transformer_blocks' in key and 'attentions' not in key:
            is_flux = True
        
        # Z-Image: has 'layers.N' pattern without SDXL or Flux markers
        is_z_image = (not is_sdxl and not is_flux and 
                      'layers.' in key and 
                      ('attention' in key or 'feed_forward' in key))
        
        if is_sdxl:
            result, mask_type = self._check_sdxl_layer(key)
        elif is_flux:
            result, mask_type = self._check_flux_layer(key)
        elif is_z_image:
            result, mask_type = self._check_z_image_layer(key)
        else:
            # Unknown model, apply mask by default
            result, mask_type = True, 'img_with_text'
        
        self._should_apply_mask_cached = result
        self._mask_type_cached = mask_type
        
        if not result:
            logging.debug(f"[OffsetBypass] Skipping spatial mask for layer: {self.module_key}")

        # Debug summary for Z-Image layer mask decisions
        if is_z_image and os.environ.get("FREEFUSE_DEBUG_ZIMAGE") == "1":
            if not hasattr(self, "_debug_zimage_mask_logged"):
                self._debug_zimage_mask_logged = True
                print(f"[FreeFuse Z-Image Debug] mask_apply={result} mask_type={mask_type} layer={self.module_key}")
        
        return result
    
    def _check_flux_layer(self, key: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a Flux layer should have spatial mask applied.
        
        Returns:
            Tuple of (should_apply, mask_type) where mask_type is 'img_only' or 'img_with_text'
        """
        # Single stream blocks (single_blocks / single_transformer_blocks)
        # These process joint txt+img sequence, so mask needs to include text portion
        if 'single_block' in key or 'single_transformer_block' in key:
            # Apply to: to_q, to_k, to_v, proj_mlp, proj_out (in attn module or directly)
            if any(s in key for s in ['to_q', 'to_k', 'to_v', 'proj_mlp', 'proj_out', 'qkv']):
                return True, 'img_with_text'
            # Linear layers in single blocks
            if 'linear' in key or 'proj' in key:
                return True, 'img_with_text'
            return False, None
        
        # Double stream blocks (double_blocks / transformer_blocks)
        if 'double_block' in key or 'transformer_block' in key:
            # Context (text) path - should NOT have spatial mask, only token position masking
            # Excludes: ff_context, add_q_proj, add_k_proj, add_v_proj, to_add_out, txt_attn, txt_mlp
            if 'ff_context' in key:
                return False, None
            if any(s in key for s in ['add_q', 'add_k', 'add_v', 'to_add_out']):
                return False, None
            # txt_attn and txt_mlp are for text path, should NOT have spatial mask
            if 'txt_attn' in key or 'txt_mlp' in key:
                return False, None
            
            # Image path - should have spatial mask (img only, no text in sequence)
            # Includes: to_q, to_k, to_v, to_out, ff (but not ff_context)
            # Also handles fused qkv weights
            if any(s in key for s in ['to_q', 'to_k', 'to_v', 'to_out', 'qkv']):
                return True, 'img_only'
            if 'ff' in key and 'ff_context' not in key:
                return True, 'img_only'
            if 'img_attn' in key or 'img_mlp' in key:
                return True, 'img_only'
            
            return False, None
        
        # Default for unknown Flux layers
        return True, 'img_with_text'
    
    def _check_sdxl_layer(self, key: str) -> Tuple[bool, Optional[str]]:
        """
        Check if an SDXL UNet layer should have spatial mask applied.
        
        Returns:
            Tuple of (should_apply, mask_type)
        """
        # Cross-attention (attn2) - processes text-to-image attention
        # IMPORTANT: Check attn2 specific patterns FIRST before general attn checks
        if 'attn2' in key:
            # to_k and to_v process text tokens (77 tokens), NOT image features
            # These should NOT have spatial mask applied
            if '.to_k' in key or '.to_v' in key or 'to_k.' in key or 'to_v.' in key:
                return False, None
            # to_q and to_out process image features, should have mask
            if '.to_q' in key or '.to_out' in key or 'to_q.' in key or 'to_out.' in key:
                return True, 'img_only'
            return False, None
        
        # Self-attention (attn1) - all layers process image features
        if 'attn1' in key:
            return True, 'img_only'
        
        # FeedForward layers - process image features
        if 'ff.net' in key or 'ff_net' in key:
            return True, 'img_only'
        
        # Default: don't apply mask to unknown SDXL layers
        return False, None
    
    def _check_z_image_layer(self, key: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a Z-Image (Lumina2/NextDiT) layer should have spatial mask applied.
        
        Z-Image uses a unified [txt, img] sequence in ComfyUI (text FIRST, then image).
        Note: In diffusers, the order is [img, txt], but ComfyUI's NextDiT concatenates
        as torch.cat(feats + (x,), dim=1) where feats=(cap_feats,) and x=img.
        All transformer layers process this unified sequence.
        
        Returns:
            Tuple of (should_apply, mask_type)
        """
        # adaLN_modulation is global conditioning, not per-token
        if 'adaln_modulation' in key or 'adaln' in key:
            return False, None
        
        # Attention layers (qkv, out_proj) process the full unified sequence
        if 'attention' in key:
            if any(s in key for s in ['qkv', 'to_q', 'to_k', 'to_v', 'out_proj', 'proj']):
                return True, 'z_image_unified'
            return True, 'z_image_unified'
        
        # Feed-forward layers process the full unified sequence
        if 'feed_forward' in key or 'ff' in key:
            return True, 'z_image_unified'
        
        # Default for unknown Z-Image layers: apply
        return True, 'z_image_unified'
    
    def _get_mask_type(self) -> Optional[str]:
        """Get the mask type for this layer (after _should_apply_spatial_mask has been called)."""
        if self._mask_type_cached is None:
            self._should_apply_spatial_mask()  # Populate cache
        return self._mask_type_cached
    
    def _get_mask_for_sequence(
        self,
        adapter_name: str,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """
        Get mask resized for the current sequence length.
        
        Behavior depends on mask_type (determined by _should_apply_spatial_mask):
        - 'img_with_text': For Flux single stream blocks where txt+img are concatenated.
                          Returns mask with txt portion = 1.0 and img portion = spatial mask.
                          Layout: [txt_tokens, img_tokens]
        - 'z_image_unified': For Z-Image unified sequence where txt comes FIRST.
                            Returns mask with txt portion = 1.0 and img portion = spatial mask.
                            Layout: [txt_tokens, img_tokens]
        - 'img_only': For Flux double stream (img path) and SDXL layers.
                      Returns mask for image sequence only (no text portion).
        
        Args:
            adapter_name: Name of the adapter to get mask for
            seq_len: Total sequence length
            device: Target device
            dtype: Target dtype
            
        Returns:
            Mask tensor of shape (seq_len,) or None if not available
        """
        if not self.mask_enabled or self.masks is None:
            return None
        
        if adapter_name not in self.masks:
            return None
        
        mask = self.masks[adapter_name]
        mask_type = self._get_mask_type()
        
        # Determine img_len based on mask_type
        if mask_type == 'img_with_text':
            # Flux single stream: seq_len = txt_len + img_len
            img_len = seq_len - self.txt_len
            if img_len <= 0:
                # This might be a txt-only layer, don't apply mask
                return None
        elif mask_type == 'z_image_unified':
            # Z-Image unified: seq_len = cap_len + img_len
            # In ComfyUI's NextDiT, TEXT comes FIRST, then IMAGE
            # (padded_full_embed = torch.cat(feats + (x,), dim=1))
            # Infer img_len from mask shape
            if mask.dim() == 2:
                img_len = mask.shape[0] * mask.shape[1]
            else:
                img_len = mask.numel()
            cap_len = seq_len - img_len
            if cap_len < 0:
                # Mask is larger than sequence, fallback: entire sequence is image
                img_len = seq_len
                cap_len = 0
        else:
            # img_only or SDXL: seq_len = img_len (no text in sequence)
            img_len = seq_len
        
        # mask shape: (H, W) - spatial mask for img tokens
        if mask.dim() == 2:
            h, w = mask.shape
            mask_flat = mask.view(-1)  # (H*W,)

            # If this is a single-stream block, txt_len may be inaccurate.
            # Prefer preserving the original mask shape by inferring txt_len
            # from seq_len and the existing mask size.
            if mask_type == 'img_with_text':
                expected_img_len = img_len
                if mask_flat.shape[0] != expected_img_len:
                    inferred_txt_len = seq_len - mask_flat.shape[0]
                    if inferred_txt_len >= 0:
                        img_len = mask_flat.shape[0]
                        full_mask = torch.ones(seq_len, device=device, dtype=dtype)
                        full_mask[inferred_txt_len:inferred_txt_len + img_len] = mask_flat.to(
                            device=device, dtype=dtype
                        )
                        return full_mask
            
            # Check if we need to resize
            if mask_flat.shape[0] != img_len:
                # Resize mask to match img_len exactly
                # Find dimensions that multiply to img_len
                if self.latent_size is not None:
                    ratio = self.latent_size[1] / self.latent_size[0]  # W/H
                else:
                    ratio = w / h  # Use original mask ratio
                
                # Find new_h such that new_h * new_w = img_len
                # new_w = new_h * ratio, so new_h^2 * ratio = img_len
                new_h = int((img_len / ratio) ** 0.5)
                new_w = img_len // new_h if new_h > 0 else img_len
                
                # Adjust to get exact match
                while new_h > 1 and new_h * new_w != img_len:
                    new_h -= 1
                    new_w = img_len // new_h
                
                # Final fallback: if still not exact, use 1 x img_len
                if new_h * new_w != img_len:
                    new_h = 1
                    new_w = img_len
                
                # Resize using interpolation
                mask_2d = mask.unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
                mask_resized = F.interpolate(
                    mask_2d, size=(new_h, new_w), mode='nearest'
                )
                mask_flat = mask_resized.view(-1)  # Now exactly img_len elements
        else:
            # For non-2D masks, resize to img_len
            if mask.numel() != img_len:
                mask_flat = F.interpolate(
                    mask.view(1, 1, -1).float(), size=img_len, mode='nearest'
                ).view(-1)
            else:
                mask_flat = mask.view(-1)
        
        # Build final mask based on mask_type
        if mask_type == 'z_image_unified':
            # Z-Image unified in ComfyUI: TEXT tokens come FIRST, then IMAGE tokens
            # Layout: [txt_tokens(1.0), img_tokens(spatial mask)]
            # (ComfyUI's NextDiT: padded_full_embed = torch.cat(feats + (x,), dim=1))
            full_mask = torch.ones(seq_len, device=device, dtype=dtype)
            full_mask[cap_len:cap_len + img_len] = mask_flat[:img_len].to(device=device, dtype=dtype)
        elif mask_type == 'img_with_text':
            # Flux single stream: txt tokens come first, then img tokens
            # Layout: [txt_tokens(1.0), img_tokens(spatial mask)]
            full_mask = torch.ones(seq_len, device=device, dtype=dtype)
            full_mask[self.txt_len:self.txt_len + img_len] = mask_flat.to(device=device, dtype=dtype)
        else:
            # img_only: mask covers entire sequence (which is only image tokens)
            full_mask = mask_flat.to(device=device, dtype=dtype)

        # Debug: log resolved lengths for Z-Image once per adapter
        if mask_type == 'z_image_unified' and os.environ.get("FREEFUSE_DEBUG_ZIMAGE") == "1":
            debug_key = f"_debug_zimage_mask_shape_{adapter_name}"
            if not hasattr(self, debug_key):
                setattr(self, debug_key, True)
                print(
                    f"[FreeFuse Z-Image Debug] seq_len={seq_len} cap_len={cap_len} img_len={img_len} "
                    f"mask_shape={tuple(mask.shape)} adapter={adapter_name} layer={self.module_key}"
                )
        
        return full_mask
    
    def _bypass_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Multi-adapter bypass forward with FreeFuse mask support.
        
        Computes base output, then adds each adapter's contribution
        to the appropriate slice, optionally applying spatial masks.
        
        For fused weights (e.g., Flux QKV), each adapter targets a different
        slice of the output. The offset (dim, start, size) specifies which
        slice of the OUTPUT FEATURES to modify:
        - For Linear: output shape is [..., out_features], offset applies to last dim
        - offset=(0, 0, 3072) means: output[..., 0:3072] (Q portion of QKV)
        
        With FreeFuse masks:
        - Each adapter_name has an associated spatial mask
        - Mask is applied to h(x) before adding to base output
        - Formula: output = g(f(x) + mask1 * h1(x) + mask2 * h2(x) + ...)
        
        When lora_enabled=False (Phase 1):
        - Skip all adapter contributions, return only base model output
        - This allows collecting attention patterns from pure base model
        """
        base_out = self.original_forward(x, *args, **kwargs)
        
        # If LoRA is disabled (Phase 1), return base output directly
        if not self.lora_enabled:
            return base_out
        
        # Accumulate all adapter contributions
        total_h = torch.zeros_like(base_out)
        
        # Determine sequence length for mask (for Flux: B, seq_len, hidden)
        # Typically x shape is (B, seq_len, in_features) or (B, channels, H, W)
        seq_len = None
        if base_out.dim() == 3:
            # Transformer-style: (B, seq_len, hidden)
            seq_len = base_out.shape[1]
        elif base_out.dim() == 4:
            # Conv-style: (B, C, H, W)
            seq_len = base_out.shape[2] * base_out.shape[3]
        
        # Debug logging (only once per run)
        if not hasattr(self, '_debug_logged'):
            self._debug_logged = True
            logging.debug(f"[OffsetBypass] _bypass_forward: mask_enabled={self.mask_enabled}, "
                        f"masks={list(self.masks.keys()) if self.masks else None}, "
                        f"seq_len={seq_len}, adapters={[a['adapter_name'] for a in self.adapters]}")
        
        for adapter_info in self.adapters:
            adapter = adapter_info['adapter']
            offset = adapter_info['offset']
            adapter_name = adapter_info['adapter_name']

            # Low-VRAM mode can keep this hook alive while module/device placement changes
            # across sampling steps. Ensure adapter tensors follow the current activation
            # device before calling adapter.h(...), otherwise F.linear/F.conv can hit
            # cpu-vs-cuda mismatches.
            if self._adapter_needs_device_move(adapter, x.device):
                self._move_adapter_weights_to_device(adapter, x.device, dtype=None)

            h_out = adapter.h(x, base_out)
            
            # Apply FreeFuse mask if available AND this layer should have mask applied
            # Check _should_apply_spatial_mask to match diffusers implementation
            should_apply = self._should_apply_spatial_mask()
            if self.mask_enabled and adapter_name and seq_len is not None and should_apply:
                mask = self._get_mask_for_sequence(
                    adapter_name, seq_len, h_out.device, h_out.dtype
                )
                if mask is not None:
                    # Debug: log mask application (only once per adapter)
                    debug_key = f"_mask_debug_{adapter_name}"
                    if not hasattr(self, debug_key):
                        setattr(self, debug_key, True)
                        img_len = seq_len - self.txt_len
                        txt_mask_mean = mask[:self.txt_len].mean().item() if self.txt_len > 0 else 0
                        img_mask_mean = mask[self.txt_len:].mean().item() if img_len > 0 else 0
                        mask_type = self._get_mask_type()
                        logging.debug(f"[OffsetBypass] Mask applied for {adapter_name} at {self.module_key}: "
                                    f"seq_len={seq_len}, txt_len={self.txt_len}, img_len={img_len}, "
                                    f"txt_mask_mean={txt_mask_mean:.4f}, img_mask_mean={img_mask_mean:.4f}, "
                                    f"mask_type={mask_type}")
                    
                    # Apply mask to h_out
                    # h_out shape: (B, seq_len, features) for transformer
                    # mask shape: (seq_len,)
                    if h_out.dim() == 3:
                        # Expand mask to match: (1, seq_len, 1)
                        mask_expanded = mask.view(1, -1, 1)
                        h_out = h_out * mask_expanded
                    elif h_out.dim() == 4:
                        # Conv: (B, C, H, W), mask needs to be (1, 1, H, W)
                        h_w = h_out.shape[3]
                        h_h = h_out.shape[2]
                        mask_2d = mask.view(1, 1, h_h, h_w)
                        h_out = h_out * mask_2d
            
            if offset is not None:
                # offset = (weight_dim, start, size)
                # weight_dim=0 means output features dimension
                # For Linear/Conv, output features are in the LAST dimension of the output tensor
                weight_dim, start, size = offset
                
                # Map weight dimension to output tensor dimension
                # weight_dim=0 (output features) -> output tensor's last dim (-1)
                if weight_dim == 0:
                    output_dim = -1  # Last dimension for output features
                else:
                    # For other cases (bias, etc.), use as-is
                    output_dim = weight_dim
                
                # Build slice for the output tensor
                # h_out shape matches the slice size, so we place it directly
                slices = [slice(None)] * base_out.dim()
                slices[output_dim] = slice(start, start + size)
                
                # h_out should already have the correct size for this slice
                # due to how LoRA weights are structured
                total_h[tuple(slices)] = total_h[tuple(slices)] + h_out
            else:
                total_h = total_h + h_out
        
        # g() should be identity for LoRA, but call it for completeness
        # Use first adapter's g() - they should all be the same
        if self.adapters:
            return self.adapters[0]['adapter'].g(base_out + total_h)
        return base_out + total_h

    def _iter_adapter_tensors(self, adapter):
        """Yield tensors owned by an adapter (weights/params/buffers)."""
        if isinstance(adapter, nn.Module):
            for p in adapter.parameters():
                yield p
            for b in adapter.buffers():
                yield b
            return

        if not hasattr(adapter, "weights") or adapter.weights is None:
            return

        weights = adapter.weights
        if isinstance(weights, (list, tuple)):
            for w in weights:
                if isinstance(w, torch.Tensor):
                    yield w
        elif isinstance(weights, torch.Tensor):
            yield weights

    def _adapter_needs_device_move(self, adapter, device: torch.device) -> bool:
        """
        Return True if any adapter tensor is not on the requested device.

        In low-vram/offload scenarios, module and adapter placement can diverge:
        module activations run on CUDA while adapter tensors remain on CPU.
        """
        for t in self._iter_adapter_tensors(adapter):
            if t.device != device:
                return True
        return False
    
    def inject(self):
        """Replace module forward with bypass version."""
        # IMPORTANT: After ModelPatcher.clone()/deepcopy, this hook object can be
        # copied with a non-None `original_forward` that points to a previous
        # module instance. That stale state would make LoRA appear to have no
        # effect if we early-return here.
        current_forward = getattr(self.module, "forward", None)
        if current_forward is not None and hasattr(current_forward, "__self__"):
            if current_forward.__self__ is self:
                return  # Already injected for this module instance
        
        # Move all adapter weights to device
        device = None
        dtype = None
        if hasattr(self.module, "weight") and self.module.weight is not None:
            device = self.module.weight.device
            dtype = self.module.weight.dtype
        
        if device is not None:
            for adapter_info in self.adapters:
                self._move_adapter_weights_to_device(adapter_info['adapter'], device, dtype)
        
        self.original_forward = self.module.forward
        self.module.forward = self._bypass_forward
        logging.debug(f"[OffsetBypass] Injected multi-adapter hook for {type(self.module).__name__} "
                     f"({len(self.adapters)} adapters)")
    
    def _move_adapter_weights_to_device(self, adapter, device, dtype=None):
        """Move adapter weights to specified device."""
        if isinstance(adapter, nn.Module):
            adapter.to(device=device)
            return
        
        if not hasattr(adapter, "weights") or adapter.weights is None:
            return
        
        weights = adapter.weights
        if isinstance(weights, (list, tuple)):
            new_weights = []
            for w in weights:
                if isinstance(w, torch.Tensor):
                    if dtype is not None:
                        new_weights.append(w.to(device=device, dtype=dtype))
                    else:
                        new_weights.append(w.to(device=device))
                else:
                    new_weights.append(w)
            adapter.weights = tuple(new_weights) if isinstance(weights, tuple) else new_weights
        elif isinstance(weights, torch.Tensor):
            if dtype is not None:
                adapter.weights = weights.to(device=device, dtype=dtype)
            else:
                adapter.weights = weights.to(device=device)
    
    def eject(self):
        """Restore original module forward."""
        if self.original_forward is None:
            return
        # Only restore if we are actually the current forward hook.
        current_forward = getattr(self.module, "forward", None)
        if current_forward is not None and hasattr(current_forward, "__self__") and current_forward.__self__ is self:
            self.module.forward = self.original_forward
        self.original_forward = None
        logging.debug(f"[OffsetBypass] Ejected multi-adapter hook for {type(self.module).__name__}")


class OffsetBypassInjectionManager:
    """
    Bypass injection manager with offset support for fused weights.
    
    Key difference from ComfyUI's BypassInjectionManager:
    - Supports offset parameter for each adapter
    - Allows multiple adapters per module (collected into MultiAdapterBypassForwardHook)
    - Correctly handles Flux's fused QKV weights
    - Stores masks configuration for deferred application (works before/after inject)
    """
    
    def __init__(self):
        # Map from module_key to list of adapter dicts
        self.adapters: Dict[str, List[Dict[str, Any]]] = {}  # key -> [{'adapter', 'strength', 'offset', 'adapter_name'}, ...]
        self.hooks: List[MultiAdapterBypassForwardHook] = []
        
        # Stored mask configuration (applied when hooks are injected or set_masks is called)
        self._pending_masks: Optional[Dict[str, torch.Tensor]] = None
        self._pending_latent_size: Optional[Tuple[int, int]] = None
        self._pending_txt_len: int = 256

        # Track which model instance the current `hooks` list was built for.
        # ModelPatcher.clone() deepcopies model_options, which can leave hooks
        # pointing at modules from a previous model instance, causing LoRA to
        # appear to have no effect.
        self._hooks_model_id: Optional[int] = None
    
    def add_adapter(
        self,
        key: str,
        adapter: BypassAdapter,
        strength: float = 1.0,
        offset: Optional[Tuple[int, int, int]] = None,
        adapter_name: str = None,
    ):
        """
        Add an adapter for a specific weight key.
        
        Args:
            key: Weight key (string, not tuple)
            adapter: The weight adapter
            strength: Multiplier for adapter effect
            offset: Optional (dim, start, size) for fused weights
            adapter_name: Name for tracking (used by FreeFuse masks)
        """
        module_key = key
        if module_key.endswith(".weight"):
            module_key = module_key[:-7]
        
        if module_key not in self.adapters:
            self.adapters[module_key] = []
        
        self.adapters[module_key].append({
            'adapter': adapter,
            'strength': strength,
            'offset': offset,
            'adapter_name': adapter_name,
        })
        logging.debug(f"[OffsetBypass] Added adapter: {module_key} "
                     f"(strength={strength}, offset={offset}, name={adapter_name})")
    
    def clear_adapters(self):
        """Remove all adapters."""
        self.adapters.clear()
    
    def _get_module_by_key(self, model: nn.Module, key: str) -> Optional[nn.Module]:
        """Get a submodule by dot-separated key."""
        parts = key.split(".")
        module = model
        try:
            for part in parts:
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
            return module
        except (AttributeError, IndexError, KeyError) as e:
            logging.error(f"[OffsetBypass] Failed to find module for key {key}: {e}")
            return None
    
    def create_injections(self, model: nn.Module) -> List[PatcherInjection]:
        """
        Create PatcherInjection objects for all registered adapters.
        
        Groups multiple adapters targeting the same module into a single
        MultiAdapterBypassForwardHook.
        """
        self.hooks.clear()
        self._hooks_model_id = id(model)
        
        logging.debug(f"[OffsetBypass] Creating injections for {len(self.adapters)} module keys")
        
        for module_key, adapter_list in self.adapters.items():
            module = self._get_module_by_key(model, module_key)
            
            if module is None:
                logging.warning(f"[OffsetBypass] Module not found: {module_key}")
                continue
            
            if not hasattr(module, "weight"):
                logging.warning(f"[OffsetBypass] Module {module_key} has no weight")
                continue
            
            # Create multi-adapter hook for this module, passing module_key for mask filtering
            hook = MultiAdapterBypassForwardHook(module, module_key=module_key)
            
            for adapter_info in adapter_list:
                hook.add_adapter(
                    adapter_info['adapter'],
                    adapter_info['strength'],
                    adapter_info['offset'],
                    adapter_info['adapter_name'],
                )
            
            self.hooks.append(hook)
        
        logging.debug(f"[OffsetBypass] Created {len(self.hooks)} hooks")

        if os.environ.get("FREEFUSE_DEBUG_ZIMAGE") == "1":
            # Summarize Z-Image mask application across hooked modules
            total = 0
            apply_count = 0
            mask_type_counts: Dict[str, int] = {}
            sample = []
            for hook in self.hooks:
                total += 1
                apply = hook._should_apply_spatial_mask()
                if apply:
                    apply_count += 1
                mask_type = hook._mask_type_cached or "none"
                mask_type_counts[mask_type] = mask_type_counts.get(mask_type, 0) + 1
                if len(sample) < 10:
                    sample.append((hook.module_key, mask_type, apply))
            print(f"[FreeFuse Z-Image Debug] hooks={total} apply={apply_count} mask_types={mask_type_counts}")
            for mk, mt, ap in sample:
                print(f"[FreeFuse Z-Image Debug] sample layer={mk} mask_type={mt} apply={ap}")
        
        # IMPORTANT: Don't capture self in closures!
        # When model is cloned, model_options is deepcopied creating new manager,
        # but injections still reference the old manager via closure.
        # Instead, we retrieve the current manager from model_patcher.model_options.
        
        def inject_all(model_patcher):
            # Get the current manager from model_options (handles clone correctly)
            transformer_options = model_patcher.model_options.get("transformer_options", {})
            current_manager = transformer_options.get("freefuse_bypass_manager")
            
            if current_manager is None:
                logging.warning("[OffsetBypass] No manager found in model_options during inject")
                return
            
            logging.debug(f"[OffsetBypass] inject_all: manager id={id(current_manager)}, "
                        f"adapters={len(current_manager.adapters)}, hooks={len(current_manager.hooks)}, "
                        f"pending_masks={list(current_manager._pending_masks.keys()) if current_manager._pending_masks else None}")
            
            # IMPORTANT: ModelPatcher.clone() deepcopies model_options. If `hooks` is
            # non-empty after deepcopy, it can still reference modules from the old
            # model instance. Always ensure hooks are built for *this* model.
            if current_manager.adapters:
                current_model_id = id(model_patcher.model)
                if current_manager._hooks_model_id != current_model_id:
                    logging.debug(
                        f"[OffsetBypass] Rebuilding hooks for current model "
                        f"(prev_model_id={current_manager._hooks_model_id}, current_model_id={current_model_id}, "
                        f"modules={len(current_manager.adapters)})"
                    )
                    current_manager._rebuild_hooks(model_patcher.model)
            
            for hook in current_manager.hooks:
                hook.inject()
            
            # Apply pending masks if they were set before injection
            if current_manager._pending_masks is not None:
                logging.debug(f"[OffsetBypass] Applying pending masks to {len(current_manager.hooks)} hooks")
                for hook in current_manager.hooks:
                    hook.set_masks(
                        current_manager._pending_masks,
                        current_manager._pending_latent_size,
                        current_manager._pending_txt_len
                    )
        
        def eject_all(model_patcher):
            transformer_options = model_patcher.model_options.get("transformer_options", {})
            current_manager = transformer_options.get("freefuse_bypass_manager")
            
            if current_manager is None:
                return
            
            for hook in current_manager.hooks:
                hook.eject()
        
        return [PatcherInjection(inject=inject_all, eject=eject_all)]
    
    def _rebuild_hooks(self, model: nn.Module):
        """Rebuild hooks list from adapters (used after deepcopy)."""
        self.hooks.clear()
        self._hooks_model_id = id(model)
        
        for module_key, adapter_list in self.adapters.items():
            module = self._get_module_by_key(model, module_key)
            
            if module is None:
                logging.warning(f"[OffsetBypass] Module not found during rebuild: {module_key}")
                continue
            
            if not hasattr(module, "weight"):
                continue
            
            # Pass module_key for mask filtering
            hook = MultiAdapterBypassForwardHook(module, module_key=module_key)
            
            for adapter_info in adapter_list:
                hook.add_adapter(
                    adapter_info['adapter'],
                    adapter_info['strength'],
                    adapter_info['offset'],
                    adapter_info['adapter_name'],
                )
            
            self.hooks.append(hook)
        
        logging.debug(f"[OffsetBypass] Rebuilt {len(self.hooks)} hooks")
    
    def set_masks(
        self,
        masks: Dict[str, torch.Tensor],
        latent_size: Tuple[int, int],
        txt_len: int = 256,
    ):
        """
        Set FreeFuse masks on all hooks.
        
        This method stores masks both as pending (for hooks not yet injected)
        and applies to any existing hooks immediately.
        
        Args:
            masks: Dict mapping adapter_name -> spatial mask (H, W)
            latent_size: (H, W) of the latent space
            txt_len: Length of text tokens (for Flux: CLIP + T5)
        """
        # Store as pending for future inject() calls
        self._pending_masks = masks
        self._pending_latent_size = latent_size
        self._pending_txt_len = txt_len
        
        # Apply to existing hooks immediately (if already injected)
        hooks_updated = 0
        for hook in self.hooks:
            current_forward = getattr(hook.module, "forward", None)
            is_injected = (
                current_forward is not None
                and hasattr(current_forward, "__self__")
                and current_forward.__self__ is hook
            )
            if is_injected:
                hook.set_masks(masks, latent_size, txt_len)
                hooks_updated += 1
        
        if hooks_updated > 0:
            logging.debug(f"[OffsetBypass] Applied masks to {hooks_updated} injected hooks")
        else:
            logging.debug(f"[OffsetBypass] Stored pending masks for {len(masks)} adapters (hooks not yet injected)")
    
    def clear_masks(self):
        """Clear masks from all hooks."""
        self._pending_masks = None
        self._pending_latent_size = None
        for hook in self.hooks:
            hook.clear_masks()
    
    def disable_lora(self):
        """
        Disable LoRA output on all hooks (for Phase 1 mask collection).
        
        When disabled, adapters' h(x) contributions are skipped and only
        the base model output is returned. This allows collecting attention
        patterns from the pure base model without LoRA interference.
        """
        for hook in self.hooks:
            hook.disable_lora()
        logging.debug(f"[OffsetBypass] Disabled LoRA on {len(self.hooks)} hooks")
    
    def enable_lora(self):
        """
        Enable LoRA output on all hooks (for Phase 2 generation).
        
        Re-enables adapter contributions after Phase 1 collection.
        """
        for hook in self.hooks:
            hook.enable_lora()
        logging.debug(f"[OffsetBypass] Enabled LoRA on {len(self.hooks)} hooks")
    
    def get_hook_count(self) -> int:
        """Return number of hooks."""
        return len(self.hooks)
    
    def get_total_adapter_count(self) -> int:
        """Return total number of adapters across all modules."""
        return sum(len(adapters) for adapters in self.adapters.values())


def load_bypass_lora_for_models_fixed(
    model, 
    clip, 
    lora, 
    strength_model: float, 
    strength_clip: float,
    adapter_name: str = None,
):
    """
    Fixed version of ComfyUI's load_bypass_lora_for_models().
    
    Key fixes:
    1. Correctly parses tuple keys for Flux fused QKV weights
    2. Supports offset parameter for applying adapter to correct slice
    3. Groups multiple adapters targeting same module
    
    Args:
        model: Model patcher
        clip: CLIP patcher
        lora: LoRA state dict
        strength_model: Strength for model adapters
        strength_clip: Strength for CLIP adapters
        adapter_name: Optional name for FreeFuse tracking
    
    Returns:
        Tuple of (new_model_patcher, new_clip_patcher)
    """
    key_map = {}
    
    if model is not None:
        key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
    
    if clip is not None:
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)
    
    # Convert and load LoRA
    lora = comfy.lora_convert.convert_lora(lora)
    loaded = comfy.lora.load_lora(lora, key_map)
    
    logging.debug(f"[FixedBypassLoRA] Loaded {len(loaded)} entries")
    
    # Separate adapters from other patches
    bypass_patches = {}  # WeightAdapterBase -> bypass mode
    regular_patches = {}  # Other patches -> regular patching
    
    for key, patch_data in loaded.items():
        if isinstance(patch_data, comfy.weight_adapter.WeightAdapterBase):
            bypass_patches[key] = patch_data
        else:
            regular_patches[key] = patch_data
    
    logging.debug(f"[FixedBypassLoRA] {len(bypass_patches)} bypass adapters, "
                 f"{len(regular_patches)} regular patches")
    
    k = set()  # Loaded model keys
    k1 = set()  # Loaded clip keys
    
    if model is not None:
        new_modelpatcher = model.clone()
        
        # Apply regular patches normally
        if regular_patches:
            patched_keys = new_modelpatcher.add_patches(regular_patches, strength_model)
            k.update(patched_keys)
        
        # === FIX: Merge with existing bypass adapters ===
        # Check if there's an existing manager from previous LoRAs
        existing_manager = None
        if "transformer_options" in new_modelpatcher.model_options:
            existing_manager = new_modelpatcher.model_options["transformer_options"].get("freefuse_bypass_manager")
        
        # Create new manager that will hold ALL adapters (existing + new)
        manager = OffsetBypassInjectionManager()
        model_sd_keys = set(new_modelpatcher.model.state_dict().keys())
        
        # First, collect existing adapters from previous manager
        if existing_manager is not None:
            logging.info(f"[FixedBypassLoRA] Found existing manager with {existing_manager.get_total_adapter_count()} adapters, merging...")
            # Copy all adapters from existing manager to new manager
            for module_key, adapter_list in existing_manager.adapters.items():
                for adapter_info in adapter_list:
                    manager.add_adapter(
                        module_key,
                        adapter_info['adapter'],
                        strength=adapter_info['strength'],
                        offset=adapter_info.get('offset'),
                        adapter_name=adapter_info.get('adapter_name'),
                    )
            # Clear old injections
            new_modelpatcher.remove_injections("bypass_lora")
        
        # Now add new adapters from current LoRA
        for key, adapter in bypass_patches.items():
            # === FIX: Handle tuple keys for fused weights ===
            offset = None
            if isinstance(key, str):
                actual_key = key
            else:
                # Tuple key: (actual_key, (dim, start, size))
                actual_key = key[0]
                offset = key[1]  # (dim, start, size)
            
            if actual_key in model_sd_keys:
                manager.add_adapter(
                    actual_key, 
                    adapter, 
                    strength=strength_model,
                    offset=offset,
                    adapter_name=adapter_name,
                )
                k.add(key)
            else:
                logging.warning(f"[FixedBypassLoRA] Adapter key not in model: {actual_key}")
        
        # Create unified injections with ALL adapters
        injections = manager.create_injections(new_modelpatcher.model)
        
        if manager.get_hook_count() > 0:
            new_modelpatcher.set_injections("bypass_lora", injections)
            
            # Store manager in model_options for FreeFuse mask application
            if "transformer_options" not in new_modelpatcher.model_options:
                new_modelpatcher.model_options["transformer_options"] = {}
            
            # Support multiple LoRAs - accumulate managers
            if "freefuse_bypass_managers" not in new_modelpatcher.model_options["transformer_options"]:
                new_modelpatcher.model_options["transformer_options"]["freefuse_bypass_managers"] = []
            
            new_modelpatcher.model_options["transformer_options"]["freefuse_bypass_managers"].append({
                "manager": manager,
                "adapter_name": adapter_name,
            })
            
            # Store unified manager (contains ALL adapters from all LoRAs)
            new_modelpatcher.model_options["transformer_options"]["freefuse_bypass_manager"] = manager
            
            logging.info(f"[FixedBypassLoRA] Model: {manager.get_total_adapter_count()} adapters "
                        f"in {manager.get_hook_count()} hooks (merged)")
    else:
        new_modelpatcher = None
    
    if clip is not None:
        new_clip = clip.clone()
        
        # Apply regular patches
        if regular_patches:
            patched_keys = new_clip.add_patches(regular_patches, strength_clip)
            k1.update(patched_keys)
        
        # Apply bypass adapters
        clip_manager = OffsetBypassInjectionManager()
        clip_sd_keys = set(new_clip.cond_stage_model.state_dict().keys())
        
        for key, adapter in bypass_patches.items():
            # Handle tuple keys
            offset = None
            if isinstance(key, str):
                actual_key = key
            else:
                actual_key = key[0]
                offset = key[1]
            
            if actual_key in clip_sd_keys:
                clip_manager.add_adapter(
                    actual_key,
                    adapter,
                    strength=strength_clip,
                    offset=offset,
                    adapter_name=adapter_name,
                )
                k1.add(key)
        
        clip_injections = clip_manager.create_injections(new_clip.cond_stage_model)
        if clip_manager.get_hook_count() > 0:
            new_clip.patcher.set_injections("bypass_lora", clip_injections)
            logging.info(f"[FixedBypassLoRA] CLIP: {clip_manager.get_total_adapter_count()} adapters "
                        f"in {clip_manager.get_hook_count()} hooks")
    else:
        new_clip = None
    
    # Log any unloaded keys
    for x in loaded:
        if (x not in k) and (x not in k1):
            patch_data = loaded[x]
            patch_type = type(patch_data).__name__
            if isinstance(patch_data, tuple):
                patch_type = f"tuple({patch_data[0]})"
            # Extract actual key for logging
            if isinstance(x, tuple):
                log_key = f"{x[0]} (offset={x[1]})"
            else:
                log_key = x
            logging.warning(f"[FixedBypassLoRA] NOT LOADED: {log_key} (type={patch_type})")
    
    return (new_modelpatcher, new_clip)


__all__ = [
    "OffsetBypassForwardHook",
    "MultiAdapterBypassForwardHook",
    "OffsetBypassInjectionManager",
    "load_bypass_lora_for_models_fixed",
]
