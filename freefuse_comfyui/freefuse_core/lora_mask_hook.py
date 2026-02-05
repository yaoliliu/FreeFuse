"""
FreeFuse LoRA Mask Hook

Applies spatial masks to LoRA outputs during bypass mode.
This implements the core FreeFuse concept: each LoRA is spatially constrained
to its corresponding region in the image.

Key features:
- Wraps ComfyUI's BypassForwardHook with mask application
- Supports multiple adapters with different masks
- Zero out LoRA results at other concepts' token positions
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class FreeFuseAdapterConfig:
    """Configuration for a FreeFuse adapter."""
    name: str
    strength: float = 1.0
    mask: Optional[torch.Tensor] = None
    token_positions: Optional[List[int]] = None


class FreeFuseMaskedBypassHook:
    """
    Custom bypass hook that applies FreeFuse spatial masks to LoRA outputs.
    
    This wraps the standard bypass computation:
        output = base_forward(x) + lora_path(x)
    
    And applies spatial masks:
        output = base_forward(x) + mask * lora_path(x)
    
    Where mask is the spatial activation mask for this adapter.
    """
    
    def __init__(
        self,
        module: nn.Module,
        adapter,
        adapter_name: str,
        multiplier: float = 1.0,
    ):
        self.module = module
        self.adapter = adapter
        self.adapter_name = adapter_name
        self.multiplier = multiplier
        self.original_forward = None
        
        # FreeFuse specific state
        self.freefuse_mask: Optional[torch.Tensor] = None
        self.mask_enabled: bool = False
        self.latent_size: Optional[Tuple[int, int]] = None
        
        # Token position masking for other adapters
        self.other_adapter_positions: Dict[str, List[int]] = {}
        
        # Setup adapter properties similar to BypassForwardHook
        module_info = self._get_module_type_info(module)
        adapter.multiplier = multiplier
        adapter.is_conv = module_info["is_conv"]
        adapter.conv_dim = module_info["conv_dim"]
        adapter.kernel_size = module_info.get("kernel_size", (1,))
        adapter.in_channels = module_info.get("in_channels", None)
        adapter.out_channels = module_info.get("out_channels", None)
        
        if module_info["is_conv"]:
            adapter.kw_dict = {
                "stride": module_info["stride"],
                "padding": module_info["padding"],
                "dilation": module_info["dilation"],
                "groups": module_info["groups"],
            }
        else:
            adapter.kw_dict = {}
    
    def _get_module_type_info(self, module: nn.Module) -> dict:
        """Get module type info (conv vs linear, conv params)."""
        info = {
            "is_conv": False,
            "conv_dim": 0,
            "stride": (1,),
            "padding": (0,),
            "dilation": (1,),
            "groups": 1,
            "kernel_size": (1,),
            "in_channels": None,
            "out_channels": None,
        }
        
        if isinstance(module, nn.Conv1d):
            info["is_conv"] = True
            info["conv_dim"] = 1
        elif isinstance(module, nn.Conv2d):
            info["is_conv"] = True
            info["conv_dim"] = 2
        elif isinstance(module, nn.Conv3d):
            info["is_conv"] = True
            info["conv_dim"] = 3
        elif isinstance(module, nn.Linear):
            info["is_conv"] = False
            info["conv_dim"] = 0
        
        if info["is_conv"]:
            info["stride"] = getattr(module, "stride", (1,) * info["conv_dim"])
            info["padding"] = getattr(module, "padding", (0,) * info["conv_dim"])
            info["dilation"] = getattr(module, "dilation", (1,) * info["conv_dim"])
            info["groups"] = getattr(module, "groups", 1)
            info["kernel_size"] = getattr(module, "kernel_size", (1,) * info["conv_dim"])
            info["in_channels"] = getattr(module, "in_channels", None)
            info["out_channels"] = getattr(module, "out_channels", None)
        
        return info
    
    def set_mask(self, mask: torch.Tensor, latent_size: Tuple[int, int]):
        """Set the spatial mask for this adapter.
        
        Args:
            mask: Spatial mask tensor of shape (H, W) with values in [0, 1]
            latent_size: (H, W) of the latent space
        """
        self.freefuse_mask = mask
        self.latent_size = latent_size
        self.mask_enabled = True
    
    def set_other_adapter_positions(self, positions: Dict[str, List[int]]):
        """Set token positions for other adapters (for zeroing out).
        
        Args:
            positions: Dict mapping adapter names to their token positions
        """
        self.other_adapter_positions = positions
    
    def enable_mask(self):
        """Enable mask application."""
        self.mask_enabled = True
    
    def disable_mask(self):
        """Disable mask application."""
        self.mask_enabled = False
    
    def _apply_spatial_mask(self, lora_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial mask to LoRA output.
        
        The mask is applied based on spatial position. For transformer attention,
        the sequence dimension corresponds to spatial positions.
        
        Args:
            lora_output: Output from LoRA path, shape varies by layer type
            x: Input tensor (for reference dimensions)
        
        Returns:
            Masked LoRA output
        """
        if self.freefuse_mask is None or not self.mask_enabled:
            return lora_output
        
        mask = self.freefuse_mask
        
        # Get output shape info
        if lora_output.dim() == 2:
            # Linear: (seq_len, hidden_dim) - typically batch is fused
            # Or: (batch, hidden_dim)
            return self._apply_mask_2d(lora_output, mask)
        
        elif lora_output.dim() == 3:
            # Transformer: (batch, seq_len, hidden_dim)
            return self._apply_mask_3d(lora_output, mask)
        
        elif lora_output.dim() == 4:
            # Conv2d: (batch, channels, height, width)
            return self._apply_mask_4d(lora_output, mask)
        
        else:
            # Unknown shape, return as-is
            logging.warning(f"[FreeFuse] Unknown lora_output shape: {lora_output.shape}")
            return lora_output
    
    def _apply_mask_2d(self, output: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply mask to 2D output."""
        seq_len, hidden_dim = output.shape
        
        if self.latent_size is not None:
            h, w = self.latent_size
            img_len = h * w
            
            if seq_len == img_len:
                # Flatten mask and apply
                mask_flat = mask.view(-1).to(output.device, output.dtype)
                return output * mask_flat.unsqueeze(-1)
        
        # Can't determine spatial mapping, return as-is
        return output
    
    def _apply_mask_3d(self, output: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply mask to 3D output (transformer attention)."""
        batch, seq_len, hidden_dim = output.shape
        
        if self.latent_size is not None:
            h, w = self.latent_size
            img_len = h * w
            
            if seq_len == img_len:
                # Pure image sequence
                mask_flat = mask.view(-1).to(output.device, output.dtype)
                mask_expanded = mask_flat.unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
                return output * mask_expanded
            
            elif seq_len > img_len:
                # Joint sequence (text + image for Flux)
                # Apply mask only to image tokens
                txt_len = seq_len - img_len
                
                mask_flat = mask.view(-1).to(output.device, output.dtype)
                
                # Create full mask: ones for text, spatial mask for image
                full_mask = torch.ones(seq_len, device=output.device, dtype=output.dtype)
                full_mask[txt_len:] = mask_flat
                
                return output * full_mask.unsqueeze(0).unsqueeze(-1)
        
        # Can't determine spatial mapping, return as-is
        return output
    
    def _apply_mask_4d(self, output: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply mask to 4D output (conv2d)."""
        batch, channels, height, width = output.shape
        
        # Resize mask to match output spatial dimensions
        if mask.shape[-2:] != (height, width):
            mask_resized = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=(height, width),
                mode='bilinear',
                align_corners=False
            ).squeeze()
        else:
            mask_resized = mask
        
        mask_resized = mask_resized.to(output.device, output.dtype)
        return output * mask_resized.unsqueeze(0).unsqueeze(0)
    
    def _apply_token_position_masking(self, lora_output: torch.Tensor) -> torch.Tensor:
        """Zero out LoRA results at other adapters' token positions.
        
        This prevents concept bleeding at the token level.
        """
        if not self.other_adapter_positions or not self.mask_enabled:
            return lora_output
        
        if lora_output.dim() < 2:
            return lora_output
        
        # Collect all positions to zero out
        all_positions = []
        for name, positions in self.other_adapter_positions.items():
            all_positions.extend(positions)
        
        if not all_positions:
            return lora_output
        
        # Zero out at specified positions
        if lora_output.dim() == 2:
            # (seq_len, hidden_dim)
            for pos in all_positions:
                if pos < lora_output.shape[0]:
                    lora_output[pos, :] = 0
        
        elif lora_output.dim() == 3:
            # (batch, seq_len, hidden_dim)
            for i in range(lora_output.shape[0]):
                for pos in all_positions:
                    if pos < lora_output.shape[1]:
                        lora_output[i, pos, :] = 0
        
        return lora_output
    
    def _freefuse_bypass_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """FreeFuse bypass forward with mask application.
        
        Computes: output = base_forward(x) + mask * lora_path(x)
        """
        # Compute base output
        base_out = self.original_forward(x, *args, **kwargs)
        
        # Compute LoRA output using adapter's h function
        lora_out = self.adapter.h(x, base_out)
        
        # Apply FreeFuse spatial mask
        if self.mask_enabled and self.freefuse_mask is not None:
            lora_out = self._apply_spatial_mask(lora_out, x)
            lora_out = self._apply_token_position_masking(lora_out)
        
        # Apply adapter's g function (usually identity) and combine
        return self.adapter.g(base_out + lora_out)
    
    def inject(self):
        """Replace module forward with FreeFuse bypass version."""
        if self.original_forward is not None:
            logging.debug(f"[FreeFuse] Already injected for {type(self.module).__name__}")
            return
        
        # Move adapter weights to module's device
        device = None
        dtype = None
        if hasattr(self.module, "weight") and self.module.weight is not None:
            device = self.module.weight.device
            dtype = self.module.weight.dtype
        
        if device is not None:
            self._move_adapter_weights(device, dtype)
        
        self.original_forward = self.module.forward
        self.module.forward = self._freefuse_bypass_forward
        
        logging.debug(
            f"[FreeFuse] Injected masked bypass for {self.adapter_name} "
            f"({type(self.module).__name__})"
        )
    
    def _move_adapter_weights(self, device, dtype=None):
        """Move adapter weights to device."""
        if isinstance(self.adapter, nn.Module):
            self.adapter.to(device=device)
            return
        
        if not hasattr(self.adapter, "weights") or self.adapter.weights is None:
            return
        
        weights = self.adapter.weights
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
            self.adapter.weights = tuple(new_weights) if isinstance(weights, tuple) else new_weights
        elif isinstance(weights, torch.Tensor):
            if dtype is not None:
                self.adapter.weights = weights.to(device=device, dtype=dtype)
            else:
                self.adapter.weights = weights.to(device=device)
    
    def eject(self):
        """Restore original module forward."""
        if self.original_forward is None:
            return
        
        self.module.forward = self.original_forward
        self.original_forward = None
        logging.debug(f"[FreeFuse] Ejected masked bypass for {self.adapter_name}")


class FreeFuseMaskManager:
    """
    Manages FreeFuse mask application across all LoRA adapters.
    
    Usage:
        manager = FreeFuseMaskManager()
        manager.add_hooks_from_model(model_patcher, adapter_mapping)
        manager.set_masks(masks, latent_size)
        manager.inject_all()
        
        # ... run inference ...
        
        manager.eject_all()
    """
    
    def __init__(self):
        self.hooks: Dict[str, List[FreeFuseMaskedBypassHook]] = {}  # adapter_name -> hooks
        self.masks: Dict[str, torch.Tensor] = {}
        self.latent_size: Optional[Tuple[int, int]] = None
        self.token_pos_maps: Dict[str, List[int]] = {}
    
    def add_hook(self, adapter_name: str, hook: FreeFuseMaskedBypassHook):
        """Add a hook for an adapter."""
        if adapter_name not in self.hooks:
            self.hooks[adapter_name] = []
        self.hooks[adapter_name].append(hook)
    
    def set_masks(self, masks: Dict[str, torch.Tensor], latent_size: Tuple[int, int]):
        """Set masks for all adapters.
        
        Args:
            masks: Dict mapping adapter names to spatial masks
            latent_size: (H, W) of the latent space
        """
        self.masks = masks
        self.latent_size = latent_size
        
        # Apply masks to all hooks
        for adapter_name, hooks in self.hooks.items():
            mask = masks.get(adapter_name)
            if mask is not None:
                for hook in hooks:
                    hook.set_mask(mask, latent_size)
    
    def set_token_positions(self, token_pos_maps: Dict[str, List[int]]):
        """Set token position maps for cross-adapter masking."""
        self.token_pos_maps = token_pos_maps
        
        # For each adapter, set the OTHER adapters' positions
        for adapter_name, hooks in self.hooks.items():
            other_positions = {
                name: pos for name, pos in token_pos_maps.items() 
                if name != adapter_name
            }
            for hook in hooks:
                hook.set_other_adapter_positions(other_positions)
    
    def enable_masks(self):
        """Enable mask application for all hooks."""
        for hooks in self.hooks.values():
            for hook in hooks:
                hook.enable_mask()
    
    def disable_masks(self):
        """Disable mask application for all hooks."""
        for hooks in self.hooks.values():
            for hook in hooks:
                hook.disable_mask()
    
    def inject_all(self):
        """Inject all hooks."""
        for hooks in self.hooks.values():
            for hook in hooks:
                hook.inject()
    
    def eject_all(self):
        """Eject all hooks."""
        for hooks in self.hooks.values():
            for hook in hooks:
                hook.eject()
    
    def get_hook_count(self) -> int:
        """Get total number of hooks."""
        return sum(len(hooks) for hooks in self.hooks.values())


def create_freefuse_injections(
    model: nn.Module,
    adapter_mapping: Dict[str, Dict],
    masks: Optional[Dict[str, torch.Tensor]] = None,
    latent_size: Optional[Tuple[int, int]] = None,
    token_pos_maps: Optional[Dict[str, List[int]]] = None,
):
    """
    Create FreeFuse injections for a model with loaded adapters.
    
    Args:
        model: The model with adapters loaded in bypass mode
        adapter_mapping: Dict mapping weight keys to adapter info:
            {weight_key: {"adapter": adapter, "name": adapter_name, "strength": float}}
        masks: Optional masks to apply immediately
        latent_size: Latent space dimensions (H, W)
        token_pos_maps: Token position maps for cross-adapter masking
    
    Returns:
        Tuple of (inject_fn, eject_fn, manager) for use with ModelPatcher
    """
    from comfy.patcher_extension import PatcherInjection
    
    manager = FreeFuseMaskManager()
    
    for key, info in adapter_mapping.items():
        adapter = info["adapter"]
        adapter_name = info["name"]
        strength = info.get("strength", 1.0)
        
        # Get module from model
        module = _get_module_by_key(model, key)
        if module is None:
            logging.warning(f"[FreeFuse] Module not found: {key}")
            continue
        
        hook = FreeFuseMaskedBypassHook(
            module=module,
            adapter=adapter,
            adapter_name=adapter_name,
            multiplier=strength,
        )
        manager.add_hook(adapter_name, hook)
    
    # Set masks if provided
    if masks is not None and latent_size is not None:
        manager.set_masks(masks, latent_size)
    
    if token_pos_maps is not None:
        manager.set_token_positions(token_pos_maps)
    
    def inject_all(model_patcher):
        manager.inject_all()
    
    def eject_all(model_patcher):
        manager.eject_all()
    
    injections = [PatcherInjection(inject=inject_all, eject=eject_all)]
    
    return injections, manager


def _get_module_by_key(model: nn.Module, key: str) -> Optional[nn.Module]:
    """Get submodule by dot-separated key."""
    # Remove .weight suffix if present
    if key.endswith(".weight"):
        key = key[:-7]
    
    parts = key.split(".")
    module = model
    
    try:
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module
    except (AttributeError, IndexError, KeyError):
        return None
