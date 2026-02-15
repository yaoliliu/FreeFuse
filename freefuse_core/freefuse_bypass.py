"""
FreeFuse Bypass Mode Implementation

Custom bypass hooks that apply spatial masks to LoRA outputs during forward pass.
This enables the core FreeFuse concept: each LoRA is spatially constrained
to its corresponding region in the image.

Based on ComfyUI's weight_adapter.bypass module, extended with:
- Spatial mask application to LoRA outputs
- Token position masking to prevent concept bleeding
- Multi-adapter mask management

Key formula:
    Original bypass: output = g(f(x) + h(x))
    FreeFuse bypass: output = g(f(x) + mask * h(x))

Where mask is the spatial activation mask for each adapter.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import ComfyUI's bypass infrastructure
from comfy.weight_adapter.bypass import (
    BypassForwardHook,
    BypassInjectionManager,
    get_module_type_info,
)
from comfy.weight_adapter.base import WeightAdapterBase, WeightAdapterTrainBase
from comfy.patcher_extension import PatcherInjection


# Type alias
BypassAdapter = Union[WeightAdapterBase, WeightAdapterTrainBase]


class FreeFuseBypassForwardHook(BypassForwardHook):
    """
    FreeFuse version of BypassForwardHook with spatial mask support.
    
    Extends ComfyUI's BypassForwardHook to apply spatial masks to LoRA outputs,
    enabling multi-concept LoRA composition where each LoRA affects only its
    designated spatial region.
    
    Key features:
    - Applies spatial masks to h(x) output before adding to base
    - Supports token position masking to zero out LoRA at other concepts' positions
    - Automatically handles different tensor shapes (2D, 3D, 4D)
    """
    
    def __init__(
        self,
        module: nn.Module,
        adapter: BypassAdapter,
        multiplier: float = 1.0,
        adapter_name: str = None,
    ):
        super().__init__(module, adapter, multiplier)
        self.adapter_name = adapter_name
        
        # FreeFuse specific state
        self.freefuse_mask: Optional[torch.Tensor] = None
        self.latent_size: Optional[Tuple[int, int]] = None
        self.mask_enabled: bool = True
        
        # Token position masking for other adapters
        self.other_adapter_positions: Dict[str, List[int]] = {}
        
        # Track txt_len for joint attention (Flux)
        self.txt_len: Optional[int] = None
    
    def set_mask(self, mask: torch.Tensor, latent_size: Tuple[int, int]):
        """
        Set the spatial mask for this adapter.
        
        Args:
            mask: Spatial mask tensor of shape (H, W) with values in [0, 1]
            latent_size: (H, W) of the latent space
        """
        self.freefuse_mask = mask
        self.latent_size = latent_size
        logging.debug(f"[FreeFuse] Set mask for {self.adapter_name}: "
                     f"shape={mask.shape}, latent_size={latent_size}")
    
    def set_other_adapter_positions(self, positions: Dict[str, List[int]]):
        """
        Set token positions for other adapters (for zeroing out).
        
        This prevents LoRA from affecting tokens that belong to other concepts.
        
        Args:
            positions: Dict mapping adapter names to their token positions
        """
        self.other_adapter_positions = positions
    
    def set_txt_len(self, txt_len: int):
        """Set text sequence length for joint attention models (Flux)."""
        self.txt_len = txt_len
    
    def enable_mask(self):
        """Enable mask application."""
        self.mask_enabled = True
    
    def disable_mask(self):
        """Disable mask application."""
        self.mask_enabled = False
    
    def _bypass_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        FreeFuse bypass forward: g(f(x) + mask * h(x))
        
        Applies spatial mask to the LoRA contribution before adding to base output.
        """
        # Check if adapter has custom bypass_forward
        adapter_bypass = getattr(self.adapter, "bypass_forward", None)
        if adapter_bypass is not None:
            adapter_type = type(self.adapter)
            is_default_bypass = (
                adapter_type.bypass_forward is WeightAdapterBase.bypass_forward
                or adapter_type.bypass_forward is WeightAdapterTrainBase.bypass_forward
            )
            if not is_default_bypass:
                # Custom bypass - we can't easily apply masks here
                # Fall back to original behavior
                return adapter_bypass(self.original_forward, x, *args, **kwargs)
        
        # Default FreeFuse bypass: g(f(x) + mask * h(x))
        base_out = self.original_forward(x, *args, **kwargs)

        # Low-VRAM/offload can move modules between CPU/GPU after hook injection.
        # Ensure adapter tensors follow the current activation device before h(x).
        if self._adapter_needs_device_move(x.device):
            self._move_adapter_weights_to_device(x.device)

        h_out = self.adapter.h(x, base_out)
        
        # === FreeFuse: Apply spatial mask to LoRA output ===
        if self.mask_enabled and self.freefuse_mask is not None:
            h_out = self._apply_spatial_mask(h_out, x)
        
        # === FreeFuse: Zero out at other adapters' token positions ===
        if self.mask_enabled and self.other_adapter_positions:
            h_out = self._apply_token_position_masking(h_out)
        
        return self.adapter.g(base_out + h_out)

    def _iter_adapter_tensors(self):
        """Yield tensors owned by this hook's adapter."""
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

    def _adapter_needs_device_move(self, device: torch.device) -> bool:
        for t in self._iter_adapter_tensors():
            if t.device != device:
                return True
        return False

    def _move_adapter_weights_to_device(self, device: torch.device):
        """Move adapter tensors to device while keeping their current dtype."""
        adapter = self.adapter
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
                    new_weights.append(w.to(device=device))
                else:
                    new_weights.append(w)
            adapter.weights = tuple(new_weights) if isinstance(weights, tuple) else new_weights
        elif isinstance(weights, torch.Tensor):
            adapter.weights = weights.to(device=device)
    
    def _apply_spatial_mask(self, h_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial mask to LoRA output based on tensor shape.
        
        Handles:
        - 2D: (seq_len, hidden_dim) - flattened spatial
        - 3D: (batch, seq_len, hidden_dim) - transformer attention
        - 4D: (batch, channels, height, width) - conv layers
        
        Args:
            h_out: LoRA output tensor
            x: Input tensor (for reference)
        
        Returns:
            Masked LoRA output
        """
        if self.freefuse_mask is None or self.latent_size is None:
            return h_out
        
        mask = self.freefuse_mask
        h, w = self.latent_size
        img_len = h * w
        
        # Move mask to same device/dtype
        mask = mask.to(device=h_out.device, dtype=h_out.dtype)
        
        if h_out.dim() == 2:
            # (seq_len, hidden_dim) - typically batch fused or single sample
            return self._apply_mask_2d(h_out, mask, img_len)
        
        elif h_out.dim() == 3:
            # (batch, seq_len, hidden_dim) - transformer attention
            return self._apply_mask_3d(h_out, mask, img_len)
        
        elif h_out.dim() == 4:
            # (batch, channels, height, width) - conv2d
            return self._apply_mask_4d(h_out, mask)
        
        else:
            logging.warning(f"[FreeFuse] Unknown h_out shape: {h_out.shape}, "
                          f"skipping mask application")
            return h_out
    
    def _apply_mask_2d(self, h_out: torch.Tensor, mask: torch.Tensor, 
                       img_len: int) -> torch.Tensor:
        """Apply mask to 2D output (seq_len, hidden_dim)."""
        seq_len, hidden_dim = h_out.shape
        h, w = self.latent_size
        
        if seq_len == img_len:
            # Pure image sequence - reshape mask and apply
            mask_flat = mask.view(-1)
            if mask_flat.shape[0] != seq_len:
                mask_flat = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0),
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze().view(-1)
            return h_out * mask_flat.unsqueeze(-1)
        
        elif seq_len > img_len and self.txt_len is not None:
            # Joint sequence (text + image for Flux)
            txt_len = self.txt_len
            if seq_len == txt_len + img_len:
                mask_flat = mask.view(-1)
                # Apply mask only to image part, keep text unchanged
                h_out_img = h_out[txt_len:, :] * mask_flat.unsqueeze(-1)
                h_out = torch.cat([h_out[:txt_len, :], h_out_img], dim=0)
                return h_out
        
        # Can't determine spatial mapping
        logging.debug(f"[FreeFuse] 2D mask: seq_len={seq_len} != img_len={img_len}")
        return h_out
    
    def _apply_mask_3d(self, h_out: torch.Tensor, mask: torch.Tensor,
                       img_len: int) -> torch.Tensor:
        """Apply mask to 3D output (batch, seq_len, hidden_dim)."""
        batch, seq_len, hidden_dim = h_out.shape
        h, w = self.latent_size
        
        if seq_len == img_len:
            # Pure image sequence
            mask_flat = mask.view(-1)
            if mask_flat.shape[0] != seq_len:
                mask_flat = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0),
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze().view(-1)
            # (1, seq_len, 1) for broadcasting
            mask_expanded = mask_flat.unsqueeze(0).unsqueeze(-1)
            return h_out * mask_expanded
        
        elif seq_len > img_len and self.txt_len is not None:
            # Joint sequence (text + image)
            txt_len = self.txt_len
            if seq_len == txt_len + img_len:
                mask_flat = mask.view(-1)
                # Create full mask: ones for text, spatial mask for image
                full_mask = torch.ones(seq_len, device=h_out.device, dtype=h_out.dtype)
                full_mask[txt_len:] = mask_flat
                # (1, seq_len, 1) for broadcasting
                mask_expanded = full_mask.unsqueeze(0).unsqueeze(-1)
                return h_out * mask_expanded
        
        logging.debug(f"[FreeFuse] 3D mask: seq_len={seq_len} != img_len={img_len}")
        return h_out
    
    def _apply_mask_4d(self, h_out: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply mask to 4D output (batch, channels, height, width)."""
        batch, channels, height, width = h_out.shape
        
        # Resize mask to match output spatial dimensions
        if mask.shape[-2:] != (height, width):
            mask_resized = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=(height, width),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)
        else:
            mask_resized = mask
        
        # (1, 1, H, W) for broadcasting across batch and channels
        mask_expanded = mask_resized.unsqueeze(0).unsqueeze(0)
        return h_out * mask_expanded
    
    def _apply_token_position_masking(self, h_out: torch.Tensor) -> torch.Tensor:
        """
        Zero out LoRA results at other adapters' token positions.
        
        This prevents concept bleeding at the token level.
        """
        if not self.other_adapter_positions:
            return h_out
        
        # Collect all positions to zero out
        all_positions = []
        for name, positions in self.other_adapter_positions.items():
            all_positions.extend(positions)
        
        if not all_positions:
            return h_out
        
        # Zero out at specified positions
        if h_out.dim() == 2:
            # (seq_len, hidden_dim)
            for pos in all_positions:
                if 0 <= pos < h_out.shape[0]:
                    h_out[pos, :] = 0
        
        elif h_out.dim() == 3:
            # (batch, seq_len, hidden_dim)
            for pos in all_positions:
                if 0 <= pos < h_out.shape[1]:
                    h_out[:, pos, :] = 0
        
        return h_out


class FreeFuseBypassInjectionManager(BypassInjectionManager):
    """
    FreeFuse version of BypassInjectionManager.
    
    Manages bypass mode injection with spatial mask support.
    Uses FreeFuseBypassForwardHook instead of BypassForwardHook.
    
    Usage:
        manager = FreeFuseBypassInjectionManager()
        manager.add_adapter("diffusion_model.double_blocks.0.img_attn.qkv", 
                           adapter, strength=0.8, adapter_name="character1")
        
        injections = manager.create_injections(model)
        model_patcher.set_injections("freefuse_bypass", injections)
        
        # After Phase 1, set masks
        manager.set_masks(masks_dict, latent_size=(64, 64))
        manager.set_token_positions(token_pos_maps)
    """
    
    def __init__(self):
        super().__init__()
        # Override hooks type for FreeFuse
        self.hooks: List[FreeFuseBypassForwardHook] = []
        self.adapter_to_hooks: Dict[str, List[FreeFuseBypassForwardHook]] = {}
        self.masks: Dict[str, torch.Tensor] = {}
        self.latent_size: Optional[Tuple[int, int]] = None
        self.token_pos_maps: Dict[str, List[int]] = {}
        self.txt_len: Optional[int] = None
    
    def add_adapter(
        self,
        key: str,
        adapter: BypassAdapter,
        strength: float = 1.0,
        adapter_name: str = None,
    ):
        """
        Add an adapter for a specific weight key.
        
        Args:
            key: Weight key (e.g., "diffusion_model.layers.0.self_attn.q_proj.weight")
            adapter: The weight adapter (LoRAAdapter, etc.)
            strength: Multiplier for adapter effect
            adapter_name: Name used for mask lookup (e.g., "character1")
        """
        # Remove .weight suffix if present
        module_key = key
        if module_key.endswith(".weight"):
            module_key = module_key[:-7]
        
        # Store with adapter_name for later mask application
        self.adapters[module_key] = (adapter, strength, adapter_name)
        
        logging.debug(f"[FreeFuse] Added adapter: {module_key} "
                     f"(name={adapter_name}, strength={strength})")
    
    def create_injections(self, model: nn.Module) -> List[PatcherInjection]:
        """
        Create PatcherInjection objects for all registered adapters.
        
        Uses FreeFuseBypassForwardHook instead of standard BypassForwardHook.
        """
        self.hooks.clear()
        self.adapter_to_hooks.clear()
        
        logging.debug(f"[FreeFuse] Creating injections for {len(self.adapters)} adapters")
        
        for key, value in self.adapters.items():
            adapter, strength, adapter_name = value
            
            module = self._get_module_by_key(model, key)
            if module is None:
                logging.warning(f"[FreeFuse] Module not found: {key}")
                continue
            
            if not hasattr(module, "weight"):
                logging.warning(f"[FreeFuse] Module {key} has no weight")
                continue
            
            # Create FreeFuse hook instead of standard hook
            hook = FreeFuseBypassForwardHook(
                module, adapter, 
                multiplier=strength,
                adapter_name=adapter_name
            )
            self.hooks.append(hook)
            
            # Track hooks by adapter name for mask application
            if adapter_name:
                if adapter_name not in self.adapter_to_hooks:
                    self.adapter_to_hooks[adapter_name] = []
                self.adapter_to_hooks[adapter_name].append(hook)
        
        logging.debug(f"[FreeFuse] Created {len(self.hooks)} hooks")
        
        # Create injection
        def inject_all(model_patcher):
            logging.debug(f"[FreeFuse] Injecting {len(self.hooks)} hooks")
            for hook in self.hooks:
                hook.inject()
        
        def eject_all(model_patcher):
            logging.debug(f"[FreeFuse] Ejecting {len(self.hooks)} hooks")
            for hook in self.hooks:
                hook.eject()
        
        return [PatcherInjection(inject=inject_all, eject=eject_all)]
    
    def set_masks(self, masks: Dict[str, torch.Tensor], latent_size: Tuple[int, int]):
        """
        Set spatial masks for all adapters.
        
        Args:
            masks: Dict mapping adapter names to spatial mask tensors (H, W)
            latent_size: (H, W) of the latent space
        """
        self.masks = masks
        self.latent_size = latent_size
        
        # Apply masks to hooks
        for adapter_name, hooks in self.adapter_to_hooks.items():
            mask = masks.get(adapter_name)
            if mask is not None:
                for hook in hooks:
                    hook.set_mask(mask, latent_size)
                logging.debug(f"[FreeFuse] Set mask for {adapter_name}: {len(hooks)} hooks")
            else:
                logging.warning(f"[FreeFuse] No mask found for adapter: {adapter_name}")
    
    def set_token_positions(self, token_pos_maps: Dict[str, List[int]]):
        """
        Set token position maps for cross-adapter masking.
        
        Each adapter will zero out its LoRA output at other adapters' token positions.
        
        Args:
            token_pos_maps: Dict mapping adapter names to their token positions
        """
        self.token_pos_maps = token_pos_maps
        
        # For each adapter, set the OTHER adapters' positions
        for adapter_name, hooks in self.adapter_to_hooks.items():
            other_positions = {
                name: pos_list[0] if pos_list else []
                for name, pos_list in token_pos_maps.items()
                if name != adapter_name and not name.startswith("__")
            }
            for hook in hooks:
                hook.set_other_adapter_positions(other_positions)
    
    def set_txt_len(self, txt_len: int):
        """Set text sequence length for joint attention models."""
        self.txt_len = txt_len
        for hook in self.hooks:
            hook.set_txt_len(txt_len)
    
    def enable_masks(self):
        """Enable mask application for all hooks."""
        for hook in self.hooks:
            hook.enable_mask()
    
    def disable_masks(self):
        """Disable mask application for all hooks."""
        for hook in self.hooks:
            hook.disable_mask()
    
    def get_adapter_names(self) -> List[str]:
        """Get list of registered adapter names."""
        return list(self.adapter_to_hooks.keys())


def create_freefuse_bypass_from_model(
    model,
    freefuse_data: Dict,
    masks: Dict[str, torch.Tensor] = None,
    latent_size: Tuple[int, int] = None,
) -> Tuple[List[PatcherInjection], FreeFuseBypassInjectionManager]:
    """
    Create FreeFuse bypass injections from a model that already has LoRAs loaded.
    
    This function inspects the model for existing bypass adapters and wraps them
    with FreeFuse's mask-aware hooks.
    
    Args:
        model: ComfyUI ModelPatcher with LoRAs loaded in bypass mode
        freefuse_data: FreeFuse data dict containing adapter info
        masks: Optional masks to apply immediately
        latent_size: Latent space dimensions
    
    Returns:
        Tuple of (injections_list, manager)
    """
    manager = FreeFuseBypassInjectionManager()
    
    adapters_info = freefuse_data.get("adapters", [])
    adapter_keys = freefuse_data.get("adapter_keys", {})
    
    # Get the model's existing patches
    if hasattr(model, 'patches'):
        patches = model.patches
    else:
        patches = {}
    
    # Look for bypass adapters in model options
    model_options = getattr(model, 'model_options', {})
    
    # Find injections that were set by load_bypass_lora_for_models
    existing_injections = model_options.get('injections', {})
    
    logging.info(f"[FreeFuse] Creating bypass from model with {len(adapters_info)} adapters")
    
    # For each adapter, find its keys and create FreeFuse hooks
    for adapter_info in adapters_info:
        adapter_name = adapter_info.get("name")
        keys = adapter_keys.get(adapter_name, [])
        
        # Get adapter patches from model
        for key in keys:
            if key in patches:
                patch_list = patches[key]
                for patch in patch_list:
                    if len(patch) >= 2:
                        patch_data = patch[1]
                        if isinstance(patch_data, (WeightAdapterBase, WeightAdapterTrainBase)):
                            strength = patch[0] if len(patch) > 0 else 1.0
                            manager.add_adapter(
                                key, 
                                patch_data, 
                                strength=strength,
                                adapter_name=adapter_name
                            )
    
    # Create injections
    diffusion_model = model.model.diffusion_model if hasattr(model.model, 'diffusion_model') else model.model
    injections = manager.create_injections(diffusion_model)
    
    # Apply masks if provided
    if masks is not None and latent_size is not None:
        manager.set_masks(masks, latent_size)
    
    # Apply token positions if available
    token_pos_maps = freefuse_data.get("token_pos_maps", {})
    if token_pos_maps:
        manager.set_token_positions(token_pos_maps)
    
    return injections, manager


__all__ = [
    "FreeFuseBypassForwardHook",
    "FreeFuseBypassInjectionManager",
    "create_freefuse_bypass_from_model",
]
