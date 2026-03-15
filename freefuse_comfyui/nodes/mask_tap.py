"""
FreeFuse Mask Tap Utility Nodes

Utility nodes for tapping into and reassembling FreeFuse mask banks.
- FreeFuseMaskTap: Extracts individual masks from the mask bank
- FreeFuseMaskReassemble: Reassembles edited masks back into a mask bank
"""

import torch
import torch.nn.functional as F

class FreeFuseMaskTap:
    """
    Taps into the mask flow and exposes individual masks for editing.
    Outputs 8 masks in 2D format [H, W] for mask editing nodes.
    Also passes through freefuse_data to preserve adapter names.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "freefuse_data": ("FREEFUSE_DATA",),
            },
            "optional": {
                "masks": ("FREEFUSE_MASKS",),
            }
        }

    # 8 masks for editing + passthrough
    RETURN_TYPES = ("MASK", "MASK", "MASK", "MASK", "MASK", "MASK", "MASK", "MASK",
                   "FREEFUSE_DATA",)

    RETURN_NAMES = ("mask_00", "mask_01", "mask_02", "mask_03",
                   "mask_04", "mask_05", "mask_06", "mask_07",
                   "freefuse_data")

    FUNCTION = "tap_masks"
    CATEGORY = "FreeFuse/Utils"

    def tap_masks(self, freefuse_data, masks=None):
        masks_out = []

        # Get latent size from masks or use default
        latent_h, latent_w = 64, 64  # Default fallback

        # Get all mask names from the mask bank (includes __background__)
        mask_names = []
        if masks is not None and isinstance(masks, dict):
            mask_dict = masks.get("masks", {})
            if isinstance(mask_dict, dict):
                mask_names = list(mask_dict.keys())

        print(f"[FreeFuseMaskTap] Mask names from mask bank: {mask_names}")

        if masks is not None and isinstance(masks, dict):
            mask_dict = masks.get("masks", {})
            if isinstance(mask_dict, dict):
                print(f"[FreeFuseMaskTap] Found {len(mask_dict)} masks in bank")

                # Get size from first mask
                for mask in mask_dict.values():
                    if mask.dim() == 2:
                        latent_h, latent_w = mask.shape
                    elif mask.dim() == 3:
                        latent_h, latent_w = mask.shape[1], mask.shape[2]
                    elif mask.dim() == 4:
                        latent_h, latent_w = mask.shape[2], mask.shape[3]
                    break

                # Output first 8 masks from the mask bank
                for i in range(8):
                    if i < len(mask_names):
                        mask_name = mask_names[i]
                        mask = mask_dict.get(mask_name)

                        if mask is not None:
                            try:
                                # Convert to 2D spatial format [H, W]
                                if mask.dim() == 4:  # [B, C, H, W]
                                    mask_2d = mask[0, 0]
                                elif mask.dim() == 3 and mask.shape[0] == 1:  # [1, H, W]
                                    mask_2d = mask[0]
                                elif mask.dim() == 3 and mask.shape[0] > 1:  # [B, H, W]
                                    mask_2d = mask[0]
                                elif mask.dim() == 2:  # [H, W]
                                    mask_2d = mask
                                else:
                                    mask_2d = torch.zeros((latent_h, latent_w), dtype=torch.float32)

                                masks_out.append(mask_2d)
                            except Exception as e:
                                print(f"[FreeFuseMaskTap] Error processing mask {i}: {e}")
                                empty_2d = torch.zeros((latent_h, latent_w), dtype=torch.float32)
                                masks_out.append(empty_2d)
                        else:
                            empty_2d = torch.zeros((latent_h, latent_w), dtype=torch.float32)
                            masks_out.append(empty_2d)
                    else:
                        empty_2d = torch.zeros((latent_h, latent_w), dtype=torch.float32)
                        masks_out.append(empty_2d)
            else:
                print("[FreeFuseMaskTap] Invalid masks dict, using empty masks")
                for i in range(8):
                    empty_2d = torch.zeros((latent_h, latent_w), dtype=torch.float32)
                    masks_out.append(empty_2d)
        else:
            print("[FreeFuseMaskTap] No masks provided, outputting empty masks for manual editing")
            for i in range(8):
                empty_2d = torch.zeros((latent_h, latent_w), dtype=torch.float32)
                masks_out.append(empty_2d)

        return (*masks_out, freefuse_data)


class FreeFuseMaskReassemble:
    """
    Takes edited masks and puts them back into a mask bank.
    Uses freefuse_data to get correct adapter names.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_00": ("MASK",),
                "mask_01": ("MASK",),
                "mask_02": ("MASK",),
                "mask_03": ("MASK",),
                "mask_04": ("MASK",),
                "mask_05": ("MASK",),
                "mask_06": ("MASK",),
                "mask_07": ("MASK",),
                "freefuse_data": ("FREEFUSE_DATA",),
            },
            "optional": {
                "latent": ("LATENT",),
                "normalize_masks": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Make masks mutually exclusive using argmax (prevents bleeding)"
                }),
            }
        }

    RETURN_TYPES = ("FREEFUSE_MASKS",)
    RETURN_NAMES = ("masks",)
    FUNCTION = "reassemble_masks"
    CATEGORY = "FreeFuse/Utils"

    def reassemble_masks(self, mask_00, mask_01, mask_02, mask_03, mask_04, mask_05, mask_06, mask_07, freefuse_data, latent=None, normalize_masks=True):
        # Collect all input masks
        input_masks = [mask_00, mask_01, mask_02, mask_03, mask_04, mask_05, mask_06, mask_07]

        # Get mask names from freefuse_data adapters (for LoRA masks)
        # These are the names that MaskApplicator will look for
        adapter_names = []
        seen = set()
        token_pos_maps = {}
        
        # Calculate target size from latent input (most reliable method)
        # For LTX-Video: attention uses 32x32 pixel blocks, not latent resolution
        # 1024x1024 input -> 32x32 attention tokens (1024/32 = 32)
        target_size = 64  # fallback
        latent_spatial_dims = None  # Store actual latent spatial dims for reference
        attention_spatial_dims = None  # Store attention resolution

        if latent is not None and isinstance(latent, dict):
            samples = latent.get("samples")
            if samples is not None and len(samples.shape) >= 3:
                # Check if 5D (LTX-Video: [B, C, T, H, W]) or 4D (image: [B, C, H, W])
                if len(samples.shape) == 5:
                    # LTX-Video: [B, C, T, H, W]
                    latent_t = samples.shape[2]
                    latent_h = samples.shape[3]
                    latent_w = samples.shape[4]
                    latent_spatial_dims = (latent_h, latent_w)
                    
                    # 🔥 CRITICAL FIX: LTX-Video attention uses 32x32 pixel blocks
                    # For 1024x1024 input: attention = 32x32 tokens
                    # We need to infer the original input resolution from latent
                    # LTX-Video VAE compresses: spatial / 8, temporal / 8
                    # So 16x16 latent = 128x128 input... but that's wrong
                    # Actually LTX-Video uses 32x32 pixel blocks for attention
                    # Attention resolution = input_resolution / 32
                    
                    # For 1024x1024 input with 16x16 latent:
                    # - VAE compression: 1024/16 = 64 (latent represents 64px blocks)
                    # - Attention uses: 1024/32 = 32 tokens
                    # So attention resolution = latent_size * (64/32) = latent_size * 2
                    
                    attn_h = latent_h * 2  # 16 * 2 = 32
                    attn_w = latent_w * 2  # 16 * 2 = 32
                    attention_spatial_dims = (attn_h, attn_w)
                    target_size = max(attn_h, attn_w)
                    
                    print(f"[FreeFuseMaskReassemble] LTX-Video latent: T={latent_t}, spatial={latent_w}x{latent_h}")
                    print(f"[FreeFuseMaskReassemble] LTX-Video attention: {attn_w}x{attn_h} (32px blocks)")
                    print(f"[FreeFuseMaskReassemble] Target size from attention: {target_size}x{target_size}")
                elif len(samples.shape) >= 4:
                    # Image models: [B, C, H, W]
                    latent_h = samples.shape[2]
                    latent_w = samples.shape[3] if len(samples.shape) >= 4 else latent_h
                    latent_spatial_dims = (latent_h, latent_w)
                    # Mask resolution = latent / 2
                    target_size = latent_h // 2
                    print(f"[FreeFuseMaskReassemble] Target size from latent: {target_size}x{target_size} (latent_h={latent_h})")

        # Always extract adapter names from freefuse_data (regardless of target_size source)
        if freefuse_data is not None and isinstance(freefuse_data, dict):
            adapters = freefuse_data.get("adapters", [])
            for a in adapters:
                if isinstance(a, dict) and "name" in a:
                    name = a["name"]
                elif isinstance(a, str):
                    name = a
                else:
                    continue
                if name not in seen:
                    adapter_names.append(name)
                    seen.add(name)
            token_pos_maps = freefuse_data.get("token_pos_maps", {})
            
            # 🔥 FIX: Check for similarity maps to get actual attention resolution
            # This is more accurate than latent dimensions for LTX-Video
            if target_size == 64 or attention_spatial_dims is not None:
                similarity_maps = freefuse_data.get("similarity_maps", {})
                if similarity_maps:
                    first_sim = list(similarity_maps.values())[0]
                    if isinstance(first_sim, torch.Tensor):
                        # Get sequence length from similarity map
                        # Shape is (B, N, 1) where N = T * attn_H * attn_W for video
                        seq_len = first_sim.shape[1] if first_sim.dim() >= 2 else 0
                        
                        if seq_len > 0:
                            # For LTX-Video, need to handle spatio-temporal attention
                            latent_t = samples.shape[2] if len(samples.shape) == 5 else None
                            
                            # Try to factorize seq_len
                            attn_spatial = int(seq_len ** 0.5)
                            if attn_spatial * attn_spatial == seq_len:
                                # Perfect square: attention uses square grid (or T=1)
                                target_size = attn_spatial
                                attention_spatial_dims = (attn_spatial, attn_spatial)
                                print(f"[FreeFuseMaskReassemble] Using attention resolution: {target_size}x{target_size} (seq_len={seq_len})")
                            elif latent_t is not None and latent_t > 1 and seq_len % latent_t == 0:
                                # Video: seq_len = T * attn_H * attn_W
                                spatial_tokens = seq_len // latent_t
                                attn_spatial = int(spatial_tokens ** 0.5)
                                if attn_spatial * attn_spatial == spatial_tokens:
                                    target_size = attn_spatial
                                    attention_spatial_dims = (attn_spatial, attn_spatial)
                                    print(f"[FreeFuseMaskReassemble] Using video attention resolution: {target_size}x{target_size} (T={latent_t}, spatial_tokens={spatial_tokens})")
                                else:
                                    # Non-square spatial
                                    for i in range(int(spatial_tokens ** 0.5), 0, -1):
                                        if spatial_tokens % i == 0:
                                            target_size = max(i, spatial_tokens // i)
                                            attention_spatial_dims = (i, spatial_tokens // i)
                                            break
                                    print(f"[FreeFuseMaskReassemble] Using non-square video attention: {target_size}x{target_size}")
                            elif latent_spatial_dims is not None:
                                # Use latent spatial dimensions as fallback
                                target_size = max(latent_spatial_dims)
                                print(f"[FreeFuseMaskReassemble] Using latent spatial: {target_size}x{target_size}")
            
            # Only use settings for target_size if latent didn't provide it
            if target_size == 64:
                settings = freefuse_data.get("settings", {})
                if isinstance(settings, dict):
                    image_size = settings.get("image_size", 512)
                    target_size = image_size // 8

        # Also add __background__ if it exists in the workflow (common pattern)
        # It will be at position after all LoRA adapters
        if "__background__" not in adapter_names:
            adapter_names.append("__background__")

        print(f"[FreeFuseMaskReassemble] Mask names: {adapter_names}, target_size: {target_size}x{target_size}")

        # If no adapter names, create generic ones
        if not adapter_names:
            adapter_names = [f"adapter_{i:02d}" for i in range(8)]

        # Build new mask bank using adapter names from freefuse_data
        new_masks = {}
        for i, mask in enumerate(input_masks):
            if i < len(adapter_names):
                adapter_name = adapter_names[i]
                # Ensure mask is [1, H, W] format
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0)
                elif mask.dim() == 3 and mask.shape[0] != 1:
                    mask = mask[0].unsqueeze(0)
                
                # Resize to target latent size if different
                if mask.shape[1] != target_size or mask.shape[2] != target_size:
                    print(f"[FreeFuseMaskReassemble] Resizing {adapter_name} from {mask.shape[1]}x{mask.shape[2]} to {target_size}x{target_size}")
                    mask = F.interpolate(
                        mask.unsqueeze(0),
                        size=(target_size, target_size),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                
                new_masks[adapter_name] = mask

        # 🔥 Normalize masks to prevent bleeding (make mutually exclusive)
        if normalize_masks and len(new_masks) > 1:
            print(f"[FreeFuseMaskReassemble] Normalizing masks to prevent bleeding...")
            # Stack all masks: (num_masks, H, W)
            mask_names = list(new_masks.keys())
            stacked = torch.stack([new_masks[name].squeeze(0) for name in mask_names], dim=0)
            
            # Argmax to make mutually exclusive
            assignment = torch.argmax(stacked, dim=0)  # (H, W)
            
            # Create one-hot masks
            normalized = {}
            for i, name in enumerate(mask_names):
                one_hot = (assignment == i).float()
                normalized[name] = one_hot.unsqueeze(0)  # (1, H, W)
                print(f"  {name}: {(one_hot > 0).sum().item()} pixels")
            
            new_masks = normalized
            print(f"[FreeFuseMaskReassemble] Masks normalized (mutually exclusive)")

        new_mask_bank = {
            "masks": new_masks,
            "similarity_maps": {},
            "token_pos_maps": token_pos_maps,
            "metadata": {},
        }

        print(f"[FreeFuseMaskReassemble] Reassembled {len(new_masks)} masks with names: {list(new_masks.keys())}")
        
        # 🔥 DEBUG: Print mask statistics
        print(f"[FreeFuseMaskReassemble] === Mask Statistics ===")
        for name, mask in new_masks.items():
            mask_flat = mask.reshape(-1) if mask.dim() > 1 else mask
            print(f"  {name}: shape={mask.shape}, min={mask_flat.min():.4f}, max={mask_flat.max():.4f}, mean={mask_flat.mean():.4f}, active={(mask_flat > 0.5).sum().item()}/{mask_flat.numel()}")
        
        return (new_mask_bank,)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "FreeFuseMaskTap": FreeFuseMaskTap,
    "FreeFuseMaskReassemble": FreeFuseMaskReassemble,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeFuseMaskTap": "FreeFuse Mask Tap",
    "FreeFuseMaskReassemble": "FreeFuse Mask Reassemble",
}
