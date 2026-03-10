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
            }
        }

    RETURN_TYPES = ("FREEFUSE_MASKS",)
    RETURN_NAMES = ("masks",)
    FUNCTION = "reassemble_masks"
    CATEGORY = "FreeFuse/Utils"

    def reassemble_masks(self, mask_00, mask_01, mask_02, mask_03, mask_04, mask_05, mask_06, mask_07, freefuse_data):
        # Collect all input masks
        input_masks = [mask_00, mask_01, mask_02, mask_03, mask_04, mask_05, mask_06, mask_07]

        # Get mask names from freefuse_data adapters (for LoRA masks)
        # These are the names that MaskApplicator will look for
        adapter_names = []
        seen = set()
        token_pos_maps = {}
        target_size = 64  # Default latent size for 512px image
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
            # Get target size from settings if available
            settings = freefuse_data.get("settings", {})
            if isinstance(settings, dict):
                # Latent size is typically image_size / 8
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

        new_mask_bank = {
            "masks": new_masks,
            "similarity_maps": {},
            "token_pos_maps": token_pos_maps,
            "metadata": {},
        }

        print(f"[FreeFuseMaskReassemble] Reassembled {len(new_masks)} masks with names: {list(new_masks.keys())}")
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
