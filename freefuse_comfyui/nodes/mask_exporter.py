"""
FreeFuse Mask Export Node

Exports FreeFuse masks as individual image files for external use.
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
from typing import Dict, List


class FreeFuseMaskExporter:
    """
    Export FreeFuse masks as PNG files.
    
    Saves each mask as a separate PNG file for use in external tools
    or for manual inspection.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("FREEFUSE_MASKS", {
                    "tooltip": "FreeFuse masks data"
                }),
                "output_dir": ("STRING", {
                    "default": "output/freefuse_masks",
                    "tooltip": "Directory to save mask images"
                }),
            },
            "optional": {
                "save_format": (["png", "jpg", "webp"], {
                    "default": "png",
                    "tooltip": "Image format for export"
                }),
                "resize_to": ("INT", {
                    "default": 512, "min": 64, "max": 2048, "step": 64,
                    "tooltip": "Resize masks to this size (0 = original)"
                }),
                "save_combined": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Also save combined color-coded preview"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("export_info", "preview")
    FUNCTION = "export_masks"
    CATEGORY = "FreeFuse/Utils"

    DESCRIPTION = """Export FreeFuse masks as image files.

Saves each mask as a separate PNG/JPG file for external use.
Also returns a preview image for immediate viewing.

Output:
- export_info: Path to saved files
- preview: Combined color-coded preview
"""

    def export_masks(self, masks, output_dir="output/freefuse_masks",
                     save_format="png", resize_to=512, save_combined=True):

        mask_dict = masks.get("masks", {})
        sim_maps = masks.get("similarity_maps", {})

        if not mask_dict:
            return ("No masks to export", torch.zeros(1, 512, 512, 3))

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        saved_files = []
        print(f"\n[FreeFuse Mask Exporter] Exporting {len(mask_dict)} masks to {output_dir}")

        # Determine output size
        first_mask = next(iter(mask_dict.values()))
        if first_mask.dim() == 2:
            orig_h, orig_w = first_mask.shape
        elif first_mask.dim() == 3:
            orig_h, orig_w = first_mask.shape[1], first_mask.shape[2]
        else:
            orig_h, orig_w = 512, 512

        out_h = out_w = resize_to if resize_to > 0 else orig_h

        # Export each mask
        colors = {
            "elle": (255, 0, 0),      # Red
            "fox": (0, 255, 0),       # Green
            "__background__": (128, 128, 128),  # Gray
        }

        for name, mask in mask_dict.items():
            try:
                # Convert to numpy
                mask_cpu = mask.detach().cpu().float()
                if mask_cpu.dim() == 3:
                    mask_cpu = mask_cpu[0]  # Remove batch dim
                if mask_cpu.dim() == 3 and mask_cpu.shape[0] == 1:
                    mask_cpu = mask_cpu[0]  # Remove channel dim

                # Resize if needed
                if resize_to > 0 and (mask_cpu.shape[0] != out_h or mask_cpu.shape[1] != out_w):
                    mask_cpu = F.interpolate(
                        mask_cpu.unsqueeze(0).unsqueeze(0),
                        size=(out_h, out_w),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).squeeze(0)

                mask_np = (mask_cpu.numpy() * 255).clip(0, 255).astype(np.uint8)

                # Save as grayscale
                mask_img = Image.fromarray(mask_np, mode='L')
                filename = f"{name.replace('/', '_').replace('\\', '_')}.{save_format}"
                filepath = os.path.join(output_dir, filename)
                mask_img.save(filepath)
                saved_files.append(filepath)
                print(f"  Saved: {filepath}")

                # Also save colored version
                if save_combined:
                    color = colors.get(name, (255, 255, 255))
                    colored_np = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                    for c in range(3):
                        colored_np[:, :, c] = (mask_np * color[c] / 255).astype(np.uint8)
                    colored_img = Image.fromarray(colored_np, mode='RGB')
                    color_filepath = os.path.join(output_dir, f"{name}_color.{save_format}")
                    colored_img.save(color_filepath)
                    print(f"  Saved colored: {color_filepath}")

            except Exception as e:
                print(f"  Error exporting {name}: {e}")

        # Save combined preview
        if save_combined:
            combined = self._create_combined_preview(mask_dict, out_w, out_h)
            combined_np = (combined[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            combined_img = Image.fromarray(combined_np, mode='RGB')
            combined_path = os.path.join(output_dir, f"combined_preview.{save_format}")
            combined_img.save(combined_path)
            saved_files.append(combined_path)
            print(f"  Saved combined preview: {combined_path}")

        info = f"Exported {len(saved_files)} files to {output_dir}"
        print(f"\n[FreeFuse Mask Exporter] {info}")

        # Return preview
        preview = self._create_combined_preview(mask_dict, 512, 512)

        return (info, preview)

    def _create_combined_preview(self, mask_dict, width, height):
        """Create combined color-coded preview."""
        colors = [
            (1.0, 0.0, 0.0),  # Red
            (0.0, 1.0, 0.0),  # Green
            (0.0, 0.0, 1.0),  # Blue
            (1.0, 1.0, 0.0),  # Yellow
            (1.0, 0.0, 1.0),  # Magenta
            (0.0, 1.0, 1.0),  # Cyan
            (1.0, 0.5, 0.0),  # Orange
            (0.5, 0.5, 0.5),  # Gray
        ]

        device = None
        for name in mask_dict:
            if hasattr(mask_dict[name], 'device'):
                device = mask_dict[name].device
                break

        if device is None:
            return torch.zeros(1, height, width, 3)

        preview = torch.zeros(3, height, width, device=device)

        sorted_names = [n for n in mask_dict.keys() if not (n.startswith("_") and "background" in n.lower())]
        bg_names = [n for n in mask_dict.keys() if n.startswith("_") and "background" in n.lower()]
        sorted_names.extend(bg_names)

        for i, name in enumerate(sorted_names):
            mask = mask_dict.get(name)
            if mask is None or not hasattr(mask, 'dim'):
                continue

            color = colors[i % len(colors)]

            if mask.dim() == 2:
                m = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                m = mask.unsqueeze(0)
            else:
                m = mask

            mask_resized = F.interpolate(
                m.float().to(device), size=(height, width), mode='nearest'
            ).squeeze()

            for c in range(3):
                preview[c] += mask_resized * color[c]

        preview = preview.clamp(0, 1).permute(1, 2, 0).unsqueeze(0)
        return preview


# Export node mappings
NODE_CLASS_MAPPINGS = {
    "FreeFuseMaskExporter": FreeFuseMaskExporter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeFuseMaskExporter": "💾 FreeFuse Mask Exporter",
}
