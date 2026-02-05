"""
FreeFuse Mask Preview - Simplified
"""

import torch
import torch.nn.functional as F


class FreeFuseMaskPreview:
    """Visualize FreeFuse spatial masks."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("FREEFUSE_MASKS",),
            },
            "optional": {
                "width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)
    FUNCTION = "preview"
    CATEGORY = "FreeFuse"
    OUTPUT_NODE = True
    
    COLORS = [
        (1.0, 0.0, 0.0),  # Red
        (0.0, 1.0, 0.0),  # Green  
        (0.0, 0.0, 1.0),  # Blue
        (1.0, 1.0, 0.0),  # Yellow
        (1.0, 0.0, 1.0),  # Magenta
        (0.25, 0.25, 0.25),  # Gray
    ]
    
    def preview(self, masks, width=512, height=512):
        mask_dict = masks.get("masks", {})
        
        if not mask_dict:
            return (torch.zeros(1, height, width, 3),)
        
        preview = torch.zeros(3, height, width)
        
        for i, (name, mask) in enumerate(mask_dict.items()):
            color = self.COLORS[i % len(self.COLORS)]
            
            # Handle different mask dimensions
            if mask.dim() == 2:
                m = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                m = mask.unsqueeze(0)
            else:
                m = mask
            
            mask_resized = F.interpolate(
                m.float(), size=(height, width), mode='nearest'
            ).squeeze()
            
            for c in range(3):
                preview[c] += mask_resized * color[c]
        
        # (C, H, W) -> (B, H, W, C)
        preview = preview.clamp(0, 1).permute(1, 2, 0).unsqueeze(0)
        
        return (preview,)
