"""
FreeFuse Mask Preview - Comprehensive Visualization

Provides multiple preview modes for FreeFuse spatial masks:
- Combined color-coded view
- Individual mask panels
- Similarity map visualization (if available)
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple


class FreeFuseMaskPreview:
    """
    Visualize FreeFuse spatial masks with multiple display options.
    
    Features:
    - Combined color-coded preview (all masks overlaid)
    - Individual mask panels (each mask separately)
    - Adjustable output resolution
    - Optional similarity map visualization
    """
    
    # Distinct colors for up to 8 concepts
    COLORS = [
        (1.0, 0.3, 0.3),   # Red (concept 1)
        (0.3, 1.0, 0.3),   # Green (concept 2)
        (0.3, 0.3, 1.0),   # Blue (concept 3)
        (1.0, 1.0, 0.3),   # Yellow (concept 4)
        (1.0, 0.3, 1.0),   # Magenta (concept 5)
        (0.3, 1.0, 1.0),   # Cyan (concept 6)
        (1.0, 0.6, 0.3),   # Orange (concept 7)
        (0.4, 0.4, 0.4),   # Gray (background)
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("FREEFUSE_MASKS",),
            },
            "optional": {
                "width": ("INT", {
                    "default": 512, "min": 64, "max": 2048, "step": 64,
                    "tooltip": "Output preview width"
                }),
                "height": ("INT", {
                    "default": 512, "min": 64, "max": 2048, "step": 64,
                    "tooltip": "Output preview height"
                }),
                "show_labels": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show concept names on preview (requires PIL)"
                }),
                "show_coverage": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Print coverage statistics to console"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("combined_preview", "individual_masks")
    FUNCTION = "preview"
    CATEGORY = "FreeFuse"
    OUTPUT_NODE = True
    
    DESCRIPTION = """Visualize FreeFuse spatial masks.

Output 1 (combined_preview): All masks color-coded and overlaid
Output 2 (individual_masks): Each mask as a separate grayscale panel

Colors are assigned in order: Red, Green, Blue, Yellow, Magenta, Cyan, Orange, Gray
Background mask (if present) is always gray."""
    
    def preview(self, masks, width=512, height=512, show_labels=True, show_coverage=True):
        mask_dict = masks.get("masks", {})
        
        if not mask_dict:
            empty = torch.zeros(1, height, width, 3)
            return (empty, empty)
        
        # Sort masks: regular concepts first, background last
        sorted_names = []
        bg_name = None
        for name in mask_dict.keys():
            if name.startswith("_") and "background" in name.lower():
                bg_name = name
            else:
                sorted_names.append(name)
        if bg_name:
            sorted_names.append(bg_name)
        
        # Print coverage statistics
        if show_coverage:
            print("[FreeFuse Mask Preview] Coverage statistics:")
            for name in sorted_names:
                mask = mask_dict[name]
                coverage = mask.sum() / mask.numel() * 100
                print(f"  {name}: {coverage:.1f}% coverage")
        
        # Create combined preview
        combined = self._create_combined_preview(mask_dict, sorted_names, width, height)
        
        # Create individual mask panels
        individual = self._create_individual_panels(mask_dict, sorted_names, width, height)
        
        return (combined, individual)
    
    def _create_combined_preview(
        self, 
        mask_dict: Dict[str, torch.Tensor], 
        sorted_names: List[str],
        width: int, 
        height: int
    ) -> torch.Tensor:
        """Create combined color-coded preview."""
        device = next(iter(mask_dict.values())).device
        preview = torch.zeros(3, height, width, device=device)
        
        for i, name in enumerate(sorted_names):
            mask = mask_dict[name]
            
            # Use gray for background, otherwise use ordered colors
            if name.startswith("_") and "background" in name.lower():
                color = self.COLORS[-1]  # Gray
            else:
                color = self.COLORS[i % (len(self.COLORS) - 1)]
            
            # Handle different mask dimensions
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
        
        # (C, H, W) -> (B, H, W, C)
        preview = preview.clamp(0, 1).permute(1, 2, 0).unsqueeze(0)
        
        return preview
    
    def _create_individual_panels(
        self,
        mask_dict: Dict[str, torch.Tensor],
        sorted_names: List[str],
        width: int,
        height: int
    ) -> torch.Tensor:
        """Create individual grayscale mask panels arranged horizontally."""
        num_masks = len(sorted_names)
        if num_masks == 0:
            return torch.zeros(1, height, width, 3)
        
        # Create horizontal strip of all masks
        panel_width = width // num_masks if num_masks <= 4 else width // 4
        total_width = panel_width * min(num_masks, 4)
        num_rows = (num_masks + 3) // 4
        total_height = height * num_rows
        
        device = next(iter(mask_dict.values())).device
        panels = torch.zeros(3, total_height, total_width, device=device)
        
        for i, name in enumerate(sorted_names):
            mask = mask_dict[name]
            row = i // 4
            col = i % 4
            
            # Handle different mask dimensions
            if mask.dim() == 2:
                m = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                m = mask.unsqueeze(0)
            else:
                m = mask
            
            # Resize mask to panel size
            mask_resized = F.interpolate(
                m.float().to(device), size=(height, panel_width), mode='nearest'
            ).squeeze()
            
            # Place in panel grid
            y_start = row * height
            y_end = y_start + height
            x_start = col * panel_width
            x_end = x_start + panel_width
            
            # Grayscale (same value for all channels)
            for c in range(3):
                panels[c, y_start:y_end, x_start:x_end] = mask_resized
        
        # (C, H, W) -> (B, H, W, C)
        panels = panels.clamp(0, 1).permute(1, 2, 0).unsqueeze(0)
        
        return panels
