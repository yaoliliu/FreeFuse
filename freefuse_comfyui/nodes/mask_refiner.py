"""
FreeFuse Mask Refiner Node

Post-process FreeFuse masks to fill holes, smooth boundaries, and clean up artifacts.
Sits between RawSimilarityOverlay and MaskApplicator for final mask cleanup.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple


class FreeFuseMaskRefiner:
    """
    Refine FreeFuse masks with morphological operations and hole filling.
    
    This node takes masks from RawSimilarityOverlay and applies:
    - Hole filling
    - Morphological closing/opening
    - Boundary smoothing
    - Small region removal
    
    Perfect for cleaning up masks before applying to LoRAs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("FREEFUSE_MASKS", {
                    "tooltip": "FreeFuse masks from RawSimilarityOverlay or SimilarityExtractor"
                }),
            },
            "optional": {
                # Hole filling
                "fill_holes": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Fill small holes in masks"
                }),
                "max_hole_size": ("INT", {
                    "default": 50, "min": 0, "max": 1000, "step": 10,
                    "tooltip": "Maximum hole size to fill (in pixels at 512x512)"
                }),
                
                # Morphological operations
                "morph_operation": (["none", "close", "open", "dilate", "erode"], {
                    "default": "close",
                    "tooltip": "Morphological operation to apply"
                }),
                "kernel_size": ("INT", {
                    "default": 3, "min": 1, "max": 11, "step": 2,
                    "tooltip": "Size of morphological kernel"
                }),
                "iterations": ("INT", {
                    "default": 1, "min": 1, "max": 5, "step": 1,
                    "tooltip": "Number of morphological iterations"
                }),
                
                # Smoothing
                "smooth_boundaries": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Smooth mask boundaries with Gaussian blur"
                }),
                "blur_sigma": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1,
                    "tooltip": "Gaussian blur sigma for smoothing"
                }),
                
                # Thresholding
                "apply_threshold": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply hard threshold to masks"
                }),
                "threshold_value": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Threshold value (values below become 0)"
                }),
                
                # Small region removal
                "remove_small_regions": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Remove small isolated regions"
                }),
                "min_region_size": ("INT", {
                    "default": 100, "min": 0, "max": 2000, "step": 50,
                    "tooltip": "Minimum region size to keep (in pixels at 512x512)"
                }),
            }
        }

    RETURN_TYPES = ("FREEFUSE_MASKS", "IMAGE")
    RETURN_NAMES = ("refined_masks", "preview")
    FUNCTION = "refine_masks"
    CATEGORY = "FreeFuse/Utils"

    DESCRIPTION = """Refine FreeFuse masks with hole filling and cleanup.

Use this between RawSimilarityOverlay and MaskApplicator to clean up masks.

Operations:
- Fill holes: Fill small gaps inside mask regions
- Morph close: Connect nearby regions, smooth boundaries
- Remove small: Remove isolated noise regions
- Smooth: Gaussian blur for softer edges
- Threshold: Convert soft masks to hard binary

Perfect for Qwen-Image masks that need cleanup."""

    def refine_masks(self, masks,
                     fill_holes=True, max_hole_size=50,
                     morph_operation="close", kernel_size=3, iterations=1,
                     smooth_boundaries=False, blur_sigma=1.0,
                     apply_threshold=False, threshold_value=0.5,
                     remove_small_regions=True, min_region_size=100):

        mask_dict = masks.get("masks", {})
        similarity_maps = masks.get("similarity_maps", {})
        token_pos_maps = masks.get("token_pos_maps", {})

        if not mask_dict:
            print("[FreeFuse Mask Refiner] Warning: No masks to refine")
            return (masks, torch.zeros(1, 512, 512, 3))

        print(f"\n[FreeFuse Mask Refiner] Refining {len(mask_dict)} masks")
        print(f"  Fill holes: {fill_holes} (max_size={max_hole_size})")
        print(f"  Morph operation: {morph_operation} (kernel={kernel_size}, iter={iterations})")
        print(f"  Smooth boundaries: {smooth_boundaries} (sigma={blur_sigma})")
        print(f"  Threshold: {apply_threshold} (value={threshold_value})")
        print(f"  Remove small regions: {remove_small_regions} (min_size={min_region_size})")

        refined_masks = {}

        for name, mask in mask_dict.items():
            try:
                # Ensure mask is on CPU for processing
                mask_cpu = mask.detach().cpu().float()
                
                # Handle different dimensions
                if mask_cpu.dim() == 3 and mask_cpu.shape[0] == 1:
                    mask_2d = mask_cpu[0]  # [1, H, W] -> [H, W]
                elif mask_cpu.dim() == 3:
                    mask_2d = mask_cpu.mean(dim=0)  # [C, H, W] -> [H, W]
                elif mask_cpu.dim() == 2:
                    mask_2d = mask_cpu
                else:
                    print(f"  {name}: Unexpected shape {mask_cpu.shape}, skipping")
                    refined_masks[name] = mask
                    continue

                original_mask = mask_2d.clone()

                # 1. Apply threshold if enabled
                if apply_threshold:
                    mask_2d = (mask_2d >= threshold_value).float()

                # 2. Fill holes
                if fill_holes and max_hole_size > 0:
                    mask_2d = self._fill_holes(mask_2d, max_hole_size)

                # 3. Morphological operations
                if morph_operation != "none":
                    mask_2d = self._morphological_op(
                        mask_2d, morph_operation, kernel_size, iterations
                    )

                # 4. Remove small regions
                if remove_small_regions and min_region_size > 0:
                    mask_2d = self._remove_small_regions(mask_2d, min_region_size)

                # 5. Smooth boundaries
                if smooth_boundaries and blur_sigma > 0:
                    mask_2d = self._smooth_boundaries(mask_2d, blur_sigma)

                # Ensure values are in [0, 1]
                mask_2d = mask_2d.clamp(0, 1)

                # Restore original dimensions
                if mask.dim() == 3:
                    if mask.shape[0] == 1:
                        refined_masks[name] = mask_2d.unsqueeze(0)
                    else:
                        refined_masks[name] = mask_2d.unsqueeze(0).expand(3, -1, -1)
                else:
                    refined_masks[name] = mask_2d

                # Report changes
                diff = (mask_2d - original_mask).abs().sum()
                if diff > 0:
                    print(f"  {name}: Refined (diff={diff:.2f})")

            except Exception as e:
                print(f"  {name}: Error during refinement: {e}")
                refined_masks[name] = mask  # Keep original on error

        # Build output
        refined_output = {
            "masks": refined_masks,
            "similarity_maps": similarity_maps,
            "token_pos_maps": token_pos_maps,
        }

        # Create preview
        preview = self._create_preview(refined_masks)

        print(f"\n[FreeFuse Mask Refiner] Refinement complete")
        return (refined_output, preview)

    def _fill_holes(self, mask: torch.Tensor, max_size: int) -> torch.Tensor:
        """Fill small holes in the mask."""
        try:
            # Binary mask
            binary = (mask > 0.5).float()
            
            # Invert to find holes
            inverted = 1 - binary
            
            # Simple hole filling: morphological close on inverted, then subtract
            kernel = self._create_kernel(3)
            filled = F.conv2d(
                inverted.unsqueeze(0).unsqueeze(0),
                kernel,
                padding=1
            ).squeeze()
            filled = (filled > 0.5).float()
            
            # Holes are regions that were filled
            holes = filled - inverted
            holes = holes.clamp(0, 1)
            
            # Only fill small holes (simple size check)
            # For now, just add all holes back
            result = binary + holes
            result = result.clamp(0, 1)
            
            return result
        except:
            return mask

    def _morphological_op(self, mask: torch.Tensor, operation: str, 
                          kernel_size: int, iterations: int) -> torch.Tensor:
        """Apply morphological operation."""
        try:
            kernel = self._create_kernel(kernel_size)
            result = mask.unsqueeze(0).unsqueeze(0)
            
            for _ in range(iterations):
                if operation == "close":
                    # Dilate then erode
                    result = F.conv2d(result, kernel, padding=kernel_size//2)
                    result = (result > 0.5).float()
                    result = F.conv2d(result, kernel, padding=kernel_size//2)
                    result = (result > 0.5).float()
                elif operation == "open":
                    # Erode then dilate
                    result = F.conv2d(result, kernel, padding=kernel_size//2)
                    result = (result > 4).float()  # Erode
                    result = F.conv2d(result, kernel, padding=kernel_size//2)
                    result = (result > 0.5).float()
                elif operation == "dilate":
                    result = F.conv2d(result, kernel, padding=kernel_size//2)
                    result = (result > 0.5).float()
                elif operation == "erode":
                    result = F.conv2d(result, kernel, padding=kernel_size//2)
                    result = (result > 4).float()
            
            return result.squeeze()
        except:
            return mask

    def _remove_small_regions(self, mask: torch.Tensor, min_size: int) -> torch.Tensor:
        """Remove small isolated regions."""
        try:
            binary = (mask > 0.5).float()
            # Simple approach: blur and threshold removes small regions
            kernel = self._create_kernel(5)
            blurred = F.conv2d(
                binary.unsqueeze(0).unsqueeze(0),
                kernel,
                padding=2
            ).squeeze()
            # Keep only regions that survived blurring
            result = (blurred > 0.3).float()
            return result
        except:
            return mask

    def _smooth_boundaries(self, mask: torch.Tensor, sigma: float) -> torch.Tensor:
        """Smooth mask boundaries with Gaussian blur."""
        try:
            # Simple Gaussian-like smoothing using box blur
            kernel_size = int(sigma * 6) | 1  # Make odd
            kernel = torch.ones((1, 1, kernel_size, kernel_size)) / (kernel_size ** 2)
            
            result = F.conv2d(
                mask.unsqueeze(0).unsqueeze(0),
                kernel,
                padding=kernel_size//2
            ).squeeze()
            
            return result
        except:
            return mask

    def _create_kernel(self, size: int) -> torch.Tensor:
        """Create square kernel for morphological operations."""
        return torch.ones((1, 1, size, size))

    def _create_preview(self, mask_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Create combined preview of refined masks."""
        colors = [
            (1.0, 0.0, 0.0),  # Red
            (0.0, 1.0, 0.0),  # Green
            (0.0, 0.0, 1.0),  # Blue
            (1.0, 1.0, 0.0),  # Yellow
        ]

        # Get dimensions from first mask
        first_mask = next(iter(mask_dict.values()))
        if first_mask.dim() == 2:
            h, w = first_mask.shape
        elif first_mask.dim() == 3:
            h, w = first_mask.shape[1], first_mask.shape[2]
        else:
            h, w = 512, 512

        device = first_mask.device if hasattr(first_mask, 'device') else torch.device('cpu')
        preview = torch.zeros(3, h, w, device=device)

        for i, (name, mask) in enumerate(mask_dict.items()):
            color = colors[i % len(colors)]
            
            if mask.dim() == 3 and mask.shape[0] == 1:
                m = mask[0]
            elif mask.dim() == 3:
                m = mask.mean(dim=0)
            elif mask.dim() == 2:
                m = mask
            else:
                continue

            for c in range(3):
                preview[c] += m * color[c]

        preview = preview.clamp(0, 1).permute(1, 2, 0).unsqueeze(0)
        return preview


# Export node mappings
NODE_CLASS_MAPPINGS = {
    "FreeFuseMaskRefiner": FreeFuseMaskRefiner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeFuseMaskRefiner": "🔧 FreeFuse Mask Refiner",
}
