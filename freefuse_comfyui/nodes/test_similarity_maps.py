"""
FreeFuse Test Similarity Map Generator

Generates synthetic similarity maps for testing the FreeFuseRawSimilarityOverlay node.
Useful for debugging and visualization testing without running full Phase 1 sampling.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional


class FreeFuseTestSimilarityMaps:
    """
    Generate synthetic similarity maps for testing overlay visualization.
    
    Creates realistic-looking similarity maps with various patterns:
    - Gaussian blobs at different positions
    - Gradient patterns
    - Noise patterns
    
    Useful for testing FreeFuseRawSimilarityOverlay without running Phase 1 sampling.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "freefuse_data": ("FREEFUSE_DATA", {
                    "tooltip": "Freefuse data with concept names"
                }),
                "latent": ("LATENT", {
                    "tooltip": "Latent to determine output dimensions"
                }),
            },
            "optional": {
                "pattern_type": (["gaussian", "gradient", "noise", "checkerboard"], {
                    "default": "gaussian",
                    "tooltip": "Type of pattern to generate"
                }),
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for pattern generation"
                }),
                "overlap": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Amount of overlap between concept maps"
                }),
                "contrast": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1,
                    "tooltip": "Contrast multiplier for similarity maps"
                }),
            }
        }

    RETURN_TYPES = ("FREEFUSE_MASKS",)
    RETURN_NAMES = ("raw_similarity",)
    FUNCTION = "generate"
    CATEGORY = "FreeFuse/Debug"

    DESCRIPTION = """Generates synthetic similarity maps for testing overlay visualization.

Useful for:
- Testing FreeFuseRawSimilarityOverlay without Phase 1 sampling
- Debugging visualization parameters
- Understanding how overlay combines multiple concepts

The generated maps simulate what real attention-based similarity maps would look like."""

    def generate(self,
                 freefuse_data,
                 latent,
                 pattern_type="gaussian",
                 seed=42,
                 overlap=0.3,
                 contrast=1.0):

        concepts = freefuse_data.get("concepts", {})
        
        if not concepts:
            return ({"masks": {}, "similarity_maps": {}},)

        # Get latent dimensions
        latent_tensor = latent["samples"]
        latent_h, latent_w = latent_tensor.shape[2], latent_tensor.shape[3]
        
        # Upscale to pixel space for visualization
        out_h, out_w = latent_h * 8, latent_w * 8
        
        # For similarity maps, we work at latent resolution
        sim_h, sim_w = latent_h, latent_w
        
        print(f"[FreeFuse Test] Generating test similarity maps")
        print(f"  Concepts: {list(concepts.keys())}")
        print(f"  Pattern: {pattern_type}")
        print(f"  Output size: {sim_w}x{sim_h} (latent), {out_w}x{out_h} (preview)")
        
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Get concept names (exclude background)
        concept_names = [name for name in concepts.keys() if not name.startswith("__")]
        
        # Add background if needed
        include_background = len(concept_names) > 0
        if include_background:
            concept_names.append("__background__")
        
        similarity_maps = {}
        
        if pattern_type == "gaussian":
            similarity_maps = self._generate_gaussian(concept_names, sim_h, sim_w, overlap, contrast)
        elif pattern_type == "gradient":
            similarity_maps = self._generate_gradient(concept_names, sim_h, sim_w, contrast)
        elif pattern_type == "noise":
            similarity_maps = self._generate_noise(concept_names, sim_h, sim_w, contrast)
        elif pattern_type == "checkerboard":
            similarity_maps = self._generate_checkerboard(concept_names, sim_h, sim_w, contrast)
        
        # Create output
        result = {
            "masks": {},
            "similarity_maps": similarity_maps,
        }
        
        print(f"[FreeFuse Test] Generated {len(similarity_maps)} similarity maps")
        for name, sim_map in similarity_maps.items():
            if isinstance(sim_map, torch.Tensor):
                print(f"  {name}: shape={sim_map.shape}, min={sim_map.min():.4f}, max={sim_map.max():.4f}")
        
        return (result,)
    
    def _generate_gaussian(self, concept_names: List[str], h: int, w: int, 
                           overlap: float, contrast: float) -> Dict[str, torch.Tensor]:
        """Generate Gaussian blob patterns at different positions."""
        similarity_maps = {}
        n_concepts = len(concept_names)
        
        for idx, name in enumerate(concept_names):
            if name == "__background__":
                # Background: low uniform value
                sim_map = torch.ones(1, h * w, 1) * 0.1 * contrast
            else:
                # Create Gaussian at different position for each concept
                # Arrange concepts in a grid pattern
                grid_size = int(np.ceil(np.sqrt(n_concepts - 1))) if n_concepts > 1 else 1
                
                if grid_size > 0:
                    row = idx // grid_size
                    col = idx % grid_size
                    
                    center_y = (row + 0.5) / grid_size * h
                    center_x = (col + 0.5) / grid_size * w
                else:
                    center_y, center_x = h / 2, w / 2
                
                # Create 2D Gaussian
                y = torch.linspace(0, h - 1, h)
                x = torch.linspace(0, w - 1, w)
                yy, xx = torch.meshgrid(y, x, indexing='ij')
                
                # Sigma scales with overlap
                sigma = min(h, w) * (0.2 + overlap * 0.3)
                
                gaussian = torch.exp(
                    -((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma ** 2)
                )
                
                # Normalize and apply contrast
                gaussian = gaussian / gaussian.max() * contrast
                
                # Reshape to (1, N, 1)
                sim_map = gaussian.view(1, h * w, 1)
            
            similarity_maps[name] = sim_map
        
        return similarity_maps
    
    def _generate_gradient(self, concept_names: List[str], h: int, w: int, 
                           contrast: float) -> Dict[str, torch.Tensor]:
        """Generate gradient patterns in different directions."""
        similarity_maps = {}
        n_concepts = len(concept_names)
        
        directions = [
            (1, 0),   # horizontal
            (0, 1),   # vertical
            (1, 1),   # diagonal
            (-1, 1),  # anti-diagonal
        ]
        
        for idx, name in enumerate(concept_names):
            if name == "__background__":
                sim_map = torch.ones(1, h * w, 1) * 0.1 * contrast
            else:
                # Select direction based on concept index
                dir_idx = idx % len(directions)
                dx, dy = directions[dir_idx]
                
                # Create gradient
                y = torch.linspace(0, 1, h)
                x = torch.linspace(0, 1, w)
                yy, xx = torch.meshgrid(y, x, indexing='ij')
                
                gradient = (xx * dx + yy * dy)
                gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())
                gradient = gradient * contrast
                
                sim_map = gradient.view(1, h * w, 1)
            
            similarity_maps[name] = sim_map
        
        return similarity_maps
    
    def _generate_noise(self, concept_names: List[str], h: int, w: int, 
                        contrast: float) -> Dict[str, torch.Tensor]:
        """Generate smooth noise patterns."""
        similarity_maps = {}
        
        for idx, name in enumerate(concept_names):
            if name == "__background__":
                sim_map = torch.ones(1, h * w, 1) * 0.1 * contrast
            else:
                # Generate smooth noise using low-frequency components
                noise = torch.randn(1, h, w)
                
                # Apply Gaussian blur in frequency domain for smoothness
                noise_fft = torch.fft.fft2(noise)
                freq_y = torch.fft.fftfreq(h)[:, None]
                freq_x = torch.fft.fftfreq(w)[None, :]
                
                # Low-pass filter
                sigma_freq = 0.1
                filter_mask = torch.exp(-(freq_y ** 2 + freq_x ** 2) / (2 * sigma_freq ** 2))
                noise_fft = noise_fft * filter_mask
                
                # Inverse FFT
                noise_smooth = torch.fft.ifft2(noise_fft).real
                
                # Normalize
                noise_smooth = (noise_smooth - noise_smooth.min()) / (noise_smooth.max() - noise_smooth.min())
                noise_smooth = noise_smooth * contrast
                
                sim_map = noise_smooth.view(1, h * w, 1)
            
            similarity_maps[name] = sim_map
        
        return similarity_maps
    
    def _generate_checkerboard(self, concept_names: List[str], h: int, w: int, 
                                contrast: float) -> Dict[str, torch.Tensor]:
        """Generate checkerboard patterns with different frequencies."""
        similarity_maps = {}
        
        frequencies = [4, 8, 16, 32]
        
        for idx, name in enumerate(concept_names):
            if name == "__background__":
                sim_map = torch.ones(1, h * w, 1) * 0.1 * contrast
            else:
                freq = frequencies[idx % len(frequencies)]
                
                y = torch.arange(h)
                x = torch.arange(w)
                yy, xx = torch.meshgrid(y, x, indexing='ij')
                
                checker = ((xx // (w // freq) + yy // (h // freq)) % 2).float()
                checker = checker * contrast
                
                sim_map = checker.view(1, h * w, 1)
            
            similarity_maps[name] = sim_map
        
        return similarity_maps


# Export node mappings
NODE_CLASS_MAPPINGS = {
    "FreeFuseTestSimilarityMaps": FreeFuseTestSimilarityMaps,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeFuseTestSimilarityMaps": "🧪 FreeFuse Test Similarity Maps",
}
