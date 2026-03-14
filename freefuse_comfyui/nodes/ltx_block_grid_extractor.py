"""
FreeFuse LTX-Video Block Grid Extractor

Extracts similarity maps from multiple transformer blocks in LTX-Video
and displays them as a grid for visual analysis.

This helps identify which blocks produce meaningful attention patterns.
"""

import torch
import torch.nn.functional as F
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Optional, Callable, Any
import comfy.sample
import comfy.samplers

from ..freefuse_core.token_utils import detect_model_type


class EarlyStopException(Exception):
    """Exception to signal early termination after collection."""
    pass


class FreeFuseLTXBlockGridExtractor:
    """
    Extract similarity maps from multiple LTX-Video transformer blocks
    and display as a grid for visual analysis.
    
    LTX-Video has 48 transformer blocks with 32 attention heads.
    This node helps you find the optimal block for concept separation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "LTX-Video model"
                }),
                "positive": ("CONDITIONING", {
                    "tooltip": "Positive conditioning"
                }),
                "negative": ("CONDITIONING", {
                    "tooltip": "Negative conditioning"
                }),
                "latent": ("LATENT", {
                    "tooltip": "Latent video"
                }),
                "freefuse_data": ("FREEFUSE_DATA", {
                    "tooltip": "Freefuse data with concept token positions"
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff
                }),
                "steps": ("INT", {
                    "default": 3, "min": 1, "max": 100,
                    "tooltip": "Number of sampling steps (2-3 is enough)"
                }),
                "collect_step": ("INT", {
                    "default": 1, "min": 0, "max": 99,
                    "tooltip": "Step at which to collect attention"
                }),
                "block_start": ("INT", {
                    "default": 0, "min": 0, "max": 47,
                    "tooltip": "First block to collect from (0-47)"
                }),
                "block_end": ("INT", {
                    "default": 47, "min": 0, "max": 47,
                    "tooltip": "Last block to collect from (0-47)"
                }),
                "cfg": ("FLOAT", {
                    "default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
            },
            "optional": {
                "temperature": ("FLOAT", {
                    "default": 4000.0, "min": 0.0, "max": 10000.0, "step": 100.0,
                    "tooltip": "Temperature for similarity computation"
                }),
                "top_k_ratio": ("FLOAT", {
                    "default": 0.3, "min": 0.01, "max": 1.0, "step": 0.05,
                    "tooltip": "Ratio of top-k tokens to use"
                }),
                "preview_size": ("INT", {
                    "default": 512, "min": 256, "max": 1024, "step": 64,
                    "tooltip": "Preview grid size (longest side)"
                }),
                "cell_size": ("INT", {
                    "default": 128, "min": 64, "max": 256, "step": 32,
                    "tooltip": "Size of each block preview cell in pixels"
                }),
                "low_vram_mode": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable VRAM optimization"
                }),

                # ===== ARGMAX GRID PARAMETERS =====
                "argmax_method": (["simple", "stabilized"], {
                    "default": "stabilized",
                    "tooltip": "simple = direct argmax, stabilized = balanced with spatial constraints"
                }),
                "max_iter": ("INT", {
                    "default": 15, "min": 1, "max": 50, "step": 1,
                    "tooltip": "Number of iterations for stabilized argmax"
                }),
                "balance_lr": ("FLOAT", {
                    "default": 0.01, "min": 0.001, "max": 0.1, "step": 0.001,
                    "tooltip": "Learning rate for bias updates"
                }),
                "gravity_weight": ("FLOAT", {
                    "default": 0.00004, "min": 0.0, "max": 0.001, "step": 0.00001,
                    "tooltip": "How strongly pixels are pulled to concept centroid"
                }),
                "spatial_weight": ("FLOAT", {
                    "default": 0.00004, "min": 0.0, "max": 0.001, "step": 0.00001,
                    "tooltip": "How much neighboring pixels influence assignment"
                }),
                "momentum": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 0.9, "step": 0.05,
                    "tooltip": "Smoothing between iterations"
                }),
                "anisotropy": ("FLOAT", {
                    "default": 1.3, "min": 0.5, "max": 2.0, "step": 0.1,
                    "tooltip": "Horizontal stretch factor"
                }),
                "centroid_margin": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.5, "step": 0.05,
                    "tooltip": "Keep centroids away from borders"
                }),
                "border_penalty": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Penalty for assigning pixels near borders"
                }),
                "bg_scale": ("FLOAT", {
                    "default": 0.95, "min": 0.5, "max": 2.0, "step": 0.05,
                    "tooltip": "Background channel multiplier"
                }),
            }
        }

    RETURN_TYPES = ("FREEFUSE_MASKS", "IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("all_masks", "block_grid", "info", "argmax_winner_grid")
    FUNCTION = "extract"
    CATEGORY = "FreeFuse/Debug"

    DESCRIPTION = """Extracts similarity maps from multiple LTX-Video transformer blocks.

LTX-Video has 48 transformer blocks. This node shows a grid of all collected blocks
so you can visually identify which blocks produce meaningful attention patterns.

The ARGMAX_WINNER_GRID shows which concept "wins" at each block (hard assignment).
Each pixel gets the color of the concept with highest similarity.

Recommended block ranges to test:
- 0-15: Early features (not recommended)
- 16-30: Good concept separation (recommended starting point)
- 31-40: Transition zone
- 41-47: Deep concepts (may have overlap)

Use this to find the best collect_block value for FreeFuseLTXSimilarityExtractor."""

    def extract(self,
                model,
                positive,
                negative,
                latent,
                freefuse_data,
                seed,
                steps,
                collect_step,
                block_start,
                block_end,
                cfg,
                sampler_name,
                scheduler,
                temperature=4000.0,
                top_k_ratio=0.3,
                preview_size=512,
                cell_size=128,
                low_vram_mode=True,
                argmax_method="stabilized",
                max_iter=15,
                balance_lr=0.01,
                gravity_weight=0.00004,
                spatial_weight=0.00004,
                momentum=0.2,
                anisotropy=1.3,
                centroid_margin=0.0,
                border_penalty=0.0,
                bg_scale=0.95):

        # Get token positions
        token_pos_maps = freefuse_data.get("token_pos_maps", {})
        concepts = freefuse_data.get("concepts", {})

        if not token_pos_maps:
            print("[FreeFuse LTX Block Grid] ERROR: No token positions found")
            empty_result = {"masks": {}, "similarity_maps": {}}
            grid = torch.zeros(1, cell_size, cell_size, 3)
            return (empty_result, grid, "No token positions")

        # Clamp block range to LTX-Video limits (0-47)
        block_start = max(0, block_start)
        block_end = min(47, block_end)
        
        if block_start > block_end:
            print(f"[FreeFuse LTX Block Grid] ERROR: block_start ({block_start}) > block_end ({block_end})")
            block_start, block_end = block_end, block_start

        num_blocks = block_end - block_start + 1
        print(f"[FreeFuse LTX Block Grid] Starting extraction")
        print(f"  Concepts: {list(concepts.keys())}")
        print(f"  Block range: {block_start}-{block_end} ({num_blocks} blocks)")
        print(f"  Collect step: {collect_step}/{steps}")
        print(f"  Temperature: {temperature}, top_k_ratio: {top_k_ratio}")
        print(f"  Low VRAM mode: {low_vram_mode}")

        # VRAM cleanup
        if low_vram_mode and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            free_mem, total_mem = torch.cuda.mem_get_info()
            print(f"[FreeFuse LTX Block Grid] VRAM: {free_mem / 1024:.0f} MB free / {total_mem / 1024:.0f} MB total")

        # Clone model
        model_clone = model.clone()

        # Get latent dimensions
        latent_tensor = latent["samples"]
        
        # LTX-Video uses 5D latents: (B, C, T, H, W)
        if latent_tensor.dim() == 4:
            print(f"[FreeFuse LTX Block Grid] Adding temporal dimension to latent")
            latent_tensor = latent_tensor.unsqueeze(2)
        
        B, C, T, H, W = latent_tensor.shape
        img_len = T * H * W
        
        print(f"[FreeFuse LTX Block Grid] Latent shape: {latent_tensor.shape}, seq_len: {img_len}")

        # Storage for all blocks
        all_sim_maps = {}  # {block_idx: {concept_name: sim_map}}

        # Get the diffusion model
        diffusion_model = model.model.diffusion_model

        # Find transformer blocks
        target_blocks = None
        if hasattr(diffusion_model, 'transformer_blocks'):
            target_blocks = diffusion_model.transformer_blocks
        elif hasattr(diffusion_model, 'layers'):
            target_blocks = diffusion_model.layers

        if target_blocks is None:
            print(f"[FreeFuse LTX Block Grid] ERROR: No transformer blocks found")
            empty_result = {"masks": {}, "similarity_maps": {}}
            grid = torch.zeros(1, cell_size, cell_size, 3)
            return (empty_result, grid, "No transformer blocks")

        print(f"[FreeFuse LTX Block Grid] Found {len(target_blocks)} transformer blocks")

        # Create noise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        noise = comfy.sample.prepare_noise(latent_tensor, seed, None)

        # Track current step
        current_step = [0]
        collection_done = [False]

        def step_callback(step, x0, x, total_steps):
            current_step[0] = step
            
            # Stop after collect_step
            if collection_done[0] and step > collect_step:
                raise EarlyStopException("Collection done")

        # Collect from each block
        for block_idx in range(block_start, block_end + 1):
            print(f"\n[FreeFuse LTX Block Grid] Collecting from block {block_idx}/{47}")
            
            # VRAM cleanup before each block
            if low_vram_mode and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Clone model for this block
            block_model = model_clone.clone()

            # Install hook
            target_block = target_blocks[block_idx]
            hook = LTXBlockGridHook(
                target_block=target_block,
                block_index=block_idx,
                token_pos_maps=token_pos_maps,
                temperature=temperature,
                top_k_ratio=top_k_ratio,
                img_len=img_len,
                all_sim_maps=all_sim_maps,
            )
            hook.install(block_model)

            # Reset collection flag
            collection_done[0] = False

            # Run sampling
            try:
                samples = comfy.sample.sample(
                    block_model,
                    noise,
                    steps,
                    cfg,
                    sampler_name,
                    scheduler,
                    positive,
                    negative,
                    latent_tensor,
                    denoise=1.0,
                    disable_noise=False,
                    start_step=0,
                    last_step=steps,
                    force_full_denoise=False,
                    noise_mask=None,
                    callback=step_callback,
                    seed=seed,
                )
            except EarlyStopException:
                pass
            except Exception as e:
                print(f"[FreeFuse LTX Block Grid] Sampling error at block {block_idx}: {e}")

            # Remove hook
            hook.remove()

            # VRAM cleanup
            if low_vram_mode and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Check if we got maps
            if block_idx in all_sim_maps:
                print(f"[FreeFuse LTX Block Grid] Block {block_idx}: collected {len(all_sim_maps[block_idx])} maps")
            else:
                print(f"[FreeFuse LTX Block Grid] Block {block_idx}: NO MAPS COLLECTED")

        # Mark collection as complete
        collection_done[0] = True

        # Create result
        result = {
            "masks": {},
            "similarity_maps": {},
        }

        # Flatten all maps
        for block_idx, block_maps in all_sim_maps.items():
            for name, sim_map in block_maps.items():
                result["similarity_maps"][f"block{block_idx}_{name}"] = sim_map

        # Create block grid
        block_grid = self._create_block_grid(
            all_sim_maps, block_start, block_end,
            T, H, W, preview_size, cell_size
        )

        # Create argmax winner grid
        argmax_grid = self._create_argmax_grid(
            all_sim_maps, block_start, block_end,
            T, H, W, cell_size,
            argmax_method, max_iter, balance_lr,
            gravity_weight, spatial_weight, momentum,
            anisotropy, centroid_margin, border_penalty, bg_scale
        )

        # Create info string
        info = self._create_info_string(all_sim_maps, block_start, block_end)

        return (result, block_grid, info, argmax_grid)

    def _create_block_grid(self, all_sim_maps, block_start, block_end, T, H, W, preview_size, cell_size):
        """Create a grid of similarity map previews for each block."""
        import numpy as np

        num_blocks = block_end - block_start + 1
        num_concepts = len([k for k in all_sim_maps.get(block_start, {}).keys() if not k.startswith("__")])
        
        if num_concepts == 0:
            return torch.zeros(1, cell_size, cell_size, 3)

        # Calculate grid layout
        cols = min(num_blocks, 6)
        rows = (num_blocks + cols - 1) // cols
        
        # Each cell shows one block (averaged over concepts)
        cell_h = cell_size
        cell_w = cell_size
        
        grid_img = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.float32)

        # Get spatial dimensions
        first_block = all_sim_maps.get(block_start, {})
        if not first_block:
            return torch.zeros(1, cell_size, cell_size, 3)
        
        first_sim = list(first_block.values())[0]
        seq_len = first_sim.shape[1]
        spatial_len = seq_len // T if T > 0 else seq_len
        actual_H = actual_W = int(spatial_len ** 0.5)

        # Colors for concepts
        colors = [
            (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0),
        ]

        for block_idx in range(block_start, block_end + 1):
            if block_idx not in all_sim_maps:
                continue
            
            block_maps = all_sim_maps[block_idx]
            col_idx = block_idx - block_start
            
            row = col_idx // cols
            col = col_idx % cols
            
            y_start = row * cell_h
            y_end = (row + 1) * cell_h
            x_start = col * cell_w
            x_end = (col + 1) * cell_w
            
            # Average over concepts for this block
            avg_sim = None
            concept_list = [(name, sim) for name, sim in block_maps.items() if not name.startswith("__")]
            
            if concept_list:
                for name, sim in concept_list[:len(colors)]:
                    sim_cpu = sim.cpu()
                    if sim_cpu.dim() == 3:
                        sim_3d = sim_cpu[0, :, :].view(T, actual_H, actual_W)
                    else:
                        sim_3d = sim_cpu.view(T, actual_H, actual_W)
                    
                    sim_2d = sim_3d[0]  # First frame
                    
                    sim_resized = F.interpolate(
                        sim_2d.unsqueeze(0).unsqueeze(0),
                        size=(cell_h, cell_w),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).squeeze(0)
                    
                    # Normalize
                    sim_min = sim_resized.min()
                    sim_max = sim_resized.max()
                    if sim_max > sim_min:
                        sim_norm = (sim_resized - sim_min) / (sim_max - sim_min)
                    else:
                        sim_norm = torch.ones_like(sim_resized) * 0.5
                    
                    if avg_sim is None:
                        avg_sim = sim_norm
                    else:
                        avg_sim = (avg_sim + sim_norm) / 2
                
                # Convert to RGB using first concept color (or average)
                if avg_sim is not None:
                    cell_data = np.zeros((cell_h, cell_w, 3), dtype=np.float32)
                    cell_data[:, :, 0] = avg_sim.numpy() * colors[0][0]
                    cell_data[:, :, 1] = avg_sim.numpy() * colors[0][1]
                    cell_data[:, :, 2] = avg_sim.numpy() * colors[0][2]
                    grid_img[y_start:y_end, x_start:x_end] = cell_data

        # Add block labels
        pil_img = Image.fromarray((grid_img * 255).astype(np.uint8))
        draw = ImageDraw.Draw(pil_img)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except:
            font = ImageFont.load_default()
        
        for block_idx in range(block_start, block_end + 1):
            col_idx = block_idx - block_start
            row = col_idx // cols
            col = col_idx % cols
            
            x = col * cell_w + 5
            y = row * cell_h + 12
            
            draw.text((x, y), f"B{block_idx}", fill=(255, 255, 255), font=font)

        # Resize to preview size
        pil_img = pil_img.resize((preview_size, int(preview_size * rows / cols)), Image.LANCZOS)
        
        grid_tensor = torch.from_numpy(np.array(pil_img).astype(np.float32) / 255.0).unsqueeze(0)
        
        return grid_tensor

    def _create_argmax_grid(self, all_sim_maps, block_start, block_end, T, H, W, cell_size,
                           argmax_method, max_iter, balance_lr, gravity_weight, spatial_weight,
                           momentum, anisotropy, centroid_margin, border_penalty, bg_scale):
        """Create argmax winner grid showing which concept wins at each block."""
        import numpy as np

        num_blocks = block_end - block_start + 1
        
        # Calculate grid layout
        cols = min(num_blocks, 6)
        rows = (num_blocks + cols - 1) // cols
        
        cell_h = cell_size
        cell_w = cell_size
        
        grid_img = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.float32)

        # Get spatial dimensions
        first_block = all_sim_maps.get(block_start, {})
        if not first_block:
            return torch.zeros(1, cell_size, cell_size, 3)
        
        first_sim = list(first_block.values())[0]
        seq_len = first_sim.shape[1]
        spatial_len = seq_len // T if T > 0 else seq_len
        actual_H = actual_W = int(spatial_len ** 0.5)

        # Colors for concepts
        colors = [
            (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0),
        ]

        for block_idx in range(block_start, block_end + 1):
            if block_idx not in all_sim_maps:
                continue
            
            block_maps = all_sim_maps[block_idx]
            col_idx = block_idx - block_start
            
            row = col_idx // cols
            col = col_idx % cols
            
            y_start = row * cell_h
            y_end = (row + 1) * cell_h
            x_start = col * cell_w
            x_end = (col + 1) * cell_w
            
            # Get concept maps for this block
            concept_list = [(name, sim) for name, sim in block_maps.items() if not name.startswith("__")]
            
            if len(concept_list) == 0:
                continue
            
            # Stack similarity maps
            sim_stack = []
            for name, sim in concept_list:
                sim_cpu = sim.cpu()
                if sim_cpu.dim() == 3:
                    sim_3d = sim_cpu[0, :, :].view(T, actual_H, actual_W)
                else:
                    sim_3d = sim_cpu.view(T, actual_H, actual_W)
                sim_2d = sim_3d[0]  # First frame
                
                sim_resized = F.interpolate(
                    sim_2d.unsqueeze(0).unsqueeze(0),
                    size=(cell_h, cell_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)
                
                sim_stack.append(sim_resized)
            
            # Stack: (num_concepts, H, W)
            sim_tensor = torch.stack(sim_stack, dim=0)
            
            # Apply argmax
            if argmax_method == "simple":
                # Simple argmax
                winner = torch.argmax(sim_tensor, dim=0)
            else:
                # Stabilized argmax with spatial constraints
                winner = self._stabilized_argmax(
                    sim_tensor, max_iter, balance_lr,
                    gravity_weight, spatial_weight, momentum,
                    anisotropy, centroid_margin, border_penalty, bg_scale
                )
            
            # Convert to RGB
            cell_data = np.zeros((cell_h, cell_w, 3), dtype=np.float32)
            winner_np = winner.numpy()
            
            for c_idx, color in enumerate(colors[:len(concept_list)]):
                mask = (winner_np == c_idx)
                cell_data[mask, 0] = color[0]
                cell_data[mask, 1] = color[1]
                cell_data[mask, 2] = color[2]
            
            grid_img[y_start:y_end, x_start:x_end] = cell_data

        # Add labels
        pil_img = Image.fromarray((grid_img * 255).astype(np.uint8))
        draw = ImageDraw.Draw(pil_img)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except:
            font = ImageFont.load_default()
        
        for block_idx in range(block_start, block_end + 1):
            col_idx = block_idx - block_start
            row = col_idx // cols
            col = col_idx % cols
            
            x = col * cell_w + 5
            y = row * cell_h + 12
            
            draw.text((x, y), f"B{block_idx}", fill=(255, 255, 255), font=font)

        grid_tensor = torch.from_numpy(grid_img).unsqueeze(0)
        
        return grid_tensor

    def _stabilized_argmax(self, sim_tensor, max_iter, balance_lr, gravity_weight,
                          spatial_weight, momentum, anisotropy, centroid_margin,
                          border_penalty, bg_scale):
        """Stabilized argmax with spatial constraints."""
        num_concepts, H, W = sim_tensor.shape
        
        # Initialize assignment
        assignment = torch.argmax(sim_tensor, dim=0).float()
        velocity = torch.zeros_like(assignment)
        
        # Create spatial grid
        y_grid, x_grid = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        y_grid = y_grid.float() / H
        x_grid = x_grid.float() / W
        
        # Apply anisotropy
        x_grid = x_grid * anisotropy
        
        for iteration in range(max_iter):
            new_assignment = assignment.clone()
            
            for c in range(num_concepts):
                # Get current mask for this concept
                mask = (assignment == c).float()
                
                # Compute centroid
                total = mask.sum()
                if total > 0:
                    cy = (mask * y_grid).sum() / total
                    cx = (mask * x_grid).sum() / total
                    
                    # Distance to centroid
                    dist_y = (y_grid - cy) ** 2
                    dist_x = (x_grid - cx) ** 2
                    dist = dist_y * gravity_weight + dist_x * gravity_weight * anisotropy
                    
                    # Border penalty
                    if border_penalty > 0:
                        border_dist = torch.minimum(
                            torch.minimum(y_grid, 1 - y_grid),
                            torch.minimum(x_grid, 1 - x_grid)
                        )
                        dist = dist + border_penalty * (1 - border_dist)
                    
                    # Combine with similarity
                    combined = sim_tensor[c] - dist
                    
                    # Spatial smoothing
                    if spatial_weight > 0:
                        # Simple 3x3 smoothing
                        padded = F.pad(combined.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
                        kernel = torch.ones(1, 1, 3, 3) / 9.0
                        smoothed = F.conv2d(padded, kernel).squeeze()
                        combined = combined + spatial_weight * smoothed
                    
                    # Update assignment where this concept wins
                    winner = torch.argmax(sim_tensor - dist.unsqueeze(0).expand(num_concepts, H, W), dim=0)
                    new_assignment = torch.where(winner == c, torch.ones_like(assignment) * c, new_assignment)
            
            # Apply momentum
            delta = new_assignment - assignment
            velocity = momentum * velocity + (1 - momentum) * delta
            assignment = assignment + balance_lr * velocity
            assignment = assignment.clamp(0, num_concepts - 1).round()
        
        return assignment.long()

    def _create_info_string(self, all_sim_maps, block_start, block_end):
        """Create an info string with statistics for each block."""
        lines = ["LTX-Video Block Analysis", "=" * 40]
        
        for block_idx in range(block_start, block_end + 1):
            if block_idx not in all_sim_maps:
                continue
            
            block_maps = all_sim_maps[block_idx]
            lines.append(f"\nBlock {block_idx}:")
            
            for name, sim in block_maps.items():
                if name.startswith("__"):
                    continue
                
                sim_cpu = sim.cpu()
                min_val = sim_cpu.min().item()
                max_val = sim_cpu.max().item()
                mean_val = sim_cpu.mean().item()
                
                lines.append(f"  {name}: min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}")
        
        return "\n".join(lines)


class LTXBlockGridHook:
    """Hook for extracting attention from LTX-Video transformer blocks."""

    def __init__(
        self,
        target_block,
        block_index: int,
        token_pos_maps: Dict[str, List[List[int]]],
        temperature: float,
        top_k_ratio: float,
        img_len: int,
        all_sim_maps: Dict[int, Dict[str, torch.Tensor]],
    ):
        self.target_block = target_block
        self.block_index = block_index
        self.token_pos_maps = token_pos_maps
        self.temperature = temperature
        self.top_k_ratio = top_k_ratio
        self.img_len = img_len
        self.all_sim_maps = all_sim_maps

        self.cached_input = None
        self.hook_handle = None
        self.pre_hook_handle = None
        self.collected = False

    def install(self, model_patcher):
        """Install hooks on the target block."""
        self.pre_hook_handle = self.target_block.register_forward_pre_hook(
            self._pre_hook,
            with_kwargs=True
        )

        self.hook_handle = self.target_block.register_forward_hook(
            self._forward_hook
        )

        logging.info(f"[LTXBlockGridHook] Installed on block {self.block_index}")

    def remove(self):
        """Remove hooks."""
        if self.pre_hook_handle is not None:
            self.pre_hook_handle.remove()
            self.pre_hook_handle = None
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
        logging.info(f"[LTXBlockGridHook] Removed from block {self.block_index}")

    def _pre_hook(self, module, args, kwargs=None):
        """Capture LTX-Video block inputs before forward pass."""
        if self.collected:
            return

        if kwargs:
            self.cached_input = {
                "hidden_states": kwargs.get("hidden_states"),
                "encoder_hidden_states": kwargs.get("encoder_hidden_states"),
                "temb": kwargs.get("temb"),
                "image_rotary_emb": kwargs.get("image_rotary_emb"),
            }
        elif len(args) >= 2:
            self.cached_input = {
                "hidden_states": args[0] if len(args) > 0 else None,
                "encoder_hidden_states": args[1] if len(args) > 1 else None,
                "temb": kwargs.get("temb"),
                "image_rotary_emb": kwargs.get("image_rotary_emb"),
            }

        if self.cached_input:
            for key, value in self.cached_input.items():
                if isinstance(value, torch.Tensor):
                    self.cached_input[key] = value.detach()

    def _forward_hook(self, module, args, output):
        """Compute similarity maps from attention output."""
        if self.collected or self.cached_input is None:
            return output

        try:
            attn = getattr(module, 'attn', None)
            if attn is None:
                attn = getattr(module, 'attention', None)
            
            if attn is None:
                logging.warning("[LTXBlockGridHook] No attention module found")
                return output

            img_hidden = self.cached_input.get("hidden_states")
            txt_hidden = self.cached_input.get("encoder_hidden_states")

            if img_hidden is None or txt_hidden is None:
                logging.warning("[LTXBlockGridHook] Missing cached inputs")
                return output

            img_len = img_hidden.shape[1]
            cap_len = txt_hidden.shape[1]

            logging.info(f"[LTXBlockGridHook] Computing similarity maps at block {self.block_index}")

            # Compute similarity maps (same as LTXSimilarityExtractor)
            sim_maps = self._compute_similarity_from_block(
                module, attn, img_hidden, txt_hidden,
                img_len, cap_len
            )

            if sim_maps:
                self.all_sim_maps[self.block_index] = sim_maps
                self.collected = True
                logging.info(f"[LTXBlockGridHook] Collected {len(sim_maps)} similarity maps")

        except Exception as e:
            logging.error(f"[LTXBlockGridHook] Error: {e}")
            import traceback
            traceback.print_exc()

        self.cached_input = None
        return output

    def _compute_similarity_from_block(
        self,
        module,
        attn,
        img_hidden: torch.Tensor,
        txt_hidden: torch.Tensor,
        img_len: int,
        cap_len: int,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Compute similarity maps from block's attention computation."""

        try:
            # Get QKV projection layers
            to_q = getattr(attn, 'to_q', None)
            to_k = getattr(attn, 'to_k', None)
            to_v = getattr(attn, 'to_v', None)

            add_q_proj = getattr(attn, 'add_q_proj', None)
            add_k_proj = getattr(attn, 'add_k_proj', None)
            add_v_proj = getattr(attn, 'add_v_proj', None)

            if to_q is None or to_k is None or to_v is None:
                return None

            if add_q_proj is None or add_k_proj is None or add_v_proj is None:
                return None

            # Apply normalization if available
            img_attn_in = img_hidden
            txt_attn_in = txt_hidden

            if hasattr(module, 'norm1'):
                img_attn_in = module.norm1(img_hidden)
            if hasattr(module, 'norm1_context'):
                txt_attn_in = module.norm1_context(txt_hidden)

            # Project to QKV
            q = to_q(img_attn_in)
            k = to_k(img_attn_in)
            v = to_v(img_attn_in)

            txt_q = add_q_proj(txt_attn_in)
            txt_k = add_k_proj(txt_attn_in)
            txt_v = add_v_proj(txt_attn_in)

            # Apply QK norms if available
            norm_q = getattr(attn, 'norm_q', None)
            norm_k = getattr(attn, 'norm_k', None)
            norm_added_q = getattr(attn, 'norm_added_q', None)
            norm_added_k = getattr(attn, 'norm_added_k', None)

            if norm_q is not None:
                q = norm_q(q)
            if norm_k is not None:
                k = norm_k(k)
            if norm_added_q is not None:
                txt_q = norm_added_q(txt_q)
            if norm_added_k is not None:
                txt_k = norm_added_k(txt_k)

            # Reshape for multi-head attention (LTX: 32 heads, head dim 128)
            num_heads = 32
            head_dim = 128

            q = q.view(q.shape[0], q.shape[1], num_heads, head_dim).transpose(1, 2)
            k = k.view(k.shape[0], k.shape[1], num_heads, head_dim).transpose(1, 2)
            txt_q = txt_q.view(txt_q.shape[0], txt_q.shape[1], num_heads, head_dim).transpose(1, 2)
            txt_k = txt_k.view(txt_k.shape[0], txt_k.shape[1], num_heads, head_dim).transpose(1, 2)

            # Compute similarity: img_q @ txt_k^T
            scale = 1.0 / (head_dim ** 0.5)
            similarity = torch.einsum('bhqd,bhkd->bhqk', q, txt_k) * scale
            similarity = similarity * (self.temperature / 1000.0)

            attention_weights = F.softmax(similarity, dim=-1)

            # Compute similarity maps for each concept
            sim_maps = {}

            for concept_name, positions_list in self.token_pos_maps.items():
                if not positions_list or not positions_list[0]:
                    continue

                positions = positions_list[0]

                concept_mask = torch.zeros_like(attention_weights[..., -cap_len:])
                for pos in positions:
                    if pos < cap_len:
                        concept_mask[..., pos] = 1.0

                concept_attention = (attention_weights[..., -cap_len:] * concept_mask).sum(dim=-1)

                # Apply top-k filtering
                if self.top_k_ratio < 1.0:
                    k_val = max(1, int(concept_attention.shape[-1] * self.top_k_ratio))
                    top_k_vals, top_k_indices = torch.topk(concept_attention, k_val, dim=-1)
                    filtered = torch.zeros_like(concept_attention).scatter_(-1, top_k_indices, top_k_vals)
                    concept_attention = filtered

                # Average over heads
                concept_attention = concept_attention.mean(dim=1, keepdim=True)
                concept_attention = concept_attention.squeeze(1).unsqueeze(-1)

                sim_maps[f"sim_{concept_name}"] = concept_attention

            return sim_maps

        except Exception as e:
            logging.error(f"[LTXBlockGridHook] Error: {e}")
            return None


# Node registration
NODE_CLASS_MAPPINGS = {
    "FreeFuseLTXBlockGridExtractor": FreeFuseLTXBlockGridExtractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeFuseLTXBlockGridExtractor": "🎬 FreeFuse LTX Block Grid Extractor",
}
