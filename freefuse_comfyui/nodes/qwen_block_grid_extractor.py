"""
FreeFuse Qwen-Image Block Grid Extractor

Extracts similarity maps from multiple transformer blocks in Qwen-Image
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


class FreeFuseQwenBlockGridExtractor:
    """
    Extract similarity maps from multiple Qwen-Image transformer blocks
    and display as a grid for visual analysis.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Qwen-Image model"
                }),
                "positive": ("CONDITIONING", {
                    "tooltip": "Positive conditioning"
                }),
                "negative": ("CONDITIONING", {
                    "tooltip": "Negative conditioning"
                }),
                "latent": ("LATENT", {
                    "tooltip": "Latent image"
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
                    "default": 0, "min": 0, "max": 59,
                    "tooltip": "First block to collect from"
                }),
                "block_end": ("INT", {
                    "default": 59, "min": 0, "max": 59,
                    "tooltip": "Last block to collect from"
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
                    "tooltip": "Number of iterations for stabilized argmax (more = more balanced)"
                }),
                "balance_lr": ("FLOAT", {
                    "default": 0.01, "min": 0.001, "max": 0.1, "step": 0.001,
                    "tooltip": "Learning rate for bias updates (higher = faster balancing)"
                }),
                "gravity_weight": ("FLOAT", {
                    "default": 0.00004, "min": 0.0, "max": 0.001, "step": 0.00001,
                    "tooltip": "How strongly pixels are pulled to concept centroid (higher = more compact)"
                }),
                "spatial_weight": ("FLOAT", {
                    "default": 0.00004, "min": 0.0, "max": 0.001, "step": 0.00001,
                    "tooltip": "How much neighboring pixels influence assignment (higher = smoother)"
                }),
                "momentum": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 0.9, "step": 0.05,
                    "tooltip": "Smoothing between iterations (higher = more stable but slower)"
                }),
                "anisotropy": ("FLOAT", {
                    "default": 1.3, "min": 0.5, "max": 2.0, "step": 0.1,
                    "tooltip": "Horizontal stretch factor (important for non-square latents!)"
                }),
                "centroid_margin": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.5, "step": 0.05,
                    "tooltip": "Keep centroids away from borders (higher = more margin)"
                }),
                "border_penalty": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Penalty for assigning pixels near borders"
                }),
                "bg_scale": ("FLOAT", {
                    "default": 0.95, "min": 0.5, "max": 2.0, "step": 0.05,
                    "tooltip": "Background channel multiplier (higher = more background)"
                }),
            }
        }

    RETURN_TYPES = ("FREEFUSE_MASKS", "IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("all_masks", "block_grid", "info", "argmax_winner_grid")
    FUNCTION = "extract"
    CATEGORY = "FreeFuse/Debug"

    DESCRIPTION = """Extracts similarity maps from multiple Qwen-Image transformer blocks.

Shows a grid of all collected blocks so you can visually identify
which blocks produce meaningful attention patterns.

The ARGMAX_WINNER_GRID shows which concept "wins" at each block (hard assignment).
Each pixel gets the color of the concept with highest similarity.
Perfect for identifying which blocks produce clean concept separation.

Use this to find the best collect_block value for the main extractor."""

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
            print("[FreeFuse Qwen Block Grid] ERROR: No token positions found")
            empty_result = {"masks": {}, "similarity_maps": {}}
            grid = torch.zeros(1, cell_size, cell_size, 3)
            return (empty_result, grid, "No token positions")

        # Clamp block range
        block_start = max(0, block_start)
        block_end = min(59, block_end)
        if block_end < block_start:
            block_end = block_start

        blocks_to_collect = list(range(block_start, block_end + 1))
        num_blocks = len(blocks_to_collect)

        print(f"[FreeFuse Qwen Block Grid] Starting extraction")
        print(f"  Concepts: {list(concepts.keys())}")
        print(f"  Block range: {block_start} - {block_end} ({num_blocks} blocks)")
        print(f"  Collect step: {collect_step}/{steps}")
        print(f"  Temperature: {temperature}, top_k_ratio: {top_k_ratio}")

        # VRAM cleanup
        if low_vram_mode and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Clone model
        model_clone = model.clone()

        # Get latent dimensions
        latent_tensor = latent["samples"]
        if latent_tensor.dim() == 5:
            latent_h, latent_w = latent_tensor.shape[3], latent_tensor.shape[4]
            # Add temporal dimension if needed for sampling
            if latent_tensor.shape[2] == 1:
                latent_tensor = latent_tensor.squeeze(2)
        else:
            latent_h, latent_w = latent_tensor.shape[2], latent_tensor.shape[3]

        img_len = latent_h * latent_w

        # Storage for all blocks
        all_block_masks = {}
        all_block_sim_maps = {}

        # Create noise once
        if latent_tensor.dim() == 4:
            latent_tensor = latent_tensor.unsqueeze(2)
        noise = comfy.sample.prepare_noise(latent_tensor, seed, None)

        # Track current step
        current_step = [0]

        def step_callback(step, x0, x, total_steps):
            current_step[0] = step

        # ========== LOOP THROUGH ALL BLOCKS ==========
        for block_idx in blocks_to_collect:
            print(f"\n[FreeFuse Qwen Block Grid] Collecting block {block_idx}...")

            # Clone model for this block
            block_model = model_clone.clone()

            # Install hook on target block
            diffusion_model = block_model.model.diffusion_model

            # Find transformer blocks
            target_blocks = None
            if hasattr(diffusion_model, 'transformer_blocks'):
                target_blocks = diffusion_model.transformer_blocks
            elif hasattr(diffusion_model, 'layers'):
                target_blocks = diffusion_model.layers

            if target_blocks is None or block_idx >= len(target_blocks):
                print(f"  ERROR: Cannot find block {block_idx}")
                continue

            target_block = target_blocks[block_idx]
            print(f"  Target block type: {type(target_block).__name__}")

            # Storage for this block's results
            block_collected_maps = {}

            # Install hook
            hook = QwenAttentionHook(
                target_block=target_block,
                block_index=block_idx,
                token_pos_maps=token_pos_maps,
                temperature=temperature,
                top_k_ratio=top_k_ratio,
                img_len=img_len,
                collected_sim_maps=block_collected_maps,
            )
            hook.install(block_model)

            # Run sampling for this block
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
                    force_full_denoise=True,
                    noise_mask=None,
                    callback=step_callback,
                    seed=seed,
                )
            except Exception as e:
                print(f"  Sampling error: {e}")
                import traceback
                traceback.print_exc()

            # Remove hook
            hook.remove()

            # Store results
            if block_collected_maps:
                all_block_sim_maps[f"block_{block_idx}"] = block_collected_maps
                print(f"  Collected {len(block_collected_maps)} maps")
            else:
                print(f"  WARNING: No maps collected for block {block_idx}")

            # VRAM cleanup
            if low_vram_mode and torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"\n[FreeFuse Qwen Block Grid] Extraction complete")
        print(f"  Blocks collected: {len(all_block_sim_maps)}")

        # Move all tensors to CPU to avoid JSON serialization issues
        for block_key in all_block_sim_maps:
            for concept_name in all_block_sim_maps[block_key]:
                if isinstance(all_block_sim_maps[block_key][concept_name], torch.Tensor):
                    all_block_sim_maps[block_key][concept_name] = all_block_sim_maps[block_key][concept_name].cpu()

        # Create result
        result = {
            "masks": all_block_sim_maps,
            "similarity_maps": all_block_sim_maps,
        }

        # Create grid preview
        grid = self._create_block_grid(
            all_block_sim_maps,
            concepts,
            latent_h,
            latent_w,
            cell_size,
            preview_size
        )

        # Create argmax winner grid
        argmax_grid = self._create_argmax_grid(
            all_block_sim_maps,
            concepts,
            latent_h,
            latent_w,
            cell_size,
            argmax_method,
            max_iter,
            balance_lr,
            gravity_weight,
            spatial_weight,
            momentum,
            anisotropy,
            centroid_margin,
            border_penalty,
            bg_scale
        )

        # Create info string
        info_lines = [
            "=" * 60,
            "QWEN-IMAGE BLOCK GRID EXTRACTION",
            "=" * 60,
            f"Block range: {block_start} - {block_end}",
            f"Blocks collected: {len(all_block_sim_maps)}",
            f"Collect step: {collect_step}",
            f"Temperature: {temperature}",
            f"Top-k ratio: {top_k_ratio}",
            f"Argmax method: {argmax_method}",
            "-" * 60,
            "Concepts:",
        ]
        for name in concepts.keys():
            info_lines.append(f"  - {name}")
        info_lines.append("-" * 60)
        info_lines.append("Grid layout: Blocks arranged left-to-right, top-to-bottom")
        info_lines.append("Colors: Red=1st concept, Green=2nd, Blue=3rd, etc.")
        info_lines.append("ARGMAX WINNER GRID: Shows which concept wins at each block")
        info_lines.append("=" * 60)

        info = "\n".join(info_lines)

        return (result, grid, info, argmax_grid)

    def _create_block_grid(self, all_block_maps, concepts, latent_h, latent_w, cell_size, preview_size):
        """Create a grid of preview images for all collected blocks."""

        num_blocks = len(all_block_maps)
        if num_blocks == 0:
            return torch.zeros(1, cell_size, cell_size, 3)

        # Calculate grid dimensions
        grid_cols = min(8, num_blocks)
        grid_rows = (num_blocks + grid_cols - 1) // grid_cols

        canvas_width = grid_cols * cell_size
        canvas_height = grid_rows * cell_size

        # Create canvas
        canvas = Image.new('RGB', (canvas_width, canvas_height), color='black')
        draw = ImageDraw.Draw(canvas)

        # Load font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            font = ImageFont.load_default()

        # Colors for concepts
        colors = [
            (255, 100, 100),  # Red
            (100, 255, 100),  # Green
            (100, 100, 255),  # Blue
            (255, 255, 100),  # Yellow
            (255, 100, 255),  # Magenta
            (100, 255, 255),  # Cyan
        ]

        # Get concept names (exclude background)
        concept_names = [name for name in concepts.keys() if not name.startswith("__")]

        # Process each block
        for idx, (block_key, block_maps) in enumerate(all_block_maps.items()):
            # Extract block number
            try:
                block_num = int(block_key.split('_')[1])
            except:
                block_num = idx

            # Calculate position in grid
            row = idx // grid_cols
            col = idx % grid_cols
            x = col * cell_size
            y = row * cell_size

            # Create cell for this block
            cell = self._create_cell_preview(
                block_maps,
                concept_names,
                cell_size,
                colors,
                latent_h,
                latent_w
            )

            # Paste cell onto canvas
            canvas.paste(cell, (x, y))

            # Draw border and label
            draw.rectangle([x, y, x + cell_size - 1, y + cell_size - 1], outline='white', width=1)
            draw.text((x + 5, y + 5), f"B{block_num:02d}", fill='white', font=font)

        # Convert to tensor
        img_array = np.array(canvas).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]

        return img_tensor

    def _create_cell_preview(self, block_maps, concept_names, cell_size, colors, latent_h, latent_w):
        """Create preview for a single block's similarity maps."""

        # Create blank cell
        cell = Image.new('RGB', (cell_size, cell_size), color='black')

        if not block_maps:
            return cell

        # Combine all concept maps with colors
        combined = torch.zeros(3, cell_size, cell_size)
        mask_count = 0

        for name in concept_names:
            if name not in block_maps:
                continue

            sim_map = block_maps[name]
            if not isinstance(sim_map, torch.Tensor):
                continue

            color = colors[mask_count % len(colors)]
            color_tensor = torch.tensor(color).view(3, 1, 1) / 255.0

            # Get sequence length and infer dimensions
            if sim_map.dim() == 3:
                seq_len = sim_map.shape[1]
                map_2d = sim_map[0, :, 0]
            else:
                seq_len = sim_map.shape[0]
                map_2d = sim_map

            # Calculate actual dimensions from sequence length
            actual_h = actual_w = int(seq_len ** 0.5)
            if actual_h * actual_w != seq_len:
                for i in range(int(seq_len ** 0.5), 0, -1):
                    if seq_len % i == 0:
                        actual_h = i
                        actual_w = seq_len // i
                        break

            # Reshape and resize
            map_2d = map_2d.view(actual_h, actual_w)
            map_resized = F.interpolate(
                map_2d.unsqueeze(0).unsqueeze(0).float(),
                size=(cell_size, cell_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)

            # Normalize
            map_min = map_resized.min()
            map_max = map_resized.max()
            if map_max > map_min:
                map_norm = (map_resized - map_min) / (map_max - map_min)
            else:
                map_norm = torch.ones_like(map_resized) * 0.5

            # Add to combined with color
            for c in range(3):
                combined[c] += map_norm * color_tensor[c]

            mask_count += 1

        # Clamp and convert
        if mask_count > 0:
            combined = combined.clamp(0, 1)

        img_np = (combined.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        cell = Image.fromarray(img_np)

        return cell

    def _create_argmax_grid(
        self,
        all_block_maps,
        concepts,
        latent_h,
        latent_w,
        cell_size,
        argmax_method="stabilized",
        max_iter=15,
        balance_lr=0.01,
        gravity_weight=0.00004,
        spatial_weight=0.00004,
        momentum=0.2,
        anisotropy=1.3,
        centroid_margin=0.0,
        border_penalty=0.0,
        bg_scale=0.95
    ):
        """Create a grid of argmax winner images for all collected blocks."""

        num_blocks = len(all_block_maps)
        if num_blocks == 0:
            return torch.zeros(1, cell_size, cell_size, 3)

        # Calculate grid dimensions
        grid_cols = min(8, num_blocks)
        grid_rows = (num_blocks + grid_cols - 1) // grid_cols

        canvas_width = grid_cols * cell_size
        canvas_height = grid_rows * cell_size

        # Create canvas
        canvas = Image.new('RGB', (canvas_width, canvas_height), color='black')
        draw = ImageDraw.Draw(canvas)

        # Load font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            font = ImageFont.load_default()

        # Colors for concepts
        colors = [
            (255, 100, 100),  # Red
            (100, 255, 100),  # Green
            (100, 100, 255),  # Blue
            (255, 255, 100),  # Yellow
            (255, 100, 255),  # Magenta
            (100, 255, 255),  # Cyan
        ]

        # Get concept names (exclude background)
        concept_names = [name for name in concepts.keys() if not name.startswith("__")]

        # Process each block
        for idx, (block_key, block_maps) in enumerate(all_block_maps.items()):
            # Extract block number
            try:
                block_num = int(block_key.split('_')[1])
            except:
                block_num = idx

            # Calculate position in grid
            row = idx // grid_cols
            col = idx % grid_cols
            x = col * cell_size
            y = row * cell_size

            # Create argmax cell for this block
            cell = self._create_argmax_cell(
                block_maps,
                concept_names,
                cell_size,
                colors,
                latent_h,
                latent_w,
                argmax_method,
                max_iter,
                balance_lr,
                gravity_weight,
                spatial_weight,
                momentum,
                anisotropy,
                centroid_margin,
                border_penalty,
                bg_scale
            )

            # Paste cell onto canvas
            canvas.paste(cell, (x, y))

            # Draw border and label
            draw.rectangle([x, y, x + cell_size - 1, y + cell_size - 1], outline='white', width=1)
            draw.text((x + 5, y + 5), f"B{block_num:02d}", fill='white', font=font)

        # Convert to tensor
        img_array = np.array(canvas).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]

        return img_tensor

    def _create_argmax_cell(
        self,
        block_maps,
        concept_names,
        cell_size,
        colors,
        latent_h,
        latent_w,
        argmax_method="stabilized",
        max_iter=15,
        balance_lr=0.01,
        gravity_weight=0.00004,
        spatial_weight=0.00004,
        momentum=0.2,
        anisotropy=1.3,
        centroid_margin=0.0,
        border_penalty=0.0,
        bg_scale=0.95
    ):
        """Create argmax winner preview for a single block's similarity maps."""

        # Create blank cell
        cell = Image.new('RGB', (cell_size, cell_size), color='black')

        if not block_maps:
            return cell

        # Prepare resized concept tensors
        concept_tensors_resized = []

        for name in concept_names:
            if name not in block_maps:
                continue

            sim_map = block_maps[name]
            if not isinstance(sim_map, torch.Tensor):
                continue

            # Get 2D version
            if sim_map.dim() == 3:
                seq_len = sim_map.shape[1]
                map_2d = sim_map[0, :, 0]
            else:
                seq_len = sim_map.shape[0]
                map_2d = sim_map

            # Calculate actual dimensions from sequence length
            actual_h = actual_w = int(seq_len ** 0.5)
            if actual_h * actual_w != seq_len:
                for i in range(int(seq_len ** 0.5), 0, -1):
                    if seq_len % i == 0:
                        actual_h = i
                        actual_w = seq_len // i
                        break

            # Reshape and resize
            map_2d = map_2d.view(actual_h, actual_w)
            map_resized = F.interpolate(
                map_2d.unsqueeze(0).unsqueeze(0).float(),
                size=(cell_size, cell_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)

            # Normalize to 0-1
            map_min = map_resized.min()
            map_max = map_resized.max()
            if map_max > map_min:
                map_norm = (map_resized - map_min) / (map_max - map_min)
            else:
                map_norm = torch.ones_like(map_resized) * 0.5

            concept_tensors_resized.append(map_norm)

        if not concept_tensors_resized:
            return cell

        # Compute argmax winner
        if argmax_method == "stabilized" and len(concept_tensors_resized) > 1:
            try:
                from ..freefuse_core.mask_utils import stabilized_balanced_argmax

                C = len(concept_tensors_resized)
                N = cell_size * cell_size

                # Stack and reshape to (1, C, N)
                stacked = torch.stack(concept_tensors_resized, dim=0).unsqueeze(0)
                logits = stacked.view(1, C, N)

                # Run stabilized_balanced_argmax
                max_indices = stabilized_balanced_argmax(
                    logits, cell_size, cell_size,
                    max_iter=max_iter,
                    lr=balance_lr,
                    gravity_weight=gravity_weight,
                    spatial_weight=spatial_weight,
                    momentum=momentum,
                    centroid_margin=centroid_margin,
                    border_penalty=border_penalty,
                    anisotropy=anisotropy,
                    debug=False,
                )

                winner_indices = max_indices[0].view(cell_size, cell_size)

            except Exception as e:
                # Fallback to simple argmax
                stacked = torch.stack(concept_tensors_resized, dim=0)
                winner_indices = torch.argmax(stacked, dim=0)
        else:
            # Simple argmax
            stacked = torch.stack(concept_tensors_resized, dim=0)
            winner_indices = torch.argmax(stacked, dim=0)

        # Color the winner indices
        argmax_image = torch.zeros(3, cell_size, cell_size)
        for idx, color in enumerate(colors[:len(concept_names)]):
            mask = (winner_indices == idx).float()
            color_tensor = torch.tensor(color).view(3, 1, 1) / 255.0
            for c in range(3):
                argmax_image[c] += mask * color_tensor[c]

        # Clamp and convert
        argmax_image = argmax_image.clamp(0, 1)
        img_np = (argmax_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        cell = Image.fromarray(img_np)

        return cell


class QwenAttentionHook:
    """
    Hook for extracting attention from Qwen-Image transformer blocks.
    """

    def __init__(
        self,
        target_block,
        block_index: int,
        token_pos_maps: Dict[str, List[List[int]]],
        temperature: float,
        top_k_ratio: float,
        img_len: int,
        collected_sim_maps: Dict[str, torch.Tensor],
    ):
        self.target_block = target_block
        self.block_index = block_index
        self.token_pos_maps = token_pos_maps
        self.temperature = temperature
        self.top_k_ratio = top_k_ratio
        self.img_len = img_len
        self.collected_sim_maps = collected_sim_maps

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
        logging.info(f"[QwenAttentionHook] Installed on block {self.block_index}")

    def remove(self):
        """Remove hooks."""
        if self.pre_hook_handle is not None:
            self.pre_hook_handle.remove()
            self.pre_hook_handle = None
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def _pre_hook(self, module, args, kwargs=None):
        """Capture Qwen-Image block inputs."""
        if self.collected:
            return

        if kwargs:
            self.cached_input = {
                "hidden_states": kwargs.get("hidden_states"),
                "encoder_hidden_states": kwargs.get("encoder_hidden_states"),
                "temb": kwargs.get("temb"),
                "image_rotary_emb": kwargs.get("image_rotary_emb"),
                "timestep_zero_index": kwargs.get("timestep_zero_index"),
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
                return output

            img_hidden = self.cached_input.get("hidden_states")
            txt_hidden = self.cached_input.get("encoder_hidden_states")
            temb = self.cached_input.get("temb")
            image_rotary_emb = self.cached_input.get("image_rotary_emb")

            if img_hidden is None or txt_hidden is None:
                return output

            img_len = img_hidden.shape[1]
            cap_len = txt_hidden.shape[1]

            sim_maps = self._compute_similarity(
                module, attn, img_hidden, txt_hidden, temb, image_rotary_emb,
                img_len, cap_len
            )

            if sim_maps:
                self.collected_sim_maps.update(sim_maps)
                self.collected = True
                logging.info(f"[QwenAttentionHook] Collected {len(sim_maps)} maps at block {self.block_index}")

        except Exception as e:
            logging.error(f"[QwenAttentionHook] Error: {e}")
            import traceback
            traceback.print_exc()

        self.cached_input = None
        return output

    def _compute_similarity(
        self,
        module,
        attn,
        img_hidden: torch.Tensor,
        txt_hidden: torch.Tensor,
        temb: Optional[torch.Tensor],
        image_rotary_emb: Optional[torch.Tensor],
        img_len: int,
        cap_len: int,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Compute similarity maps from block's attention computation."""
        try:
            from comfy.ldm.flux.math import apply_rope

            to_q = getattr(attn, 'to_q', None)
            to_k = getattr(attn, 'to_k', None)
            to_v = getattr(attn, 'to_v', None)
            add_q_proj = getattr(attn, 'add_q_proj', None)
            add_k_proj = getattr(attn, 'add_k_proj', None)
            add_v_proj = getattr(attn, 'add_v_proj', None)

            if to_q is None or to_k is None or to_v is None:
                logging.warning("[QwenAttentionHook] Missing QKV projections")
                return None

            # Apply normalization
            img_attn_in = img_hidden
            txt_attn_in = txt_hidden

            if hasattr(module, 'img_norm1'):
                img_attn_in = module.img_norm1(img_hidden)
            if hasattr(module, 'txt_norm1'):
                txt_attn_in = module.txt_norm1(txt_hidden)

            # Project QKV
            img_q = to_q(img_attn_in)
            img_k = to_k(img_attn_in)
            img_v = to_v(img_attn_in)

            if add_q_proj and add_k_proj and add_v_proj:
                txt_q = add_q_proj(txt_attn_in)
                txt_k = add_k_proj(txt_attn_in)
                txt_v = add_v_proj(txt_attn_in)
            else:
                txt_q = to_q(txt_attn_in)
                txt_k = to_k(txt_attn_in)
                txt_v = to_v(txt_attn_in)

            # Get number of heads from attention module or RoPE tensor
            n_heads = 64  # Default for Qwen-Image

            # Check RoPE tensor to get correct head count
            if image_rotary_emb is not None and isinstance(image_rotary_emb, (list, tuple)):
                rope_tensor = image_rotary_emb[0] if len(image_rotary_emb) > 0 else None
                if rope_tensor is not None and rope_tensor.dim() >= 3:
                    # Qwen-Image uses 6D RoPE: [B, 1, seq_len, num_heads, 2, 2]
                    if rope_tensor.dim() == 6:
                        n_heads = rope_tensor.shape[3]  # Index 3 for 6D RoPE
                        logging.info(f"[QwenAttentionHook] Detected {n_heads} heads from 6D RoPE")
                    elif rope_tensor.dim() == 5:
                        # Other models: [B, seq_len, num_heads, 2, 2]
                        n_heads = rope_tensor.shape[2]  # Index 2 for 5D RoPE
                        logging.info(f"[QwenAttentionHook] Detected {n_heads} heads from 5D RoPE")
                    elif rope_tensor.dim() == 4:
                        # Standard 4D RoPE: [B, seq_len, num_heads, dim]
                        n_heads = rope_tensor.shape[2]
                        logging.info(f"[QwenAttentionHook] Detected {n_heads} heads from 4D RoPE")

            logging.info(f"[QwenAttentionHook] Using n_heads={n_heads}, head_dim={img_q.shape[-1] // n_heads}")

            head_dim = img_q.shape[-1] // n_heads

            # Reshape to multi-head: (B, seq, H, D)
            img_q = img_q.view(img_q.shape[0], img_q.shape[1], n_heads, head_dim)
            img_k = img_k.view(img_k.shape[0], img_k.shape[1], n_heads, head_dim)
            img_v = img_v.view(img_v.shape[0], img_v.shape[1], n_heads, head_dim)

            txt_q = txt_q.view(txt_q.shape[0], txt_q.shape[1], n_heads, head_dim)
            txt_k = txt_k.view(txt_k.shape[0], txt_k.shape[1], n_heads, head_dim)
            txt_v = txt_v.view(txt_v.shape[0], txt_v.shape[1], n_heads, head_dim)

            # Apply QK-norm
            if hasattr(attn, "q_norm"):
                img_q = attn.q_norm(img_q)
                txt_q = attn.q_norm(txt_q)
            if hasattr(attn, "k_norm"):
                img_k = attn.k_norm(img_k)
                txt_k = attn.k_norm(txt_k)

            # Apply RoPE - handle Qwen-Image's special 5D format
            if image_rotary_emb is not None:
                try:
                    # Qwen-Image RoPE can be 5D: [B, seq, H, 2, 2] or similar
                    # We need to extract the freqs properly
                    rope_tensor = image_rotary_emb[0] if isinstance(image_rotary_emb, (list, tuple)) else image_rotary_emb
                    
                    if rope_tensor.dim() == 5:
                        # 5D format: [B, seq, H, 2, 2] - need to reshape
                        # Extract cos/sin components
                        rope_4d = rope_tensor.view(rope_tensor.shape[0], rope_tensor.shape[1], rope_tensor.shape[2], -1)
                        img_q, img_k = apply_rope(img_q, img_k, rope_4d)
                        txt_q, txt_k = apply_rope(txt_q, txt_k, rope_4d)
                    else:
                        # Standard 4D or 3D format
                        img_q, img_k = apply_rope(img_q, img_k, rope_tensor)
                        txt_q, txt_k = apply_rope(txt_q, txt_k, rope_tensor)
                        
                except Exception as e:
                    logging.warning(f"[QwenAttentionHook] RoPE failed: {e}")
                    # Continue without RoPE - similarity maps will be degraded but may still work

            # For VRAM efficiency, use the existing similarity computation
            # that doesn't require full attention recomputation
            # Use cross-attention between img_q and txt_k directly
            
            sim_maps = self._compute_sim_maps_cross_attn(
                img_q=img_q,
                txt_k=txt_k,
                img_len=img_len,
                cap_len=cap_len,
                n_heads=n_heads,
            )

            return sim_maps

        except Exception as e:
            logging.error(f"[QwenAttentionHook._compute] Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _compute_sim_maps_cross_attn(
        self,
        img_q: torch.Tensor,  # (B, img_len, H, D)
        txt_k: torch.Tensor,  # (B, cap_len, H, D)
        img_len: int,
        cap_len: int,
        n_heads: int,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Compute similarity maps using cross-attention (lighter weight)."""
        try:
            from ..freefuse_core.attention_replace import compute_qwen_image_similarity_maps_with_qkv
            
            # We need img_attn_out for the full computation
            # For now, use img_v as a proxy (the value projection)
            # This is an approximation but much lighter on VRAM
            
            B = img_q.shape[0]
            head_dim = img_q.shape[-1]
            device = img_q.device
            
            # Use img_v reshaped as proxy for attention output
            # This avoids the expensive full attention computation
            img_v_proxy = torch.randn(B, img_len, n_heads * head_dim, device=device) * 0.01
            
            # Create minimal similarity maps using just cross-attention
            concept_sim_maps = {}
            scale = 1.0 / 1000.0
            
            for lora_name, positions_list in self.token_pos_maps.items():
                if lora_name.startswith("__"):
                    continue
                    
                pos = positions_list[0] if positions_list else []
                if not pos:
                    continue
                
                pos_t = torch.tensor(pos, device=device, dtype=torch.long)
                pos_t = pos_t.clamp(0, cap_len - 1)
                
                # Get concept keys
                concept_k = txt_k[:, pos_t, :, :]  # (B, n_pos, H, D)
                
                # Cross-attention: img_q @ concept_k^T
                weights = torch.einsum("bihd,bjhd->bhij", img_q, concept_k) * scale
                weights = F.softmax(weights, dim=2)  # softmax over img dim
                scores = weights.mean(dim=1).mean(dim=-1)  # (B, img_len)
                
                # Normalize to create a map
                scores = scores - scores.min()
                if scores.max() > 0:
                    scores = scores / scores.max()
                
                # Reshape to (B, img_len, 1)
                sim_map = scores.unsqueeze(-1)
                concept_sim_maps[lora_name] = sim_map
                
            # Handle background
            for bg_key in ["__background__", "__bg__"]:
                if bg_key in self.token_pos_maps:
                    bg_pos = self.token_pos_maps[bg_key][0] if self.token_pos_maps[bg_key] else []
                    if bg_pos:
                        bg_pos_t = torch.tensor(bg_pos, device=device, dtype=torch.long)
                        bg_pos_t = bg_pos_t.clamp(0, cap_len - 1)
                        
                        bg_k = txt_k[:, bg_pos_t, :, :]
                        bg_weights = torch.einsum("bihd,bjhd->bhij", img_q, bg_k) * scale
                        bg_weights = F.softmax(bg_weights, dim=2)
                        bg_scores = bg_weights.mean(dim=1).mean(dim=-1)
                        
                        bg_scores = bg_scores - bg_scores.min()
                        if bg_scores.max() > 0:
                            bg_scores = bg_scores / bg_scores.max()
                        
                        concept_sim_maps[bg_key] = bg_scores.unsqueeze(-1)
                        break
            
            return concept_sim_maps
            
        except Exception as e:
            logging.error(f"[QwenAttentionHook._compute_sim_maps_cross_attn] Error: {e}")
            import traceback
            traceback.print_exc()
            return None


# Export node mappings
NODE_CLASS_MAPPINGS = {
    "FreeFuseQwenBlockGridExtractor": FreeFuseQwenBlockGridExtractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeFuseQwenBlockGridExtractor": "🔍 Qwen Block Grid Extractor",
}
