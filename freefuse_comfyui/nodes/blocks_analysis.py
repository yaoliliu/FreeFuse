"""
FreeFuse Research Node - Blocks Analysis

RESEARCH PURPOSES ONLY - SLOW!
Loopt door meerdere blocks en verzamelt masks.
Gebaseerd op werkende FreeFusePhase1Sampler code.
"""

import torch
import torch.nn.functional as F
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Optional, List, Tuple

import comfy.model_patcher
import comfy.sample
import comfy.samplers

from ..freefuse_core.bypass_lora_loader import (
    OffsetBypassInjectionManager,
    MultiAdapterBypassForwardHook,
)
from ..freefuse_core.attention_replace import (
    FreeFuseState,
    FreeFuseFluxBlockReplace,
    FreeFuseSDXLAttnReplace,
    FreeFuseZImageBlockReplace,
    apply_freefuse_replace_patches,
)
from ..freefuse_core.mask_utils import generate_masks
from ..freefuse_core.token_utils import detect_model_type
from ..freefuse_core.voting import create_consensus_image_rgb

# SDXL collect region map (zelfde als origineel)
SDXL_COLLECT_REGION_MAP = {
    "output_early ★ (recommended)": ("output", 0, 10),
    "output_mid": ("output", 1, 10),
    "output_late": ("output", 2, 10),
    "middle": ("middle", 0, 10),
    "input_deep": ("input", 7, 10),
    "input_deep_2": ("input", 8, 10),
    "output_shallow": ("output", 3, 2),
    "input_shallow": ("input", 4, 2),
}

class EarlyStopException(Exception):
    """Exception to signal early termination of sampling after mask collection."""
    pass


class FreeFuseBlocksAnalysis:
    """
    RESEARCH NODE - SLOW!
    Loopt door meerdere blocks en verzamelt masks.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "neg_conditioning": ("CONDITIONING",),
                "latent": ("LATENT",),
                "freefuse_data": ("FREEFUSE_DATA",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 150}),
                "collect_step": ("INT", {"default": 5, "min": 1, "max": 20}),
                "cfg": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                # Block range voor analyse
                "block_start": ("INT", {"default": 0, "min": 0, "max": 56}),
                "block_end": ("INT", {"default": 56, "min": 0, "max": 56}),
            },
            "optional": {
                # Block selection — Flux
                "collect_block": ("INT", {
                    "default": 18, "min": 0, "max": 56,
                }),
                # Block selection — SDXL
                "collect_region": (list(SDXL_COLLECT_REGION_MAP.keys()), {
                    "default": "output_early ★ (recommended)",
                }),
                "collect_tf_index": ("INT", {
                    "default": 3, "min": 0, "max": 9,
                }),
                # Similarity map computation parameters
                "temperature": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 10000.0, "step": 100.0,
                }),
                "top_k_ratio": ("FLOAT", {
                    "default": 0.3, "min": 0.01, "max": 1.0, "step": 0.05,
                }),
                # LoRA control
                "disable_lora_phase1": ("BOOLEAN", {
                    "default": True,
                }),
                # Mask post-processing parameters
                "bg_scale": ("FLOAT", {
                    "default": 0.95, "min": 0.0, "max": 2.0, "step": 0.05,
                }),
                "use_morphological_cleaning": ("BOOLEAN", {
                    "default": False,
                }),
                "balance_iterations": ("INT", {
                    "default": 15, "min": 0, "max": 50,
                }),
                # stabilized_balanced_argmax parameters
                "balance_lr": ("FLOAT", {
                    "default": 0.01, "min": 0.001, "max": 0.1, "step": 0.001,
                }),
                "gravity_weight": ("FLOAT", {
                    "default": 0.00004, "min": 0.0, "max": 1.0, "step": 0.00001,
                }),
                "spatial_weight": ("FLOAT", {
                    "default": 0.00004, "min": 0.0, "max": 1.0, "step": 0.00001,
                }),
                "momentum": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 0.9, "step": 0.05,
                }),
                "centroid_margin": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.5, "step": 0.05,
                }),
                "border_penalty": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "anisotropy": ("FLOAT", {
                    "default": 1.3, "min": 0.5, "max": 3.0, "step": 0.1,
                }),
                # Preview size
                "preview_size": ("INT", {
                    "default": 1024, "min": 512, "max": 2048,
                }),
                # 👈 NIEUW: Consensus bereik
                "consensus_start": ("INT", {
                    "default": 5, "min": 0, "max": 56,
                    "tooltip": "Start block voor consensus voting (skip vroege ruis)"
                }),
                "consensus_end": ("INT", {
                    "default": 56, "min": 0, "max": 56,
                    "tooltip": "Eind block voor consensus voting"
                }),
            }
        }
    
    # 👈 4 outputs: blocks_grid, info, all_blocks_data, consensus_image
    RETURN_TYPES = ("IMAGE", "STRING", "FREEFUSE_MASKS", "IMAGE")
    RETURN_NAMES = ("blocks_grid", "analysis_info", "all_blocks_data", "consensus_image")
    FUNCTION = "analyze_blocks"
    CATEGORY = "FreeFuse/Research"
    
    DESCRIPTION = """RESEARCH ONLY: Loop door meerdere blocks.
    
WARNING: Dit is traag! Gebruik block_start/block_end om bereik te beperken."""
    
    def analyze_blocks(
        self,
        model,
        conditioning,
        neg_conditioning,
        latent,
        freefuse_data,
        seed,
        steps,
        collect_step,
        cfg,
        sampler_name,
        scheduler,
        block_start=0,
        block_end=56,
        # Optional parameters met defaults
        collect_block=18,
        collect_region="output_early ★ (recommended)",
        collect_tf_index=3,
        temperature=0.0,
        top_k_ratio=0.3,
        disable_lora_phase1=True,
        bg_scale=0.95,
        use_morphological_cleaning=False,
        balance_iterations=15,
        balance_lr=0.01,
        gravity_weight=0.00004,
        spatial_weight=0.00004,
        momentum=0.2,
        centroid_margin=0.0,
        border_penalty=0.0,
        anisotropy=1.3,
        preview_size=1024,
        # 👈 Nieuwe consensus parameters
        consensus_start=5,
        consensus_end=56,
    ):
        # Extract data
        concepts = freefuse_data.get("concepts", {})
        token_pos_maps = freefuse_data.get("token_pos_maps", {})
        settings = freefuse_data.get("settings", {})
        include_background = settings.get("enable_background", True)
        
        if not concepts:
            print("[FreeFuseBlocksAnalysis] Warning: No concepts defined")
            empty_preview = torch.zeros(1, preview_size, preview_size, 3)
            return (empty_preview, "No concepts defined", {"masks": {}}, empty_preview)
        
        # Check token positions
        if not token_pos_maps:
            print("[FreeFuseBlocksAnalysis] Warning: No token positions computed.")
            latent_image = latent["samples"]
            latent_h, latent_w = latent_image.shape[2], latent_image.shape[3]
            masks = {name: torch.ones(latent_h, latent_w) for name in concepts}
            empty_preview = self._create_preview(masks, latent_w*8, latent_h*8)
            return (empty_preview, "No token positions", {"masks": masks}, empty_preview)
        
        # Detect model type
        model_type = detect_model_type(model=model)
        print(f"[FreeFuseBlocksAnalysis] Detected model type: {model_type}")

        # Get latent info
        latent_image = latent["samples"]
        
        # Handle 5D latents for Qwen-Image (B, C, T, H, W)
        if latent_image.dim() == 5:
            print(f"[FreeFuseBlocksAnalysis] Qwen-Image 5D latent detected: {latent_image.shape}")
            latent_h, latent_w = latent_image.shape[3], latent_image.shape[4]
        else:
            latent_h, latent_w = latent_image.shape[2], latent_image.shape[3]
        
        print(f"[FreeFuseBlocksAnalysis] Latent dimensions: H={latent_h}, W={latent_w}, img_len={latent_h*latent_w}")
        
        # Sanity check collect_step
        if collect_step > steps:
            print(f"[FreeFuseBlocksAnalysis] Warning: collect_step ({collect_step}) > steps ({steps}), clamping")
            collect_step = steps
        if collect_step < 1:
            collect_step = 1
        
        # Auto temperature
        if temperature == 0.0:
            if model_type == "z_image" or model_type == "qwen_image":
                auto_temperature = 4000.0
            elif model_type == "flux":
                auto_temperature = 4000.0
            else:
                auto_temperature = 300.0
        else:
            auto_temperature = temperature
        
        # Build SDXL collect_blocks
        sdxl_collect_blocks = None
        if model_type == "sdxl":
            region_info = SDXL_COLLECT_REGION_MAP.get(collect_region)
            if region_info:
                block_name, block_num, max_tf = region_info
                tf_idx = min(collect_tf_index, max_tf - 1)
                sdxl_collect_blocks = [(block_name, block_num, tf_idx)]
        
        # Storage voor alle blocks
        all_blocks_masks = {}
        all_similarity_maps = {}
        blocks_to_analyze = range(block_start, min(block_end + 1, 57))
        
        info_lines = []
        info_lines.append("=" * 60)
        info_lines.append(f"FREEFUSE BLOCKS ANALYSIS: {block_start}-{block_end}")
        info_lines.append("=" * 60)
        info_lines.append(f"Model Type: {model_type}")
        info_lines.append(f"Concepts: {list(concepts.keys())}")
        info_lines.append(f"Total blocks: {len(list(blocks_to_analyze))}")
        info_lines.append(f"Collect step: {collect_step}")
        info_lines.append("-" * 40)
        
        # Grid setup voor visualisatie
        grid_cols = 8
        grid_rows = (len(list(blocks_to_analyze)) + grid_cols - 1) // grid_cols
        cell_size = preview_size // 8
        canvas_width = grid_cols * cell_size
        canvas_height = grid_rows * cell_size
        
        canvas = Image.new('RGB', (canvas_width, canvas_height), color='black')
        draw = ImageDraw.Draw(canvas)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        # Create noise once (zelfde voor alle blocks)
        noise = comfy.sample.prepare_noise(latent_image, seed, None)
        
        # ========== LOOP DOOR ALLE BLOCKS ==========
        for idx, block_idx in enumerate(blocks_to_analyze):
            print(f"[FreeFuseBlocksAnalysis] Analyzing block {block_idx}...")
            
            # Clone model voor deze block
            model_clone = model.clone()
            
            # Setup FreeFuse state
            freefuse_state = FreeFuseState()
            freefuse_state.phase = "collect"
            freefuse_state.collect_step = collect_step - 1
            freefuse_state.collect_block = block_idx  # 👈 GEBRUIK LOOP INDEX
            freefuse_state.token_pos_maps = token_pos_maps
            freefuse_state.include_background = include_background
            freefuse_state.top_k_ratio = top_k_ratio
            freefuse_state.temperature = auto_temperature
            
            # Z-Image specific
            if model_type == "z_image":
                z_img_seq_len = latent_h * latent_w
                z_cap_seq_len = 256
                for positions_list in token_pos_maps.values():
                    if positions_list and positions_list[0]:
                        max_pos = max(positions_list[0])
                        z_cap_seq_len = max(z_cap_seq_len, max_pos + 10)
                freefuse_state.collected_outputs["img_seq_len"] = z_img_seq_len
                freefuse_state.collected_outputs["cap_seq_len"] = z_cap_seq_len
                freefuse_state.collected_outputs["latent_h"] = latent_h
                freefuse_state.collected_outputs["latent_w"] = latent_w

            # Qwen-Image specific (dual-stream MMDiT)
            if model_type == "qwen_image":
                qwen_img_seq_len = latent_h * latent_w
                qwen_cap_seq_len = 256  # Qwen2.5-VL default
                for positions_list in token_pos_maps.values():
                    if positions_list and positions_list[0]:
                        max_pos = max(positions_list[0])
                        qwen_cap_seq_len = max(qwen_cap_seq_len, max_pos + 10)
                freefuse_state.collected_outputs["img_seq_len"] = qwen_img_seq_len
                freefuse_state.collected_outputs["cap_seq_len"] = qwen_cap_seq_len
                freefuse_state.collected_outputs["latent_h"] = latent_h
                freefuse_state.collected_outputs["latent_w"] = latent_w
            
            # Apply replace patches
            apply_freefuse_replace_patches(
                model_clone,
                freefuse_state,
                model_type=model_type,
                sdxl_collect_blocks=sdxl_collect_blocks,
                flux_collect_blocks=[block_idx] if model_type == "flux" else None,
                qwen_collect_blocks=[block_idx] if model_type == "qwen_image" else None,
            )

            # Debug logging for qwen_image
            if model_type == "qwen_image":
                print(f"[FreeFuseBlocksAnalysis] Applied qwen_collect_blocks=[{block_idx}] for block {block_idx}")
            
            # Optioneel LoRA uitschakelen
            bypass_manager = model_clone.model_options.get("transformer_options", {}).get("freefuse_bypass_manager")
            if bypass_manager is not None and disable_lora_phase1:
                bypass_manager.disable_lora()
            
            # Step callback
            def step_callback(step, x0, x, total_steps):
                freefuse_state.current_step = step
                if freefuse_state.similarity_maps and step > freefuse_state.collect_step:
                    raise EarlyStopException()
            
            try:
                # Run sampling
                comfy.sample.sample(
                    model_clone,
                    noise,
                    steps,
                    cfg,
                    sampler_name,
                    scheduler,
                    conditioning,
                    neg_conditioning,
                    latent_image,
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
                print(f"[FreeFuseBlocksAnalysis] Block {block_idx} early stop")
            except Exception as e:
                print(f"[FreeFuseBlocksAnalysis] Error in block {block_idx}: {e}")
                continue
            
            # LoRA weer inschakelen
            if bypass_manager is not None and disable_lora_phase1:
                bypass_manager.enable_lora()

            # Get similarity maps
            similarity_maps = freefuse_state.similarity_maps
            
            # Debug logging for qwen_image
            if model_type == "qwen_image":
                print(f"[FreeFuseBlocksAnalysis] Block {block_idx}: Collected {len(similarity_maps)} similarity maps")
                if similarity_maps:
                    for name, sim_map in similarity_maps.items():
                        if isinstance(sim_map, torch.Tensor):
                            print(f"  {name}: shape={sim_map.shape}, min={sim_map.min():.6f}, max={sim_map.max():.6f}")
                else:
                    print(f"[FreeFuseBlocksAnalysis] WARNING: No similarity maps collected for block {block_idx}")

            # Generate masks
            if similarity_maps:
                masks = generate_masks(
                    similarity_maps,
                    include_background=include_background,
                    method="stabilized",
                    bg_scale=bg_scale,
                    use_morphological_cleaning=use_morphological_cleaning,
                    max_iter=balance_iterations,
                    latent_h=latent_h,
                    latent_w=latent_w,
                    balance_lr=balance_lr,
                    gravity_weight=gravity_weight,
                    spatial_weight=spatial_weight,
                    momentum=momentum,
                    centroid_margin=centroid_margin,
                    border_penalty=border_penalty,
                    anisotropy=anisotropy,
                )
            else:
                print(f"[FreeFuseBlocksAnalysis] WARNING: No similarity maps for block {block_idx}")
                masks = {}
                for name in concepts.keys():
                    masks[name] = torch.ones(latent_h, latent_w)
                if include_background:
                    masks["__background__"] = torch.zeros(latent_h, latent_w)
            
            # Store data
            all_blocks_masks[f"block_{block_idx}"] = masks
            all_similarity_maps[f"block_{block_idx}"] = similarity_maps
            
            # Create preview cell
            row = idx // grid_cols
            col = idx % grid_cols
            x = col * cell_size
            y = row * cell_size
            
            cell = self._create_block_preview(masks, block_idx, cell_size, font)
            canvas.paste(cell, (x, y))
            
            info_lines.append(f"Block {block_idx:02d}: ✓ Collected ({len(masks)} masks)")
        
        info_lines.append("=" * 60)
        info = "\n".join(info_lines)
        
        # Convert canvas to tensor
        img_array = np.array(canvas).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]

        # 👈 Maak een consensus image met instelbaar bereik
        # Use shared voting utility (same algorithm as sampler.py)
        consensus_image = self._create_consensus_image(
            all_blocks_masks,
            concepts,
            preview_size,
            consensus_start,
            consensus_end
        )
        
        output_data = {
            "masks": all_blocks_masks,
            "similarity_maps": all_similarity_maps,
        }
        
        return (img_tensor, info, output_data, consensus_image)
    
    def _create_block_preview(self, masks, block_idx, cell_size, font):
        """Create preview for one block."""
        cell = Image.new('RGB', (cell_size, cell_size), color='black')
        cell_draw = ImageDraw.Draw(cell)
        
        colors = [
            (255, 100, 100), (100, 255, 100), (100, 100, 255),
            (255, 255, 100), (255, 100, 255), (100, 255, 255),
        ]
        
        combined = torch.zeros(3, cell_size, cell_size)
        mask_count = 0
        
        for name, mask in masks.items():
            if name == "__background__":
                continue
            
            color = colors[mask_count % len(colors)]
            color_tensor = torch.tensor(color).view(3, 1, 1) / 255.0
            
            if isinstance(mask, torch.Tensor):
                if mask.dim() == 3:
                    mask_2d = mask[0]
                else:
                    mask_2d = mask
                mask_2d = mask_2d.float().cpu()
                
                mask_resized = F.interpolate(
                    mask_2d.unsqueeze(0).unsqueeze(0),
                    size=(cell_size, cell_size),
                    mode='bilinear'
                ).squeeze(0).squeeze(0)
                
                for c in range(3):
                    combined[c] += mask_resized * color_tensor[c]
                mask_count += 1
        
        if mask_count > 0:
            combined = combined.clamp(0, 1)
        
        img_np = (combined.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_preview = Image.fromarray(img_np)
        cell.paste(img_preview, (0, 0))
        
        cell_draw.rectangle([0, 0, cell_size-1, cell_size-1], outline='white', width=1)
        cell_draw.text((5, 5), f"B{block_idx:02d}", fill='white', font=font)
        
        return cell
    
    def _create_preview(self, masks, width, height):
        """Create a simple preview for fallback case."""
        colors = [
            (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0),
        ]
        
        device = next(iter(masks.values())).device if masks else "cpu"
        preview = torch.zeros(3, height, width, device=device)
        
        for i, (name, mask) in enumerate(masks.items()):
            color = colors[i % len(colors)]
            color_tensor = torch.tensor(color, device=device).view(3, 1, 1)
            
            if mask.dim() == 2:
                m = mask.unsqueeze(0).unsqueeze(0)
            else:
                m = mask.unsqueeze(0)
            
            mask_resized = F.interpolate(
                m.float(),
                size=(height, width),
                mode='nearest'
            ).squeeze()
            
            for c in range(3):
                preview[c] += mask_resized * color_tensor[c]
        
        preview = preview.clamp(0, 1).permute(1, 2, 0).unsqueeze(0).cpu()
        return preview
    
    # 👈 VERBETERDE FUNCTIE: Majority voting met instelbaar bereik
    def _create_consensus_image(self, all_blocks_masks: Dict, concepts: Dict,
                                preview_size: int, start_block: int, end_block: int) -> torch.Tensor:
        """
        Create a consensus image using majority voting across a range of blocks.
        Uses shared voting utility from freefuse_core.voting.

        Args:
            start_block: Eerste block om mee te nemen in voting
            end_block: Laatste block om mee te nemen in voting
        """
        if not all_blocks_masks:
            return torch.zeros(1, preview_size, preview_size, 3)

        # Get all concept names (including background)
        concept_names = list(concepts.keys())
        if "__background__" not in concept_names:
            concept_names.append("__background__")

        if not concept_names:
            return torch.zeros(1, preview_size, preview_size, 3)

        # Filter blocks by range
        filtered_blocks = {}
        blocks_used = 0

        for block_key, block_masks in all_blocks_masks.items():
            # Extract block number from key (e.g., "block_5" -> 5)
            try:
                block_num = int(block_key.split('_')[1])
            except:
                block_num = 0

            # Skip blocks outside the range
            if block_num < start_block or block_num > end_block:
                continue

            filtered_blocks[block_key] = block_masks
            blocks_used += 1

        print(f"[Consensus] Using {blocks_used} blocks from range {start_block}-{end_block}")

        # Use shared voting utility (same algorithm as sampler.py)
        device = next(iter(next(iter(all_blocks_masks.values())).values())).device
        return create_consensus_image_rgb(
            all_blocks_masks=filtered_blocks,
            concept_names=concept_names,
            preview_size=preview_size,
            device=device,
        )


# Export node mappings
NODE_CLASS_MAPPINGS = {
    "FreeFuseBlocksAnalysis": FreeFuseBlocksAnalysis,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeFuseBlocksAnalysis": "🔬 FreeFuse Blocks Analysis",
}
