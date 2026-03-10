"""
FreeFuse Research Node - Base Model Analysis

RESEARCH PURPOSES ONLY - SLOW!
Captures attention from the BASE model only (LoRAs disabled).
Use this to understand where the model naturally places concepts.
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


class FreeFuseBaseAnalysis:
    """
    RESEARCH NODE - SLOW!
    Captures BASE model attention (LoRAs disabled).
    Shows where the model naturally places concepts without LoRA influence.
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
                # Block range
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
            }
        }
    
    # 👈 VERANDERD: Nu 4 outputs in plaats van 3
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "FREEFUSE_MASKS")
    RETURN_NAMES = ("blocks_grid", "average_preview", "analysis_info", "all_blocks_data")
    FUNCTION = "analyze_base_attention"
    CATEGORY = "FreeFuse/Research"
    
    DESCRIPTION = """RESEARCH ONLY: Capture BASE model attention (LoRAs disabled).
    
Use this to see where the model naturally places concepts without LoRA influence.
Compare with FreeFuseBlocksAnalysis to see how LoRAs modify attention patterns."""
    
    def analyze_base_attention(
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
    ):
        # Extract data
        concepts = freefuse_data.get("concepts", {})
        token_pos_maps = freefuse_data.get("token_pos_maps", {})
        settings = freefuse_data.get("settings", {})
        include_background = settings.get("enable_background", True)
        
        if not concepts:
            print("[FreeFuseBaseAnalysis] Warning: No concepts defined")
            empty_preview = torch.zeros(1, preview_size, preview_size, 3)
            return (empty_preview, empty_preview, "No concepts defined", {"masks": {}})
        
        # Check token positions
        if not token_pos_maps:
            print("[FreeFuseBaseAnalysis] Warning: No token positions computed.")
            latent_image = latent["samples"]
            latent_h, latent_w = latent_image.shape[2], latent_image.shape[3]
            masks = {name: torch.ones(latent_h, latent_w) for name in concepts}
            empty_preview = self._create_preview(masks, latent_w*8, latent_h*8)
            return (empty_preview, empty_preview, "No token positions", {"masks": masks})
        
        # Detect model type
        model_type = detect_model_type(model=model)
        print(f"[FreeFuseBaseAnalysis] Detected model type: {model_type}")
        
        # Get latent info
        latent_image = latent["samples"]
        latent_h, latent_w = latent_image.shape[2], latent_image.shape[3]
        
        # Sanity check collect_step
        if collect_step > steps:
            print(f"[FreeFuseBaseAnalysis] Warning: collect_step ({collect_step}) > steps ({steps}), clamping")
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
        info_lines.append(f"BASE MODEL ANALYSIS: {block_start}-{block_end}")
        info_lines.append("=" * 60)
        info_lines.append(f"Model Type: {model_type}")
        info_lines.append(f"Concepts: {list(concepts.keys())}")
        info_lines.append(f"Total blocks: {len(list(blocks_to_analyze))}")
        info_lines.append(f"Collect step: {collect_step}")
        info_lines.append(f"LoRAs: DISABLED (base model only)")
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
            print(f"[FreeFuseBaseAnalysis] Analyzing BASE model at block {block_idx}...")
            
            # Clone model voor deze block
            model_clone = model.clone()
            
            # ===== DISABLE LoRAs =====
            transformer_options = model_clone.model_options.get("transformer_options", {})
            bypass_manager = transformer_options.get("freefuse_bypass_manager")
            if bypass_manager:
                bypass_manager.disable_lora()
                print(f"[FreeFuseBaseAnalysis] Block {block_idx}: LoRAs DISABLED")
            
            # Setup FreeFuse state
            freefuse_state = FreeFuseState()
            freefuse_state.phase = "collect"
            freefuse_state.collect_step = collect_step - 1
            freefuse_state.collect_block = block_idx
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
                print(f"[FreeFuseBaseAnalysis] Block {block_idx} early stop")
            except Exception as e:
                print(f"[FreeFuseBaseAnalysis] Error in block {block_idx}: {e}")
                continue
            
            # ===== RE-ENABLE LoRAs =====
            if bypass_manager:
                bypass_manager.enable_lora()
            
            # Get similarity maps
            similarity_maps = freefuse_state.similarity_maps
            
            # Generate masks (still needed for visualization)
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
                print(f"[FreeFuseBaseAnalysis] WARNING: No similarity maps for block {block_idx}")
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
            
            # cell = self._create_block_preview(masks, block_idx, cell_size, font)
            cell = self._create_block_preview(similarity_maps, block_idx, cell_size, font, token_pos_maps)
            canvas.paste(cell, (x, y))
            
            info_lines.append(f"Block {block_idx:02d}: ✓ Base attention collected")
        
        info_lines.append("=" * 60)
        info = "\n".join(info_lines)
        
        # Convert canvas to tensor
        img_array = np.array(canvas).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]
        
        # 👈 NIEUW: Maak een gemiddelde afbeelding van alle similarity maps
        avg_image = self._create_average_image(all_similarity_maps, preview_size)
        
        output_data = {
            "masks": all_blocks_masks,
            "similarity_maps": all_similarity_maps,
        }
        
        # 👈 VERANDERD: Return 4 outputs
        return (img_tensor, avg_image, info, output_data)
        
        
        
    def _create_average_image(self, all_similarity_maps: Dict, preview_size: int, contrast_boost: float = 3.0) -> torch.Tensor:
        """
        Create an average image from all similarity maps across blocks.
        Shows the "ideal" composition with enhanced contrast.
        
        Args:
            contrast_boost: Higher values = more contrast (2.0-5.0 recommended)
        """
        if not all_similarity_maps:
            return torch.zeros(1, preview_size, preview_size, 3)
        
        # Verzamel alle afbeeldingen van alle blocks
        all_images = []
        
        for block_key, sim_maps in all_similarity_maps.items():
            if not sim_maps:
                continue
                
            # Gebruik de eerste similarity map voor deze block
            first_map = next(iter(sim_maps.values()))
            
            # Converteer naar 2D afbeelding
            if first_map.dim() == 3:
                N = first_map.shape[1]
                h = w = int(N ** 0.5)
                if h * w != N:
                    for i in range(int(N ** 0.5), 0, -1):
                        if N % i == 0:
                            h = i
                            w = N // i
                            break
                img_2d = first_map[0, :, 0].view(h, w).float().cpu()
            else:
                img_2d = first_map.float().cpu()
            
            # Resize naar preview_size
            img_resized = F.interpolate(
                img_2d.unsqueeze(0).unsqueeze(0),
                size=(preview_size, preview_size),
                mode='bilinear'
            ).squeeze(0).squeeze(0)
            
            all_images.append(img_resized)
        
        if not all_images:
            return torch.zeros(1, preview_size, preview_size, 3)
        
        # Stack en average
        stacked = torch.stack(all_images)  # (num_blocks, H, W)
        averaged = stacked.mean(dim=0)      # (H, W)
        
        # 🔥 MEGA CONTRAST VERSTERKING
        # Optie 1: Log transform met extra boost
        averaged = torch.log1p(averaged * 100000) / 8
        
        # Optie 2: Power transform (gamma < 1 versterkt donkere gebieden)
        averaged = averaged ** 0.3
        
        # Optie 3: Combineer beide voor maximaal contrast
        # averaged = (torch.log1p(averaged * 100000) / 8) ** 0.4
        
        # Normaliseer naar 0-1 met clipping
        if averaged.max() > averaged.min():
            averaged = (averaged - averaged.min()) / (averaged.max() - averaged.min())
            # Extra contrast via sigmoid
            averaged = torch.sigmoid((averaged - 0.5) * contrast_boost * 2)
        else:
            averaged = torch.ones_like(averaged) * 0.5
        
        # Converteer naar 3-channel RGB (grijswaarden)
        averaged_rgb = averaged.unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)
        averaged_rgb = averaged_rgb.permute(1, 2, 0).unsqueeze(0)  # (1, H, W, 3)
        
        return averaged_rgb
    
    def _create_block_preview(self, similarity_maps, block_idx, cell_size, font, token_pos_maps):
        """
        Create preview showing attention for EACH INDIVIDUAL TOKEN.
        """
        cell = Image.new('RGB', (cell_size, cell_size), color='black')
        cell_draw = ImageDraw.Draw(cell)
        
        # Meer kleuren voor meer tokens
        colors = [
            (1.0, 0.0, 0.0),  # Rood
            (0.0, 1.0, 0.0),  # Groen
            (0.0, 0.0, 1.0),  # Blauw
            (1.0, 1.0, 0.0),  # Geel
            (1.0, 0.0, 1.0),  # Magenta
            (0.0, 1.0, 1.0),  # Cyaan
            (0.5, 0.0, 0.0),  # Donkerrood
            (0.0, 0.5, 0.0),  # Donkergroen
            (0.0, 0.0, 0.5),  # Donkerblauw
        ]
        
        combined = torch.zeros(3, cell_size, cell_size)
        token_count = 0
        
        # similarity_maps is per concept, maar we willen per token
        # We moeten de data herinterpreteren
        
        # Voor Z-Image: similarity_maps[concept] is (1, N, 1) 
        # waarbij N = img_len * cap_len? Nee, N = img_len (alleen image tokens)
        
        first_map = next(iter(similarity_maps.values()))
        if first_map.dim() == 3:
            N = first_map.shape[1]  # Aantal image tokens
            img_h = img_w = int(N ** 0.5)
            if img_h * img_w != N:
                # Zoek juiste aspect ratio
                for i in range(int(N ** 0.5), 0, -1):
                    if N % i == 0:
                        img_h = i
                        img_w = N // i
                        break
        
        # Nu maken we voor ELK token een aparte visualisatie
        token_idx = 0
        for concept_name, positions_list in token_pos_maps.items():
            if not positions_list or not positions_list[0]:
                continue
                
            concept_map = similarity_maps.get(concept_name)
            if concept_map is None:
                continue
                
            # Haal de attention voor dit concept
            if concept_map.dim() == 3:
                concept_attention = concept_map[0, :, 0].float().cpu()  # (N,)
            else:
                concept_attention = concept_map.float().cpu()
            
            # Voor ELKE token positie in dit concept
            for token_pos in positions_list[0]:  # [0] = eerste prompt
                color = colors[token_idx % len(colors)]
                color_tensor = torch.tensor(color).view(3, 1, 1)
                
                # Gebruik DEZELFDE attention voor elke token? 
                # Nee, we hebben alleen 1 attention map per concept
                # Maar we kunnen het visualiseren alsof elke token zijn eigen map heeft
                
                # Reshape naar 2D
                attention_2d = concept_attention.view(img_h, img_w)
                
                # Resize naar cell_size
                attention_resized = F.interpolate(
                    attention_2d.unsqueeze(0).unsqueeze(0),
                    size=(cell_size, cell_size),
                    mode='bilinear'
                ).squeeze(0).squeeze(0)
                
                # Log transform voor kleine waarden
                attention_resized = torch.log1p(attention_resized * 10000) / 10
                attention_resized = attention_resized.clamp(0, 1)
                
                # Voeg toe met de kleur van deze token
                for c in range(3):
                    combined[c] += attention_resized * color_tensor[c]
                
                token_idx += 1
        
        # Normaliseer per kanaal
        for c in range(3):
            if combined[c].max() > 0:
                combined[c] = combined[c] / combined[c].max()
        
        # Zet om naar PIL
        img_np = (combined.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_preview = Image.fromarray(img_np)
        cell.paste(img_preview, (0, 0))
        
        # Voeg label toe met aantal tokens
        cell_draw.rectangle([0, 0, cell_size-1, cell_size-1], outline='white', width=1)
        cell_draw.text((5, 5), f"B{block_idx:02d}\n{token_idx} tokens", fill='white', font=font)
        
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


# Export node mappings
NODE_CLASS_MAPPINGS = {
    "FreeFuseBaseAnalysis": FreeFuseBaseAnalysis,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeFuseBaseAnalysis": "🔬 FreeFuse Base Analysis (No LoRAs)",
}
