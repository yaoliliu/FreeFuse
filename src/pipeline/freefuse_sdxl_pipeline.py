# Copyright 2025 The FreeFuse Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FreeFuse SDXL Pipeline

This module provides a pipeline for FreeFuse on SDXL architecture with two-phase denoising:
- Phase 1: Collect concept similarity maps (few steps, no LoRA masks)
- Phase 2: Apply masks and regenerate (full steps, with LoRA masks)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import os

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.cm as cm

from diffusers import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.utils import logging

from src.models.freefuse_unet_sdxl import FreeFuseUNet2DConditionModel
from src.attn_processor.freefuse_sdxl_attn_processor import FreeFuseSDXLAttnProcessor
from src.tuner.freefuse_lora_layer import FreeFuseLinear


logger = logging.get_logger(__name__)


def find_concept_positions_sdxl(
    tokenizer,
    prompts: Union[str, List[str]],
    concepts: Dict[str, str],
    filter_meaningless: bool = True,
    filter_single_char: bool = True,
) -> Dict[str, List[List[int]]]:
    """
    Find token positions for each concept in the prompts.
    Works with both Fast tokenizers and regular tokenizers (no offset_mapping needed).
    
    Args:
        tokenizer: The CLIP tokenizer
        prompts: Single prompt string or list of prompt strings
        concepts: Dict mapping concept name (e.g., adapter name) to concept description text
        filter_meaningless: Whether to filter stopwords/punctuation tokens (default True)
        filter_single_char: Whether to filter single-character tokens (default True)
    
    Returns:
        concept_pos_map: Dict with structure for each concept and prompt
    """
    # Stopwords and punctuation to filter
    STOPWORDS = {
        'a', 'an', 'the', 'and', 'or', 'but', 'nor', 'so', 'yet',
        'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'as', 
        'into', 'through', 'during', 'before', 'after', 'above', 'below', 
        'between', 'under', 'over', 'it', 'its', 'this', 'that', 'these', 
        'those', 'their', 'his', 'her', 'my', 'your', 'our', 'is', 'are', 
        'was', 'were', 'be', 'been', 'being', 'has', 'have', 'had', 'having',
        'which', 'who', 'whom', 'whose', 'where', 'when', 'while',
    }
    PUNCTUATION = {',', '.', '!', '?', ';', ':', '"', "'", '(', ')', '[', ']', 
                   '{', '}', '-', '–', '—', '/', '\\', '...', '..'}
    MEANINGLESS_TOKENS = STOPWORDS | PUNCTUATION
    
    def clean_token(token_text):
        return token_text.replace('</w>', '').replace('Ġ', '').replace('▁', '').strip().lower()
    
    def is_meaningless_token(token_text, check_single_char=True):
        cleaned = clean_token(token_text)
        if not cleaned:
            return True
        # if check_single_char and len(cleaned) == 1:
        #     return True
        return cleaned in MEANINGLESS_TOKENS
    
    if isinstance(prompts, str):
        prompts = [prompts]
    
    concept_pos_map = {}
    for concept_name, concept_text in concepts.items():
        concept_pos_map[concept_name] = []
        
        # Tokenize the concept text (without special tokens)
        concept_tokens = tokenizer.encode(concept_text, add_special_tokens=False)
        
        for prompt in prompts:
            # Tokenize the full prompt (with special tokens)
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
            prompt_token_texts = [tokenizer.decode([tid]) for tid in prompt_tokens]
            
            positions = []
            positions_with_text = []
            
            # Find where concept tokens appear in prompt tokens using sliding window
            concept_len = len(concept_tokens)
            prompt_len = len(prompt_tokens)
            
            if concept_len > 0:
                for i in range(prompt_len - concept_len + 1):
                    match = all(prompt_tokens[i + j] == concept_tokens[j] for j in range(concept_len))
                    if match:
                        for j in range(concept_len):
                            pos = i + j
                            if pos not in positions:
                                positions.append(pos)
                                positions_with_text.append((pos, prompt_token_texts[pos]))
            
            # Filter meaningless tokens
            if filter_meaningless and positions_with_text:
                filtered_positions = [
                    pos for pos, text in positions_with_text 
                    if not is_meaningless_token(text, check_single_char=filter_single_char)
                ]
                
                if not filtered_positions:
                    non_punct_positions = [
                        pos for pos, text in positions_with_text
                        if clean_token(text) not in PUNCTUATION
                    ]
                    if non_punct_positions:
                        filtered_positions = [non_punct_positions[0]]
                    elif positions_with_text:
                        filtered_positions = [positions_with_text[0][0]]
                
                positions = filtered_positions
            
            positions.sort()
            concept_pos_map[concept_name].append(positions)
    
    return concept_pos_map


class FreeFuseSDXLPipeline(StableDiffusionXLPipeline):
    """
    FreeFuse Pipeline for SDXL with two-phase denoising.
    
    Phase 1 (Mask Collection):
        - Run a few denoising steps without LoRA masks
        - Collect concept similarity maps from cross-attention layers
        - Generate spatial masks based on which concept each region best matches
    
    Phase 2 (Generation with Masks):
        - Reset to initial noise
        - Apply LoRA masks so each LoRA only affects its corresponding region
        - Run full denoising for final image
    """
    
    @classmethod
    def from_pipe(cls, pipe: StableDiffusionXLPipeline) -> "FreeFuseSDXLPipeline":
        """
        Create FreeFuseSDXLPipeline from an existing StableDiffusionXLPipeline.
        
        This also upgrades the UNet to FreeFuseUNet2DConditionModel.
        """
        # Upgrade UNet
        pipe.unet = FreeFuseUNet2DConditionModel.from_unet(pipe.unet)
        
        # Fix text_encoder_projection_dim if not set (needed for from_single_file)
        if not hasattr(pipe, 'text_encoder_projection_dim') or pipe.text_encoder_projection_dim is None:
            if hasattr(pipe, 'text_encoder_2') and pipe.text_encoder_2 is not None:
                pipe.text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim
            elif hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
                pipe.text_encoder_projection_dim = pipe.text_encoder.config.projection_dim
            else:
                pipe.text_encoder_projection_dim = 1280  # Default SDXL value
        
        # Change pipeline class
        pipe.__class__ = cls
        return pipe
    
    def setup_freefuse_attention_processors(
        self,
        cal_concept_sim_blocks: Optional[List[str]] = None,
    ):
        """
        Set up FreeFuse attention processors for the UNet.
        
        Args:
            cal_concept_sim_blocks: List of block patterns where concept similarity
                                    should be calculated. Default: ['mid_block']
                                    Examples: ['mid_block', 'up_blocks.0', 'down_blocks.2']
        """
        if cal_concept_sim_blocks is None:
            cal_concept_sim_blocks = ['mid_block']
        
        attn_processors = {}
        for name in self.unet.attn_processors.keys():
            proc = FreeFuseSDXLAttnProcessor()
            # Enable similarity calculation for specified blocks
            for block_pattern in cal_concept_sim_blocks:
                if block_pattern in name and 'attn2' in name:  # Only cross-attention
                    proc.cal_concept_sim_map = True
                    break
            attn_processors[name] = proc
        
        self.unet.set_attn_processor(attn_processors)
    
    def stabilized_balanced_argmax(self, logits, h, w, target_count=None, max_iter=15, 
                                  lr=0.01,           
                                  gravity_weight=0.00003, 
                                  spatial_weight=0.00003,
                                  momentum=0.2,
                                  centroid_margin=0.0,
                                  border_penalty=0.0,
                                  anisotropy=1.3,
                                  debug=True
                                  ):
        """
        V4: 使用线性归一化代替 Softmax。
        
        核心改进：
        - 线性归一化保持 logits 的相对大小关系，不会像 softmax 那样指数放大差异
        - 避免了重复 softmax 导致的 "赢家通吃" 效应
        - 更稳定的迭代收敛
        """
        B, C, N = logits.shape
        device = logits.device
        
        # === 1. 物理空间坐标 ===
        max_dim = max(h, w)
        # scale_h = (h / max_dim)
        # scale_w = (w / max_dim)
        
        scale_h = 1
        scale_w = 1
        
        y_range = torch.linspace(-scale_h, scale_h, steps=h, device=device)
        x_range = torch.linspace(-scale_w, scale_w, steps=w, device=device)
        x_range = x_range * anisotropy 
        
        grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')
        flat_y = grid_y.reshape(1, 1, N) 
        flat_x = grid_x.reshape(1, 1, N)
        
        # 边界 Mask
        pixel_size = 2.0 / max_dim 
        is_border = (flat_y.abs() > (scale_h - pixel_size * 1.5)) | \
                    (flat_x.abs() > (scale_w - pixel_size * 1.5))
        border_mask = is_border.float()

        if target_count is None:
            target_count = N / C
        
        bias = torch.zeros(B, C, 1, device=device)
        
        # === 核心：线性归一化函数 ===
        def linear_normalize(x, dim=1):
            """将 tensor 沿指定维度线性归一化到 [0, 1]"""
            x_min = x.min(dim=dim, keepdim=True)[0]
            x_max = x.max(dim=dim, keepdim=True)[0]
            return (x - x_min) / (x_max - x_min + 1e-8)
        
        # 初始化 running_probs 使用线性归一化
        running_probs = linear_normalize(logits, dim=-1)
        
        # 计算 logit 尺度用于自适应 lr
        logit_range = (logits.max() - logits.min()).item()
        logit_scale = max(logit_range, 1e-4)
        effective_lr = lr * logit_scale
        max_bias = logit_scale * 10.0
        
        if debug:
            print(f"\n[Argmax Debug] Start. B={B}, C={C}, N={N}, Target={target_count:.1f}")
            print(f"Logits range: min={logits.min().item():.5f}, max={logits.max().item():.5f}")
            print(f"Logit scale: {logit_scale:.6f}, Effective LR: {effective_lr:.6f}")
        
        # 空间卷积核
        neighbor_kernel = torch.ones(C, 1, 3, 3, device=device, dtype=logits.dtype) / 8.0
        neighbor_kernel[:, :, 1, 1] = 0
        
        current_logits = logits.clone()

        for i in range(max_iter):
            # A. 线性归一化（代替 softmax）
            probs = linear_normalize(current_logits - bias, dim=1)
            
            # B. 动量平滑
            running_probs = (1 - momentum) * probs + momentum * running_probs
            
            # C. 计算软重心
            mass = running_probs.sum(dim=2, keepdim=True) + 1e-6
            center_y = (running_probs * flat_y).sum(dim=2, keepdim=True) / mass
            center_x = (running_probs * flat_x).sum(dim=2, keepdim=True) / mass
            
            # 重心钳制
            if centroid_margin > 0:
                limit_y = scale_h * (1.0 - centroid_margin)
                limit_x = scale_w * (1.0 - centroid_margin)
                center_y = torch.clamp(center_y, -limit_y, limit_y)
                center_x = torch.clamp(center_x, -limit_x, limit_x)
            
            # D. 距离场
            dist_sq = (flat_y - center_y)**2 + (flat_x - center_x)**2
            
            # E. Bias 更新（基于硬分配的数量统计）
            # 使用硬分配来统计实际数量，用于更准确的平衡
            hard_indices = torch.argmax(current_logits - bias, dim=1)
            hard_counts = F.one_hot(hard_indices, num_classes=C).float().sum(dim=1)  # [B, C]
            
            diff = hard_counts - target_count
            cur_lr = effective_lr * (0.95 ** i)
            bias += torch.sign(diff).unsqueeze(2) * cur_lr
            bias = torch.clamp(bias, -max_bias, max_bias)
            
            # F. 空间投票
            if spatial_weight > 0:
                probs_img = running_probs.view(B, C, h, w)
                probs_img_f32 = probs_img.float()
                kernel_f32 = neighbor_kernel.float()
                neighbor_votes = F.conv2d(probs_img_f32, kernel_f32, padding=1, groups=C)
                neighbor_votes = neighbor_votes.to(logits.dtype).view(B, C, N)
            else:
                neighbor_votes = torch.zeros_like(logits)
                
            gravity_term = dist_sq * gravity_weight
            border_term = border_mask * border_penalty
            
            current_logits = logits - bias + \
                            (neighbor_votes * spatial_weight) - \
                            gravity_term - \
                            border_term
                            
            if debug and (i == 0 or i == max_iter - 1 or i % 10 == 0):
                hard_counts_list = hard_counts[0].int().tolist()
                print(f"Iter {i:02d}: Counts={hard_counts_list}")
                print(f"  Bias range: {bias.min().item():.5f} ~ {bias.max().item():.5f}")
                print(f"  Gravity Mean: {gravity_term.mean().item():.3f}, Max: {gravity_term.max().item():.3f}")
                print(f"  Spatial Mean: {(neighbor_votes * spatial_weight).mean().item():.3f}")

        if debug:
            final_assignment = torch.argmax(current_logits, dim=1)[0]
            final_counts = final_assignment.bincount(minlength=C).tolist()
            print(f"[Argmax Debug] Final Counts: {final_counts}\n")
            
        return torch.argmax(current_logits, dim=1)
    
    def _collect_concept_sim_maps(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        add_text_embeds: torch.Tensor,
        add_time_ids: torch.Tensor,
        freefuse_token_pos_maps: Dict[str, List[List[int]]],
        latent_height: int,
        latent_width: int,
        num_collect_steps: int = 5,
        guidance_scale: float = 7.5,
        do_classifier_free_guidance: bool = True,
        debug_dir: Optional[str] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run Phase 1: Collect concept similarity maps.
        
        Returns accumulated concept_sim_maps from cross-attention layers.
        Also saves debug visualizations if debug_dir is provided.
        
        Args:
            latent_height: Height of the latent tensor
            latent_width: Width of the latent tensor
        """
        accumulated_sim_maps = {}
        num_steps = min(num_collect_steps, len(timesteps))
        
        # Create debug subdirectory for phase 1
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
        
        for i, t in enumerate(timesteps[:num_steps]):
            # Expand for CFG
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Prepare added conditions
            added_cond_kwargs = {
                "text_embeds": add_text_embeds,
                "time_ids": add_time_ids,
            }
            
            # Forward pass with FreeFuse token position maps
            # Merge user-provided cross_attention_kwargs with internal kwargs
            phase1_cross_attn_kwargs = {**(cross_attention_kwargs or {})}
            phase1_cross_attn_kwargs['freefuse_token_pos_maps'] = freefuse_token_pos_maps
            
            # The attention processors will return concept_sim_maps
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=phase1_cross_attn_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            
            # Collect concept_sim_maps from attention processors
            step_sim_maps = {}
            for name, proc in self.unet.attn_processors.items():
                if hasattr(proc, '_last_concept_sim_maps') and proc._last_concept_sim_maps:
                    for lora_name, sim_map in proc._last_concept_sim_maps.items():
                        if lora_name not in accumulated_sim_maps:
                            accumulated_sim_maps[lora_name] = []
                        accumulated_sim_maps[lora_name].append(sim_map)
                        # Store for debug saving
                        if lora_name not in step_sim_maps:
                            step_sim_maps[lora_name] = sim_map
            
            # CFG
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            # Compute predicted x0 for debug visualization  
            # For EulerDiscreteScheduler: x0_pred = x_t - sigma * noise_pred
            if debug_dir:
                try:
                    # Get sigma for current step
                    step_idx = self.scheduler.step_index if hasattr(self.scheduler, 'step_index') and self.scheduler.step_index is not None else i
                    if step_idx is None:
                        step_idx = i
                    sigma = self.scheduler.sigmas[step_idx]
                    
                    # Predicted x0 using Euler formula
                    pred_x0 = latents - sigma * noise_pred
                    
                    # Decode predicted x0 for visualization
                    pred_x0_decoded = self.vae.decode(pred_x0 / self.vae.config.scaling_factor, return_dict=False)[0]
                    pred_x0_img = self.image_processor.postprocess(pred_x0_decoded, output_type="pil")[0]
                    pred_x0_img.save(os.path.join(debug_dir, f"phase1_step{i:02d}_pred_x0.png"))
                except Exception as e:
                    logger.warning(f"Failed to save predicted x0 at step {i}: {e}")
                
                # Save sim_maps for this step
                for lora_name, sim_map in step_sim_maps.items():
                    try:
                        # sim_map shape: (B, H*W, 1) - take first batch
                        # 取第二个，因为第一个是negative导出的sim map
                        sim_values = sim_map[1, :].cpu().numpy()  # (H*W,)
                        
                        # Reshape to 2D using aspect ratio from latent dimensions
                        seq_len = sim_values.shape[0]
                        aspect_ratio = latent_width / latent_height
                        h = int(np.sqrt(seq_len / aspect_ratio))
                        w = int(h * aspect_ratio)
                        # Adjust if there's rounding error
                        if h * w != seq_len:
                            for test_h in range(int(np.sqrt(seq_len)), 0, -1):
                                if seq_len % test_h == 0:
                                    h = test_h
                                    w = seq_len // h
                                    break
                        sim_2d = sim_values.reshape(h, w)
                        
                        # Normalize to 0-1 for colormap
                        sim_min, sim_max = sim_2d.min(), sim_2d.max()
                        if sim_max - sim_min > 1e-6:
                            sim_normalized = (sim_2d - sim_min) / (sim_max - sim_min)
                        else:
                            sim_normalized = np.zeros_like(sim_2d)
                        
                        # Apply viridis colormap for heatmap visualization
                        sim_colored = cm.viridis(sim_normalized)
                        sim_colored = (sim_colored[:, :, :3] * 255).astype(np.uint8)  # Remove alpha, convert to uint8
                        sim_img = Image.fromarray(sim_colored)
                        sim_img.save(os.path.join(debug_dir, f"phase1_step{i:02d}_sim_{lora_name}.png"))
                    except Exception as e:
                        raise
                        logger.warning(f"Failed to save sim_map for {lora_name} at step {i}: {e}")
            
            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        # Average accumulated sim maps (handling different resolutions)
        # Target resolution is latent_height * latent_width
        target_seq_len = latent_height * latent_width
        final_sim_maps = {}
        for lora_name, sim_map_list in accumulated_sim_maps.items():
            # if sim_map_list:
            #     # Interpolate all sim_maps to target resolution if needed
            #     normalized_maps = []
            #     for sim_map in sim_map_list:
            #         # sim_map shape: (B, seq_len, 1)
            #         if sim_map.shape[1] != target_seq_len:
            #             batch_size = sim_map.shape[0]
            #             # Infer source h,w from seq_len
            #             src_seq_len = sim_map.shape[1]
            #             src_h = int(np.sqrt(src_seq_len * latent_height / latent_width))
            #             src_w = src_seq_len // src_h if src_h > 0 else src_seq_len
            #             if src_h * src_w != src_seq_len:
            #                 src_h = int(np.sqrt(src_seq_len))
            #                 src_w = src_seq_len // src_h
                        
            #             # Reshape to 4D, interpolate, reshape back
            #             sim_4d = sim_map.permute(0, 2, 1).view(batch_size, 1, src_h, src_w)
            #             sim_4d = F.interpolate(sim_4d, size=(latent_height, latent_width), mode='bilinear', align_corners=False)
            #             sim_map = sim_4d.view(batch_size, 1, -1).permute(0, 2, 1)
            #         normalized_maps.append(sim_map)
                
            #     final_sim_maps[lora_name] = torch.stack(normalized_maps).mean(dim=0)
            # 我们只保存最后一个sim map就好
            final_sim_maps[lora_name] = sim_map_list[-1]
        
        # Save final averaged sim_maps
        if debug_dir:
            # Save raw concept_sim_maps tensor for evaluation (Precision@k calculation)
            sim_maps_save_data = {
                'concept_sim_maps': {k: v.cpu().float() for k, v in final_sim_maps.items()},
                'latent_h': latent_height,
                'latent_w': latent_width,
            }
            torch.save(sim_maps_save_data, os.path.join(debug_dir, 'concept_sim_maps.pt'))
            
            for lora_name, sim_map in final_sim_maps.items():
                try:
                    sim_values = sim_map[1, :].cpu().numpy()
                    seq_len = sim_values.shape[0]
                    aspect_ratio = latent_width / latent_height
                    h = int(np.sqrt(seq_len / aspect_ratio))
                    w = int(h * aspect_ratio)
                    if h * w != seq_len:
                        for test_h in range(int(np.sqrt(seq_len)), 0, -1):
                            if seq_len % test_h == 0:
                                h = test_h
                                w = seq_len // h
                                break
                    sim_2d = sim_values.reshape(h, w)
                    
                    sim_min, sim_max = sim_2d.min(), sim_2d.max()
                    if sim_max - sim_min > 1e-6:
                        sim_normalized = (sim_2d - sim_min) / (sim_max - sim_min)
                    else:
                        sim_normalized = np.zeros_like(sim_2d)
                    
                    # Apply viridis colormap for heatmap visualization
                    sim_colored = cm.viridis(sim_normalized)
                    sim_colored = (sim_colored[:, :, :3] * 255).astype(np.uint8)
                    sim_img = Image.fromarray(sim_colored)
                    sim_img.save(os.path.join(debug_dir, f"phase1_final_sim_{lora_name}.png"))
                except Exception as e:
                    logger.warning(f"Failed to save final sim_map for {lora_name}: {e}")
        
        return final_sim_maps
    
    def _generate_masks_from_sim_maps(
        self,
        concept_sim_maps: Dict[str, torch.Tensor],
        latent_height: int,
        latent_width: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate binary masks from concept similarity maps using argmax.
        
        For each position, the LoRA with highest similarity gets 1, others get 0.
        
        Args:
            concept_sim_maps: Dict of sim maps, each of shape (B, H*W, 1)
            latent_height: Height of the latent space (not image height)
            latent_width: Width of the latent space (not image width)
        """
        if not concept_sim_maps:
            return {}
        
        lora_names = list(concept_sim_maps.keys())
        # 取第二个，因为第一个是negative导出的sim map
        concept_sim_maps = {name: concept_sim_maps[name][1].unsqueeze(0) for name in lora_names}
        
        # Stack all sim maps: (B, num_loras, H*W)
        stacked = torch.stack([concept_sim_maps[name].squeeze(-1) for name in lora_names], dim=1)

        # Create masks
        masks = {}
        batch_size = stacked.shape[0]
        seq_len = stacked.shape[2]
        
        # Use provided dimensions - the sim_map may come from a specific attention layer
        # which may have different resolution than the full latent
        # We compute the actual h/w based on the aspect ratio of the latent
        aspect_ratio = latent_width / latent_height
        
        # The sim_map seq_len should be h * w where h/w maintains the aspect ratio
        # So: seq_len = h * w and w/h = aspect_ratio
        # => h = sqrt(seq_len / aspect_ratio), w = h * aspect_ratio
        h = int(np.sqrt(seq_len / aspect_ratio))
        w = int(h * aspect_ratio)
        
        # Get argmax
        # max_indices = stacked.argmax(dim=1)  # (B, H*W)
        max_indices = self.stabilized_balanced_argmax(stacked, h, w)  # (B, H*W)
        
        # Adjust if there's rounding error
        if h * w != seq_len:
            # Try to find exact factors
            for test_h in range(int(np.sqrt(seq_len)), 0, -1):
                if seq_len % test_h == 0:
                    h = test_h
                    w = seq_len // h
                    break
        
        for i, lora_name in enumerate(lora_names):
            mask = (max_indices == i).float()  # (B, H*W)
            mask = mask.reshape(batch_size, 1, h, w)  # (B, 1, h, w)
            masks[lora_name] = mask
        
        return masks
    
    def _construct_attention_bias(
        self,
        lora_masks: Dict[str, torch.Tensor],
        freefuse_token_pos_maps: Dict[str, List[List[int]]],
        txt_seq_len: int,
        img_seq_len: int,
        bias_scale: float = 5.0,
        positive_bias_scale: float = 1.0,
        use_positive_bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> torch.Tensor:
        """
        Construct soft attention bias matrix for SDXL cross-attention.
        
        Unlike Flux's joint attention, SDXL uses separate cross-attention where image
        features (query) attend to text features (key/value). So we only need image->text
        direction bias.
        
        Args:
            lora_masks: Dict mapping lora_name -> (B, 1, H, W) binary mask indicating
                        which image positions belong to this LoRA
            freefuse_token_pos_maps: Dict mapping lora_name -> [[token positions in prompt], ...]
                                     Token positions in the CLIP text embedding
            txt_seq_len: Length of text sequence (usually 77 for CLIP)
            img_seq_len: Length of image sequence (H*W)
            bias_scale: Strength of the NEGATIVE bias (larger = stronger suppression)
            positive_bias_scale: Strength of the POSITIVE bias for same-LoRA attention
            use_positive_bias: If True, also add positive bias for same-LoRA attention pairs
            device: Device for the output tensor
            dtype: Data type for the output tensor
            
        Returns:
            attention_bias: (B, img_seq_len, txt_seq_len)
                           Soft bias values: positive for same-LoRA pairs (if use_positive_bias),
                           negative for cross-LoRA pairs, 0 for neutral pairs
        """
        # Get batch size from first mask
        first_mask = next(iter(lora_masks.values()))
        batch_size = first_mask.shape[0]
        
        # Initialize bias as zeros (no bias = attend freely)
        attention_bias = torch.zeros(
            batch_size, img_seq_len, txt_seq_len,
            device=device, dtype=dtype
        )
        
        # Build a mapping: for each text token position, which LoRA does it belong to?
        # -1 means no LoRA (shared/common tokens)
        text_token_to_lora = torch.full((txt_seq_len,), -1, device=device, dtype=torch.long)
        lora_name_to_idx = {name: idx for idx, name in enumerate(lora_masks.keys())}
        
        for lora_name, positions_list in freefuse_token_pos_maps.items():
            if lora_name not in lora_name_to_idx:
                continue
            lora_idx = lora_name_to_idx[lora_name]
            # positions_list is [[positions for batch 0], [positions for batch 1], ...]
            # For simplicity, use first batch's positions (assuming same across batches)
            if len(positions_list) > 0 and len(positions_list[0]) > 0:
                for pos in positions_list[0]:
                    if 0 <= pos < txt_seq_len:
                        text_token_to_lora[pos] = lora_idx
        
        # For each LoRA, get its image mask and text token positions
        for lora_name, mask_4d in lora_masks.items():
            if lora_name not in lora_name_to_idx:
                continue
            lora_idx = lora_name_to_idx[lora_name]
            
            # mask_4d: (B, 1, H, W) -> flatten to (B, H*W)
            img_mask = mask_4d.view(batch_size, -1)  # (B, img_seq_len)
            
            # Create mask for text tokens belonging to other LoRAs (not this one, not shared)
            other_lora_text_mask = (text_token_to_lora != lora_idx) & (text_token_to_lora != -1)
            other_lora_text_mask = other_lora_text_mask.float()  # (txt_seq_len,)
            
            # Image->Text bias: for each image position in this LoRA's mask,
            # suppress attention to other LoRAs' text tokens
            # img_mask: (B, img_seq_len)
            # other_lora_text_mask: (txt_seq_len,)
            # bias[b, i, j] -= scale * img_mask[b, i] * other_lora_text_mask[j]
            img_to_txt_bias = img_mask.unsqueeze(-1) * other_lora_text_mask.unsqueeze(0).unsqueeze(0)
            img_to_txt_bias = img_to_txt_bias * (-bias_scale)
            
            attention_bias += img_to_txt_bias
            
            # Positive bias: encourage this LoRA's image region to attend to this LoRA's text tokens
            if use_positive_bias:
                this_lora_text_mask = (text_token_to_lora == lora_idx).float()
                img_to_txt_positive_bias = img_mask.unsqueeze(-1) * this_lora_text_mask.unsqueeze(0).unsqueeze(0)
                img_to_txt_positive_bias = img_to_txt_positive_bias * positive_bias_scale
                
                attention_bias += img_to_txt_positive_bias
        
        return attention_bias
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        # FreeFuse specific args
        freefuse_concepts: Optional[Dict[str, str]] = None,
        freefuse_token_pos_maps: Optional[Dict[str, List[List[int]]]] = None,
        num_mask_collect_steps: int = 10,
        use_freefuse: bool = True,
        debug_dir: Optional[str] = None,
        # Attention bias params
        use_attention_bias: bool = True,
        attention_bias_scale: float = 1.0,
        attention_bias_positive_scale: float = 1.0,
        attention_bias_positive: bool = True,
        **kwargs,
    ):
        """
        FreeFuse SDXL pipeline call with two-phase denoising.
        
        FreeFuse Args:
            freefuse_concepts: Dict mapping LoRA adapter name to concept text
                              Example: {'character1': 'a woman with red hair'}
            freefuse_token_pos_maps: Pre-computed token positions (optional, will be computed if not provided)
            num_mask_collect_steps: Number of steps for Phase 1 mask collection
            use_freefuse: Whether to use FreeFuse (False = standard SDXL)
            debug_dir: Directory to save debug visualizations
        
        Attention Bias Args:
            use_attention_bias: Whether to apply attention bias in Phase 2 to constrain
                              text-image attention based on LoRA masks. Default True.
            attention_bias_scale: Strength of the NEGATIVE bias for cross-LoRA attention.
                                Larger values = stronger suppression. Default 5.0.
            attention_bias_positive_scale: Strength of the POSITIVE bias for same-LoRA attention.
                                         Default 1.0.
            attention_bias_positive: If True, add positive bias for same-LoRA attention pairs.
                                   Default True.
        """
        # Default height/width
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)
        
        # Batch size
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        
        # Encode prompts
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        )
        
        # Concat for CFG
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        else:
            add_text_embeds = pooled_prompt_embeds
        
        # Time IDs - compute text_encoder_projection_dim
        if pooled_prompt_embeds is not None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        elif hasattr(self, 'text_encoder_2') and self.text_encoder_2 is not None:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim
        else:
            text_encoder_projection_dim = 1280  # Default SDXL value
        
        add_time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, 
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
        add_time_ids = add_time_ids.to(device)
        
        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        # Prepare latents
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        
        # Store initial latents for Phase 2
        initial_latents = latents.clone()
        
        # Compute token positions if not provided
        if use_freefuse and freefuse_concepts and freefuse_token_pos_maps is None:
            # Use tokenizer (prefer tokenizer_2 if available, else tokenizer)
            tokenizer = getattr(self, 'tokenizer_2', None) or self.tokenizer
            freefuse_token_pos_maps = find_concept_positions_sdxl(
                tokenizer,
                prompt if isinstance(prompt, str) else prompt[0],
                freefuse_concepts,
            )
            logger.info(f"Computed token positions: {freefuse_token_pos_maps}")
        
        # ===== Phase 1: Mask Collection (if using FreeFuse) =====
        if use_freefuse and freefuse_token_pos_maps:
            self.disable_lora()
            logger.info(f"Phase 1: Collecting masks for {num_mask_collect_steps} steps...")
            
            # Disable LoRA masks for Phase 1
            if hasattr(self.unet, 'disable_freefuse_masks'):
                self.unet.disable_freefuse_masks()
            
            # Run Phase 1 denoising
            latent_height = height // self.vae_scale_factor
            latent_width = width // self.vae_scale_factor
            concept_sim_maps = self._collect_concept_sim_maps(
                latents=latents,
                timesteps=timesteps,
                prompt_embeds=prompt_embeds,
                add_text_embeds=add_text_embeds,
                add_time_ids=add_time_ids,
                freefuse_token_pos_maps=freefuse_token_pos_maps,
                latent_height=latent_height,
                latent_width=latent_width,
                num_collect_steps=num_mask_collect_steps,
                guidance_scale=guidance_scale,
                do_classifier_free_guidance=do_classifier_free_guidance,
                debug_dir=debug_dir,
                cross_attention_kwargs=cross_attention_kwargs,
            )
            
            # Generate masks from sim maps
            lora_masks = self._generate_masks_from_sim_maps(
                concept_sim_maps,
                height // self.vae_scale_factor,
                width // self.vae_scale_factor,
            )
            
            # Debug: save masks
            if debug_dir and lora_masks:
                os.makedirs(debug_dir, exist_ok=True)
                
                # Save lora_masks as .pt for evaluation
                lora_masks_save_data = {
                    'lora_masks': {k: v.cpu().float() for k, v in lora_masks.items()},
                    'latent_h': height // self.vae_scale_factor,
                    'latent_w': width // self.vae_scale_factor,
                }
                torch.save(lora_masks_save_data, os.path.join(debug_dir, 'lora_masks.pt'))
                
                for lora_name, mask in lora_masks.items():
                    mask_np = mask[0, 0].cpu().numpy()
                    mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
                    mask_img.save(os.path.join(debug_dir, f"mask_{lora_name}.png"))
            
            # Set masks to UNet
            if lora_masks:
                mask_h = next(iter(lora_masks.values())).shape[2]
                mask_w = next(iter(lora_masks.values())).shape[3]
                self.unet.set_freefuse_masks(
                    lora_masks, 
                    mask_height=mask_h, 
                    mask_width=mask_w,
                    derive_01_mask=True,
                    h=height // self.vae_scale_factor,
                    w=width // self.vae_scale_factor,
                )
                self.unet.enable_freefuse_masks()
            
            # Construct attention bias for Phase 2
            attention_bias = None
            if use_attention_bias and lora_masks and freefuse_token_pos_maps:
                # CLIP text sequence length (usually 77) - CFG doubles batch, not sequence
                txt_seq_len = prompt_embeds.shape[1]
                # Image sequence length from mask dimensions (mask comes from sim_map resolution)
                first_mask = next(iter(lora_masks.values()))
                mask_h, mask_w = first_mask.shape[2], first_mask.shape[3]
                img_seq_len = mask_h * mask_w
                
                attention_bias = self._construct_attention_bias(
                    lora_masks=lora_masks,
                    freefuse_token_pos_maps=freefuse_token_pos_maps,
                    txt_seq_len=txt_seq_len,
                    img_seq_len=img_seq_len,
                    bias_scale=attention_bias_scale,
                    positive_bias_scale=attention_bias_positive_scale,
                    use_positive_bias=attention_bias_positive,
                    device=device,
                    dtype=prompt_embeds.dtype,
                )
                
                # Expand attention_bias for CFG (duplicate for unconditional and conditional)
                if do_classifier_free_guidance:
                    attention_bias = torch.cat([attention_bias, attention_bias], dim=0)
                
                logger.info(f"Attention bias constructed: shape={attention_bias.shape}, scale={attention_bias_scale}, positive_scale={attention_bias_positive_scale}")
                
                # Debug: save attention bias visualization
                if debug_dir:
                    try:
                        import matplotlib.pyplot as plt
                        bias_2d = attention_bias[0].cpu().float().numpy()
                        fig, ax = plt.subplots(figsize=(12, 8))
                        im = ax.imshow(bias_2d, cmap='RdBu', vmin=-attention_bias_scale, vmax=attention_bias_positive_scale)
                        ax.set_title('Attention Bias Matrix (SDXL Cross-Attention)')
                        ax.set_xlabel('Text token position')
                        ax.set_ylabel('Image position')
                        plt.colorbar(im, ax=ax)
                        plt.savefig(os.path.join(debug_dir, 'attention_bias.png'), dpi=150, bbox_inches='tight')
                        plt.close(fig)
                    except Exception as e:
                        logger.warning(f"Failed to save attention bias visualization: {e}")
            
            # Prepare cross_attention_kwargs for Phase 2 with attention bias
            if cross_attention_kwargs is None:
                cross_attention_kwargs = {}
            if attention_bias is not None:
                cross_attention_kwargs['attention_bias'] = attention_bias
                cross_attention_kwargs['freefuse_token_pos_maps'] = freefuse_token_pos_maps
            
            # Reset latents for Phase 2
            latents = initial_latents.clone()
            # Reset scheduler state for Phase 2 (Phase 1 advanced the step index)
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps
            logger.info("Phase 2: Generating with masks and attention bias...")
        else:
            # Not using FreeFuse - ensure masks are disabled (important if pipeline was previously used with FreeFuse)
            if hasattr(self.unet, 'disable_freefuse_masks'):
                self.unet.disable_freefuse_masks()
        
        # ===== Phase 2 (or standard): Full Denoising =====
        self.enable_lora()
        for i, t in enumerate(self.progress_bar(timesteps)):
            # Expand for CFG
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Prepare added conditions
            added_cond_kwargs = {
                "text_embeds": add_text_embeds,
                "time_ids": add_time_ids,
            }
            
            # UNet forward
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            
            # CFG
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        # Decode latents
        if output_type == "latent":
            image = latents
        else:
            latents = latents / self.vae.config.scaling_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
        
        if not return_dict:
            return (image,)
        
        return StableDiffusionXLPipelineOutput(images=image)
