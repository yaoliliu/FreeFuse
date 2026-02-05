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
FreeFuse SDXL Attention Processor

This module provides attention processors for FreeFuse on SDXL (UNet) architecture.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.attention import Attention
from diffusers.utils import logging


logger = logging.get_logger(__name__)
    
class FreeFuseSDXLAttnProcessor:
    """
    FreeFuse attention processor for SDXL using CrossAttn + SelfConcept method.
    
    This processor combines two techniques:
    1. CrossAttn: Uses cross-attention weights (img_query @ txt_key) to select top-k image tokens
    2. SelfConcept: Uses hidden_states inner product to compute final similarity maps
    
    Architecture: Uses cross_attention_kwargs to pass data between attn1 (self-attn) and attn2 (cross-attn)
    within the same BasicTransformerBlock:
    - attn1: Caches hidden_states after self-attention
    - attn2: Uses cached hidden_states for SelfConcept similarity computation
    
    The SelfConcept method computes semantic similarity via hidden_states inner product,
    which captures pure feature similarity without positional bias.
    """
    
    def __init__(self, top_k_ratio: float = 0.1, temperature: float = 1000.0):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "FreeFuseCrossAttnSelfConceptSDXLAttnProcessor requires PyTorch 2.0."
            )
        self.cal_concept_sim_map: bool = False
        self._last_concept_sim_maps: Optional[Dict[str, torch.Tensor]] = None
        self.top_k_ratio = top_k_ratio
        self.temperature = temperature
    
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        freefuse_token_pos_maps: Optional[Dict[str, List[List[int]]]] = None,
        attention_bias: Optional[torch.Tensor] = None,
        _self_attn_cache: Optional[Dict[str, torch.Tensor]] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with attn1/attn2 cache mechanism for SelfConcept computation.
        
        Args:
            attn: The Attention module
            hidden_states: Image features, shape (B, H*W, C) or (B, C, H, W)
            encoder_hidden_states: Text features for cross-attention, shape (B, seq_len, C)
            attention_mask: Optional attention mask
            temb: Optional temporal embedding
            freefuse_token_pos_maps: Dict mapping lora_name -> [[positions for prompt 1], ...]
            attention_bias: Optional attention bias for cross-attention
            _self_attn_cache: Mutable dict passed via cross_attention_kwargs for attn1/attn2 communication
        
        Returns:
            hidden_states: Processed image features
        """
        self._last_concept_sim_maps = None
        is_cross_attention = encoder_hidden_states is not None
        
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size = hidden_states.shape[0]

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # === ATTN1 (Self-Attention): Cache hidden_states for later SelfConcept computation ===
        if not is_cross_attention and _self_attn_cache is not None:
            # Standard self-attention computation
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)
            
            # Output projection
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            
            # Cache hidden_states AFTER attention and projection (before residual)
            # This represents the self-attention output features
            _self_attn_cache["hidden_states"] = hidden_states.clone()
            
            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
            if attn.residual_connection:
                hidden_states = hidden_states + residual
            hidden_states = hidden_states / attn.rescale_output_factor
            return hidden_states

        # === ATTN2 (Cross-Attention): Use cached hidden_states for SelfConcept ===
        combined_mask = attention_mask
        if is_cross_attention and attention_bias is not None:
            curr_img_seq_len = query.shape[2]
            txt_seq_len = key.shape[2]
            bias_img_seq_len = attention_bias.shape[1]
            
            if bias_img_seq_len != curr_img_seq_len:
                import math
                bias_h = int(math.sqrt(bias_img_seq_len))
                bias_w = bias_img_seq_len // bias_h
                if bias_h * bias_w != bias_img_seq_len:
                    for h in range(int(math.sqrt(bias_img_seq_len)) + 1, 0, -1):
                        if bias_img_seq_len % h == 0:
                            bias_h, bias_w = h, bias_img_seq_len // h
                            break
                
                curr_h = int(math.sqrt(curr_img_seq_len))
                curr_w = curr_img_seq_len // curr_h
                if curr_h * curr_w != curr_img_seq_len:
                    for h in range(int(math.sqrt(curr_img_seq_len)) + 1, 0, -1):
                        if curr_img_seq_len % h == 0:
                            curr_h, curr_w = h, curr_img_seq_len // h
                            break
                
                bias_4d = attention_bias.permute(0, 2, 1).view(batch_size, txt_seq_len, bias_h, bias_w)
                bias_4d = F.interpolate(bias_4d, size=(curr_h, curr_w), mode='bilinear', align_corners=False)
                attention_bias = bias_4d.view(batch_size, txt_seq_len, -1).permute(0, 2, 1)
            
            attn_bias_expanded = attention_bias.unsqueeze(1)
            combined_mask = combined_mask + attn_bias_expanded if combined_mask is not None else attn_bias_expanded

        # Compute cross-attention and extract concept sim maps
        if is_cross_attention and self.cal_concept_sim_map and freefuse_token_pos_maps is not None:
            scale = head_dim ** -0.5
            attn_scores = torch.matmul(query, key.transpose(-1, -2)) * scale
            if combined_mask is not None:
                attn_scores = attn_scores + combined_mask
            attn_weights = F.softmax(attn_scores, dim=-1)
            
            # Get cached hidden_states from attn1 for SelfConcept computation
            cached_hidden_states = _self_attn_cache.get("hidden_states") if _self_attn_cache else None
            
            self._last_concept_sim_maps = self._extract_concept_sim_maps_self_concept(
                attn_weights=attn_weights,
                freefuse_token_pos_maps=freefuse_token_pos_maps,
                cached_hidden_states=cached_hidden_states,
            )
            
            hidden_states = torch.matmul(attn_weights, value)
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=combined_mask, dropout_p=0.0, is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    
    def _extract_concept_sim_maps_self_concept(
        self,
        attn_weights: torch.Tensor,
        freefuse_token_pos_maps: Dict[str, List[List[int]]],
        cached_hidden_states: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract concept similarity maps using CrossAttn for top-k selection + SelfConcept for final sim map.
        
        Two-stage approach:
        1. Use cross-attention weights to select top-k image tokens for each concept
        2. Use hidden_states inner product (SelfConcept) to compute final similarity map
        
        Args:
            attn_weights: Cross-attention weights, shape (B, heads, img_seq_len, txt_seq_len)
            freefuse_token_pos_maps: Position maps for each LoRA concept
            cached_hidden_states: Hidden states from attn1, shape (B, img_seq_len, C)
        
        Returns:
            Dict mapping lora_name -> concept similarity map of shape (B, img_seq_len)
        """
        concept_sim_maps = {}
        batch_size, num_heads, img_seq_len, txt_seq_len = attn_weights.shape
        
        # Collect all cross-attention scores for contrastive selection
        all_cross_attn_scores = {}
        for lora_name, positions_list in freefuse_token_pos_maps.items():
            pos = positions_list[0]
            pos = [p for p in pos if 0 <= p < txt_seq_len]
            if len(pos) == 0:
                continue
            
            pos_tensor = torch.tensor(pos, device=attn_weights.device, dtype=torch.long)
            concept_attn = attn_weights[:, :, :, pos_tensor]  # (B, heads, img_seq_len, concept_len)
            # Mean over heads and concept tokens -> (B, img_seq_len)
            cross_attn_scores = concept_attn.mean(dim=-1).mean(dim=1)
            all_cross_attn_scores[lora_name] = cross_attn_scores
        
        # Compute contrastive scores and final sim maps
        for lora_name in all_cross_attn_scores.keys():
            # Contrastive score: enhance current concept, subtract others
            cross_attn_scores = all_cross_attn_scores[lora_name] * len(all_cross_attn_scores)
            for other_name, other_scores in all_cross_attn_scores.items():
                if other_name != lora_name:
                    cross_attn_scores = cross_attn_scores - other_scores
            
            # Select top-k image tokens based on cross-attention scores
            k = max(1, int(img_seq_len * self.top_k_ratio))
            _, top_k_indices = torch.topk(cross_attn_scores, k, dim=-1)  # (B, k)
            
            if cached_hidden_states is not None:
                # SelfConcept: Use hidden_states inner product
                # Extract core image tokens
                top_k_indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, cached_hidden_states.shape[-1])
                core_tokens = torch.gather(cached_hidden_states, dim=1, index=top_k_indices_expanded)  # (B, k, C)
                
                # Compute self-modal similarity: core tokens @ all tokens
                self_modal_sim = torch.bmm(core_tokens, cached_hidden_states.transpose(-1, -2))  # (B, k, img_seq_len)
                
                # Average over core tokens and apply softmax with temperature
                concept_sim_map = self_modal_sim.mean(dim=1)  # (B, img_seq_len)
                concept_sim_map = F.softmax(concept_sim_map / self.temperature, dim=-1)
            else:
                # Fallback: use cross-attention scores directly
                concept_sim_map = F.softmax(cross_attn_scores, dim=-1)
            
            concept_sim_maps[lora_name] = concept_sim_map
        
        return concept_sim_maps

