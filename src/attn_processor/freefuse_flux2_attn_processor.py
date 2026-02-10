# Copyright 2025 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
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
FreeFuse Attention Processors for Flux2 Klein architecture.

This module provides two attention processors:
1. FreeFuseFlux2AttnProcessor - for Double-Stream blocks (separate text/image paths)
2. FreeFuseFlux2SingleAttnProcessor - for Single-Stream blocks (fused QKV+MLP)

Key features:
- Concept similarity map extraction via cross-attention top-k + concept attention
- Additive attention bias injection for cross-LoRA suppression
- Support for background detection via EOS token or user-defined positions
"""

from typing import Dict, List, Optional, TYPE_CHECKING

import torch
import torch.nn.functional as F

from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import apply_rotary_emb

if TYPE_CHECKING:
    from diffusers.models.transformers.transformer_flux2 import Flux2Attention, Flux2ParallelSelfAttention


def _get_qkv_projections(attn: "Flux2Attention", hidden_states, encoder_hidden_states=None):
    """Get QKV projections for Flux2 attention."""
    if attn.fused_projections:
        query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        if encoder_hidden_states is not None and hasattr(attn, "to_added_qkv"):
            encoder_query, encoder_key, encoder_value = attn.to_added_qkv(encoder_hidden_states).chunk(3, dim=-1)
        else:
            encoder_query = encoder_key = encoder_value = None
    else:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)
        else:
            encoder_query = encoder_key = encoder_value = None
    return query, key, value, encoder_query, encoder_key, encoder_value


class FreeFuseFlux2AttnProcessor:
    """
    FreeFuse attention processor for Flux2 Double-Stream blocks.
    
    Extends Flux2AttnProcessor with:
    - Concept similarity map extraction
    - Additive attention bias for cross-LoRA suppression
    """
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0.")
        
        # Sim-map extraction control
        self.cal_concept_sim_map: bool = False
        self.concept_sim_maps: Optional[Dict[str, torch.Tensor]] = None
        
        # Set by transformer before each forward
        self._freefuse_token_pos_maps: Optional[Dict[str, List[List[int]]]] = None
        self._top_k_ratio: float = 0.1
        self._eos_token_index: Optional[int] = None
        self._background_token_positions: Optional[List[int]] = None
        self._attention_bias: Optional[torch.Tensor] = None

    def __call__(
        self,
        attn: "Flux2Attention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        freefuse_token_pos_maps: Optional[Dict[str, List[List[int]]]] = None,
        top_k_ratio: float = 0.1,
        eos_token_index: Optional[int] = None,
        background_token_positions: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Forward pass with FreeFuse concept similarity map extraction.
        
        Args:
            freefuse_token_pos_maps: Dict mapping lora_name -> [[positions for prompt 1], ...]
            top_k_ratio: Ratio of top image tokens for concept attention (default: 0.1)
            eos_token_index: Index of first EOS token for background detection
            background_token_positions: User-defined background token positions
        """
        # Use passed params or fall back to stored ones
        freefuse_token_pos_maps = freefuse_token_pos_maps or self._freefuse_token_pos_maps
        top_k_ratio = top_k_ratio if freefuse_token_pos_maps else self._top_k_ratio
        eos_token_index = eos_token_index if eos_token_index is not None else self._eos_token_index
        background_token_positions = background_token_positions or self._background_token_positions
        
        # QKV projections
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            # Concatenate text and image: [text, image]
            txt_img_query = torch.cat([encoder_query, query], dim=1)
            txt_img_key = torch.cat([encoder_key, key], dim=1)
            txt_img_value = torch.cat([encoder_value, value], dim=1)

        # Apply RoPE
        if image_rotary_emb is not None:
            txt_img_query = apply_rotary_emb(txt_img_query, image_rotary_emb, sequence_dim=1)
            txt_img_key = apply_rotary_emb(txt_img_key, image_rotary_emb, sequence_dim=1)

        # Prepare attention mask with optional bias
        combined_mask = self._prepare_attention_mask(attention_mask, txt_img_query)

        # Run attention
        attention_output = dispatch_attention_fn(
            txt_img_query,
            txt_img_key,
            txt_img_value,
            attn_mask=combined_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        attention_output = attention_output.flatten(2, 3)
        attention_output = attention_output.to(query.dtype)

        if encoder_hidden_states is not None:
            txt_len = encoder_hidden_states.shape[1]
            img_len = hidden_states.shape[1]
            
            encoder_hidden_states_out, hidden_states_out = attention_output.split_with_sizes(
                [txt_len, img_len], dim=1
            )

            # Extract concept similarity maps
            concept_sim_maps = None
            if self.cal_concept_sim_map and freefuse_token_pos_maps is not None:
                # Keys with RoPE applied
                encoder_key_rope = txt_img_key[:, :txt_len, :, :]
                img_query_rope = txt_img_query[:, txt_len:, :, :]
                
                concept_sim_maps = self._extract_concept_sim_maps(
                    img_query_rope, encoder_key_rope, hidden_states_out,
                    freefuse_token_pos_maps, top_k_ratio,
                    eos_token_index, background_token_positions,
                )
                self.concept_sim_maps = concept_sim_maps

            # Output projections
            hidden_states_out = attn.to_out[0](hidden_states_out)
            hidden_states_out = attn.to_out[1](hidden_states_out)
            encoder_hidden_states_out = attn.to_add_out(encoder_hidden_states_out)

            return hidden_states_out, encoder_hidden_states_out, {}, concept_sim_maps
        else:
            return attention_output

    def _prepare_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        query: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Combine attention mask with optional additive bias."""
        if self._attention_bias is not None:
            B, S = query.shape[0], query.shape[1]
            bias = self._attention_bias

            if bias.shape[0] < B:
                repeats = (B + bias.shape[0] - 1) // bias.shape[0]
                bias = bias.repeat(repeats, 1, 1)[:B]

            if bias.shape[1] < S:
                pad_s = S - bias.shape[1]
                bias = F.pad(bias, (0, pad_s, 0, pad_s), value=0.0)

            bias = bias.unsqueeze(1).to(query.dtype)

            if attention_mask is not None and attention_mask.ndim == 2:
                float_key = torch.zeros(B, 1, 1, S, device=attention_mask.device, dtype=query.dtype)
                float_key.masked_fill_(~attention_mask.bool().unsqueeze(1).unsqueeze(1), torch.finfo(query.dtype).min)
                return float_key + bias
            return bias

        if attention_mask is not None and attention_mask.ndim == 2:
            return attention_mask[:, None, None, :]
        return attention_mask

    def _extract_concept_sim_maps(
        self,
        img_query_rope: torch.Tensor,
        encoder_key_rope: torch.Tensor,
        hidden_states_out: torch.Tensor,
        freefuse_token_pos_maps: Dict[str, List[List[int]]],
        top_k_ratio: float,
        eos_token_index: Optional[int],
        background_token_positions: Optional[List[int]],
    ) -> Dict[str, torch.Tensor]:
        """
        Extract concept similarity maps using cross-attention top-k + concept attention.
        
        Strategy:
        1. Cross-attention: img_query @ concept_text_key -> select top-k image tokens
        2. Concept attention: top-k hidden-states @ all image hidden-states
        3. Softmax normalization for spatial mask
        """
        concept_sim_maps = {}
        img_len = img_query_rope.shape[1]
        scale = 1.0 / 1000.0

        # Step 1: Compute cross-attention scores for each concept
        all_cross_attn_scores = {}
        all_concept_keys = {}
        
        for lora_name, positions_list in freefuse_token_pos_maps.items():
            pos = positions_list[0]  # First prompt's positions
            if len(pos) > 0:
                pos_tensor = torch.tensor(pos, device=encoder_key_rope.device)
                concept_key = encoder_key_rope[:, pos_tensor, :, :]
                all_concept_keys[lora_name] = concept_key
                
                cross_attn_weights = torch.einsum('bihd,bjhd->bhij', img_query_rope, concept_key) * scale
                cross_attn_weights = F.softmax(cross_attn_weights, dim=2)
                cross_attn_scores = cross_attn_weights.mean(dim=1).mean(dim=-1)
                all_cross_attn_scores[lora_name] = cross_attn_scores

        # Step 2: Contrastive top-k + concept attention
        n_concepts = len(all_cross_attn_scores)
        for lora_name in all_cross_attn_scores.keys():
            # Contrastive scoring
            scores = all_cross_attn_scores[lora_name] * n_concepts
            for other in all_cross_attn_scores:
                if other != lora_name:
                    scores = scores - all_cross_attn_scores[other]

            k = max(1, int(img_len * top_k_ratio))
            _, topk_idx = torch.topk(scores, k, dim=-1)

            # Extract core image tokens
            expanded = topk_idx.unsqueeze(-1).expand(-1, -1, hidden_states_out.shape[-1])
            core = torch.gather(hidden_states_out, dim=1, index=expanded)

            # Concept attention: core @ all
            sim = core @ hidden_states_out.transpose(-1, -2)
            sim_avg = sim.mean(dim=1, keepdim=True).transpose(1, 2)
            sim_map = F.softmax(sim_avg / 4000.0, dim=1)
            concept_sim_maps[lora_name] = sim_map

        # Background sim-map
        if background_token_positions is not None and len(background_token_positions) > 0:
            concept_sim_maps['__bg__'] = self._compute_background_sim_map(
                img_query_rope, encoder_key_rope, hidden_states_out,
                background_token_positions, top_k_ratio, scale
            )
        elif eos_token_index is not None:
            concept_sim_maps['__eos__'] = self._compute_background_sim_map(
                img_query_rope, encoder_key_rope, hidden_states_out,
                [eos_token_index], top_k_ratio, scale
            )

        return concept_sim_maps

    def _compute_background_sim_map(
        self,
        img_query_rope: torch.Tensor,
        encoder_key_rope: torch.Tensor,
        hidden_states_out: torch.Tensor,
        positions: List[int],
        top_k_ratio: float,
        scale: float,
    ) -> torch.Tensor:
        """Compute background similarity map from specified token positions."""
        img_len = img_query_rope.shape[1]
        pos_tensor = torch.tensor(positions, device=encoder_key_rope.device)
        bg_key = encoder_key_rope[:, pos_tensor, :, :]

        bg_weights = torch.einsum('bihd,bjhd->bhij', img_query_rope, bg_key) * scale
        bg_weights = F.softmax(bg_weights, dim=2)
        bg_scores = bg_weights.mean(dim=1).mean(dim=-1)

        k = max(1, int(img_len * top_k_ratio))
        _, topk_idx = torch.topk(bg_scores, k, dim=-1)

        expanded = topk_idx.unsqueeze(-1).expand(-1, -1, hidden_states_out.shape[-1])
        core = torch.gather(hidden_states_out, dim=1, index=expanded)

        sim = core @ hidden_states_out.transpose(-1, -2)
        sim_avg = sim.mean(dim=1, keepdim=True).transpose(1, 2)
        return F.softmax(sim_avg / 4000.0, dim=1)


class FreeFuseFlux2SingleAttnProcessor:
    """
    FreeFuse attention processor for Flux2 Single-Stream blocks.
    
    Handles fused QKV+MLP projections and:
    - Concept similarity map extraction from unified [text, image] sequence
    - Additive attention bias injection
    """
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0.")
        
        # Sim-map extraction control
        self.cal_concept_sim_map: bool = False
        self.concept_sim_maps: Optional[Dict[str, torch.Tensor]] = None
        
        # Set by transformer before each forward
        self._freefuse_token_pos_maps: Optional[Dict[str, List[List[int]]]] = None
        self._txt_len: int = 0  # Text sequence length
        self._top_k_ratio: float = 0.1
        self._eos_token_index: Optional[int] = None
        self._background_token_positions: Optional[List[int]] = None
        self._attention_bias: Optional[torch.Tensor] = None

    def __call__(
        self,
        attn: "Flux2ParallelSelfAttention",
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        freefuse_token_pos_maps: Optional[Dict[str, List[List[int]]]] = None,
        txt_len: int = 0,
        top_k_ratio: float = 0.1,
        eos_token_index: Optional[int] = None,
        background_token_positions: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Forward pass for single-stream block with FreeFuse support.
        
        In single-stream blocks, hidden_states is already [text, image] concatenated.
        """
        # Use passed params or fall back to stored ones
        freefuse_token_pos_maps = freefuse_token_pos_maps or self._freefuse_token_pos_maps
        txt_len = txt_len if txt_len > 0 else self._txt_len
        top_k_ratio = top_k_ratio if freefuse_token_pos_maps else self._top_k_ratio
        eos_token_index = eos_token_index if eos_token_index is not None else self._eos_token_index
        background_token_positions = background_token_positions or self._background_token_positions

        # Fused QKV + MLP projection
        proj_output = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(
            proj_output, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1
        )

        query, key, value = qkv.chunk(3, dim=-1)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # Apply RoPE
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # Prepare attention mask with optional bias
        combined_mask = self._prepare_attention_mask(attention_mask, query)

        # Run attention
        attn_output = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=combined_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        attn_output = attn_output.flatten(2, 3)
        attn_output = attn_output.to(query.dtype)

        # Extract concept similarity maps if enabled
        if self.cal_concept_sim_map and freefuse_token_pos_maps is not None and txt_len > 0:
            img_len = hidden_states.shape[1] - txt_len
            
            # Split unified sequence into text and image parts
            txt_key = key[:, :txt_len, :, :]
            img_query = query[:, txt_len:, :, :]
            img_attn_out = attn_output[:, txt_len:, :]
            
            self.concept_sim_maps = self._extract_concept_sim_maps(
                img_query, txt_key, img_attn_out,
                freefuse_token_pos_maps, top_k_ratio,
                eos_token_index, background_token_positions,
            )

        # MLP activation
        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)

        # Fused output projection
        combined_output = torch.cat([attn_output, mlp_hidden_states], dim=-1)
        output = attn.to_out(combined_output)

        return output

    def _prepare_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        query: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Combine attention mask with optional additive bias."""
        if self._attention_bias is not None:
            B, S = query.shape[0], query.shape[1]
            bias = self._attention_bias

            if bias.shape[0] < B:
                repeats = (B + bias.shape[0] - 1) // bias.shape[0]
                bias = bias.repeat(repeats, 1, 1)[:B]

            if bias.shape[1] < S:
                pad_s = S - bias.shape[1]
                bias = F.pad(bias, (0, pad_s, 0, pad_s), value=0.0)

            bias = bias.unsqueeze(1).to(query.dtype)

            if attention_mask is not None and attention_mask.ndim == 2:
                float_key = torch.zeros(B, 1, 1, S, device=attention_mask.device, dtype=query.dtype)
                float_key.masked_fill_(~attention_mask.bool().unsqueeze(1).unsqueeze(1), torch.finfo(query.dtype).min)
                return float_key + bias
            return bias

        if attention_mask is not None and attention_mask.ndim == 2:
            return attention_mask[:, None, None, :]
        return attention_mask

    def _extract_concept_sim_maps(
        self,
        img_query: torch.Tensor,
        txt_key: torch.Tensor,
        img_attn_out: torch.Tensor,
        freefuse_token_pos_maps: Dict[str, List[List[int]]],
        top_k_ratio: float,
        eos_token_index: Optional[int],
        background_token_positions: Optional[List[int]],
    ) -> Dict[str, torch.Tensor]:
        """Extract concept sim maps from unified sequence (similar to Z-Image)."""
        concept_sim_maps = {}
        img_len = img_query.shape[1]
        scale = 1.0 / 1000.0

        # Cross-attention scores for each concept
        all_cross_attn_scores = {}
        
        for lora_name, positions_list in freefuse_token_pos_maps.items():
            pos = positions_list[0]
            if len(pos) > 0:
                pos_tensor = torch.tensor(pos, device=txt_key.device)
                concept_key = txt_key[:, pos_tensor, :, :]
                
                weights = torch.einsum('bihd,bjhd->bhij', img_query, concept_key) * scale
                weights = F.softmax(weights, dim=2)
                scores = weights.mean(dim=1).mean(dim=-1)
                all_cross_attn_scores[lora_name] = scores

        # Contrastive top-k + concept attention
        n_concepts = len(all_cross_attn_scores)
        for lora_name in all_cross_attn_scores.keys():
            scores = all_cross_attn_scores[lora_name] * n_concepts
            for other in all_cross_attn_scores:
                if other != lora_name:
                    scores = scores - all_cross_attn_scores[other]

            k = max(1, int(img_len * top_k_ratio))
            _, topk_idx = torch.topk(scores, k, dim=-1)

            expanded = topk_idx.unsqueeze(-1).expand(-1, -1, img_attn_out.shape[-1])
            core = torch.gather(img_attn_out, dim=1, index=expanded)

            sim = core @ img_attn_out.transpose(-1, -2)
            sim_avg = sim.mean(dim=1, keepdim=True).transpose(1, 2)
            sim_map = F.softmax(sim_avg / 4000.0, dim=1)
            concept_sim_maps[lora_name] = sim_map

        # Background sim-map
        if background_token_positions is not None and len(background_token_positions) > 0:
            concept_sim_maps['__bg__'] = self._compute_bg_sim_map(
                img_query, txt_key, img_attn_out, background_token_positions, top_k_ratio, scale
            )
        elif eos_token_index is not None:
            concept_sim_maps['__eos__'] = self._compute_bg_sim_map(
                img_query, txt_key, img_attn_out, [eos_token_index], top_k_ratio, scale
            )

        return concept_sim_maps

    def _compute_bg_sim_map(
        self,
        img_query: torch.Tensor,
        txt_key: torch.Tensor,
        img_attn_out: torch.Tensor,
        positions: List[int],
        top_k_ratio: float,
        scale: float,
    ) -> torch.Tensor:
        """Compute background similarity map."""
        img_len = img_query.shape[1]
        pos_tensor = torch.tensor(positions, device=txt_key.device)
        bg_key = txt_key[:, pos_tensor, :, :]

        weights = torch.einsum('bihd,bjhd->bhij', img_query, bg_key) * scale
        weights = F.softmax(weights, dim=2)
        scores = weights.mean(dim=1).mean(dim=-1)

        k = max(1, int(img_len * top_k_ratio))
        _, topk_idx = torch.topk(scores, k, dim=-1)

        expanded = topk_idx.unsqueeze(-1).expand(-1, -1, img_attn_out.shape[-1])
        core = torch.gather(img_attn_out, dim=1, index=expanded)

        sim = core @ img_attn_out.transpose(-1, -2)
        sim_avg = sim.mean(dim=1, keepdim=True).transpose(1, 2)
        return F.softmax(sim_avg / 4000.0, dim=1)
