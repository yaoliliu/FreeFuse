# Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team. All rights reserved.
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
FreeFuse Attention Processor for Z-Image's single-stream architecture.

This processor:
1. Performs standard single-stream self-attention (QKV, RMSNorm, RoPE, SDPA)
2. Optionally injects additive attention bias for cross-LoRA suppression
3. Extracts concept similarity maps using cross-attn top-k + concept attention

Key difference from Flux: Z-Image uses a unified sequence [image, text] with
complex-valued RoPE, whereas Flux has a dual-stream [text, image] with
real-valued RoPE.
"""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from diffusers.models.attention_processor import Attention
from diffusers.models.attention_dispatch import dispatch_attention_fn


class FreeFuseZImageAttnProcessor:
    """
    FreeFuse attention processor for Z-Image single-stream transformer blocks.

    Replaces `ZSingleStreamAttnProcessor` and adds:
    - Additive attention bias (for cross-LoRA text-image suppression)
    - Concept similarity map extraction (cross-attn top-k → concept attention)

    The processor stores extracted `concept_sim_maps` on itself for later
    retrieval by the transformer's forward method.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "FreeFuseZImageAttnProcessor requires PyTorch 2.0. "
                "Please upgrade PyTorch to version 2.0 or higher."
            )

        # ── Sim-map extraction control ──
        self.cal_concept_sim_map: bool = False
        self.concept_sim_maps: Optional[Dict[str, torch.Tensor]] = None

        # ── Set by the transformer before each forward loop ──
        self._freefuse_token_pos_maps: Optional[Dict[str, List[List[int]]]] = None
        self._x_seqlens: Optional[List[int]] = None      # per-batch image token lengths
        self._cap_seqlens: Optional[List[int]] = None     # per-batch text token lengths
        self._top_k_ratio: float = 0.1
        self._eos_token_index: Optional[int] = None       # text-relative
        self._background_token_positions: Optional[List[int]] = None  # text-relative
        self._attention_bias: Optional[torch.Tensor] = None  # (B, seq, seq) additive

    # ──────────────────────────────────────────────────────────────────
    # Main __call__
    # ──────────────────────────────────────────────────────────────────
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # ── QKV projections ──
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        # ── QK normalization (RMSNorm) ──
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # ── RoPE (complex-valued, Z-Image style) ──
        if freqs_cis is not None:
            query = self._apply_rotary_emb(query, freqs_cis)
            key = self._apply_rotary_emb(key, freqs_cis)

        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        # ── Build combined attention mask (key-mask + optional additive bias) ──
        combined_mask = self._prepare_attention_mask(attention_mask, query)

        # ── Scaled dot-product attention ──
        hidden_states_out = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=combined_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        hidden_states_out = hidden_states_out.flatten(2, 3)
        hidden_states_out = hidden_states_out.to(dtype)

        # ── Concept similarity map extraction ──
        if (
            self.cal_concept_sim_map
            and self._freefuse_token_pos_maps is not None
            and self._x_seqlens is not None
        ):
            self._extract_concept_sim_maps(query, key, hidden_states_out)

        # ── Output projection ──
        output = attn.to_out[0](hidden_states_out)
        if len(attn.to_out) > 1:  # dropout
            output = attn.to_out[1](output)

        return output

    # ──────────────────────────────────────────────────────────────────
    # RoPE  (Z-Image uses complex-valued rotary embeddings)
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _apply_rotary_emb(
        x_in: torch.Tensor, freqs_cis: torch.Tensor
    ) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            x = torch.view_as_complex(
                x_in.float().reshape(*x_in.shape[:-1], -1, 2)
            )
            freqs_cis = freqs_cis.unsqueeze(2)
            x_out = torch.view_as_real(x * freqs_cis).flatten(3)
            return x_out.type_as(x_in)

    # ──────────────────────────────────────────────────────────────────
    # Attention mask + bias
    # ──────────────────────────────────────────────────────────────────
    def _prepare_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        query: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Combine the boolean key-mask with the optional additive attention bias.

        Returns
        -------
        combined : Tensor | None
            - If no bias and no mask → ``None``
            - If only key-mask       → (B, 1, 1, S)  bool
            - If bias present         → (B, 1, S, S)  float  (with -inf for padding)
        """
        if self._attention_bias is not None:
            B, S = query.shape[0], query.shape[1]
            bias = self._attention_bias  # (B_bias, S, S)

            # Handle batch mismatch (e.g. CFG doubles the batch)
            if bias.shape[0] < B:
                repeats = (B + bias.shape[0] - 1) // bias.shape[0]
                bias = bias.repeat(repeats, 1, 1)[:B]

            # Handle sequence length mismatch (bias may be smaller than padded seq)
            if bias.shape[1] < S:
                pad_s = S - bias.shape[1]
                bias = F.pad(bias, (0, pad_s, 0, pad_s), value=0.0)

            bias = bias.unsqueeze(1).to(query.dtype)  # (B, 1, S, S)

            if attention_mask is not None and attention_mask.ndim == 2:
                float_key = torch.zeros(
                    B, 1, 1, S, device=attention_mask.device, dtype=query.dtype
                )
                float_key.masked_fill_(
                    ~attention_mask.bool().unsqueeze(1).unsqueeze(1),
                    torch.finfo(query.dtype).min,
                )
                return float_key + bias
            return bias

        # No bias — fall back to standard boolean key-mask
        if attention_mask is not None and attention_mask.ndim == 2:
            return attention_mask[:, None, None, :]
        return attention_mask

    # ──────────────────────────────────────────────────────────────────
    # Concept sim-map extraction
    # ──────────────────────────────────────────────────────────────────
    def _extract_concept_sim_maps(
        self,
        query: torch.Tensor,      # (B, seq, heads, head_dim)  after RoPE
        key: torch.Tensor,         # (B, seq, heads, head_dim)  after RoPE
        hidden_states_out: torch.Tensor,  # (B, seq, dim)   attention output
    ) -> None:
        """
        Compute per-concept spatial similarity maps.

        Strategy (same as Flux FreeFuse):
        1. Cross-attention: image-Q @ concept-text-K → select top-k image tokens
        2. Concept attention: top-k hidden-states inner-product with all image hidden-states
        3. Softmax → sim_map  (B, img_token_count_with_padding, 1)

        Results are stored on ``self.concept_sim_maps``.
        """
        concept_sim_maps: Dict[str, torch.Tensor] = {}
        B = query.shape[0]

        for b in range(B):
            x_len = self._x_seqlens[b]
            cap_len = self._cap_seqlens[b]

            # Split unified sequence into image / text regions
            img_query = query[b : b + 1, :x_len]               # (1, x_len, H, D)
            img_hidden = hidden_states_out[b : b + 1, :x_len]  # (1, x_len, dim)
            txt_key = key[b : b + 1, x_len : x_len + cap_len]  # (1, cap_len, H, D)

            img_len = x_len
            scale = 1.0 / 1000.0

            # ── Step 1: cross-attn scores per concept ──
            all_cross_attn_scores: Dict[str, torch.Tensor] = {}

            for lora_name, positions_list in self._freefuse_token_pos_maps.items():
                pos = positions_list[min(b, len(positions_list) - 1)]
                if not pos:
                    continue
                pos_t = torch.tensor(pos, device=query.device)
                concept_key = txt_key[:, pos_t]  # (1, n_concept, H, D)

                weights = torch.einsum(
                    "bihd,bjhd->bhij", img_query, concept_key
                ) * scale
                weights = F.softmax(weights, dim=2)
                scores = weights.mean(dim=1).mean(dim=-1)  # (1, img_len)
                all_cross_attn_scores[lora_name] = scores

            # ── Step 2: contrastive top-k + concept attention ──
            n_concepts = len(all_cross_attn_scores)
            for lora_name in list(all_cross_attn_scores.keys()):
                scores = all_cross_attn_scores[lora_name] * n_concepts
                for other in all_cross_attn_scores:
                    if other != lora_name:
                        scores = scores - all_cross_attn_scores[other]

                k = max(1, int(img_len * self._top_k_ratio))
                _, topk_idx = torch.topk(scores, k, dim=-1)

                expanded = topk_idx.unsqueeze(-1).expand(
                    -1, -1, img_hidden.shape[-1]
                )
                core = torch.gather(img_hidden, dim=1, index=expanded)

                sim = core @ img_hidden.transpose(-1, -2)       # (1, k, img_len)
                sim_avg = sim.mean(dim=1, keepdim=True).transpose(1, 2)  # (1, img_len, 1)
                sim_map = F.softmax(sim_avg / 4000.0, dim=1)

                if b == 0:
                    concept_sim_maps[lora_name] = sim_map
                else:
                    concept_sim_maps[lora_name] = torch.cat(
                        [concept_sim_maps[lora_name], sim_map], dim=0
                    )

            # ── Background sim-map (user-defined tokens) ──
            if (
                self._background_token_positions is not None
                and len(self._background_token_positions) > 0
            ):
                self._compute_background_sim_map(
                    img_query, img_hidden, txt_key, img_len,
                    torch.tensor(
                        self._background_token_positions, device=query.device
                    ),
                    scale, b, concept_sim_maps, "__bg__",
                )
            elif self._eos_token_index is not None:
                eos_pos = torch.tensor(
                    [self._eos_token_index], device=query.device
                )
                self._compute_background_sim_map(
                    img_query, img_hidden, txt_key, img_len,
                    eos_pos, scale, b, concept_sim_maps, "__eos__",
                )

        self.concept_sim_maps = concept_sim_maps if concept_sim_maps else None

    # ──────────────────────────────────────────────────────────────────
    def _compute_background_sim_map(
        self,
        img_query: torch.Tensor,
        img_hidden: torch.Tensor,
        txt_key: torch.Tensor,
        img_len: int,
        positions: torch.Tensor,
        scale: float,
        batch_idx: int,
        out_dict: Dict[str, torch.Tensor],
        key_name: str,
    ) -> None:
        """Cross-attn top-k + concept attention for background / EOS tokens."""
        bg_key = txt_key[:, positions]
        weights = torch.einsum("bihd,bjhd->bhij", img_query, bg_key) * scale
        weights = F.softmax(weights, dim=2)
        scores = weights.mean(dim=1).mean(dim=-1)  # (1, img_len)

        k = max(1, int(img_len * self._top_k_ratio))
        _, topk_idx = torch.topk(scores, k, dim=-1)

        expanded = topk_idx.unsqueeze(-1).expand(-1, -1, img_hidden.shape[-1])
        core = torch.gather(img_hidden, dim=1, index=expanded)

        sim = core @ img_hidden.transpose(-1, -2)
        sim_avg = sim.mean(dim=1, keepdim=True).transpose(1, 2)
        sim_map = F.softmax(sim_avg / 4000.0, dim=1)

        if batch_idx == 0:
            out_dict[key_name] = sim_map
        else:
            out_dict[key_name] = torch.cat(
                [out_dict[key_name], sim_map], dim=0
            )
