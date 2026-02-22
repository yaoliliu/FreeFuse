"""
FreeFuse attention processor for LTX-2.0 audiovisual transformer.

This processor extends the stock LTX2 attention processor with:
1) concept similarity map extraction (query-token -> concept-token routing)
2) additive attention bias support (for concept-aware suppression)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_ltx2 import (
    apply_interleaved_rotary_emb,
    apply_split_rotary_emb,
)
from diffusers.utils import is_torch_version


class FreeFuseLTX2AttnProcessor:
    """
    FreeFuse processor for LTX2 attention layers.

    `role` identifies which attention branch this processor belongs to, e.g.
    `video_text`, `audio_text`, `video_self`, `audio_self`, etc.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self, role: str = ""):
        if is_torch_version("<", "2.0"):
            raise ValueError(
                "LTX2 attention processors require a minimum PyTorch version of 2.0."
            )

        self._freefuse_role: str = role

        # Sim-map extraction state
        self.cal_concept_sim_map: bool = False
        self.concept_sim_maps: Optional[Dict[str, torch.Tensor]] = None

        # Runtime controls
        self._freefuse_token_pos_maps: Optional[Dict[str, List[List[int]]]] = None
        self._top_k_ratio: float = 0.1
        self._eos_token_index: Optional[int] = None
        self._background_token_positions: Optional[List[int]] = None
        self._attention_bias: Optional[torch.Tensor] = None

    @staticmethod
    def _adaptive_softmax(scores: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Normalize scores with per-sample z-score before softmax.
        This avoids over-flattening from fixed large temperatures across different models.
        """
        centered = scores - scores.mean(dim=dim, keepdim=True)
        scale = centered.std(dim=dim, keepdim=True, unbiased=False).clamp_min(1e-6)
        return F.softmax(centered / scale, dim=dim)

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        query_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        key_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        query_len = hidden_states.shape[1]
        key_len = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else query_len

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if query_rotary_emb is not None:
            if attn.rope_type == "interleaved":
                query = apply_interleaved_rotary_emb(query, query_rotary_emb)
                key = apply_interleaved_rotary_emb(
                    key, key_rotary_emb if key_rotary_emb is not None else query_rotary_emb
                )
            elif attn.rope_type == "split":
                query = apply_split_rotary_emb(query, query_rotary_emb)
                key = apply_split_rotary_emb(
                    key, key_rotary_emb if key_rotary_emb is not None else query_rotary_emb
                )

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, key_len, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        combined_mask = self._combine_attention_mask_and_bias(
            attention_mask=attention_mask,
            query=query,
            key=key,
        )

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
        hidden_states_out = hidden_states_out.to(query.dtype)

        if (
            self.cal_concept_sim_map
            and encoder_hidden_states is not None
            and self._freefuse_token_pos_maps is not None
            and self._freefuse_role in {"video_text", "audio_text"}
        ):
            self.concept_sim_maps = self._extract_concept_sim_maps(
                query_tokens=query,                 # (B, q_len, H, D)
                text_keys=key,                      # (B, text_len, H, D)
                query_hidden_states=hidden_states_out,  # (B, q_len, C)
                token_pos_maps=self._freefuse_token_pos_maps,
                top_k_ratio=self._top_k_ratio,
                eos_token_index=self._eos_token_index,
                background_token_positions=self._background_token_positions,
            )

        hidden_states_out = attn.to_out[0](hidden_states_out)
        hidden_states_out = attn.to_out[1](hidden_states_out)
        return hidden_states_out

    def _combine_attention_mask_and_bias(
        self,
        attention_mask: Optional[torch.Tensor],
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Merge the stock attention mask with optional additive FreeFuse bias.
        """
        combined = attention_mask
        if combined is not None:
            if combined.ndim == 2:
                combined = combined[:, None, None, :]
            elif combined.ndim == 3:
                combined = combined[:, None, :, :]
            combined = combined.to(dtype=query.dtype)

            # Broadcast query length if needed.
            if combined.shape[-2] == 1 and query.shape[1] > 1:
                combined = combined.expand(-1, combined.shape[1], query.shape[1], -1)

        if self._attention_bias is None:
            return combined

        bias = self._attention_bias  # (B, q_len, k_len)
        if bias.ndim != 3:
            raise ValueError(f"Expected `_attention_bias` to be 3D, got {tuple(bias.shape)}.")

        q_len = query.shape[1]
        k_len = key.shape[1]
        bsz = query.shape[0]

        if bias.shape[0] < bsz:
            repeats = (bsz + bias.shape[0] - 1) // bias.shape[0]
            bias = bias.repeat(repeats, 1, 1)
        bias = bias[:bsz]

        if bias.shape[1] < q_len:
            bias = F.pad(bias, (0, 0, 0, q_len - bias.shape[1]), value=0.0)
        elif bias.shape[1] > q_len:
            bias = bias[:, :q_len, :]

        if bias.shape[2] < k_len:
            bias = F.pad(bias, (0, k_len - bias.shape[2], 0, 0), value=0.0)
        elif bias.shape[2] > k_len:
            bias = bias[:, :, :k_len]

        bias = bias.unsqueeze(1).to(dtype=query.dtype)
        if combined is None:
            return bias
        return combined + bias

    def _extract_concept_sim_maps(
        self,
        query_tokens: torch.Tensor,        # (B, q_len, H, D)
        text_keys: torch.Tensor,           # (B, text_len, H, D)
        query_hidden_states: torch.Tensor, # (B, q_len, C)
        token_pos_maps: Dict[str, List[List[int]]],
        top_k_ratio: float,
        eos_token_index: Optional[int],
        background_token_positions: Optional[List[int]],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Cross-attn top-k + concept attention map extraction.
        """
        B, q_len = query_tokens.shape[0], query_tokens.shape[1]
        text_len = text_keys.shape[1]
        head_dim = query_tokens.shape[-1]
        scale = head_dim ** -0.5

        if query_hidden_states.ndim != 3:
            raise ValueError(
                f"Expected `query_hidden_states` to be 3D (B, q_len, C), got {tuple(query_hidden_states.shape)}."
            )
        if query_hidden_states.shape[1] != q_len:
            raise ValueError(
                "Mismatched query lengths between attention tokens and hidden states: "
                f"{q_len} vs {query_hidden_states.shape[1]}."
            )

        result_by_name: Dict[str, List[torch.Tensor]] = {}
        bg_result: List[torch.Tensor] = []
        bg_key_name: Optional[str] = None

        for b in range(B):
            per_concept_scores: Dict[str, torch.Tensor] = {}

            for lora_name, positions_list in token_pos_maps.items():
                if not positions_list:
                    continue
                cur_positions = positions_list[min(b, len(positions_list) - 1)]
                cur_positions = [p for p in cur_positions if 0 <= p < text_len]
                if not cur_positions:
                    continue

                pos_tensor = torch.tensor(cur_positions, device=query_tokens.device, dtype=torch.long)
                concept_key = text_keys[b : b + 1, pos_tensor]  # (1, n_pos, H, D)

                weights = torch.einsum("bihd,bjhd->bhij", query_tokens[b : b + 1], concept_key) * scale
                weights = F.softmax(weights, dim=2)
                scores = weights.mean(dim=1).mean(dim=-1)  # (1, q_len)
                per_concept_scores[lora_name] = scores

            if not per_concept_scores:
                continue

            num_concepts = len(per_concept_scores)
            k = max(1, int(q_len * float(top_k_ratio)))

            for lora_name, base_scores in per_concept_scores.items():
                contrastive = base_scores * num_concepts
                for other_name, other_scores in per_concept_scores.items():
                    if other_name != lora_name:
                        contrastive = contrastive - other_scores

                _, topk_idx = torch.topk(contrastive, k, dim=-1)
                gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, query_hidden_states.shape[-1])
                core = torch.gather(query_hidden_states[b : b + 1], dim=1, index=gather_idx)

                sim = core @ query_hidden_states[b : b + 1].transpose(-1, -2)  # (1, k, q_len)
                sim_avg = sim.mean(dim=1, keepdim=True).transpose(1, 2)         # (1, q_len, 1)
                sim_map = self._adaptive_softmax(sim_avg.float(), dim=1).to(dtype=query_hidden_states.dtype)
                if sim_map.ndim != 3 or sim_map.shape[1] != q_len or sim_map.shape[-1] != 1:
                    raise ValueError(
                        f"Unexpected concept sim-map shape for `{lora_name}`: {tuple(sim_map.shape)}."
                    )
                if not torch.isfinite(sim_map).all():
                    raise ValueError(f"Non-finite concept sim-map detected for `{lora_name}`.")
                sim_mass = sim_map.float().sum(dim=1)  # (1, 1)
                if not torch.allclose(sim_mass, torch.ones_like(sim_mass), atol=1e-2, rtol=1e-2):
                    raise ValueError(
                        f"Concept sim-map for `{lora_name}` is not normalized over token dimension. "
                        f"sum range=({float(sim_mass.min().item()):.6f}, {float(sim_mass.max().item()):.6f})"
                    )

                result_by_name.setdefault(lora_name, []).append(sim_map)

            bg_positions = None
            if background_token_positions is not None and len(background_token_positions) > 0:
                bg_positions = [p for p in background_token_positions if 0 <= p < text_len]
                bg_key_name = "__bg__"
            elif eos_token_index is not None and 0 <= eos_token_index < text_len:
                bg_positions = [int(eos_token_index)]
                bg_key_name = "__eos__"

            if bg_positions:
                bg_pos_tensor = torch.tensor(bg_positions, device=query_tokens.device, dtype=torch.long)
                bg_key = text_keys[b : b + 1, bg_pos_tensor]  # (1, n_bg, H, D)

                bg_weights = torch.einsum("bihd,bjhd->bhij", query_tokens[b : b + 1], bg_key) * scale
                bg_weights = F.softmax(bg_weights, dim=2)
                bg_scores = bg_weights.mean(dim=1).mean(dim=-1)  # (1, q_len)

                _, topk_idx = torch.topk(bg_scores, k, dim=-1)
                gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, query_hidden_states.shape[-1])
                core = torch.gather(query_hidden_states[b : b + 1], dim=1, index=gather_idx)
                sim = core @ query_hidden_states[b : b + 1].transpose(-1, -2)
                sim_avg = sim.mean(dim=1, keepdim=True).transpose(1, 2)
                bg_map = self._adaptive_softmax(sim_avg.float(), dim=1).to(dtype=query_hidden_states.dtype)
                if bg_map.ndim != 3 or bg_map.shape[1] != q_len or bg_map.shape[-1] != 1:
                    raise ValueError(f"Unexpected background sim-map shape: {tuple(bg_map.shape)}.")
                if not torch.isfinite(bg_map).all():
                    raise ValueError("Non-finite background sim-map detected.")
                bg_mass = bg_map.float().sum(dim=1)
                if not torch.allclose(bg_mass, torch.ones_like(bg_mass), atol=1e-2, rtol=1e-2):
                    raise ValueError(
                        "Background sim-map is not normalized over token dimension. "
                        f"sum range=({float(bg_mass.min().item()):.6f}, {float(bg_mass.max().item()):.6f})"
                    )
                bg_result.append(bg_map)

        if not result_by_name and not bg_result:
            return None

        concept_maps: Dict[str, torch.Tensor] = {}
        for lora_name, maps in result_by_name.items():
            concept_maps[lora_name] = torch.cat(maps, dim=0)

        if bg_result and bg_key_name is not None:
            concept_maps[bg_key_name] = torch.cat(bg_result, dim=0)

        return concept_maps
