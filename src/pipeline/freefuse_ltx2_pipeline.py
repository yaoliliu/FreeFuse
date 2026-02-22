"""
FreeFuse pipeline for LTX-2.0.

This pipeline wraps the stock `LTX2Pipeline` with a two-phase FreeFuse flow:
1) Phase 1: collect concept sim-maps from video/audio text cross-attn
2) Phase 2: regenerate from the same initial noise with LoRA routing masks + bias
"""

import os
import string
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from diffusers.models.autoencoders import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video
from diffusers.models.transformers import LTX2VideoTransformer3DModel
from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
from diffusers.pipelines.ltx2.pipeline_ltx2 import LTX2Pipeline
from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import Gemma3ForConditionalGeneration, GemmaTokenizer, GemmaTokenizerFast

from src.models.freefuse_transformer_ltx2 import FreeFuseLTX2VideoTransformer3DModel
from src.tuner.freefuse_lora_layer import convert_peft_lora_to_freefuse_lora


class FreeFuseLTX2Pipeline(LTX2Pipeline):
    """
    LTX2 pipeline with FreeFuse two-phase routing for multi-LoRA composition.
    """

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLLTX2Video,
        audio_vae: AutoencoderKLLTX2Audio,
        text_encoder: Gemma3ForConditionalGeneration,
        tokenizer: Union[GemmaTokenizer, GemmaTokenizerFast],
        connectors: LTX2TextConnectors,
        transformer: LTX2VideoTransformer3DModel,
        vocoder: LTX2Vocoder,
    ):
        super().__init__(
            scheduler=scheduler,
            vae=vae,
            audio_vae=audio_vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            connectors=connectors,
            transformer=transformer,
            vocoder=vocoder,
        )
        # IMPORTANT:
        # Do not class-swap transformer here. `set_adapters()` from diffusers
        # relies on the original class name to find adapter-scale handlers.
        # We swap to FreeFuse transformer lazily in `__call__` or explicitly
        # after LoRA adapters are configured.

        self._freefuse_last_concept_sim_maps: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
        self._freefuse_last_video_masks: Optional[Dict[str, torch.Tensor]] = None
        self._freefuse_last_audio_masks: Optional[Dict[str, torch.Tensor]] = None
        self._freefuse_last_attention_biases: Optional[Dict[str, torch.Tensor]] = None
        self._freefuse_last_phase1_debug_steps: Optional[List[Dict[str, Any]]] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup_freefuse_attention_processors(self) -> None:
        if not isinstance(self.transformer, FreeFuseLTX2VideoTransformer3DModel):
            self.transformer.__class__ = FreeFuseLTX2VideoTransformer3DModel
        self.transformer.setup_freefuse_attention_processors()

    def convert_lora_layers(self, include_connectors: bool = False) -> None:
        """
        Upgrade PEFT LoRA linear layers to `FreeFuseLinear`.
        """
        convert_peft_lora_to_freefuse_lora(self.transformer)
        if include_connectors and getattr(self, "connectors", None) is not None:
            convert_peft_lora_to_freefuse_lora(self.connectors)

    @contextmanager
    def _ltx2_tokenizer_alignment(self):
        """
        Align tokenizer behavior with `LTX2Pipeline._get_gemma_prompt_embeds`.
        """
        old_padding_side = getattr(self.tokenizer, "padding_side", "right")
        old_pad_token = getattr(self.tokenizer, "pad_token", None)

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            yield
        finally:
            self.tokenizer.padding_side = old_padding_side
            try:
                self.tokenizer.pad_token = old_pad_token
            except Exception:
                # Some tokenizer variants may not allow resetting pad token to None.
                pass

    # ------------------------------------------------------------------
    # Token helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _infer_connector_token_span(
        attention_mask: Optional[List[int]],
        seq_len: int,
    ) -> Tuple[int, int]:
        """
        Infer valid-token span before connector register replacement.

        LTX2 connectors first gather non-padding tokens and then place them at the
        beginning of the sequence. For left padding, this is equivalent to:
        `new_idx = old_idx - first_valid_idx`.
        """
        if attention_mask is None or len(attention_mask) != seq_len:
            return 0, seq_len

        valid_indices = [i for i, v in enumerate(attention_mask) if int(v) > 0]
        if not valid_indices:
            return seq_len, 0

        first_valid_idx = valid_indices[0]
        valid_token_count = len(valid_indices)
        return first_valid_idx, valid_token_count

    @staticmethod
    def _remap_positions_to_connector_indices(
        positions: List[int],
        first_valid_idx: int,
        valid_token_count: int,
    ) -> List[int]:
        if valid_token_count <= 0:
            return []

        remapped: List[int] = []
        for pos in positions:
            new_pos = int(pos) - int(first_valid_idx)
            if 0 <= new_pos < valid_token_count:
                remapped.append(new_pos)
        return sorted(set(remapped))

    def find_concept_positions(
        self,
        prompt: Union[str, List[str]],
        concept_map: Dict[str, str],
        max_sequence_length: int = 1024,
        filter_meaningless: bool = True,
        filter_single_char: bool = True,
    ) -> Dict[str, List[List[int]]]:
        """
        Find concept token positions in LTX2 (Gemma) tokenization space.

        Returned indices are in connector-attention space (after left-padding is
        removed and valid tokens are packed to the front).
        """
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        prompts = [p.strip() for p in prompts]

        stopwords = {
            "a",
            "an",
            "the",
            "and",
            "or",
            "but",
            "of",
            "in",
            "on",
            "at",
            "to",
            "for",
            "with",
            "by",
            "from",
            "as",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
        }

        def normalize_token_text(token_text: str) -> str:
            return (
                token_text.replace("▁", " ")
                .replace("Ġ", " ")
                .replace("_", " ")
                .strip()
                .lower()
            )

        def is_pure_punctuation(token_text: str) -> bool:
            cleaned = normalize_token_text(token_text)
            if not cleaned:
                return True
            return all(ch in string.punctuation for ch in cleaned)

        def is_meaningless_token(token_text: str) -> bool:
            cleaned = normalize_token_text(token_text)
            if not cleaned:
                return True
            if filter_single_char and len(cleaned) == 1:
                return True
            if cleaned in stopwords:
                return True
            if is_pure_punctuation(token_text):
                return True
            return False

        def find_subsequence_positions(haystack: List[int], needle: List[int]) -> List[int]:
            if not needle:
                return []
            n = len(needle)
            found: List[int] = []
            for start in range(0, len(haystack) - n + 1):
                if haystack[start : start + n] == needle:
                    found.extend(range(start, start + n))
            return found

        out: Dict[str, List[List[int]]] = {name: [] for name in concept_map}

        with self._ltx2_tokenizer_alignment():
            for prompt_text in prompts:
                offsets: Optional[List[Tuple[int, int]]] = None
                try:
                    tokenized = self.tokenizer(
                        prompt_text,
                        padding="max_length",
                        max_length=max_sequence_length,
                        truncation=True,
                        add_special_tokens=True,
                        return_offsets_mapping=True,
                        return_tensors="pt",
                    )
                    offsets = tokenized.offset_mapping[0].tolist()
                except (TypeError, ValueError, NotImplementedError):
                    tokenized = self.tokenizer(
                        prompt_text,
                        padding="max_length",
                        max_length=max_sequence_length,
                        truncation=True,
                        add_special_tokens=True,
                        return_tensors="pt",
                    )
                input_ids = tokenized.input_ids[0].tolist()
                attention_mask = (
                    tokenized.attention_mask[0].tolist()
                    if hasattr(tokenized, "attention_mask") and tokenized.attention_mask is not None
                    else None
                )
                first_valid_idx, valid_token_count = self._infer_connector_token_span(
                    attention_mask=attention_mask,
                    seq_len=len(input_ids),
                )

                for concept_name, concept_text in concept_map.items():
                    positions: List[int] = []
                    positions_with_text: List[Tuple[int, str]] = []
                    concept_text = concept_text.strip()
                    if not concept_text:
                        out[concept_name].append([])
                        continue

                    search_start = 0
                    if offsets is not None:
                        while True:
                            char_start = prompt_text.find(concept_text, search_start)
                            if char_start == -1:
                                break
                            char_end = char_start + len(concept_text)

                            for token_idx, (token_start, token_end) in enumerate(offsets):
                                if token_end <= char_start or token_start >= char_end:
                                    continue
                                if token_idx in positions:
                                    continue
                                positions.append(token_idx)
                                token_text = self.tokenizer.decode([input_ids[token_idx]], skip_special_tokens=False)
                                positions_with_text.append((token_idx, token_text))

                            search_start = char_start + 1
                    else:
                        concept_ids = self.tokenizer(
                            concept_text,
                            add_special_tokens=False,
                            return_tensors=None,
                        )["input_ids"]
                        matched_positions = find_subsequence_positions(input_ids, concept_ids)
                        for token_idx in matched_positions:
                            if token_idx in positions:
                                continue
                            positions.append(token_idx)
                            token_text = self.tokenizer.decode([input_ids[token_idx]], skip_special_tokens=False)
                            positions_with_text.append((token_idx, token_text))

                    if filter_meaningless and positions_with_text:
                        filtered_positions = [
                            pos
                            for pos, token_text in positions_with_text
                            if not is_meaningless_token(token_text)
                        ]
                        if not filtered_positions:
                            non_punct_positions = [
                                pos
                                for pos, token_text in positions_with_text
                                if not is_pure_punctuation(token_text)
                            ]
                            filtered_positions = [non_punct_positions[0]] if non_punct_positions else [positions_with_text[0][0]]
                        positions = filtered_positions

                    positions = self._remap_positions_to_connector_indices(
                        positions=positions,
                        first_valid_idx=first_valid_idx,
                        valid_token_count=valid_token_count,
                    )
                    out[concept_name].append(positions)

        return out

    def find_eos_token_index(
        self,
        prompt: str,
        max_sequence_length: int = 1024,
    ) -> Optional[int]:
        prompt = prompt.strip()
        with self._ltx2_tokenizer_alignment():
            tokenized = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            input_ids = tokenized.input_ids[0]
            attention_mask = tokenized.attention_mask[0] if hasattr(tokenized, "attention_mask") else None
            attention_mask_list = (
                attention_mask.tolist() if attention_mask is not None else None
            )
            first_valid_idx, valid_token_count = self._infer_connector_token_span(
                attention_mask=attention_mask_list,
                seq_len=input_ids.shape[0],
            )
            eos_id = self.tokenizer.eos_token_id
            eos_pos = (input_ids == eos_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                eos_valid_pos: Optional[int] = None
                for pos in eos_pos.tolist():
                    if attention_mask is None or int(attention_mask[pos].item()) > 0:
                        eos_valid_pos = int(pos)
                        break

                if eos_valid_pos is None:
                    return None

                remapped = self._remap_positions_to_connector_indices(
                    positions=[eos_valid_pos],
                    first_valid_idx=first_valid_idx,
                    valid_token_count=valid_token_count,
                )
                return remapped[0] if remapped else None
        return None

    # ------------------------------------------------------------------
    # Mask and bias builders
    # ------------------------------------------------------------------
    @staticmethod
    def _squeeze_sim_map(x: torch.Tensor, seq_len: int) -> torch.Tensor:
        if x.dim() == 3 and x.shape[-1] == 1:
            cur = x[:, :, 0]
        elif x.dim() == 2:
            cur = x
        else:
            raise ValueError(f"Unsupported sim-map shape: {tuple(x.shape)}")

        if cur.shape[1] < seq_len:
            pad = seq_len - cur.shape[1]
            cur = torch.nn.functional.pad(cur, (0, pad), value=0.0)
        elif cur.shape[1] > seq_len:
            cur = cur[:, :seq_len]
        return cur

    @staticmethod
    def stabilized_balanced_argmax(
        logits: torch.Tensor,
        max_iter: int = 12,
        lr: float = 0.05,
        momentum: float = 0.3,
    ) -> torch.Tensor:
        """
        Balanced hard assignment for `(B, C, N)` logits.
        Reduces collapse where one concept dominates most tokens.
        """
        if logits.ndim != 3:
            raise ValueError(f"Expected logits with shape (B, C, N), got {tuple(logits.shape)}")

        B, C, N = logits.shape
        if C <= 1:
            return torch.zeros((B, N), device=logits.device, dtype=torch.long)

        bias = torch.zeros((B, C, 1), device=logits.device, dtype=logits.dtype)
        target_count = N / float(C)
        running_probs = F.softmax(logits.float(), dim=1)

        logit_range = (
            logits.detach().float().amax(dim=(1, 2), keepdim=True)
            - logits.detach().float().amin(dim=(1, 2), keepdim=True)
        ).clamp_min(1e-4)
        max_bias = (logit_range * 5.0).to(logits.dtype)

        for i in range(max_iter):
            probs = F.softmax((logits - bias).float(), dim=1)
            if momentum > 0:
                running_probs = (1.0 - momentum) * probs + momentum * running_probs
                assign_source = running_probs
            else:
                assign_source = probs

            hard_indices = assign_source.argmax(dim=1)  # (B, N)
            hard_counts = F.one_hot(hard_indices, num_classes=C).float().sum(dim=1)  # (B, C)
            diff = hard_counts - target_count

            cur_lr = float(lr) * (0.95 ** i)
            bias = bias + torch.sign(diff).to(dtype=logits.dtype).unsqueeze(-1) * cur_lr
            bias = torch.max(torch.min(bias, max_bias), -max_bias)

        return (logits - bias).argmax(dim=1)

    @staticmethod
    def morphological_clean_mask_3d(
        mask: torch.Tensor,
        grid_t: int,
        grid_h: int,
        grid_w: int,
        opening_kernel: Tuple[int, int, int] = (1, 3, 3),
        closing_kernel: Tuple[int, int, int] = (1, 3, 3),
    ) -> torch.Tensor:
        """
        Clean binary mask `(B, N)` on 3D token grids `(T, H, W)` with opening+closing.
        """
        if mask.ndim != 2:
            raise ValueError(f"Expected mask shape (B, N), got {tuple(mask.shape)}")

        expected = int(grid_t) * int(grid_h) * int(grid_w)
        if mask.shape[1] != expected:
            return mask

        def _dilate(x: torch.Tensor, kernel: Tuple[int, int, int]) -> torch.Tensor:
            pad = tuple(k // 2 for k in kernel)
            out = F.max_pool3d(x, kernel_size=kernel, stride=1, padding=pad)
            if out.shape[-3:] != x.shape[-3:]:
                out = F.interpolate(out, size=x.shape[-3:], mode="nearest")
            return out

        def _erode(x: torch.Tensor, kernel: Tuple[int, int, int]) -> torch.Tensor:
            return 1.0 - _dilate(1.0 - x, kernel)

        x = mask.view(mask.shape[0], 1, grid_t, grid_h, grid_w).float()

        if max(opening_kernel) > 1:
            x = _dilate(_erode(x, opening_kernel), opening_kernel)
        if max(closing_kernel) > 1:
            x = _erode(_dilate(x, closing_kernel), closing_kernel)

        return x.view(mask.shape[0], -1)

    def sim_maps_to_masks_1d(
        self,
        sim_maps: Dict[str, torch.Tensor],
        seq_len: int,
        exclude_background: bool = True,
        eos_bg_scale: float = 0.95,
        use_balanced_assignment: bool = True,
        balanced_max_iter: int = 12,
        balanced_lr: float = 0.05,
        video_token_grid: Optional[Tuple[int, int, int]] = None,
        clean_video_foreground: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert `concept_sim_maps` to hard 1D token masks.

        Returns `{lora_name: (B, seq_len, 1)}`.
        """
        if not sim_maps:
            return {}

        local_maps = dict(sim_maps)
        bg_map = local_maps.pop("__bg__", None)
        if bg_map is None:
            bg_map = local_maps.pop("__eos__", None)

        concept_items = list(local_maps.items())
        if not concept_items:
            return {}

        concept_names = [name for name, _ in concept_items]
        concept_tensor = torch.stack(
            [self._squeeze_sim_map(cur, seq_len) for _, cur in concept_items],
            dim=1,
        )  # (B, C, N)

        B, C, N = concept_tensor.shape
        if N != seq_len:
            raise ValueError(f"Unexpected sequence mismatch, got N={N}, expected seq_len={seq_len}.")

        if exclude_background:
            if bg_map is not None:
                bg_channel = self._squeeze_sim_map(bg_map, seq_len) * float(eos_bg_scale)
            else:
                bg_channel = concept_tensor.mean(dim=1)
            with_bg = torch.cat([concept_tensor, bg_channel.unsqueeze(1)], dim=1)  # (B, C+1, N)
            fg_mask = (with_bg.argmax(dim=1) != C)

            if (
                clean_video_foreground
                and video_token_grid is not None
                and int(video_token_grid[0]) * int(video_token_grid[1]) * int(video_token_grid[2]) == N
            ):
                fg_mask = self.morphological_clean_mask_3d(
                    fg_mask.float(),
                    grid_t=int(video_token_grid[0]),
                    grid_h=int(video_token_grid[1]),
                    grid_w=int(video_token_grid[2]),
                ).bool()
        else:
            fg_mask = torch.ones((B, N), device=concept_tensor.device, dtype=torch.bool)

        if use_balanced_assignment and C > 1:
            max_indices = self.stabilized_balanced_argmax(
                concept_tensor,
                max_iter=balanced_max_iter,
                lr=balanced_lr,
            )
        else:
            max_indices = concept_tensor.argmax(dim=1)  # (B, N)

        masks: Dict[str, torch.Tensor] = {}
        for idx, name in enumerate(concept_names):
            hard = ((max_indices == idx) & fg_mask).float().unsqueeze(-1)  # (B, N, 1)
            masks[name] = hard
        return masks

    @staticmethod
    def build_text_attention_bias(
        token_masks: Dict[str, torch.Tensor],
        token_pos_maps: Dict[str, List[List[int]]],
        text_len: int,
        bias_scale: float = 4.0,
        positive_bias_scale: float = 2.0,
        use_positive_bias: bool = True,
    ) -> Optional[torch.Tensor]:
        """
        Build additive bias for query(tokens) -> text attention.

        Returns shape `(B, query_len, text_len)`.
        """
        if not token_masks or not token_pos_maps:
            return None

        lora_names = list(token_masks.keys())
        lora_to_idx = {name: idx for idx, name in enumerate(lora_names)}
        first_mask = next(iter(token_masks.values()))
        B, query_len = first_mask.shape[0], first_mask.shape[1]
        device = first_mask.device
        dtype = first_mask.dtype

        bias = torch.zeros((B, query_len, text_len), device=device, dtype=dtype)
        neg_scale = abs(float(bias_scale))
        pos_scale = float(positive_bias_scale)

        for b in range(B):
            text_owner = torch.full((text_len,), -1, device=device, dtype=torch.long)
            for lora_name, positions_list in token_pos_maps.items():
                if lora_name not in lora_to_idx:
                    continue
                idx = lora_to_idx[lora_name]
                if not positions_list:
                    continue
                pos = positions_list[min(b, len(positions_list) - 1)]
                for p in pos:
                    if 0 <= int(p) < text_len:
                        text_owner[int(p)] = idx

            for lora_name, query_mask in token_masks.items():
                idx = lora_to_idx[lora_name]
                mb = b % query_mask.shape[0]
                q_mask = query_mask[mb, :, 0] if query_mask.dim() == 3 else query_mask[mb]

                other_text = ((text_owner != idx) & (text_owner != -1)).float()
                bias[b] += q_mask.unsqueeze(-1) * other_text.unsqueeze(0) * (-neg_scale)

                if use_positive_bias:
                    same_text = (text_owner == idx).float()
                    bias[b] += q_mask.unsqueeze(-1) * same_text.unsqueeze(0) * pos_scale

        return bias

    @staticmethod
    def build_cross_modal_attention_bias(
        video_masks: Dict[str, torch.Tensor],
        audio_masks: Dict[str, torch.Tensor],
        bias_scale: float = 2.0,
        positive_bias_scale: float = 1.0,
        use_positive_bias: bool = True,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Build cross-modal additive biases:
        - video_audio: video query -> audio key, shape `(B, video_len, audio_len)`
        - audio_video: audio query -> video key, shape `(B, audio_len, video_len)`
        """
        if not video_masks or not audio_masks:
            return None, None

        shared_loras = [name for name in video_masks.keys() if name in audio_masks]
        if not shared_loras:
            return None, None

        v0 = video_masks[shared_loras[0]]
        a0 = audio_masks[shared_loras[0]]
        B = max(v0.shape[0], a0.shape[0])
        video_len = v0.shape[1]
        audio_len = a0.shape[1]
        device = v0.device
        dtype = v0.dtype

        video_audio = torch.zeros((B, video_len, audio_len), device=device, dtype=dtype)
        audio_video = torch.zeros((B, audio_len, video_len), device=device, dtype=dtype)

        neg_scale = abs(float(bias_scale))
        pos_scale = float(positive_bias_scale)

        for name in shared_loras:
            v_mask = video_masks[name]
            a_mask = audio_masks[name]
            for b in range(B):
                vb = b % v_mask.shape[0]
                ab = b % a_mask.shape[0]

                v = v_mask[vb, :, 0] if v_mask.dim() == 3 else v_mask[vb]
                a = a_mask[ab, :, 0] if a_mask.dim() == 3 else a_mask[ab]

                video_audio[b] += v.unsqueeze(-1) * (1.0 - a).unsqueeze(0) * (-neg_scale)
                audio_video[b] += a.unsqueeze(-1) * (1.0 - v).unsqueeze(0) * (-neg_scale)

                if use_positive_bias:
                    video_audio[b] += v.unsqueeze(-1) * a.unsqueeze(0) * pos_scale
                    audio_video[b] += a.unsqueeze(-1) * v.unsqueeze(0) * pos_scale

        return video_audio, audio_video

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_file_name(name: str) -> str:
        return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in str(name))

    @staticmethod
    def _token_pos_map_stats(token_pos_maps: Optional[Dict[str, List[List[int]]]]) -> Dict[str, Dict[str, int]]:
        if not token_pos_maps:
            return {}
        stats: Dict[str, Dict[str, int]] = {}
        for concept_name, positions_list in token_pos_maps.items():
            per_prompt_lens = [len(x) for x in (positions_list or [])]
            stats[concept_name] = {
                "num_prompts": int(len(per_prompt_lens)),
                "non_empty_prompts": int(sum(1 for x in per_prompt_lens if x > 0)),
                "total_positions": int(sum(per_prompt_lens)),
            }
        return stats

    @staticmethod
    def _normalize_token_tensor_for_debug(
        x: torch.Tensor, seq_len: int, allow_matrix_flatten: bool = False
    ) -> torch.Tensor:
        cur = x
        if cur.dim() == 4:
            cur = cur.reshape(cur.shape[0], -1)
        elif cur.dim() == 3:
            if cur.shape[-1] == 1:
                cur = cur[:, :, 0]
            elif allow_matrix_flatten:
                cur = cur.reshape(cur.shape[0], -1)
            else:
                raise ValueError(
                    "Expected token tensor with shape (B, N, 1) for debug projection, "
                    f"got {tuple(cur.shape)}."
                )
        elif cur.dim() != 2:
            raise ValueError(f"Unsupported token tensor for debug: {tuple(cur.shape)}")

        if cur.shape[1] < seq_len:
            cur = torch.nn.functional.pad(cur, (0, seq_len - cur.shape[1]), value=0.0)
        elif cur.shape[1] > seq_len:
            cur = cur[:, :seq_len]
        return cur

    def _infer_video_token_grid(
        self,
        num_frames: int,
        height: int,
        width: int,
        video_seq_len: int,
    ) -> Optional[Tuple[int, int, int]]:
        try:
            latent_num_frames = (int(num_frames) - 1) // int(self.vae_temporal_compression_ratio) + 1
            latent_height = int(height) // int(self.vae_spatial_compression_ratio)
            latent_width = int(width) // int(self.vae_spatial_compression_ratio)
            patch_t = int(getattr(self, "transformer_temporal_patch_size", 1))
            patch = int(getattr(self, "transformer_spatial_patch_size", 1))
            if patch_t <= 0 or patch <= 0:
                return None
            if latent_num_frames % patch_t != 0 or latent_height % patch != 0 or latent_width % patch != 0:
                return None
            vt = latent_num_frames // patch_t
            vh = latent_height // patch
            vw = latent_width // patch
            if vt * vh * vw != int(video_seq_len):
                return None
            return vt, vh, vw
        except Exception:
            return None

    @staticmethod
    def _choose_factor_pair_with_ratio(
        product: int,
        target_ratio: float,
    ) -> Optional[Tuple[int, int]]:
        if product <= 0:
            return None

        best_pair: Optional[Tuple[int, int]] = None
        best_err = float("inf")
        target = max(float(target_ratio), 1e-8)

        upper = int(product ** 0.5)
        for a in range(1, upper + 1):
            if product % a != 0:
                continue
            b = product // a
            for h, w in ((a, b), (b, a)):
                ratio = float(h) / max(float(w), 1.0)
                err = abs(ratio - target)
                if err < best_err:
                    best_err = err
                    best_pair = (int(h), int(w))

        return best_pair

    def _resolve_packed_video_geometry(
        self,
        latents: Optional[torch.Tensor],
        num_frames: int,
        height: int,
        width: int,
    ) -> Tuple[int, int, int]:
        """
        Resolve geometry for packed video latents `(B, S, D)`.

        Diffusers cannot infer latent `(F, H, W)` from packed latents, so it relies
        on caller-provided `height/width/num_frames`. This helper reconstructs
        spatial geometry from sequence length to avoid RoPE/token-length mismatch.
        """
        eff_num_frames = int(num_frames)
        eff_height = int(height)
        eff_width = int(width)

        if latents is None or latents.ndim != 3:
            return eff_num_frames, eff_height, eff_width

        seq_len = int(latents.shape[1])
        if seq_len <= 0:
            return eff_num_frames, eff_height, eff_width

        try:
            vae_t = int(self.vae_temporal_compression_ratio)
            vae_s = int(self.vae_spatial_compression_ratio)
            patch_t = int(getattr(self, "transformer_temporal_patch_size", 1))
            patch = int(getattr(self, "transformer_spatial_patch_size", 1))
        except Exception:
            return eff_num_frames, eff_height, eff_width

        if vae_t <= 0 or vae_s <= 0 or patch_t <= 0 or patch <= 0:
            return eff_num_frames, eff_height, eff_width

        latent_num_frames = (eff_num_frames - 1) // vae_t + 1
        if latent_num_frames <= 0 or latent_num_frames % patch_t != 0:
            return eff_num_frames, eff_height, eff_width

        temporal_tokens = latent_num_frames // patch_t
        if temporal_tokens <= 0 or seq_len % temporal_tokens != 0:
            return eff_num_frames, eff_height, eff_width

        spatial_tokens = seq_len // temporal_tokens
        if spatial_tokens <= 0:
            return eff_num_frames, eff_height, eff_width

        target_latent_h = max(1, eff_height // vae_s)
        target_latent_w = max(1, eff_width // vae_s)
        target_ratio = float(target_latent_h) / max(float(target_latent_w), 1.0)
        post_patch_hw = self._choose_factor_pair_with_ratio(spatial_tokens, target_ratio)
        if post_patch_hw is None:
            return eff_num_frames, eff_height, eff_width

        post_h, post_w = post_patch_hw
        latent_h = int(post_h) * patch
        latent_w = int(post_w) * patch
        if latent_h <= 0 or latent_w <= 0:
            return eff_num_frames, eff_height, eff_width

        eff_height = latent_h * vae_s
        eff_width = latent_w * vae_s
        return eff_num_frames, eff_height, eff_width

    def _save_freefuse_step_debug(
        self,
        debug_save_path: str,
        step_index: int,
        timestep: Union[int, float, torch.Tensor, str],
        video_maps: Optional[Dict[str, torch.Tensor]],
        audio_maps: Optional[Dict[str, torch.Tensor]],
        video_masks: Optional[Dict[str, torch.Tensor]],
        audio_masks: Optional[Dict[str, torch.Tensor]],
        video_seq_len: int,
        audio_seq_len: int,
        video_token_grid: Optional[Tuple[int, int, int]],
    ) -> None:
        if not debug_save_path:
            return

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"[FreeFuse][debug] Skip plotting: {exc}")
            return

        if isinstance(timestep, torch.Tensor):
            timestep_value = float(timestep.detach().float().flatten()[0].item())
            timestep_tag = f"{timestep_value:.4f}".replace(".", "p")
        elif isinstance(timestep, str):
            timestep_value = None
            timestep_tag = self._safe_file_name(timestep)
        else:
            timestep_value = float(timestep)
            timestep_tag = f"{timestep_value:.4f}".replace(".", "p")

        step_dir = os.path.join(
            debug_save_path, f"phase1_step_{int(step_index) + 1:02d}_t_{timestep_tag}"
        )
        os.makedirs(step_dir, exist_ok=True)

        def save_curve(values: torch.Tensor, title: str, out_name: str, y_min: Optional[float] = None, y_max: Optional[float] = None):
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(values.numpy(), linewidth=1.8)
            ax.set_title(title)
            ax.set_xlabel("token index")
            ax.set_ylabel("value")
            if y_min is not None and y_max is not None:
                ax.set_ylim(y_min, y_max)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(step_dir, out_name), dpi=150, bbox_inches="tight")
            plt.close(fig)

        def save_video_views(values_1d: torch.Tensor, title_prefix: str, out_prefix: str, is_mask: bool):
            if video_token_grid is None:
                return
            t_len, h_len, w_len = video_token_grid
            expected = t_len * h_len * w_len
            if values_1d.numel() < expected:
                return
            values_3d = values_1d[:expected].reshape(t_len, h_len, w_len)
            temporal_curve = values_3d.mean(dim=(1, 2))
            save_curve(
                temporal_curve,
                f"{title_prefix} temporal mean",
                f"{out_prefix}_temporal_curve.png",
                y_min=0.0 if is_mask else None,
                y_max=1.0 if is_mask else None,
            )
            frames_dir = os.path.join(step_dir, f"{out_prefix}_all_frames")
            os.makedirs(frames_dir, exist_ok=True)
            vmin = 0.0 if is_mask else float(values_3d.min().item())
            vmax = 1.0 if is_mask else float(values_3d.max().item())
            if not is_mask and abs(vmax - vmin) < 1e-8:
                vmax = vmin + 1e-8

            for frame_idx in range(t_len):
                fig, ax = plt.subplots(figsize=(4, 4))
                if is_mask:
                    ax.imshow(values_3d[frame_idx].numpy(), cmap="gray", vmin=vmin, vmax=vmax)
                else:
                    ax.imshow(values_3d[frame_idx].numpy(), cmap="viridis", vmin=vmin, vmax=vmax)
                ax.set_title(f"{title_prefix} frame {frame_idx}")
                ax.axis("off")
                fig.tight_layout()
                fig.savefig(
                    os.path.join(frames_dir, f"frame_{frame_idx:04d}.png"),
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close(fig)

        summary_lines: List[str] = [
            f"phase1_step={int(step_index) + 1}",
            f"timestep={timestep_value if timestep_value is not None else timestep_tag}",
            f"video_seq_len={int(video_seq_len)}",
            f"audio_seq_len={int(audio_seq_len)}",
            f"video_token_grid={video_token_grid}",
        ]

        def process_dict(data: Optional[Dict[str, torch.Tensor]], modality: str, kind: str, seq_len: int):
            if not data:
                return
            for concept_name, tensor in data.items():
                safe_name = self._safe_file_name(concept_name)
                try:
                    values = self._normalize_token_tensor_for_debug(
                        tensor.detach().float().cpu(),
                        seq_len=seq_len,
                        allow_matrix_flatten=False,
                    )
                except Exception as exc:
                    summary_lines.append(
                        f"{modality}_{kind}_{concept_name}_debug_error={self._safe_file_name(str(exc))}"
                    )
                    summary_lines.append(
                        f"{modality}_{kind}_{concept_name}_raw_shape={tuple(int(x) for x in tensor.shape)}"
                    )
                    continue
                if values.shape[0] == 0:
                    continue
                first = values[0]
                mean_value = float(first.mean().item())
                min_value = float(first.min().item())
                max_value = float(first.max().item())
                sum_value = float(first.sum().item())
                summary_lines.append(f"{modality}_{kind}_{concept_name}_mean={mean_value:.6f}")
                summary_lines.append(f"{modality}_{kind}_{concept_name}_sum={sum_value:.6f}")
                summary_lines.append(f"{modality}_{kind}_{concept_name}_min={min_value:.6f}")
                summary_lines.append(f"{modality}_{kind}_{concept_name}_max={max_value:.6f}")
                summary_lines.append(
                    f"{modality}_{kind}_{concept_name}_raw_shape={tuple(int(x) for x in tensor.shape)}"
                )
                summary_lines.append(
                    f"{modality}_{kind}_{concept_name}_debug_shape={tuple(int(x) for x in values.shape)}"
                )

                curve_name = f"{modality}_{kind}_{safe_name}_token_curve.png"
                save_curve(
                    first,
                    f"{modality} {kind} {concept_name} token profile",
                    curve_name,
                    y_min=0.0 if kind == "mask" else None,
                    y_max=1.0 if kind == "mask" else None,
                )

                if modality == "video":
                    save_video_views(
                        first,
                        title_prefix=f"{modality} {kind} {concept_name}",
                        out_prefix=f"{modality}_{kind}_{safe_name}",
                        is_mask=(kind == "mask"),
                    )
                elif modality == "audio":
                    save_curve(
                        first,
                        f"{modality} {kind} {concept_name} (time axis)",
                        f"{modality}_{kind}_{safe_name}_time_curve.png",
                        y_min=0.0 if kind == "mask" else None,
                        y_max=1.0 if kind == "mask" else None,
                    )

        process_dict(video_maps, "video", "sim_map", video_seq_len)
        process_dict(audio_maps, "audio", "sim_map", audio_seq_len)
        process_dict(video_masks, "video", "mask", video_seq_len)
        process_dict(audio_masks, "audio", "mask", audio_seq_len)

        with open(os.path.join(step_dir, "summary.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines) + "\n")

    def _save_freefuse_attention_bias_debug(
        self,
        debug_save_path: str,
        attention_biases: Dict[str, torch.Tensor],
        max_query: int = 256,
        max_key: int = 256,
    ) -> None:
        if not debug_save_path or not attention_biases:
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"[FreeFuse][debug] Skip attention-bias plotting: {exc}")
            return

        out_dir = os.path.join(debug_save_path, "phase1_5_attention_bias")
        os.makedirs(out_dir, exist_ok=True)

        for bias_name, bias_tensor in attention_biases.items():
            if bias_tensor is None or bias_tensor.ndim != 3 or bias_tensor.shape[0] == 0:
                continue
            bias = bias_tensor.detach().float().cpu()[0]
            view = bias[: min(max_query, bias.shape[0]), : min(max_key, bias.shape[1])]
            vmax = float(view.abs().max().item())
            if vmax < 1e-8:
                vmax = 1.0

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(view.numpy(), cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_title(f"{bias_name} bias (query x key)")
            ax.set_xlabel("key index")
            ax.set_ylabel("query index")
            fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
            fig.tight_layout()
            fig.savefig(
                os.path.join(out_dir, f"{self._safe_file_name(bias_name)}_bias.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)

    # ------------------------------------------------------------------
    # Two-phase FreeFuse call
    # ------------------------------------------------------------------
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        frame_rate: float = 24.0,
        num_inference_steps: int = 40,
        sigmas: Optional[List[float]] = None,
        timesteps: List[int] = None,
        guidance_scale: float = 4.0,
        guidance_rescale: float = 0.0,
        noise_scale: float = 0.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        audio_latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        decode_timestep: Union[float, List[float]] = 0.0,
        decode_noise_scale: Optional[Union[float, List[float]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 1024,
        # FreeFuse controls
        freefuse_enabled: bool = False,
        freefuse_token_pos_maps: Optional[Dict[str, List[List[int]]]] = None,
        freefuse_concept_map: Optional[Dict[str, str]] = None,
        freefuse_background_token_positions: Optional[List[int]] = None,
        freefuse_background_text: Optional[str] = None,
        freefuse_eos_token_index: Optional[int] = None,
        freefuse_top_k_ratio: float = 0.1,
        freefuse_phase1_step: int = 3,
        freefuse_phase1_guidance_scale: float = 1.0,
        freefuse_video_collect_block: Optional[str] = None,
        freefuse_audio_collect_block: Optional[str] = None,
        freefuse_use_attention_bias: bool = True,
        freefuse_attention_bias_scale: float = 4.0,
        freefuse_attention_bias_positive_scale: float = 2.0,
        freefuse_attention_bias_positive: bool = True,
        freefuse_use_av_cross_attention_bias: bool = False,
        freefuse_av_attention_bias_scale: float = 2.0,
        freefuse_attention_bias_blocks: Optional[List[str]] = None,
        freefuse_debug_save_path: Optional[str] = None,
        freefuse_debug_collect_per_step: bool = True,
    ):
        num_frames, height, width = self._resolve_packed_video_geometry(
            latents=latents,
            num_frames=num_frames,
            height=height,
            width=width,
        )

        self.setup_freefuse_attention_processors()
        debug_enabled = bool(freefuse_debug_save_path)
        if debug_enabled:
            os.makedirs(freefuse_debug_save_path, exist_ok=True)
        self._freefuse_last_phase1_debug_steps = None

        if not freefuse_enabled:
            return super().__call__(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=num_inference_steps,
                sigmas=sigmas,
                timesteps=timesteps,
                guidance_scale=guidance_scale,
                guidance_rescale=guidance_rescale,
                noise_scale=noise_scale,
                num_videos_per_prompt=num_videos_per_prompt,
                generator=generator,
                latents=latents,
                audio_latents=audio_latents,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                decode_timestep=decode_timestep,
                decode_noise_scale=decode_noise_scale,
                output_type=output_type,
                return_dict=return_dict,
                attention_kwargs=attention_kwargs,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                max_sequence_length=max_sequence_length,
            )

        if freefuse_token_pos_maps is None:
            if freefuse_concept_map is not None and prompt is not None:
                freefuse_token_pos_maps = self.find_concept_positions(
                    prompt=prompt,
                    concept_map=freefuse_concept_map,
                    max_sequence_length=max_sequence_length,
                )
            else:
                raise ValueError(
                    "FreeFuse is enabled but no token map is provided. "
                    "Pass `freefuse_token_pos_maps` or `freefuse_concept_map`."
                )

        if freefuse_background_token_positions is None and freefuse_background_text is not None and prompt is not None:
            bg_map = self.find_concept_positions(
                prompt=prompt,
                concept_map={"__bg__": freefuse_background_text},
                max_sequence_length=max_sequence_length,
            )
            freefuse_background_token_positions = bg_map["__bg__"][0] if "__bg__" in bg_map else None

        if freefuse_eos_token_index is None and isinstance(prompt, str):
            freefuse_eos_token_index = self.find_eos_token_index(prompt, max_sequence_length=max_sequence_length)

        token_pos_stats = self._token_pos_map_stats(freefuse_token_pos_maps)
        if not token_pos_stats:
            raise ValueError("FreeFuse token position map is empty.")
        total_positions = sum(x["total_positions"] for x in token_pos_stats.values())
        if total_positions <= 0:
            raise ValueError(
                "All FreeFuse concept token positions are empty after matching/filtering. "
                f"stats={token_pos_stats}"
            )

        # ------------------------------------------------------------------
        # Prepare shared initial latents once, then clone for both phases
        # ------------------------------------------------------------------
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        elif prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        else:
            raise ValueError("`prompt` or `prompt_embeds` must be provided.")

        device = self._execution_device
        batch = batch_size * num_videos_per_prompt

        num_channels_latents = self.transformer.config.in_channels
        init_video_latents = self.prepare_latents(
            batch_size=batch,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            noise_scale=noise_scale,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
        )

        duration_s = num_frames / frame_rate
        audio_latents_per_second = self.audio_sampling_rate / self.audio_hop_length / float(self.audio_vae_temporal_compression_ratio)
        audio_num_frames = round(duration_s * audio_latents_per_second)
        if audio_latents is not None and audio_latents.ndim == 4:
            _, _, audio_num_frames, _ = audio_latents.shape

        num_mel_bins = self.audio_vae.config.mel_bins if getattr(self, "audio_vae", None) is not None else 64
        num_channels_audio = self.audio_vae.config.latent_channels if getattr(self, "audio_vae", None) is not None else 8
        init_audio_latents = self.prepare_audio_latents(
            batch_size=batch,
            num_channels_latents=num_channels_audio,
            audio_latent_length=audio_num_frames,
            num_mel_bins=num_mel_bins,
            noise_scale=noise_scale,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=audio_latents,
        )

        video_seq_len = init_video_latents.shape[1]
        audio_seq_len = init_audio_latents.shape[1]
        video_token_grid = self._infer_video_token_grid(
            num_frames=num_frames,
            height=height,
            width=width,
            video_seq_len=video_seq_len,
        )

        # ------------------------------------------------------------------
        # Phase 1: collect sim-maps
        # ------------------------------------------------------------------
        self.transformer.clear_freefuse_state()
        self.transformer.set_freefuse_token_pos_maps(freefuse_token_pos_maps)
        self.transformer.set_freefuse_top_k_ratio(freefuse_top_k_ratio)
        self.transformer.set_freefuse_background_info(
            eos_token_index=freefuse_eos_token_index,
            background_token_positions=freefuse_background_token_positions,
        )
        self.transformer.enable_concept_sim_map_extraction(
            video_block_name=freefuse_video_collect_block,
            audio_block_name=freefuse_audio_collect_block,
        )

        enabled_extract_processors: List[str] = []
        if hasattr(self.transformer, "_iter_named_freefuse_processors"):
            for module_name, processor in self.transformer._iter_named_freefuse_processors():
                if getattr(processor, "cal_concept_sim_map", False):
                    enabled_extract_processors.append(module_name)
        if not enabled_extract_processors:
            raise RuntimeError("No FreeFuse attention processors were enabled for concept sim-map extraction.")

        if debug_enabled:
            with open(os.path.join(freefuse_debug_save_path, "phase1_metadata.txt"), "w", encoding="utf-8") as f:
                f.write(f"phase1_step={int(freefuse_phase1_step)}\n")
                f.write(f"phase1_effective_step={min(max(1, int(freefuse_phase1_step)), int(num_inference_steps))}\n")
                f.write(f"video_collect_block={freefuse_video_collect_block}\n")
                f.write(f"audio_collect_block={freefuse_audio_collect_block}\n")
                f.write(f"enabled_extract_processors={enabled_extract_processors}\n")
                f.write(f"token_pos_stats={token_pos_stats}\n")
                f.write(f"token_pos_maps={freefuse_token_pos_maps}\n")
                f.write(f"eos_token_index={freefuse_eos_token_index}\n")
                f.write(f"background_token_positions={freefuse_background_token_positions}\n")

        if hasattr(self, "disable_lora"):
            self.disable_lora()

        captured_maps: Dict[str, Dict[str, torch.Tensor]] = {}
        captured_timestep: Optional[float] = None
        phase1_debug_steps: List[Dict[str, Any]] = []
        target_step = min(
            max(0, int(freefuse_phase1_step) - 1),
            max(0, int(num_inference_steps) - 1),
        )
        extraction_armed = target_step == 0
        if not extraction_armed:
            self.transformer.disable_concept_sim_map_extraction()

        def _phase1_callback(pipeline, step_index, timestep, callback_kwargs):
            nonlocal captured_timestep, extraction_armed

            if not extraction_armed and step_index == target_step - 1:
                self.transformer.enable_concept_sim_map_extraction(
                    video_block_name=freefuse_video_collect_block,
                    audio_block_name=freefuse_audio_collect_block,
                )
                extraction_armed = True
                return callback_kwargs

            if step_index != target_step:
                return callback_kwargs

            maps = self.transformer.get_concept_sim_maps(clear_after_read=True)
            timestep_value = (
                float(timestep.detach().float().flatten()[0].item())
                if isinstance(timestep, torch.Tensor)
                else float(timestep)
            )

            if maps:
                captured_maps.clear()
                captured_maps.update(maps)
                captured_timestep = timestep_value

                if debug_enabled and freefuse_debug_collect_per_step:
                    cur_video_maps = maps.get("video", None)
                    cur_audio_maps = maps.get("audio", None)
                    cur_video_masks = self.sim_maps_to_masks_1d(
                        cur_video_maps or {},
                        seq_len=video_seq_len,
                        exclude_background=True,
                        video_token_grid=video_token_grid,
                    )
                    cur_audio_masks = self.sim_maps_to_masks_1d(
                        cur_audio_maps or {}, seq_len=audio_seq_len, exclude_background=True
                    )
                    self._save_freefuse_step_debug(
                        debug_save_path=freefuse_debug_save_path,
                        step_index=step_index,
                        timestep=timestep_value,
                        video_maps=cur_video_maps,
                        audio_maps=cur_audio_maps,
                        video_masks=cur_video_masks if cur_video_masks else None,
                        audio_masks=cur_audio_masks if cur_audio_masks else None,
                        video_seq_len=video_seq_len,
                        audio_seq_len=audio_seq_len,
                        video_token_grid=video_token_grid,
                    )
                    phase1_debug_steps.append(
                        {"step_index": int(step_index), "timestep": float(timestep_value)}
                    )

            pipeline._interrupt = True
            return callback_kwargs

        _ = super().__call__(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            sigmas=sigmas,
            timesteps=timesteps,
            guidance_scale=float(freefuse_phase1_guidance_scale),
            guidance_rescale=guidance_rescale,
            noise_scale=noise_scale,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=generator,
            latents=init_video_latents.clone(),
            audio_latents=init_audio_latents.clone(),
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            decode_timestep=decode_timestep,
            decode_noise_scale=decode_noise_scale,
            output_type="latent",
            return_dict=False,
            attention_kwargs=attention_kwargs,
            callback_on_step_end=_phase1_callback,
            callback_on_step_end_tensor_inputs=["latents"],
            max_sequence_length=max_sequence_length,
        )
        self._freefuse_last_phase1_debug_steps = phase1_debug_steps if phase1_debug_steps else None
        if not captured_maps:
            raise RuntimeError(
                "Phase 1 finished but no concept sim-maps were captured. "
                "Check `phase1_metadata.txt` and concept token positions."
            )

        # ------------------------------------------------------------------
        # Phase 1.5: sim-maps -> masks/bias
        # ------------------------------------------------------------------
        text_seq_len = prompt_embeds.shape[1] if prompt_embeds is not None else max_sequence_length

        video_maps = captured_maps.get("video", None)
        audio_maps = captured_maps.get("audio", None)

        video_masks = self.sim_maps_to_masks_1d(
            video_maps or {},
            seq_len=video_seq_len,
            exclude_background=True,
            video_token_grid=video_token_grid,
        )
        audio_masks = self.sim_maps_to_masks_1d(
            audio_maps or {}, seq_len=audio_seq_len, exclude_background=True
        )

        attention_biases: Dict[str, torch.Tensor] = {}
        if freefuse_use_attention_bias:
            if video_masks:
                video_text_bias = self.build_text_attention_bias(
                    token_masks=video_masks,
                    token_pos_maps=freefuse_token_pos_maps,
                    text_len=text_seq_len,
                    bias_scale=freefuse_attention_bias_scale,
                    positive_bias_scale=freefuse_attention_bias_positive_scale,
                    use_positive_bias=freefuse_attention_bias_positive,
                )
                if video_text_bias is not None:
                    attention_biases["video_text"] = video_text_bias

            if audio_masks:
                audio_text_bias = self.build_text_attention_bias(
                    token_masks=audio_masks,
                    token_pos_maps=freefuse_token_pos_maps,
                    text_len=text_seq_len,
                    bias_scale=freefuse_attention_bias_scale,
                    positive_bias_scale=freefuse_attention_bias_positive_scale,
                    use_positive_bias=freefuse_attention_bias_positive,
                )
                if audio_text_bias is not None:
                    attention_biases["audio_text"] = audio_text_bias

            if freefuse_use_av_cross_attention_bias and video_masks and audio_masks:
                video_audio_bias, audio_video_bias = self.build_cross_modal_attention_bias(
                    video_masks=video_masks,
                    audio_masks=audio_masks,
                    bias_scale=freefuse_av_attention_bias_scale,
                    positive_bias_scale=freefuse_attention_bias_positive_scale,
                    use_positive_bias=freefuse_attention_bias_positive,
                )
                if video_audio_bias is not None:
                    attention_biases["video_audio"] = video_audio_bias
                if audio_video_bias is not None:
                    attention_biases["audio_video"] = audio_video_bias

        self.transformer.set_freefuse_video_masks(video_masks if video_masks else None)
        self.transformer.set_freefuse_audio_masks(audio_masks if audio_masks else None)
        self.transformer.set_freefuse_token_pos_maps(freefuse_token_pos_maps)
        self.transformer.set_freefuse_top_k_ratio(freefuse_top_k_ratio)
        self.transformer.set_freefuse_background_info(
            eos_token_index=freefuse_eos_token_index,
            background_token_positions=freefuse_background_token_positions,
        )
        self.transformer.set_freefuse_attention_biases(
            attention_biases=attention_biases,
            attention_bias_blocks=freefuse_attention_bias_blocks,
        )
        self.transformer.disable_concept_sim_map_extraction()

        if debug_enabled:
            self._save_freefuse_step_debug(
                debug_save_path=freefuse_debug_save_path,
                step_index=target_step,
                timestep=captured_timestep if captured_timestep is not None else "phase1_final",
                video_maps=video_maps,
                audio_maps=audio_maps,
                video_masks=video_masks if video_masks else None,
                audio_masks=audio_masks if audio_masks else None,
                video_seq_len=video_seq_len,
                audio_seq_len=audio_seq_len,
                video_token_grid=video_token_grid,
            )
            self._save_freefuse_attention_bias_debug(
                debug_save_path=freefuse_debug_save_path,
                attention_biases=attention_biases,
            )

        # Keep debug state for caller access
        self._freefuse_last_concept_sim_maps = captured_maps if captured_maps else None
        self._freefuse_last_video_masks = video_masks if video_masks else None
        self._freefuse_last_audio_masks = audio_masks if audio_masks else None
        self._freefuse_last_attention_biases = attention_biases if attention_biases else None

        # ------------------------------------------------------------------
        # Phase 2: regenerate with FreeFuse routing
        # ------------------------------------------------------------------
        if hasattr(self, "enable_lora"):
            self.enable_lora()

        return super().__call__(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            sigmas=sigmas,
            timesteps=timesteps,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            noise_scale=noise_scale,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=generator,
            latents=init_video_latents.clone(),
            audio_latents=init_audio_latents.clone(),
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            decode_timestep=decode_timestep,
            decode_noise_scale=decode_noise_scale,
            output_type=output_type,
            return_dict=return_dict,
            attention_kwargs=attention_kwargs,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )
