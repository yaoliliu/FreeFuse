"""
FreeFuse wrapper for LTX-2.0 audiovisual transformer.

This module keeps the original LTX2 architecture but adds:
1) FreeFuse-aware attention processors
2) runtime APIs for mask/token/bias routing
3) concept sim-map extraction controls
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

from src.attn_processor.freefuse_ltx2_attn_processor import FreeFuseLTX2AttnProcessor
from src.tuner.freefuse_lora_layer import FreeFuseLinear


class FreeFuseLTX2VideoTransformer3DModel(LTX2VideoTransformer3DModel):
    """
    Adds FreeFuse controls on top of `LTX2VideoTransformer3DModel`.

    Note: this class is commonly activated via class-swap on a loaded model:
    `pipe.transformer.__class__ = FreeFuseLTX2VideoTransformer3DModel`.
    """

    def set_adapters(
        self,
        adapter_names: Union[List[str], str],
        weights: Optional[Union[float, Dict[str, Any], List[float], List[Dict[str, Any]], List[None]]] = None,
    ):
        """
        Compatibility wrapper for PEFT `set_adapters()` after class-swap.

        Diffusers' default implementation selects a scale-expansion function by
        `self.__class__.__name__`. After swapping from
        `LTX2VideoTransformer3DModel` to `FreeFuseLTX2VideoTransformer3DModel`,
        older mappings may miss the FreeFuse class name and raise `KeyError`.
        """
        try:
            return super().set_adapters(adapter_names, weights)
        except KeyError as exc:
            # Fallback only for the known class-name mapping issue.
            from diffusers.loaders.peft import (
                USE_PEFT_BACKEND,
                _SET_ADAPTER_SCALE_FN_MAPPING,
                set_weights_and_activate_adapters,
            )

            if not USE_PEFT_BACKEND:
                raise ValueError("PEFT backend is required for `set_adapters()`.") from exc

            class_name = self.__class__.__name__
            missing_key = exc.args[0] if len(exc.args) > 0 else None
            if missing_key != class_name:
                raise

            base_name = LTX2VideoTransformer3DModel.__name__
            if base_name not in _SET_ADAPTER_SCALE_FN_MAPPING:
                raise

            names = [adapter_names] if isinstance(adapter_names, str) else adapter_names
            scale_weights = weights
            if not isinstance(scale_weights, list):
                scale_weights = [scale_weights] * len(names)

            if len(names) != len(scale_weights):
                raise ValueError(
                    f"Length of adapter names {len(names)} is not equal to the length of their weights {len(scale_weights)}."
                )

            scale_weights = [w if w is not None else 1.0 for w in scale_weights]
            scale_expansion_fn = _SET_ADAPTER_SCALE_FN_MAPPING[base_name]
            scale_weights = scale_expansion_fn(self, scale_weights)
            set_weights_and_activate_adapters(self, names, scale_weights)

    def _ensure_freefuse_state(self) -> None:
        if not hasattr(self, "_freefuse_video_masks"):
            self._freefuse_video_masks: Optional[Dict[str, torch.Tensor]] = None
        if not hasattr(self, "_freefuse_audio_masks"):
            self._freefuse_audio_masks: Optional[Dict[str, torch.Tensor]] = None
        if not hasattr(self, "_freefuse_token_pos_maps"):
            self._freefuse_token_pos_maps: Optional[Dict[str, List[List[int]]]] = None
        if not hasattr(self, "_freefuse_attention_biases"):
            self._freefuse_attention_biases: Dict[str, torch.Tensor] = {}
        if not hasattr(self, "_freefuse_attention_bias_blocks"):
            self._freefuse_attention_bias_blocks: Optional[set[str]] = None
        if not hasattr(self, "_freefuse_top_k_ratio"):
            self._freefuse_top_k_ratio: float = 0.1
        if not hasattr(self, "_freefuse_eos_token_index"):
            self._freefuse_eos_token_index: Optional[int] = None
        if not hasattr(self, "_freefuse_background_token_positions"):
            self._freefuse_background_token_positions: Optional[List[int]] = None

    # ------------------------------------------------------------------
    # Processor setup
    # ------------------------------------------------------------------
    def setup_freefuse_attention_processors(self) -> None:
        """
        Install FreeFuse processors for all LTX2 attention branches.
        """
        self._ensure_freefuse_state()

        for block in self.transformer_blocks:
            block.attn1.set_processor(FreeFuseLTX2AttnProcessor(role="video_self"))
            block.audio_attn1.set_processor(FreeFuseLTX2AttnProcessor(role="audio_self"))
            block.attn2.set_processor(FreeFuseLTX2AttnProcessor(role="video_text"))
            block.audio_attn2.set_processor(FreeFuseLTX2AttnProcessor(role="audio_text"))
            block.audio_to_video_attn.set_processor(FreeFuseLTX2AttnProcessor(role="video_audio"))
            block.video_to_audio_attn.set_processor(FreeFuseLTX2AttnProcessor(role="audio_video"))

    def _iter_named_freefuse_processors(self):
        for module_name, module in self.named_modules():
            if hasattr(module, "processor") and isinstance(module.processor, FreeFuseLTX2AttnProcessor):
                yield module_name, module.processor

    # ------------------------------------------------------------------
    # Public FreeFuse APIs
    # ------------------------------------------------------------------
    def set_freefuse_token_pos_maps(
        self, token_pos_maps: Optional[Dict[str, List[List[int]]]]
    ) -> None:
        self._ensure_freefuse_state()
        self._freefuse_token_pos_maps = token_pos_maps

    def set_freefuse_video_masks(
        self, video_masks: Optional[Dict[str, torch.Tensor]]
    ) -> None:
        self._ensure_freefuse_state()
        self._freefuse_video_masks = video_masks

    def set_freefuse_audio_masks(
        self, audio_masks: Optional[Dict[str, torch.Tensor]]
    ) -> None:
        self._ensure_freefuse_state()
        self._freefuse_audio_masks = audio_masks

    def set_freefuse_masks(
        self,
        video_masks: Optional[Dict[str, torch.Tensor]] = None,
        audio_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        self.set_freefuse_video_masks(video_masks)
        self.set_freefuse_audio_masks(audio_masks)

    def set_freefuse_attention_biases(
        self,
        attention_biases: Optional[Dict[str, torch.Tensor]] = None,
        attention_bias_blocks: Optional[List[str]] = None,
    ) -> None:
        """
        Set additive biases by role:
        - `video_text`: (B, video_len, text_len)
        - `audio_text`: (B, audio_len, text_len)
        - `video_audio`: (B, video_len, audio_len)
        - `audio_video`: (B, audio_len, video_len)
        """
        self._ensure_freefuse_state()
        self._freefuse_attention_biases = attention_biases or {}
        self._freefuse_attention_bias_blocks = set(attention_bias_blocks) if attention_bias_blocks else None

    def set_freefuse_attention_bias(self, attention_bias: Optional[torch.Tensor]) -> None:
        """
        Compatibility helper: apply one bias tensor to both text cross-attn branches.
        """
        if attention_bias is None:
            self.set_freefuse_attention_biases({})
        else:
            self.set_freefuse_attention_biases(
                {
                    "video_text": attention_bias,
                    "audio_text": attention_bias,
                }
            )

    def set_freefuse_top_k_ratio(self, top_k_ratio: float) -> None:
        self._ensure_freefuse_state()
        self._freefuse_top_k_ratio = float(top_k_ratio)

    def set_freefuse_background_info(
        self,
        eos_token_index: Optional[int] = None,
        background_token_positions: Optional[List[int]] = None,
    ) -> None:
        self._ensure_freefuse_state()
        self._freefuse_eos_token_index = eos_token_index
        self._freefuse_background_token_positions = background_token_positions

    def clear_freefuse_state(self) -> None:
        self._ensure_freefuse_state()
        self._freefuse_video_masks = None
        self._freefuse_audio_masks = None
        self._freefuse_token_pos_maps = None
        self._freefuse_attention_biases = {}
        self._freefuse_attention_bias_blocks = None
        self._freefuse_top_k_ratio = 0.1
        self._freefuse_eos_token_index = None
        self._freefuse_background_token_positions = None

        for _, processor in self._iter_named_freefuse_processors():
            processor.cal_concept_sim_map = False
            processor.concept_sim_maps = None
            processor._freefuse_token_pos_maps = None
            processor._attention_bias = None
            processor._top_k_ratio = 0.1
            processor._eos_token_index = None
            processor._background_token_positions = None

        for _, module in self.named_modules():
            if isinstance(module, FreeFuseLinear):
                module.set_freefuse_masks(None)
                module.set_freefuse_token_pos_maps(None)

    def enable_concept_sim_map_extraction(
        self,
        video_block_name: Optional[str] = None,
        audio_block_name: Optional[str] = None,
    ) -> None:
        """
        Enable concept sim-map extraction on one video-text block and one audio-text block.
        """
        self._ensure_freefuse_state()

        for _, processor in self._iter_named_freefuse_processors():
            processor.cal_concept_sim_map = False

        if video_block_name is None:
            video_block_name = f"transformer_blocks.{len(self.transformer_blocks) - 1}.attn2"
        if audio_block_name is None:
            audio_block_name = f"transformer_blocks.{len(self.transformer_blocks) - 1}.audio_attn2"

        for module_name, processor in self._iter_named_freefuse_processors():
            if processor._freefuse_role == "video_text" and video_block_name in module_name:
                processor.cal_concept_sim_map = True
            if processor._freefuse_role == "audio_text" and audio_block_name in module_name:
                processor.cal_concept_sim_map = True

    def disable_concept_sim_map_extraction(self) -> None:
        self._ensure_freefuse_state()
        for _, processor in self._iter_named_freefuse_processors():
            processor.cal_concept_sim_map = False

    def get_concept_sim_maps(self, clear_after_read: bool = True) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """
        Returns `{"video": maps?, "audio": maps?}` if available.
        """
        out: Dict[str, Dict[str, torch.Tensor]] = {}
        for _, processor in self._iter_named_freefuse_processors():
            if processor.concept_sim_maps is None:
                continue
            if processor._freefuse_role == "video_text" and "video" not in out:
                out["video"] = processor.concept_sim_maps
                if clear_after_read:
                    processor.concept_sim_maps = None
            elif processor._freefuse_role == "audio_text" and "audio" not in out:
                out["audio"] = processor.concept_sim_maps
                if clear_after_read:
                    processor.concept_sim_maps = None

        return out if out else None

    # ------------------------------------------------------------------
    # Runtime helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _expand_mask_dict(
        masks: Optional[Dict[str, torch.Tensor]],
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if masks is None:
            return None

        out: Dict[str, torch.Tensor] = {}
        for lora_name, mask in masks.items():
            cur = mask
            if cur.dim() == 2:
                cur = cur.unsqueeze(-1)
            elif cur.dim() == 3 and cur.shape[-1] != 1:
                cur = cur.reshape(cur.shape[0], -1, 1)
            elif cur.dim() == 4 and cur.shape[1] == 1:
                cur = cur.reshape(cur.shape[0], -1, 1)
            elif cur.dim() not in (3, 4):
                raise ValueError(f"Unsupported mask shape for `{lora_name}`: {tuple(cur.shape)}")

            if cur.dim() == 4:
                cur = cur.reshape(cur.shape[0], -1, 1)

            if cur.shape[1] < seq_len:
                pad_n = seq_len - cur.shape[1]
                cur = torch.nn.functional.pad(cur, (0, 0, 0, pad_n), value=0.0)
            elif cur.shape[1] > seq_len:
                cur = cur[:, :seq_len]

            if cur.shape[0] < batch_size:
                repeats = (batch_size + cur.shape[0] - 1) // cur.shape[0]
                cur = cur.repeat(repeats, 1, 1)
            cur = cur[:batch_size]
            out[lora_name] = cur.to(device=device, dtype=dtype)

        return out

    @staticmethod
    def _expand_token_pos_maps(
        token_pos_maps: Optional[Dict[str, List[List[int]]]],
        batch_size: int,
        text_len: int,
    ) -> Optional[Dict[str, List[List[int]]]]:
        if token_pos_maps is None:
            return None

        out: Dict[str, List[List[int]]] = {}
        for lora_name, positions_list in token_pos_maps.items():
            per_batch: List[List[int]] = []
            if not positions_list:
                positions_list = [[]]
            for b in range(batch_size):
                src = positions_list[min(b, len(positions_list) - 1)]
                cur = [int(p) for p in src if 0 <= int(p) < text_len]
                per_batch.append(cur)
            out[lora_name] = per_batch
        return out

    @staticmethod
    def _classify_lora_target(module_name: str) -> str:
        """
        Classify which sequence a FreeFuseLinear target belongs to:
        - video
        - audio
        - text
        - none
        """
        # Top-level projections
        if module_name.startswith("proj_in") or module_name.startswith("proj_out"):
            return "video"
        if module_name.startswith("audio_proj_in") or module_name.startswith("audio_proj_out"):
            return "audio"

        # Cross-attn text-side projections
        if ".audio_attn2.to_k" in module_name or ".audio_attn2.to_v" in module_name:
            return "text"
        if ".attn2.to_k" in module_name or ".attn2.to_v" in module_name:
            return "text"

        # Cross-modal attention routing
        if ".audio_to_video_attn.to_q" in module_name or ".audio_to_video_attn.to_out" in module_name:
            return "video"
        if ".audio_to_video_attn.to_k" in module_name or ".audio_to_video_attn.to_v" in module_name:
            return "audio"
        if ".video_to_audio_attn.to_q" in module_name or ".video_to_audio_attn.to_out" in module_name:
            return "audio"
        if ".video_to_audio_attn.to_k" in module_name or ".video_to_audio_attn.to_v" in module_name:
            return "video"

        # Self/cross-attn branches
        if ".audio_attn1." in module_name:
            return "audio"
        if ".audio_attn2.to_q" in module_name or ".audio_attn2.to_out" in module_name:
            return "audio"
        if ".attn1." in module_name:
            return "video"
        if ".attn2.to_q" in module_name or ".attn2.to_out" in module_name:
            return "video"

        # FF branches
        if ".audio_ff." in module_name:
            return "audio"
        if ".ff." in module_name:
            return "video"

        return "none"

    def _apply_runtime_freefuse_to_processors(self) -> None:
        self._ensure_freefuse_state()
        for module_name, processor in self._iter_named_freefuse_processors():
            processor._freefuse_token_pos_maps = self._freefuse_token_pos_maps
            processor._top_k_ratio = self._freefuse_top_k_ratio
            processor._eos_token_index = self._freefuse_eos_token_index
            processor._background_token_positions = self._freefuse_background_token_positions

            if self._freefuse_attention_bias_blocks is not None:
                allow_bias = any(block_name in module_name for block_name in self._freefuse_attention_bias_blocks)
            else:
                allow_bias = True

            if not allow_bias:
                processor._attention_bias = None
                continue

            bias_key = processor._freefuse_role
            processor._attention_bias = self._freefuse_attention_biases.get(bias_key, None)

    def _apply_runtime_freefuse_to_lora_layers(
        self,
        video_batch_size: int,
        video_seq_len: int,
        audio_batch_size: int,
        audio_seq_len: int,
        text_seq_len: int,
        device: torch.device,
        video_dtype: torch.dtype,
        audio_dtype: torch.dtype,
    ) -> None:
        self._ensure_freefuse_state()

        video_masks = self._expand_mask_dict(
            self._freefuse_video_masks, video_batch_size, video_seq_len, device, video_dtype
        )
        audio_masks = self._expand_mask_dict(
            self._freefuse_audio_masks, audio_batch_size, audio_seq_len, device, audio_dtype
        )
        text_pos_maps = self._expand_token_pos_maps(
            self._freefuse_token_pos_maps, max(video_batch_size, audio_batch_size), text_seq_len
        )

        for module_name, module in self.named_modules():
            if not isinstance(module, FreeFuseLinear):
                continue

            target = self._classify_lora_target(module_name)
            if target == "video":
                module.set_freefuse_masks(video_masks)
                module.set_freefuse_token_pos_maps(None)
            elif target == "audio":
                module.set_freefuse_masks(audio_masks)
                module.set_freefuse_token_pos_maps(None)
            elif target == "text":
                module.set_freefuse_masks(None)
                module.set_freefuse_token_pos_maps(text_pos_maps)
            else:
                module.set_freefuse_masks(None)
                module.set_freefuse_token_pos_maps(None)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        audio_timestep: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        audio_encoder_attention_mask: Optional[torch.Tensor] = None,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        fps: float = 24.0,
        audio_num_frames: Optional[int] = None,
        video_coords: Optional[torch.Tensor] = None,
        audio_coords: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict] = None,
        return_dict: bool = True,
    ):
        self._apply_runtime_freefuse_to_processors()
        self._apply_runtime_freefuse_to_lora_layers(
            video_batch_size=hidden_states.shape[0],
            video_seq_len=hidden_states.shape[1],
            audio_batch_size=audio_hidden_states.shape[0],
            audio_seq_len=audio_hidden_states.shape[1],
            text_seq_len=encoder_hidden_states.shape[1],
            device=hidden_states.device,
            video_dtype=hidden_states.dtype,
            audio_dtype=audio_hidden_states.dtype,
        )

        return super().forward(
            hidden_states=hidden_states,
            audio_hidden_states=audio_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            audio_encoder_hidden_states=audio_encoder_hidden_states,
            timestep=timestep,
            audio_timestep=audio_timestep,
            encoder_attention_mask=encoder_attention_mask,
            audio_encoder_attention_mask=audio_encoder_attention_mask,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
            audio_num_frames=audio_num_frames,
            video_coords=video_coords,
            audio_coords=audio_coords,
            attention_kwargs=attention_kwargs,
            return_dict=return_dict,
        )
