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

import inspect
import os
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, PreTrainedModel

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, ZImageLoraLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import ZImageTransformer2DModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.z_image.pipeline_output import ZImagePipelineOutput

from src.models.freefuse_transformer_z_image import SEQ_MULTI_OF


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import ZImagePipeline

        >>> pipe = ZImagePipeline.from_pretrained("Z-a-o/Z-Image-Turbo", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")

        >>> # Optionally, set the attention backend to flash-attn 2 or 3, default is SDPA in PyTorch.
        >>> # (1) Use flash attention 2
        >>> # pipe.transformer.set_attention_backend("flash")
        >>> # (2) Use flash attention 3
        >>> # pipe.transformer.set_attention_backend("_flash_3")

        >>> prompt = "一幅为名为“造相「Z-IMAGE-TURBO」”的项目设计的创意海报。画面巧妙地将文字概念视觉化：一辆复古蒸汽小火车化身为巨大的拉链头，正拉开厚厚的冬日积雪，展露出一个生机盎然的春天。"
        >>> image = pipe(
        ...     prompt,
        ...     height=1024,
        ...     width=1024,
        ...     num_inference_steps=9,
        ...     guidance_scale=0.0,
        ...     generator=torch.Generator("cuda").manual_seed(42),
        ... ).images[0]
        >>> image.save("zimage.png")
        ```
"""


# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class FreeFuseZImagePipeline(DiffusionPipeline, ZImageLoraLoaderMixin, FromSingleFileMixin):
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: PreTrainedModel,
        tokenizer: AutoTokenizer,
        transformer: ZImageTransformer2DModel,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            transformer=transformer,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_embeds = self._encode_prompt(
            prompt=prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            max_sequence_length=max_sequence_length,
        )

        if do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = ["" for _ in prompt]
            else:
                negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            assert len(prompt) == len(negative_prompt)
            negative_prompt_embeds = self._encode_prompt(
                prompt=negative_prompt,
                device=device,
                prompt_embeds=negative_prompt_embeds,
                max_sequence_length=max_sequence_length,
            )
        else:
            negative_prompt_embeds = []
        return prompt_embeds, negative_prompt_embeds

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        max_sequence_length: int = 512,
    ) -> List[torch.FloatTensor]:
        device = device or self._execution_device

        if prompt_embeds is not None:
            return prompt_embeds

        if isinstance(prompt, str):
            prompt = [prompt]

        for i, prompt_item in enumerate(prompt):
            messages = [
                {"role": "user", "content": prompt_item},
            ]
            prompt_item = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            prompt[i] = prompt_item

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)
        prompt_masks = text_inputs.attention_mask.to(device).bool()

        prompt_embeds = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]

        embeddings_list = []

        for i in range(len(prompt_embeds)):
            embeddings_list.append(prompt_embeds[i][prompt_masks[i]])

        return embeddings_list

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    # ──────────────────────────────────────────────────────────────────
    # FreeFuse helper methods
    # ──────────────────────────────────────────────────────────────────
    def _construct_attention_bias(
        self,
        lora_masks: Dict[str, torch.Tensor],
        freefuse_token_pos_maps: Dict[str, List[List[int]]],
        x_seqlen: int,
        cap_seqlen: int,
        bias_scale: float = 5.0,
        positive_bias_scale: float = 1.0,
        bidirectional: bool = True,
        use_positive_bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> torch.Tensor:
        """
        Build an additive attention-bias matrix for the unified sequence
        ``[image(x_seqlen), text(cap_seqlen)]``.

        Negative bias suppresses cross-LoRA image↔text attention.
        Positive bias encourages same-LoRA image↔text attention.

        Returns
        -------
        attention_bias : ``(B, total_seq, total_seq)`` float tensor.
        """
        batch_size = next(iter(lora_masks.values())).shape[0]
        total = x_seqlen + cap_seqlen
        bias = torch.zeros(batch_size, total, total, device=device, dtype=dtype)

        lora_name_to_idx = {n: i for i, n in enumerate(lora_masks.keys())}

        # text-token → LoRA owner  (-1 = shared / common)
        text_owner = torch.full((cap_seqlen,), -1, device=device, dtype=torch.long)
        for lora_name, positions_list in freefuse_token_pos_maps.items():
            if lora_name not in lora_name_to_idx:
                continue
            idx = lora_name_to_idx[lora_name]
            if positions_list and positions_list[0]:
                for p in positions_list[0]:
                    if 0 <= p < cap_seqlen:
                        text_owner[p] = idx

        for lora_name, img_mask in lora_masks.items():
            if lora_name not in lora_name_to_idx:
                continue
            idx = lora_name_to_idx[lora_name]

            # img_mask: (B, img_token_count) — pad to x_seqlen
            if img_mask.shape[1] < x_seqlen:
                img_mask = F.pad(img_mask, (0, x_seqlen - img_mask.shape[1]))

            other_text = ((text_owner != idx) & (text_owner != -1)).float()   # (cap,)

            # Image→Text: suppress cross-LoRA
            bias[:, :x_seqlen, x_seqlen:x_seqlen + cap_seqlen] += (
                img_mask.unsqueeze(-1) * other_text.unsqueeze(0).unsqueeze(0) * (-bias_scale)
            )

            if use_positive_bias:
                same_text = (text_owner == idx).float()
                bias[:, :x_seqlen, x_seqlen:x_seqlen + cap_seqlen] += (
                    img_mask.unsqueeze(-1) * same_text.unsqueeze(0).unsqueeze(0) * positive_bias_scale
                )

            if bidirectional:
                same_text = (text_owner == idx).float()
                not_this_img = 1.0 - img_mask
                bias[:, x_seqlen:x_seqlen + cap_seqlen, :x_seqlen] += (
                    same_text.unsqueeze(0).unsqueeze(-1) * not_this_img.unsqueeze(1) * (-bias_scale)
                )
                if use_positive_bias:
                    bias[:, x_seqlen:x_seqlen + cap_seqlen, :x_seqlen] += (
                        same_text.unsqueeze(0).unsqueeze(-1) * img_mask.unsqueeze(1) * positive_bias_scale
                    )

        return bias

    def stabilized_balanced_argmax(
        self, logits, h, w, target_count=None, max_iter=15,
        lr=0.0001, gravity_weight=0.000001, spatial_weight=0.00004,
        momentum=0.2, centroid_margin=0.0, border_penalty=0.0,
        anisotropy=1.1, debug=False,
    ):
        """Balanced argmax with spatial regularisation (same as Flux version)."""
        B, C, N = logits.shape
        device = logits.device

        y_range = torch.linspace(-1, 1, steps=h, device=device)
        x_range = torch.linspace(-1, 1, steps=w, device=device) * anisotropy
        grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing="ij")
        flat_y = grid_y.reshape(1, 1, N)
        flat_x = grid_x.reshape(1, 1, N)

        pixel_size = 2.0 / max(h, w)
        is_border = (flat_y.abs() > (1 - pixel_size * 1.5)) | (flat_x.abs() > (1 - pixel_size * 1.5))
        border_mask = is_border.float()

        if target_count is None:
            target_count = N / C

        bias = torch.zeros(B, C, 1, device=device)

        def linear_normalize(x, dim=1):
            x_min = x.min(dim=dim, keepdim=True)[0]
            x_max = x.max(dim=dim, keepdim=True)[0]
            return (x - x_min) / (x_max - x_min + 1e-8)

        running_probs = linear_normalize(logits, dim=1)
        logit_range = (logits.max() - logits.min()).item()
        logit_scale = max(logit_range, 1e-4)
        effective_lr = lr * logit_scale
        max_bias = logit_scale * 10.0

        neighbor_kernel = torch.ones(C, 1, 3, 3, device=device, dtype=logits.dtype) / 8.0
        neighbor_kernel[:, :, 1, 1] = 0

        current_logits = logits.clone()

        for i in range(max_iter):
            probs = linear_normalize(current_logits - bias, dim=1)
            running_probs = (1 - momentum) * probs + momentum * running_probs
            mass = running_probs.sum(dim=2, keepdim=True) + 1e-6
            center_y = (running_probs * flat_y).sum(dim=2, keepdim=True) / mass
            center_x = (running_probs * flat_x).sum(dim=2, keepdim=True) / mass

            if centroid_margin > 0:
                center_y = torch.clamp(center_y, -(1 - centroid_margin), 1 - centroid_margin)
                center_x = torch.clamp(center_x, -(1 - centroid_margin), 1 - centroid_margin)

            dist_sq = (flat_y - center_y) ** 2 + (flat_x - center_x) ** 2

            hard_indices = torch.argmax(current_logits - bias, dim=1)
            hard_counts = F.one_hot(hard_indices, num_classes=C).float().sum(dim=1)

            diff = hard_counts - target_count
            cur_lr = effective_lr * (0.95 ** i)
            bias += torch.sign(diff).unsqueeze(2) * cur_lr
            bias = torch.clamp(bias, -max_bias, max_bias)

            if spatial_weight > 0:
                probs_img = running_probs.view(B, C, h, w).float()
                neighbor_votes = F.conv2d(probs_img, neighbor_kernel.float(), padding=1, groups=C)
                neighbor_votes = neighbor_votes.to(logits.dtype).view(B, C, N)
            else:
                neighbor_votes = torch.zeros_like(logits)

            current_logits = (
                logits - bias
                + neighbor_votes * spatial_weight
                - dist_sq * gravity_weight
                - border_mask * border_penalty
            )

        return torch.argmax(current_logits, dim=1)

    def morphological_clean_mask(
        self, mask, h, w, opening_kernel_size=3, closing_kernel_size=3,
    ):
        """Opening + closing on a (B, N) binary mask."""
        B = mask.shape[0]
        mask_2d = mask.view(B, 1, h, w)

        def dilate(x, ks):
            p = ks // 2
            o = F.max_pool2d(x, kernel_size=ks, stride=1, padding=p)
            if o.shape[-2:] != x.shape[-2:]:
                o = F.interpolate(o, size=x.shape[-2:], mode="nearest")
            return o

        def erode(x, ks):
            p = ks // 2
            o = 1.0 - F.max_pool2d(1.0 - x, kernel_size=ks, stride=1, padding=p)
            if o.shape[-2:] != x.shape[-2:]:
                o = F.interpolate(o, size=x.shape[-2:], mode="nearest")
            return o

        if opening_kernel_size > 1:
            mask_2d = dilate(erode(mask_2d, opening_kernel_size), opening_kernel_size)
        if closing_kernel_size > 1:
            mask_2d = erode(dilate(mask_2d, closing_kernel_size), closing_kernel_size)

        return mask_2d.view(B, -1)

    def _process_sim_maps(
        self,
        concept_sim_maps: Dict[str, torch.Tensor],
        h_tokens: int,
        w_tokens: int,
        img_token_count: int,
        eos_bg_scale: float = 0.95,
    ):
        """
        Convert raw concept similarity maps to binary LoRA masks.

        Returns
        -------
        lora_masks : ``{name: (B, img_token_count)}``
        debug_info : dict with intermediate tensors for visualisation
        """
        debug_info: Dict[str, Any] = {}

        if concept_sim_maps is None:
            return {}, debug_info

        has_bg = "__bg__" in concept_sim_maps
        has_eos = "__eos__" in concept_sim_maps

        # Pop background channel
        if has_bg:
            bg_raw = concept_sim_maps.pop("__bg__")[:, :img_token_count, :]
        elif has_eos:
            bg_raw = concept_sim_maps.pop("__eos__")[:, :img_token_count, :]
        else:
            bg_raw = None

        # Stack concept maps  → (B, C, N, 1)
        names = list(concept_sim_maps.keys())
        maps_list = [concept_sim_maps[n][:, :img_token_count, :] for n in names]
        stacked = torch.stack(maps_list, dim=1)          # (B, C, N, 1)
        B, C, N, _ = stacked.shape
        squeezed = stacked.squeeze(-1)                    # (B, C, N)

        # Build background channel
        if bg_raw is not None:
            bg_channel = bg_raw.squeeze(-1) * eos_bg_scale  # (B, N)
        else:
            bg_channel = squeezed.mean(dim=1)               # fallback: average

        # Argmax with background
        with_bg = torch.cat([squeezed, bg_channel.unsqueeze(1)], dim=1)  # (B, C+1, N)
        bg_argmax = with_bg.argmax(dim=1)
        raw_fg = (bg_argmax != C).float()

        # Morphological cleaning
        fg_mask = self.morphological_clean_mask(
            raw_fg, h_tokens, w_tokens,
            opening_kernel_size=2, closing_kernel_size=2,
        ).bool()

        # Balanced argmax among concepts
        max_indices = self.stabilized_balanced_argmax(
            squeezed, h_tokens, w_tokens,
        )

        lora_masks = {}
        for idx, name in enumerate(names):
            lora_masks[name] = ((max_indices == idx).float() * fg_mask.float())

        # Put background back into concept_sim_maps for visualization
        if has_bg:
            concept_sim_maps["__bg__"] = bg_raw
        elif has_eos:
            concept_sim_maps["__eos__"] = bg_raw

        # Collect debug info
        debug_info = {
            "concept_sim_maps": {k: v.clone() for k, v in concept_sim_maps.items()},
            "names": names,
            "raw_fg": raw_fg,               # (B, N) before morphological cleaning
            "fg_mask": fg_mask,             # (B, N) after morphological cleaning
            "max_indices": max_indices,     # (B, N) balanced argmax result
            "bg_channel": bg_channel,       # (B, N)
            "has_bg": has_bg,
            "has_eos": has_eos,
        }

        return lora_masks, debug_info

    # ──────────────────────────────────────────────────────────────────
    # __call__
    # ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 12,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 0.0,
        cfg_normalization: bool = False,
        cfg_truncation: float = 1.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        negative_prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        # ── FreeFuse params ──
        aggreate_lora_score_step: int = 3,
        sim_map_block_idx: int = 18,
        debug_save_path: Optional[str] = None,
        use_attention_bias: bool = True,
        attention_bias_scale: float = 4.0,
        attention_bias_positive_scale: float = 2.0,
        attention_bias_bidirectional: bool = True,
        attention_bias_positive: bool = True,
        attention_bias_blocks: Optional[Union[str, List[str]]] = None,
    ):
        """
        FreeFuse Z-Image pipeline with two-phase denoising.

        Phase 1: run a few steps with LoRA disabled to collect concept sim maps.
        Phase 2: restart from initial noise with LoRA + spatial masks + attention bias.
        """
        height = height or 1024
        width = width or 1024

        vae_scale = self.vae_scale_factor * 2
        if height % vae_scale != 0:
            raise ValueError(f"Height must be divisible by {vae_scale} (got {height}).")
        if width % vae_scale != 0:
            raise ValueError(f"Width must be divisible by {vae_scale} (got {width}).")

        device = self._execution_device
        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs or {}
        self._interrupt = False
        self._cfg_normalization = cfg_normalization
        self._cfg_truncation = cfg_truncation

        # ── batch size ──
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = len(prompt_embeds)

        # ── encode prompt ──
        if prompt_embeds is not None and prompt is None:
            if self.do_classifier_free_guidance and negative_prompt_embeds is None:
                raise ValueError(
                    "When `prompt_embeds` is provided without `prompt`, "
                    "`negative_prompt_embeds` must also be provided for classifier-free guidance."
                )
        else:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                device=device,
                max_sequence_length=max_sequence_length,
            )

        # ── latents ──
        num_channels_latents = self.transformer.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents, height, width,
            torch.float32, device, generator, latents,
        )

        if num_images_per_prompt > 1:
            prompt_embeds = [pe for pe in prompt_embeds for _ in range(num_images_per_prompt)]
            if self.do_classifier_free_guidance and negative_prompt_embeds:
                negative_prompt_embeds = [npe for npe in negative_prompt_embeds for _ in range(num_images_per_prompt)]

        actual_batch_size = batch_size * num_images_per_prompt

        # ── timesteps ──
        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        self.scheduler.sigma_min = 0.0
        scheduler_kwargs = {"mu": mu}
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device,
            sigmas=sigmas, **scheduler_kwargs,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        initial_latents = latents.clone()

        # ── spatial dimensions ──
        latent_h = 2 * (height // vae_scale)
        latent_w = 2 * (width // vae_scale)
        h_tokens = latent_h // 2   # after patch_size=2
        w_tokens = latent_w // 2
        img_token_count = h_tokens * w_tokens
        x_seqlen = img_token_count + ((-img_token_count) % SEQ_MULTI_OF)

        text_token_count = len(prompt_embeds[0])
        cap_seqlen = text_token_count + ((-text_token_count) % SEQ_MULTI_OF)

        # ── FreeFuse extraction from joint_attention_kwargs ──
        freefuse_token_pos_maps = self._joint_attention_kwargs.get("freefuse_token_pos_maps", None)
        background_token_positions = self._joint_attention_kwargs.get("background_token_positions", None)
        eos_token_index = self._joint_attention_kwargs.get("eos_token_index", None)
        top_k_ratio = self._joint_attention_kwargs.get("top_k_ratio", 0.1)

        # Set token_pos_maps on transformer for LoRA-layer exclusion
        if freefuse_token_pos_maps is not None:
            self.transformer.set_freefuse_token_pos_maps(freefuse_token_pos_maps)

        # ────────────────────────────────────────────
        # Helper: one denoising step (shared by both phases)
        # ────────────────────────────────────────────
        def _denoise_step(cur_latents, t, pe, npe, apply_cfg, cur_gs, extra_jk=None):
            timestep = t.expand(cur_latents.shape[0])
            timestep = (1000 - timestep) / 1000

            if apply_cfg:
                lat = cur_latents.to(self.transformer.dtype).repeat(2, 1, 1, 1)
                pe_in = pe + npe
                ts_in = timestep.repeat(2)
            else:
                lat = cur_latents.to(self.transformer.dtype)
                pe_in = pe
                ts_in = timestep

            lat = lat.unsqueeze(2)
            lat_list = list(lat.unbind(dim=0))

            out = self.transformer(
                lat_list, ts_in, pe_in,
                return_dict=False,
                joint_attention_kwargs=extra_jk,
            )
            model_out_list = out[0]
            concept_sim_maps = out[1] if len(out) > 1 else None

            if apply_cfg:
                pos_out = model_out_list[:actual_batch_size]
                neg_out = model_out_list[actual_batch_size:]
                preds = []
                for j in range(actual_batch_size):
                    p = pos_out[j].float()
                    n = neg_out[j].float()
                    pred = p + cur_gs * (p - n)
                    if self._cfg_normalization and float(self._cfg_normalization) > 0:
                        on = torch.linalg.vector_norm(p)
                        nn_ = torch.linalg.vector_norm(pred)
                        mx = on * float(self._cfg_normalization)
                        if nn_ > mx:
                            pred = pred * (mx / nn_)
                    preds.append(pred)
                noise_pred = torch.stack(preds, dim=0)
            else:
                noise_pred = torch.stack([x.float() for x in model_out_list], dim=0)

            noise_pred = noise_pred.squeeze(2)
            noise_pred = -noise_pred
            return noise_pred, concept_sim_maps

        # ════════════════════════════════════════════
        # Phase 1: collect concept similarity maps
        # ════════════════════════════════════════════
        print("[FreeFuse] Phase 1: collecting concept similarity maps …")
        self.disable_lora()

        # Enable sim-map collection on the designated block
        from src.attn_processor.freefuse_z_image_attn_processor import FreeFuseZImageAttnProcessor
        sim_proc = self.transformer.layers[sim_map_block_idx].attention.processor
        assert isinstance(sim_proc, FreeFuseZImageAttnProcessor), (
            f"Block {sim_map_block_idx} processor is not FreeFuseZImageAttnProcessor"
        )

        phase1_latents = latents.clone()
        concept_sim_maps = None

        self.scheduler._step_index = None
        for i, t in enumerate(timesteps[:aggreate_lora_score_step]):
            if self.interrupt:
                continue

            # Only collect on the LAST phase-1 step
            sim_proc.cal_concept_sim_map = (i == aggreate_lora_score_step - 1)

            noise_pred, csm = _denoise_step(
                phase1_latents, t, prompt_embeds, negative_prompt_embeds,
                apply_cfg=False, cur_gs=0.0,
                extra_jk=self._joint_attention_kwargs,
            )
            if csm is not None:
                concept_sim_maps = csm

            phase1_latents = self.scheduler.step(
                noise_pred.to(torch.float32), t, phase1_latents, return_dict=False
            )[0]

        sim_proc.cal_concept_sim_map = False

        # ════════════════════════════════════════════
        # Process sim maps → masks + attention bias
        # ════════════════════════════════════════════
        lora_masks, debug_info = self._process_sim_maps(
            concept_sim_maps, h_tokens, w_tokens, img_token_count,
        )

        if lora_masks:
            self.transformer.set_freefuse_masks(lora_masks)
            print("[FreeFuse] LoRA masks set on transformer.")

        # Attention bias
        constructed_bias = None
        if use_attention_bias and lora_masks and freefuse_token_pos_maps:
            constructed_bias = self._construct_attention_bias(
                lora_masks, freefuse_token_pos_maps,
                x_seqlen, cap_seqlen,
                bias_scale=attention_bias_scale,
                positive_bias_scale=attention_bias_positive_scale,
                bidirectional=attention_bias_bidirectional,
                use_positive_bias=attention_bias_positive,
                device=device,
                dtype=torch.float32,
            )
            print(f"[FreeFuse] Attention bias constructed ({attention_bias_scale=}).")

        # Build phase-2 joint_attention_kwargs
        phase2_jk = dict(self._joint_attention_kwargs)
        if constructed_bias is not None:
            # Resolve block name presets
            resolved_blocks = attention_bias_blocks
            if isinstance(resolved_blocks, str):
                n_layers = len(self.transformer.layers)
                if resolved_blocks == "all":
                    resolved_blocks = None
                elif resolved_blocks == "last_half":
                    resolved_blocks = [f"layers.{i}" for i in range(n_layers // 2, n_layers)]
                else:
                    resolved_blocks = [resolved_blocks]
            phase2_jk["attention_bias"] = constructed_bias
            phase2_jk["attention_bias_blocks"] = resolved_blocks

        # ════════════════════════════════════════════
        # Debug visualisation (Flux-style comprehensive)
        # ════════════════════════════════════════════
        if debug_save_path is not None and lora_masks:
            os.makedirs(debug_save_path, exist_ok=True)
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import numpy as np

                latent_h, latent_w = h_tokens, w_tokens
                step_tag = f"phase1_{aggreate_lora_score_step}"

                # ── 1. Concept similarity maps ──
                if "concept_sim_maps" in debug_info:
                    for cname, smap in debug_info["concept_sim_maps"].items():
                        # smap: (B, N, 1) or (B, N)
                        s = smap[0].squeeze(-1) if smap.dim() == 3 else smap[0]
                        s = s[:img_token_count]
                        s2d = s.view(latent_h, latent_w).cpu().float().numpy()

                        # Annotated viridis
                        fig, ax = plt.subplots(figsize=(8, 8))
                        im = ax.imshow(s2d, cmap="viridis")
                        ax.set_title(f"Concept Sim Map: {cname} ({step_tag})")
                        plt.colorbar(im, ax=ax)
                        plt.savefig(os.path.join(debug_save_path, f"concept_sim_map_{cname}_{step_tag}.png"),
                                    dpi=150, bbox_inches="tight")
                        plt.close(fig)

                        # Clean viridis (for PPT)
                        fig, ax = plt.subplots(figsize=(8, 8))
                        ax.imshow(s2d, cmap="viridis")
                        ax.axis("off")
                        plt.savefig(os.path.join(debug_save_path, f"concept_sim_map_{cname}_{step_tag}_clean_viridis.png"),
                                    dpi=150, bbox_inches="tight", pad_inches=0)
                        plt.close(fig)

                        # Clean plasma
                        fig, ax = plt.subplots(figsize=(8, 8))
                        ax.imshow(s2d, cmap="plasma")
                        ax.axis("off")
                        plt.savefig(os.path.join(debug_save_path, f"concept_sim_map_{cname}_{step_tag}_clean_plasma.png"),
                                    dpi=150, bbox_inches="tight", pad_inches=0)
                        plt.close(fig)

                # ── 2. Foreground / background mask ──
                if "fg_mask" in debug_info:
                    fg = debug_info["fg_mask"][0][:img_token_count].float().view(latent_h, latent_w).cpu().numpy()
                    fig, ax = plt.subplots(figsize=(8, 8))
                    im = ax.imshow(fg, cmap="RdYlGn", vmin=0, vmax=1)
                    ax.set_title(f"Foreground Mask ({step_tag})\nGreen=Foreground, Red=Background")
                    plt.colorbar(im, ax=ax)
                    plt.savefig(os.path.join(debug_save_path, f"foreground_mask_{step_tag}.png"),
                                dpi=150, bbox_inches="tight")
                    plt.close(fig)

                    bg = 1.0 - fg
                    fig, ax = plt.subplots(figsize=(8, 8))
                    im = ax.imshow(bg, cmap="Greys", vmin=0, vmax=1)
                    ax.set_title(f"Background Mask ({step_tag})\nWhite=Background")
                    plt.colorbar(im, ax=ax)
                    plt.savefig(os.path.join(debug_save_path, f"background_mask_{step_tag}.png"),
                                dpi=150, bbox_inches="tight")
                    plt.close(fig)

                # ── 3. Raw vs morphologically-cleaned FG mask comparison ──
                if "raw_fg" in debug_info and "fg_mask" in debug_info:
                    raw_fg_2d = debug_info["raw_fg"][0][:img_token_count].float().view(latent_h, latent_w).cpu().numpy()
                    clean_fg_2d = debug_info["fg_mask"][0][:img_token_count].float().view(latent_h, latent_w).cpu().numpy()
                    diff_2d = clean_fg_2d - raw_fg_2d

                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    im0 = axes[0].imshow(raw_fg_2d, cmap="RdYlGn", vmin=0, vmax=1)
                    axes[0].set_title("Raw Argmax FG\n(with noise/holes)")
                    plt.colorbar(im0, ax=axes[0], shrink=0.8)
                    im1 = axes[1].imshow(clean_fg_2d, cmap="RdYlGn", vmin=0, vmax=1)
                    axes[1].set_title("Morphological Cleaned FG\n(opening + closing)")
                    plt.colorbar(im1, ax=axes[1], shrink=0.8)
                    im2 = axes[2].imshow(diff_2d, cmap="coolwarm", vmin=-1, vmax=1)
                    axes[2].set_title("Difference\n(Blue=Removed, Red=Added)")
                    plt.colorbar(im2, ax=axes[2], shrink=0.8)
                    fig.suptitle(f"Morphological Cleaning Effect ({step_tag})", fontsize=14)
                    plt.tight_layout()
                    plt.savefig(os.path.join(debug_save_path, f"raw_vs_morphological_fg_{step_tag}.png"),
                                dpi=150, bbox_inches="tight")
                    plt.close(fig)

                # ── 4. Balanced argmax before BG exclusion (per concept) ──
                if "max_indices" in debug_info:
                    max_idx = debug_info["max_indices"]
                    concept_names = debug_info.get("names", list(lora_masks.keys()))
                    for cidx, cname in enumerate(concept_names):
                        argmax_2d = (max_idx[0][:img_token_count] == cidx).float().view(latent_h, latent_w).cpu().numpy()
                        fig, ax = plt.subplots(figsize=(8, 8))
                        im = ax.imshow(argmax_2d, cmap="Oranges", vmin=0, vmax=1)
                        ax.set_title(f"Argmax (Before BG): {cname} ({step_tag})")
                        plt.colorbar(im, ax=ax)
                        plt.savefig(os.path.join(debug_save_path, f"argmax_before_bg_{cname}_{step_tag}.png"),
                                    dpi=150, bbox_inches="tight")
                        plt.close(fig)

                # ── 5. Final LoRA masks (after BG exclusion) ──
                for lora_name, mask in lora_masks.items():
                    m2d = mask[0][:img_token_count].view(latent_h, latent_w).cpu().float().numpy()

                    # Annotated gray
                    fig, ax = plt.subplots(figsize=(8, 8))
                    im = ax.imshow(m2d, cmap="gray", vmin=0, vmax=1)
                    ax.set_title(f"LoRA Mask: {lora_name} ({step_tag})")
                    plt.colorbar(im, ax=ax)
                    plt.savefig(os.path.join(debug_save_path, f"lora_mask_{lora_name}_{step_tag}.png"),
                                dpi=150, bbox_inches="tight")
                    plt.close(fig)

                    # Clean version (for PPT)
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(m2d, cmap="gray", vmin=0, vmax=1)
                    ax.axis("off")
                    plt.savefig(os.path.join(debug_save_path, f"lora_mask_{lora_name}_{step_tag}_clean.png"),
                                dpi=150, bbox_inches="tight", pad_inches=0)
                    plt.close(fig)

                # ── 6. Combined overview (all masks side-by-side) ──
                n_masks = len(lora_masks)
                if n_masks > 0:
                    fig, axes = plt.subplots(1, n_masks + 1, figsize=(6 * (n_masks + 1), 6))
                    if n_masks + 1 == 1:
                        axes = [axes]
                    for midx, (mname, mmask) in enumerate(lora_masks.items()):
                        m2d = mmask[0][:img_token_count].view(latent_h, latent_w).cpu().float().numpy()
                        axes[midx].imshow(m2d, cmap="gray", vmin=0, vmax=1)
                        axes[midx].set_title(mname)
                        axes[midx].axis("off")
                    # Last subplot: overlay all masks with distinct colours
                    overlay = np.zeros((latent_h, latent_w, 3), dtype=np.float32)
                    colours = [(1, 0.2, 0.2), (0.2, 0.5, 1), (0.2, 0.9, 0.3),
                               (1, 0.8, 0.2), (0.8, 0.2, 0.8), (0.2, 0.9, 0.9)]
                    for midx, (mname, mmask) in enumerate(lora_masks.items()):
                        m2d = mmask[0][:img_token_count].view(latent_h, latent_w).cpu().float().numpy()
                        c = colours[midx % len(colours)]
                        overlay += m2d[:, :, None] * np.array(c)[None, None, :]
                    overlay = np.clip(overlay, 0, 1)
                    axes[-1].imshow(overlay)
                    axes[-1].set_title("Overlay")
                    axes[-1].axis("off")
                    fig.suptitle(f"All LoRA Masks ({step_tag})", fontsize=14)
                    plt.tight_layout()
                    plt.savefig(os.path.join(debug_save_path, f"all_masks_overview_{step_tag}.png"),
                                dpi=150, bbox_inches="tight")
                    plt.close(fig)

                # ── 7. Attention bias matrix ──
                if constructed_bias is not None:
                    bias_2d = constructed_bias[0].cpu().float().numpy()
                    fig, ax = plt.subplots(figsize=(12, 12))
                    vmax = max(abs(attention_bias_scale), abs(attention_bias_positive_scale))
                    im = ax.imshow(bias_2d, cmap="RdBu", vmin=-vmax, vmax=vmax)
                    ax.set_title(f"Attention Bias Matrix ({step_tag})")
                    ax.set_xlabel("Key (image | text)")
                    ax.set_ylabel("Query (image | text)")
                    ax.axhline(y=x_seqlen - 0.5, color="white", linestyle="--", linewidth=1)
                    ax.axvline(x=x_seqlen - 0.5, color="white", linestyle="--", linewidth=1)
                    plt.colorbar(im, ax=ax)
                    plt.savefig(os.path.join(debug_save_path, f"attention_bias_{step_tag}.png"),
                                dpi=150, bbox_inches="tight")
                    plt.close(fig)

                    # Zoomed-in: image→text quadrant only
                    img2txt = bias_2d[:x_seqlen, x_seqlen:x_seqlen + cap_seqlen]
                    fig, ax = plt.subplots(figsize=(12, 8))
                    im = ax.imshow(img2txt, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
                    ax.set_title(f"Attention Bias: Image→Text ({step_tag})")
                    ax.set_xlabel("Text token position")
                    ax.set_ylabel("Image token position")
                    plt.colorbar(im, ax=ax)
                    plt.savefig(os.path.join(debug_save_path, f"attention_bias_img2txt_{step_tag}.png"),
                                dpi=150, bbox_inches="tight")
                    plt.close(fig)

                # ── 8. Predicted x0 from Phase-1 final latents ──
                try:
                    p1_decode = phase1_latents.to(self.vae.dtype)
                    p1_decode = (p1_decode / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                    decoded = self.vae.decode(p1_decode, return_dict=False)[0]
                    decoded = decoded.clamp(-1, 1)
                    decoded = (decoded + 1) / 2
                    decoded_np = decoded[0].permute(1, 2, 0).cpu().float().numpy()
                    fig, ax = plt.subplots(figsize=(10, 10))
                    ax.imshow(decoded_np)
                    ax.set_title(f"Phase-1 Predicted x0 ({step_tag})")
                    ax.axis("off")
                    plt.savefig(os.path.join(debug_save_path, f"predicted_x0_{step_tag}.png"),
                                dpi=150, bbox_inches="tight")
                    plt.close(fig)
                except Exception as e:
                    print(f"[debug] Could not decode Phase-1 latents: {e}")

                # ── 9. Save raw tensors for offline analysis ──
                torch.save({
                    "concept_sim_maps": {k: v.cpu() for k, v in debug_info.get("concept_sim_maps", {}).items()},
                    "lora_masks": {k: v.cpu() for k, v in lora_masks.items()},
                    "raw_fg": debug_info.get("raw_fg", torch.tensor([])).cpu(),
                    "fg_mask": debug_info.get("fg_mask", torch.tensor([])).cpu(),
                    "max_indices": debug_info.get("max_indices", torch.tensor([])).cpu(),
                    "h_tokens": h_tokens,
                    "w_tokens": w_tokens,
                    "attention_bias": constructed_bias.cpu() if constructed_bias is not None else None,
                }, os.path.join(debug_save_path, f"debug_tensors_{step_tag}.pt"))

                print(f"[debug] Saved {len(lora_masks)} mask visualisations + tensors → {debug_save_path}/")

            except ImportError:
                print("[debug] matplotlib not available, skipping visualisation.")

        # ════════════════════════════════════════════
        # Phase 2: regenerate with masks & bias
        # ════════════════════════════════════════════
        print("[FreeFuse] Phase 2: generating with LoRA masks …")
        self.enable_lora()
        latents = initial_latents.clone()

        # Reset scheduler
        self.scheduler.sigma_min = 0.0
        timesteps2, _ = retrieve_timesteps(
            self.scheduler, num_inference_steps, device,
            sigmas=sigmas, **scheduler_kwargs,
        )

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps2):
                if self.interrupt:
                    continue

                timestep_val = t.expand(latents.shape[0])
                t_norm = ((1000 - timestep_val) / 1000)[0].item()

                cur_gs = self.guidance_scale
                if (
                    self.do_classifier_free_guidance
                    and self._cfg_truncation is not None
                    and float(self._cfg_truncation) <= 1
                ):
                    if t_norm > self._cfg_truncation:
                        cur_gs = 0.0

                apply_cfg = self.do_classifier_free_guidance and cur_gs > 0

                noise_pred, _ = _denoise_step(
                    latents, t, prompt_embeds, negative_prompt_embeds,
                    apply_cfg=apply_cfg, cur_gs=cur_gs,
                    extra_jk=phase2_jk,
                )

                latents = self.scheduler.step(
                    noise_pred.to(torch.float32), t, latents, return_dict=False
                )[0]
                assert latents.dtype == torch.float32

                if callback_on_step_end is not None:
                    cb_kw = {}
                    for k in callback_on_step_end_tensor_inputs:
                        cb_kw[k] = locals().get(k)
                    cb_out = callback_on_step_end(self, i, t, cb_kw)
                    latents = cb_out.pop("latents", latents)

                if i == len(timesteps2) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # ── decode ──
        if output_type == "latent":
            image = latents
        else:
            latents = latents.to(self.vae.dtype)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)
        return ZImagePipelineOutput(images=image)
