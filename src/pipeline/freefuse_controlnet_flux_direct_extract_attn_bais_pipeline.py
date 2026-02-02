# Copyright 2025 Black Forest Labs and The HuggingFace Team. All rights reserved.
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
FreeFuse Flux Pipeline with Direct Concept Extraction.

This pipeline extracts concept embeddings directly from encoder_hidden_states
using position maps, avoiding the token symmetry problem caused by separate encoding."""

import inspect
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
    T5TokenizerFast,
)

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FluxIPAdapterMixin, FluxLoraLoaderMixin, FromSingleFileMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL
from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel, FluxMultiControlNetModel
from src.models.freefuse_transformer_flux import FluxTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


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


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
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


class FreeFuseFluxDirectExtractAttnBaisPipeline(
    DiffusionPipeline,
    FluxLoraLoaderMixin,
    FromSingleFileMixin,
    TextualInversionLoaderMixin,
    FluxIPAdapterMixin,
):
    """
    FreeFuse Flux Pipeline with Direct Concept Extraction.
    
    This pipeline extracts concept embeddings directly from encoder_hidden_states
    using position maps (freefuse_token_pos_maps), avoiding the token symmetry problem
    caused by separate encoding.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->controlnet->transformer->vae"
    _optional_components = ["image_encoder", "feature_extractor", "controlnet"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "control_image"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
        controlnet: Union[
            FluxControlNetModel, List[FluxControlNetModel], Tuple[FluxControlNetModel], FluxMultiControlNetModel
        ] = None,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
    ):
        super().__init__()

        if controlnet is not None and isinstance(controlnet, (list, tuple)):
            controlnet = FluxMultiControlNetModel(controlnet)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
            controlnet=controlnet,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = 128

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer_2)

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]

        dtype = self.text_encoder_2.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False)

        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        lora_scale: Optional[float] = None,
    ):
        device = device or self._execution_device

        if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
            self._lora_scale = lora_scale

            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        if self.text_encoder is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

        return prompt_embeds, pooled_prompt_embeds, text_ids

    def check_inputs(
        self,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}. Dimensions will be resized accordingly"
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )
        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}")

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents

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

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

        latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

        return latents, latent_image_ids

    # Copied from diffusers.pipelines.controlnet_sd3.pipeline_stable_diffusion_3_controlnet.StableDiffusion3ControlNetPipeline.prepare_image
    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        if isinstance(image, torch.Tensor):
            pass
        else:
            image = self.image_processor.preprocess(image, height=height, width=width)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    def _construct_attention_bias(
        self,
        lora_masks: Dict[str, torch.Tensor],
        freefuse_token_pos_maps: Dict[str, List[List[int]]],
        txt_seq_len: int,
        img_seq_len: int,
        bias_scale: float = 5.0,
        positive_bias_scale: float = 1.0,
        bidirectional: bool = True,
        use_positive_bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> torch.Tensor:
        """
        Construct soft attention bias matrix to encourage image tokens to attend to their
        corresponding LoRA's text tokens and discourage attention to other LoRAs' text tokens.
        
        Args:
            lora_masks: Dict mapping lora_name -> (B, img_seq_len) binary mask indicating
                        which image positions belong to this LoRA
            freefuse_token_pos_maps: Dict mapping lora_name -> [[token positions in prompt], ...]
                                     Token positions in the T5 text embedding
            txt_seq_len: Length of text sequence
            img_seq_len: Length of image sequence (H*W after packing)
            bias_scale: Strength of the bias (larger = stronger effect)
            bidirectional: If True, also apply bias for text->image direction
            use_positive_bias: If True, also add positive bias for same-LoRA attention pairs,
                              in addition to negative bias for cross-LoRA pairs. Default True.
            device: Device for the output tensor
            dtype: Data type for the output tensor
            
        Returns:
            attention_bias: (B, txt_seq_len + img_seq_len, txt_seq_len + img_seq_len)
                           Soft bias values: positive for same-LoRA pairs (if use_positive_bias),
                           negative for cross-LoRA pairs, 0 for neutral pairs
        """
        # Get batch size from first mask
        batch_size = next(iter(lora_masks.values())).shape[0]
        total_seq_len = txt_seq_len + img_seq_len
        
        # Initialize bias as zeros (no bias = attend freely)
        attention_bias = torch.zeros(
            batch_size, total_seq_len, total_seq_len,
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
        for lora_name, img_mask in lora_masks.items():
            if lora_name not in lora_name_to_idx:
                continue
            lora_idx = lora_name_to_idx[lora_name]
            
            # img_mask: (B, img_seq_len) - 1 where this LoRA is active
            # We want: for image positions in this LoRA's region,
            # add negative bias to text tokens belonging to OTHER LoRAs
            
            # Create mask for text tokens belonging to other LoRAs (not this one, not shared)
            other_lora_text_mask = (text_token_to_lora != lora_idx) & (text_token_to_lora != -1)
            other_lora_text_mask = other_lora_text_mask.float()  # (txt_seq_len,)
            
            # Image->Text bias: for each image position in this LoRA's mask,
            # suppress attention to other LoRAs' text tokens
            # attention_bias shape: (B, total_seq_len, total_seq_len)
            # Query indices: txt_seq_len to txt_seq_len+img_seq_len (image positions)
            # Key indices: 0 to txt_seq_len (text positions)
            
            # img_mask: (B, img_seq_len)
            # other_lora_text_mask: (txt_seq_len,)
            # We want: bias[b, txt_seq_len+i, j] -= scale * img_mask[b, i] * other_lora_text_mask[j]
            
            # Outer product: (B, img_seq_len, 1) * (1, 1, txt_seq_len) -> (B, img_seq_len, txt_seq_len)
            img_to_txt_bias = img_mask.unsqueeze(-1) * other_lora_text_mask.unsqueeze(0).unsqueeze(0)
            img_to_txt_bias = img_to_txt_bias * (-bias_scale)
            
            # Add to attention_bias at image->text positions
            attention_bias[:, txt_seq_len:, :txt_seq_len] += img_to_txt_bias
            
            # Positive bias: encourage this LoRA's image region to attend to this LoRA's text tokens
            if use_positive_bias:
                # this_lora_text_mask: (txt_seq_len,) - 1 for tokens belonging to this LoRA
                this_lora_text_mask = (text_token_to_lora == lora_idx).float()
                
                # img_to_txt_positive_bias[b, i, j] = img_mask[b, i] * this_lora_text_mask[j] * (+scale)
                img_to_txt_positive_bias = img_mask.unsqueeze(-1) * this_lora_text_mask.unsqueeze(0).unsqueeze(0)
                img_to_txt_positive_bias = img_to_txt_positive_bias * positive_bias_scale  # positive!
                
                attention_bias[:, txt_seq_len:, :txt_seq_len] += img_to_txt_positive_bias
            
            if bidirectional:
                # Text->Image bias: for text tokens belonging to this LoRA,
                # suppress attention to image positions NOT in this LoRA's mask
                
                # this_lora_text_mask: (txt_seq_len,) - 1 for tokens belonging to this LoRA
                this_lora_text_mask = (text_token_to_lora == lora_idx).float()
                
                # not_this_lora_img_mask: (B, img_seq_len) - 1 for positions NOT in this LoRA
                not_this_lora_img_mask = 1.0 - img_mask
                
                # txt_to_img_bias[b, j, i] = this_lora_text_mask[j] * not_this_lora_img_mask[b, i] * (-scale)
                # Shape: (1, txt_seq_len, 1) * (B, 1, img_seq_len) -> (B, txt_seq_len, img_seq_len)
                txt_to_img_bias = this_lora_text_mask.unsqueeze(0).unsqueeze(-1) * not_this_lora_img_mask.unsqueeze(1)
                txt_to_img_bias = txt_to_img_bias * (-bias_scale)
                
                # Add to attention_bias at text->image positions
                attention_bias[:, :txt_seq_len, txt_seq_len:] += txt_to_img_bias
                
                # Positive bias: encourage this LoRA's text tokens to attend to this LoRA's image region
                if use_positive_bias:
                    # txt_to_img_positive_bias[b, j, i] = this_lora_text_mask[j] * img_mask[b, i] * (+scale)
                    txt_to_img_positive_bias = this_lora_text_mask.unsqueeze(0).unsqueeze(-1) * img_mask.unsqueeze(1)
                    txt_to_img_positive_bias = txt_to_img_positive_bias * positive_bias_scale  # positive!
                    
                    attention_bias[:, :txt_seq_len, txt_seq_len:] += txt_to_img_positive_bias
        
        return attention_bias

    def stabilized_balanced_argmax(self, logits, h, w, target_count=None, max_iter=15, 
                                  lr=0.01,           
                                  gravity_weight=0.00004, 
                                  spatial_weight=0.00004,
                                  momentum=0.2,
                                  centroid_margin=0.0,
                                  border_penalty=0.0,
                                  anisotropy=1.1,
                                  debug=False
                                  ):
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
        running_probs = linear_normalize(logits, dim=1)
        
        # 计算 logit 尺度用于自适应 lr
        logit_range = (logits.max() - logits.min()).item()
        logit_scale = max(logit_range, 1e-4)
        effective_lr = lr * logit_scale
        max_bias = logit_scale * 10.0
        
        if debug:
            print(f"\n[ArgmaxV4 Debug] Start. B={B}, C={C}, N={N}, Target={target_count:.1f}")
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

    def morphological_clean_mask(self, mask, h, w, opening_kernel_size=3, closing_kernel_size=3):
        """
        Clean a binary mask using morphological operations.
        
        This function applies:
        1. Opening (erosion + dilation): removes small foreground noise (isolated FG pixels)
        2. Closing (dilation + erosion): fills small holes in foreground (isolated BG pixels in FG region)
        
        The order is important: Opening first to remove noise, then Closing to fill holes.
        
        Args:
            mask: (B, N) binary mask, 1 for foreground, 0 for background
            h, w: spatial dimensions (N = h * w)
            opening_kernel_size: kernel size for opening operation (removes FG noise)
            closing_kernel_size: kernel size for closing operation (fills BG holes)
            
        Returns:
            cleaned_mask: (B, N) cleaned binary mask
        """
        B = mask.shape[0]
        device = mask.device
        dtype = mask.dtype
        
        # Reshape to 2D: (B, 1, H, W)
        mask_2d = mask.view(B, 1, h, w)
        
        # Morphological operations using max_pool2d
        # Dilation: max_pool2d (expands white regions)
        # Erosion: -max_pool2d(-x) which is equivalent to min_pool (shrinks white regions)
        
        def dilate(x, kernel_size):
            padding = kernel_size // 2
            out = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)
            # Ensure output size matches input size (handle even kernel size issues)
            if out.shape[-2:] != x.shape[-2:]:
                out = F.interpolate(out, size=x.shape[-2:], mode='nearest')
            return out
        
        def erode(x, kernel_size):
            padding = kernel_size // 2
            # Erosion = invert -> dilate -> invert
            # For binary mask: erode = 1 - max_pool(1 - x)
            out = 1.0 - F.max_pool2d(1.0 - x, kernel_size=kernel_size, stride=1, padding=padding)
            if out.shape[-2:] != x.shape[-2:]:
                out = F.interpolate(out, size=x.shape[-2:], mode='nearest')
            return out
        
        # Step 1: Opening = Erosion + Dilation
        # Effect: Removes small isolated foreground pixels (noise)
        if opening_kernel_size > 1:
            opened = erode(mask_2d, opening_kernel_size)
            opened = dilate(opened, opening_kernel_size)
        else:
            opened = mask_2d
        
        # Step 2: Closing = Dilation + Erosion
        # Effect: Fills small holes in foreground
        if closing_kernel_size > 1:
            closed = dilate(opened, closing_kernel_size)
            closed = erode(closed, closing_kernel_size)
        else:
            closed = opened
        
        # Reshape back to (B, N)
        cleaned_mask = closed.view(B, -1)
        
        return cleaned_mask

    @torch.no_grad()
    def __call__(
        self,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 35,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        # ControlNet params
        control_image: PipelineImageInput = None,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        control_mode: Optional[Union[int, List[int]]] = None,
        # FreeFuse params
        concept_embeds_map: Optional[Dict[str, torch.FloatTensor]] = None,
        aggreate_lora_score_step: Optional[int] = 6,
        voting_after_step: Optional[int] = 6,
        debug_save_path: Optional[str] = None,
        # Attention bias params
        use_attention_bias: bool = True,
        attention_bias_scale: float = 3.0,
        attention_bias_positive_scale: float = 1.0,
        attention_bias_bidirectional: bool = True,
        attention_bias_positive: bool = True,
        attention_bias_blocks: Optional[List[str]] = None,
    ):
        """
        
        Args:
            use_attention_bias: Whether to apply attention bias in Phase 2 to constrain
                              text-image attention based on LoRA masks. Default True.
            attention_bias_scale: Strength of the NEGATIVE bias for cross-LoRA attention.
                                Larger values = stronger suppression. Default 5.0.
            attention_bias_positive_scale: Strength of the POSITIVE bias for same-LoRA attention.
                                         Should be much smaller than negative scale to avoid
                                         over-focusing on LoRA tokens. Default 0.5.
            attention_bias_bidirectional: If True, apply bias for both image->text and text->image
                                        directions. If False, only apply image->text bias. Default True.
            attention_bias_positive: If True, add positive bias for same-LoRA attention pairs
                                   (encouraging attention), in addition to negative bias for 
                                   cross-LoRA pairs (discouraging attention). Default True.
            attention_bias_blocks: List of block names to apply attention bias (e.g., 
                                 ["transformer_blocks.0", "transformer_blocks.18"]).
                                 If None, apply to all blocks. Supports preset strings:
                                 - "double_stream_only": only transformer_blocks (not single_transformer_blocks)
                                 - "single_stream_only": only single_transformer_blocks
                                 - "last_half_double": last half of transformer_blocks
                                 Default None (all blocks).
            (Other args same as FreeFuseFluxPipeline)
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs
        self.check_inputs(
            height,
            width,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        has_neg_prompt = negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None 
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                negative_text_ids,
            ) = self.encode_prompt(
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        initial_latents = latents.clone()

        # 5. Prepare control image (ControlNet)
        controlnet_blocks_repeat = False
        if self.controlnet is not None and control_image is not None:
            # Handle control_guidance_start/end as lists
            if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
                control_guidance_start = len(control_guidance_end) * [control_guidance_start]
            elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
                control_guidance_end = len(control_guidance_start) * [control_guidance_end]
            elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
                mult = len(self.controlnet.nets) if isinstance(self.controlnet, FluxMultiControlNetModel) else 1
                control_guidance_start, control_guidance_end = (
                    mult * [control_guidance_start],
                    mult * [control_guidance_end],
                )

            if isinstance(self.controlnet, FluxControlNetModel):
                control_image = self.prepare_image(
                    image=control_image,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=self.vae.dtype,
                )
                height_ctrl, width_ctrl = control_image.shape[-2:]

                # xlab controlnet has a input_hint_block and instantx controlnet does not
                controlnet_blocks_repeat = False if self.controlnet.input_hint_block is None else True
                if self.controlnet.input_hint_block is None:
                    # vae encode
                    control_image = retrieve_latents(self.vae.encode(control_image), generator=generator)
                    control_image = (control_image - self.vae.config.shift_factor) * self.vae.config.scaling_factor

                    # pack
                    height_control_image, width_control_image = control_image.shape[2:]
                    control_image = self._pack_latents(
                        control_image,
                        batch_size * num_images_per_prompt,
                        num_channels_latents,
                        height_control_image,
                        width_control_image,
                    )

                # Handle control_mode for single ControlNet
                if control_mode is not None:
                    if not isinstance(control_mode, int):
                        raise ValueError(" For `FluxControlNet`, `control_mode` should be an `int` or `None`")
                    control_mode = torch.tensor(control_mode).to(device, dtype=torch.long)
                    control_mode = control_mode.view(-1, 1).expand(control_image.shape[0], 1)

            elif isinstance(self.controlnet, FluxMultiControlNetModel):
                control_images = []
                controlnet_blocks_repeat = False if self.controlnet.nets[0].input_hint_block is None else True
                for i, control_image_ in enumerate(control_image):
                    control_image_ = self.prepare_image(
                        image=control_image_,
                        width=width,
                        height=height,
                        batch_size=batch_size * num_images_per_prompt,
                        num_images_per_prompt=num_images_per_prompt,
                        device=device,
                        dtype=self.vae.dtype,
                    )
                    height_ctrl, width_ctrl = control_image_.shape[-2:]

                    if self.controlnet.nets[0].input_hint_block is None:
                        # vae encode
                        control_image_ = retrieve_latents(self.vae.encode(control_image_), generator=generator)
                        control_image_ = (control_image_ - self.vae.config.shift_factor) * self.vae.config.scaling_factor

                        # pack
                        height_control_image, width_control_image = control_image_.shape[2:]
                        control_image_ = self._pack_latents(
                            control_image_,
                            batch_size * num_images_per_prompt,
                            num_channels_latents,
                            height_control_image,
                            width_control_image,
                        )
                    control_images.append(control_image_)

                control_image = control_images

                # Handle control_mode for Multi-ControlNet
                if isinstance(control_mode, list) and len(control_mode) != len(control_image):
                    raise ValueError(
                        "For Multi-ControlNet, `control_mode` must be a list of the same "
                        + " length as the number of controlnets (control images) specified"
                    )
                if not isinstance(control_mode, list):
                    control_mode = [control_mode] * len(control_image)
                # set control mode
                control_modes = []
                for cmode in control_mode:
                    if cmode is None:
                        cmode = -1
                    cmode_tensor = torch.tensor(cmode).expand(control_images[0].shape[0]).to(device, dtype=torch.long)
                    control_modes.append(cmode_tensor)
                control_mode = control_modes

        # 6. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 7. Create tensor stating which controlnets to keep
        controlnet_keep = []
        if self.controlnet is not None and control_image is not None:
            for i in range(len(timesteps)):
                keeps = [
                    1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                    for s, e in zip(control_guidance_start, control_guidance_end)
                ]
                controlnet_keep.append(keeps[0] if isinstance(self.controlnet, FluxControlNetModel) else keeps)

        # Handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        image_embeds = None
        negative_image_embeds = None

        # 8. Denoising loop helper functions
        def _run_denoising_step(
            latents,
            t,
            timestep,
            guidance,
            pooled_prompt_embeds,
            prompt_embeds,
            text_ids,
            latent_image_ids,
            do_true_cfg,
            negative_pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_text_ids,
            negative_image_embeds,
            true_cfg_scale,
            collect_masks=False,
            concept_embeds_map=None,
            # ControlNet params
            control_image=None,
            control_mode=None,
            controlnet_cond_scale=1.0,
            controlnet_blocks_repeat=False,
        ):
            """Run a single denoising step, optionally collecting concept similarity maps."""
            # Call ControlNet if available
            controlnet_block_samples = None
            controlnet_single_block_samples = None
            
            if self.controlnet is not None and control_image is not None:
                # Determine guidance for ControlNet
                if isinstance(self.controlnet, FluxMultiControlNetModel):
                    use_guidance = self.controlnet.nets[0].config.guidance_embeds
                else:
                    use_guidance = self.controlnet.config.guidance_embeds

                controlnet_guidance = torch.tensor([guidance_scale], device=device) if use_guidance else None
                controlnet_guidance = controlnet_guidance.expand(latents.shape[0]) if controlnet_guidance is not None else None

                controlnet_block_samples, controlnet_single_block_samples = self.controlnet(
                    hidden_states=latents,
                    controlnet_cond=control_image,
                    controlnet_mode=control_mode,
                    conditioning_scale=controlnet_cond_scale,
                    timestep=timestep / 1000,
                    guidance=controlnet_guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )

            transformer_kwargs = dict(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
                controlnet_block_samples=controlnet_block_samples,
                controlnet_single_block_samples=controlnet_single_block_samples,
                controlnet_blocks_repeat=controlnet_blocks_repeat,
            )
            # Also support DirectExtract mode which uses freefuse_token_pos_maps instead
            has_concept_embeds = concept_embeds_map is not None
            has_token_pos_maps = self.joint_attention_kwargs is not None and 'freefuse_token_pos_maps' in self.joint_attention_kwargs
            if collect_masks and (has_concept_embeds or has_token_pos_maps):
                transformer_kwargs["concept_embeds_map"] = concept_embeds_map
                self.transformer.attn_processors['transformer_blocks.18.attn.processor'].cal_concept_sim_map = True
            else:
                transformer_kwargs["concept_embeds_map"] = concept_embeds_map
                self.transformer.attn_processors['transformer_blocks.18.attn.processor'].cal_concept_sim_map = False

            with self.transformer.cache_context("cond"):
                noise_pred, concept_sim_maps = self.transformer(**transformer_kwargs)

            if do_true_cfg:
                if negative_image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds

                with self.transformer.cache_context("uncond"):
                    neg_noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=None,
                        return_dict=False,
                        controlnet_block_samples=controlnet_block_samples,
                        controlnet_single_block_samples=controlnet_single_block_samples,
                        controlnet_blocks_repeat=controlnet_blocks_repeat,
                    )[0]
                noise_pred = noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

            return noise_pred, concept_sim_maps

        def _process_masks_and_debug(
            concept_sim_maps,
            text_ids,
            step_idx,
            debug_save_path,
            height,
            width,
            latents=None,
            noise_pred=None,
            t=None,
            prompt_embeds=None,
            eos_bg_scale=0.95,
        ):
            """Process concept similarity maps into LoRA masks and optionally save debug visualizations.
            
            Args:
                eos_bg_scale: Scaling factor for EOS/user-defined background channel. Higher values 
                             make it easier for background to win argmax (more aggressive exclusion).
            """
            if concept_sim_maps is None:
                return None, None

            lora_masks = {}
            
            # === Background Exclusion ===
            # Priority: user-defined __bg__ > __eos__ > average fallback
            has_bg = '__bg__' in concept_sim_maps
            has_eos = '__eos__' in concept_sim_maps
            
            if has_bg:
                # Method 1: User-defined background tokens (highest priority)
                bg_sim_map = concept_sim_maps.pop('__bg__')  # B, N, 1
                
                # Stack remaining concept sim_maps
                concept_sim_maps_list = list(concept_sim_maps.values())
                concept_sim_maps_tensor = torch.stack(concept_sim_maps_list, dim=1)  # B, C, N, 1
                B, C, N, _ = concept_sim_maps_tensor.shape
                sim_map_squeezed = concept_sim_maps_tensor.squeeze(dim=-1)  # B, C, N
                
                # Use user-defined background as background channel (apply scaling factor)
                bg_channel = bg_sim_map.squeeze(-1) * eos_bg_scale  # B, N
                background_channel = bg_channel.unsqueeze(1)  # B, 1, N
            elif has_eos:
                # Method 2: EOS token
                eos_sim_map = concept_sim_maps.pop('__eos__')  # B, N, 1
                
                # Stack remaining concept sim_maps
                concept_sim_maps_list = list(concept_sim_maps.values())
                concept_sim_maps_tensor = torch.stack(concept_sim_maps_list, dim=1)  # B, C, N, 1
                B, C, N, _ = concept_sim_maps_tensor.shape
                sim_map_squeezed = concept_sim_maps_tensor.squeeze(dim=-1)  # B, C, N
                
                # Use EOS as background channel (apply scaling factor)
                eos_channel = eos_sim_map.squeeze(-1) * eos_bg_scale  # B, N
                background_channel = eos_channel.unsqueeze(1)  # B, 1, N
            else:
                # Fallback: No background exclusion (old behavior)
                concept_sim_maps_tensor = torch.stack(list(concept_sim_maps.values()), dim=1)  # B, C, N, 1
                B, C, N, _ = concept_sim_maps_tensor.shape
                sim_map_squeezed = concept_sim_maps_tensor.squeeze(dim=-1)  # B, C, N
                
                # No background channel - just use balanced argmax directly
                enable_debug = False
                max_indices = self.stabilized_balanced_argmax(
                    sim_map_squeezed, 
                    height // 16, width // 16,
                    debug=enable_debug
                )  # B, N
                
                for idx, lora_name in enumerate(concept_sim_maps.keys()):
                    lora_masks[lora_name] = (max_indices == idx).float()  # B, N
                
                self.transformer.set_freefuse_masks(lora_masks, text_length=text_ids.shape[0])
                print('freefuse masks seted (no background exclusion), running into regenerate.')
                
                # Construct attention bias if enabled
                constructed_attention_bias = None
                if use_attention_bias and self.joint_attention_kwargs is not None:
                    freefuse_token_pos_maps = self.joint_attention_kwargs.get('freefuse_token_pos_maps', None)
                    if freefuse_token_pos_maps is not None and prompt_embeds is not None:
                        txt_seq_len = prompt_embeds.shape[1]
                        img_seq_len = (height // 16) * (width // 16)
                        
                        constructed_attention_bias = self._construct_attention_bias(
                            lora_masks=lora_masks,
                            freefuse_token_pos_maps=freefuse_token_pos_maps,
                            txt_seq_len=txt_seq_len,
                            img_seq_len=img_seq_len,
                            bias_scale=attention_bias_scale,
                            positive_bias_scale=attention_bias_positive_scale,
                            bidirectional=attention_bias_bidirectional,
                            use_positive_bias=attention_bias_positive,
                            device=prompt_embeds.device,
                            dtype=prompt_embeds.dtype,
                        )
                        print(f'Attention bias constructed with scale={attention_bias_scale}, positive_scale={attention_bias_positive_scale}, bidirectional={attention_bias_bidirectional}, positive={attention_bias_positive}')
                
                if debug_save_path is not None:
                    os.makedirs(debug_save_path, exist_ok=True)
                    latent_h = height // 16
                    latent_w = width // 16
                    
                    for lora_name, mask in lora_masks.items():
                        mask_2d = mask[0].view(latent_h, latent_w).cpu().float().numpy()
                        fig, ax = plt.subplots(figsize=(8, 8))
                        im = ax.imshow(mask_2d, cmap='gray', vmin=0, vmax=1)
                        ax.set_title(f'LoRA Mask: {lora_name} (step {step_idx})')
                        plt.colorbar(im, ax=ax)
                        plt.savefig(os.path.join(debug_save_path, f'lora_mask_{lora_name}_step{step_idx}.png'), dpi=150, bbox_inches='tight')
                        plt.close(fig)
                    
                    for concept_name, sim_map in concept_sim_maps.items():
                        sim_map_2d = sim_map[0].view(latent_h, latent_w).cpu().float().numpy()
                        fig, ax = plt.subplots(figsize=(8, 8))
                        im = ax.imshow(sim_map_2d, cmap='viridis')
                        ax.set_title(f'Concept Sim Map: {concept_name} (step {step_idx})')
                        plt.colorbar(im, ax=ax)
                        plt.savefig(os.path.join(debug_save_path, f'concept_sim_map_{concept_name}_step{step_idx}.png'), dpi=150, bbox_inches='tight')
                        plt.close(fig)
                
                return lora_masks, constructed_attention_bias
            
            # === With Background Exclusion ===
            # Concatenate background channel with concept channels: B, C+1, N
            sim_map_with_bg = torch.cat([sim_map_squeezed, background_channel], dim=1)  # B, C+1, N
            
            # === Step 1: Raw Argmax for comparison visualization ===
            # Do direct argmax on the extended tensor to determine background (raw, with holes)
            # Index C (the last one) corresponds to background
            bg_argmax_indices = sim_map_with_bg.argmax(dim=1)  # B, N
            raw_not_background_mask = (bg_argmax_indices != C).float()  # B, N (raw, with holes)
            
            # === Step 2: Morphological Cleaning for FG/BG separation ===
            # Use morphological operations to clean the raw mask
            not_background_mask = self.morphological_clean_mask(
                raw_not_background_mask,
                h=height // 16, w=width // 16,
                opening_kernel_size=2,  # Use 2x2 for mild de-noising
                closing_kernel_size=2   # Use 2x2 for mild hole-filling
            )  # B, N (cleaned)
            not_background_mask = not_background_mask.bool()  # Convert to bool for consistency
            
            # Now use balanced argmax on original sim_map for concept assignment
            enable_debug = False
            max_indices = self.stabilized_balanced_argmax(
                sim_map_squeezed, 
                height // 16, width // 16,
                debug=enable_debug
            )  # B, N
            
            # Also store top_50_masks for visualization
            top_50_masks = torch.zeros(B, C, N, dtype=torch.bool, device=sim_map_squeezed.device)
            for c in range(C):
                top_50_masks[:, c, :] = (max_indices == c)

            for idx, lora_name in enumerate(concept_sim_maps.keys()):
                # Mask from balanced argmax, intersected with foreground mask to exclude background
                argmax_mask = (max_indices == idx).float()  # B, N
                lora_masks[lora_name] = argmax_mask * not_background_mask.float()  # B, N
            
            # Put background sim_maps back into concept_sim_maps for visualization
            if has_bg:
                concept_sim_maps['__bg__'] = bg_sim_map
            elif has_eos:
                concept_sim_maps['__eos__'] = eos_sim_map

            self.transformer.set_freefuse_masks(lora_masks, text_length=text_ids.shape[0])
            print('freefuse masks seted (with background exclusion), running into regenerate.')

            # Construct attention bias if enabled
            constructed_attention_bias = None
            if use_attention_bias and self.joint_attention_kwargs is not None:
                freefuse_token_pos_maps = self.joint_attention_kwargs.get('freefuse_token_pos_maps', None)
                if freefuse_token_pos_maps is not None and prompt_embeds is not None:
                    txt_seq_len = prompt_embeds.shape[1]
                    img_seq_len = (height // 16) * (width // 16)
                    
                    constructed_attention_bias = self._construct_attention_bias(
                        lora_masks=lora_masks,
                        freefuse_token_pos_maps=freefuse_token_pos_maps,
                        txt_seq_len=txt_seq_len,
                        img_seq_len=img_seq_len,
                        bias_scale=attention_bias_scale,
                        positive_bias_scale=attention_bias_positive_scale,
                        bidirectional=attention_bias_bidirectional,
                        use_positive_bias=attention_bias_positive,
                        device=prompt_embeds.device,
                        dtype=prompt_embeds.dtype,
                    )
                    print(f'Attention bias constructed with scale={attention_bias_scale}, positive_scale={attention_bias_positive_scale}, bidirectional={attention_bias_bidirectional}, positive={attention_bias_positive}')

            if debug_save_path is not None:
                os.makedirs(debug_save_path, exist_ok=True)
                latent_h = height // 16
                latent_w = width // 16

                # === Visualize Background Exclusion Process ===
                # 1. Visualize foreground mask (not_background_mask)
                fg_mask_2d = not_background_mask[0].view(latent_h, latent_w).cpu().float().numpy()
                fig, ax = plt.subplots(figsize=(8, 8))
                im = ax.imshow(fg_mask_2d, cmap='RdYlGn', vmin=0, vmax=1)
                ax.set_title(f'Foreground Mask (step {step_idx})\nGreen=Foreground, Red=Background')
                plt.colorbar(im, ax=ax)
                plt.savefig(os.path.join(debug_save_path, f'foreground_mask_step{step_idx}.png'), dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                # 2. Visualize background mask (inverse of foreground)
                bg_mask_2d = (~not_background_mask[0]).view(latent_h, latent_w).cpu().float().numpy()
                fig, ax = plt.subplots(figsize=(8, 8))
                im = ax.imshow(bg_mask_2d, cmap='Greys', vmin=0, vmax=1)
                ax.set_title(f'Background Mask (step {step_idx})\nWhite=Background, Black=Foreground')
                plt.colorbar(im, ax=ax)
                plt.savefig(os.path.join(debug_save_path, f'background_mask_step{step_idx}.png'), dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                # 2.5. === Compare Raw Argmax vs Morphologically Cleaned FG/BG Masks ===
                raw_fg_mask_2d = raw_not_background_mask[0].view(latent_h, latent_w).cpu().float().numpy()
                cleaned_fg_mask_2d = fg_mask_2d
                
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                im0 = axes[0].imshow(raw_fg_mask_2d, cmap='RdYlGn', vmin=0, vmax=1)
                axes[0].set_title(f'Raw Argmax FG Mask\n(with noise/holes)')
                plt.colorbar(im0, ax=axes[0], shrink=0.8)
                
                im1 = axes[1].imshow(cleaned_fg_mask_2d, cmap='RdYlGn', vmin=0, vmax=1)
                axes[1].set_title(f'Morphological Cleaned FG Mask\n(opening + closing)')
                plt.colorbar(im1, ax=axes[1], shrink=0.8)
                
                diff_mask = cleaned_fg_mask_2d - raw_fg_mask_2d
                im2 = axes[2].imshow(diff_mask, cmap='coolwarm', vmin=-1, vmax=1)
                axes[2].set_title(f'Difference\n(Blue=Removed FG, Red=Added FG)')
                plt.colorbar(im2, ax=axes[2], shrink=0.8)
                
                fig.suptitle(f'Morphological Cleaning Effect (step {step_idx})', fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(debug_save_path, f'raw_vs_morphological_fg_mask_step{step_idx}.png'), dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                # 3. Visualize top 50% masks for each concept (skip __eos__ and __bg__)
                concept_names = [k for k in concept_sim_maps.keys() if k not in ('__eos__', '__bg__')]
                for idx, lora_name in enumerate(concept_names):
                    top50_mask_2d = top_50_masks[0, idx].view(latent_h, latent_w).cpu().float().numpy()
                    fig, ax = plt.subplots(figsize=(8, 8))
                    im = ax.imshow(top50_mask_2d, cmap='Blues', vmin=0, vmax=1)
                    ax.set_title(f'Top 50% Mask: {lora_name} (step {step_idx})\nBlue=Top 50%')
                    plt.colorbar(im, ax=ax)
                    plt.savefig(os.path.join(debug_save_path, f'top50_mask_{lora_name}_step{step_idx}.png'), dpi=150, bbox_inches='tight')
                    plt.close(fig)
                
                # 4. Visualize argmax mask before background exclusion (skip __eos__ and __bg__)
                for idx, lora_name in enumerate(concept_names):
                    argmax_before_mask = (max_indices == idx).float()
                    argmax_before_2d = argmax_before_mask[0].view(latent_h, latent_w).cpu().float().numpy()
                    fig, ax = plt.subplots(figsize=(8, 8))
                    im = ax.imshow(argmax_before_2d, cmap='Oranges', vmin=0, vmax=1)
                    ax.set_title(f'Argmax Mask (Before BG Exclusion): {lora_name} (step {step_idx})')
                    plt.colorbar(im, ax=ax)
                    plt.savefig(os.path.join(debug_save_path, f'argmax_before_bg_{lora_name}_step{step_idx}.png'), dpi=150, bbox_inches='tight')
                    plt.close(fig)

                # Visualize lora_masks - binary 0-1 masks
                for lora_name, mask in lora_masks.items():
                    mask_2d = mask[0].view(latent_h, latent_w).cpu().float().numpy()
                    
                    # 1. Original version with annotations
                    fig, ax = plt.subplots(figsize=(8, 8))
                    im = ax.imshow(mask_2d, cmap='gray', vmin=0, vmax=1)
                    ax.set_title(f'LoRA Mask: {lora_name} (step {step_idx})')
                    plt.colorbar(im, ax=ax)
                    plt.savefig(os.path.join(debug_save_path, f'lora_mask_{lora_name}_step{step_idx}.png'), dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    
                    # 2. Clean version for PPT (no title, no colorbar, no axes) - grayscale
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(mask_2d, cmap='gray', vmin=0, vmax=1)
                    ax.axis('off')
                    plt.savefig(os.path.join(debug_save_path, f'lora_mask_{lora_name}_step{step_idx}_clean.png'), 
                                dpi=150, bbox_inches='tight', pad_inches=0)
                    plt.close(fig)

                # Visualize concept_sim_maps using 'viridis' colormap (with annotations)
                for concept_name, sim_map in concept_sim_maps.items():
                    sim_map_2d = sim_map[0].view(latent_h, latent_w).cpu().float().numpy()
                    
                    # 1. Original version with annotations (viridis)
                    fig, ax = plt.subplots(figsize=(8, 8))
                    im = ax.imshow(sim_map_2d, cmap='viridis')
                    ax.set_title(f'Concept Sim Map: {concept_name} (step {step_idx})')
                    plt.colorbar(im, ax=ax)
                    plt.savefig(os.path.join(debug_save_path, f'concept_sim_map_{concept_name}_step{step_idx}.png'), dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    
                    # 2. Clean version for PPT (no title, no colorbar, no axes) - viridis
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(sim_map_2d, cmap='viridis')
                    ax.axis('off')
                    plt.savefig(os.path.join(debug_save_path, f'concept_sim_map_{concept_name}_step{step_idx}_clean_viridis.png'), 
                                dpi=150, bbox_inches='tight', pad_inches=0)
                    plt.close(fig)
                    
                    # 3. Clean version for PPT - plasma colormap (yellow-orange-purple gradient)
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(sim_map_2d, cmap='plasma')
                    ax.axis('off')
                    plt.savefig(os.path.join(debug_save_path, f'concept_sim_map_{concept_name}_step{step_idx}_clean_plasma.png'), 
                                dpi=150, bbox_inches='tight', pad_inches=0)
                    plt.close(fig)

                # Visualize predicted x0 (final image prediction) from current step
                if latents is not None and noise_pred is not None and t is not None:
                    step_index = self.scheduler.step_index if self.scheduler.step_index is not None else 0
                    sigma = self.scheduler.sigmas[step_index]
                    predicted_x0 = latents - sigma * noise_pred
                    unpacked_latents = self._unpack_latents(predicted_x0, height, width, self.vae_scale_factor)
                    unpacked_latents = (unpacked_latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                    with torch.no_grad():
                        decoded_image = self.vae.decode(unpacked_latents, return_dict=False)[0]
                    decoded_image = decoded_image.clamp(-1, 1)
                    decoded_image = (decoded_image + 1) / 2
                    decoded_image = decoded_image[0].permute(1, 2, 0).cpu().float().numpy()
                    
                    fig, ax = plt.subplots(figsize=(10, 10))
                    ax.imshow(decoded_image)
                    ax.set_title(f'Predicted x0 (step {step_idx}, sigma={sigma:.4f})')
                    ax.axis('off')
                    plt.savefig(os.path.join(debug_save_path, f'predicted_x0_step{step_idx}.png'), dpi=150, bbox_inches='tight')
                    plt.close(fig)
                
                # Visualize attention bias if constructed
                if constructed_attention_bias is not None:
                    bias_2d = constructed_attention_bias[0].cpu().float().numpy()
                    fig, ax = plt.subplots(figsize=(12, 12))
                    im = ax.imshow(bias_2d, cmap='RdBu', vmin=-attention_bias_scale, vmax=attention_bias_scale)
                    ax.set_title(f'Attention Bias Matrix (step {step_idx})')
                    ax.set_xlabel('Key positions (text | image)')
                    ax.set_ylabel('Query positions (text | image)')
                    # Add lines to separate text and image regions
                    txt_len = prompt_embeds.shape[1] if prompt_embeds is not None else 0
                    ax.axhline(y=txt_len - 0.5, color='white', linestyle='--', linewidth=1)
                    ax.axvline(x=txt_len - 0.5, color='white', linestyle='--', linewidth=1)
                    plt.colorbar(im, ax=ax)
                    plt.savefig(os.path.join(debug_save_path, f'attention_bias_step{step_idx}.png'), dpi=150, bbox_inches='tight')
                    plt.close(fig)

            return lora_masks, constructed_attention_bias

        def _scheduler_step_and_callbacks(
            latents,
            noise_pred,
            t,
            step_idx,
            num_warmup_steps,
            callback_on_step_end,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            progress_bar,
        ):
            """Apply scheduler step and handle callbacks."""
            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

            if callback_on_step_end is not None:
                callback_kwargs = {"latents": latents, "prompt_embeds": prompt_embeds}
                callback_outputs = callback_on_step_end(self, step_idx, t, callback_kwargs)
                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            if step_idx == len(timesteps) - 1 or ((step_idx + 1) > num_warmup_steps and (step_idx + 1) % self.scheduler.order == 0):
                progress_bar.update()

            if XLA_AVAILABLE:
                xm.mark_step()

            return latents, prompt_embeds

        def _run_denoising_loop(
            latents,
            timesteps,
            num_steps,
            guidance,
            pooled_prompt_embeds,
            prompt_embeds,
            text_ids,
            latent_image_ids,
            do_true_cfg,
            negative_pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_text_ids,
            negative_image_embeds,
            true_cfg_scale,
            num_warmup_steps,
            callback_on_step_end,
            callback_on_step_end_tensor_inputs,
            collect_masks=False,
            concept_embeds_map=None,
            debug_save_path=None,
            height=None,
            width=None,
            # ControlNet params
            control_image=None,
            control_mode=None,
            controlnet_keep=None,
            controlnet_conditioning_scale=1.0,
            controlnet_blocks_repeat=False,
        ):
            """Run the denoising loop for a given number of steps. Returns (latents, attention_bias) where attention_bias is None if not collecting masks."""
            constructed_attention_bias = None
            if collect_masks:
                prev_adapters = self.get_active_adapters()
                self.disable_lora()
            self.scheduler.set_begin_index(0)
            self.scheduler._step_index = None
            with self.progress_bar(total=num_steps) as progress_bar:
                for step_idx, t in enumerate(timesteps):
                    if not collect_masks:
                        self.enable_lora()
                    if self.interrupt:
                        continue

                    self._current_timestep = t
                    timestep = t.expand(latents.shape[0]).to(latents.dtype)

                    # Calculate controlnet conditioning scale for this step
                    if controlnet_keep is not None and len(controlnet_keep) > 0:
                        if isinstance(controlnet_keep[step_idx], list):
                            cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[step_idx])]
                        else:
                            cn_cond_scale = controlnet_conditioning_scale
                            if isinstance(cn_cond_scale, list):
                                cn_cond_scale = cn_cond_scale[0]
                            cond_scale = cn_cond_scale * controlnet_keep[step_idx]
                    else:
                        cond_scale = controlnet_conditioning_scale if isinstance(controlnet_conditioning_scale, (int, float)) else controlnet_conditioning_scale[0] if isinstance(controlnet_conditioning_scale, list) else 1.0

                    # Run denoising step
                    noise_pred, concept_sim_maps = _run_denoising_step(
                        latents=latents,
                        t=t,
                        timestep=timestep,
                        guidance=guidance,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        prompt_embeds=prompt_embeds,
                        text_ids=text_ids,
                        latent_image_ids=latent_image_ids,
                        do_true_cfg=do_true_cfg,
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        negative_text_ids=negative_text_ids,
                        negative_image_embeds=negative_image_embeds,
                        true_cfg_scale=true_cfg_scale,
                        collect_masks=collect_masks if step_idx == num_steps - 1 else False,
                        concept_embeds_map=concept_embeds_map,
                        # ControlNet params
                        control_image=control_image,
                        control_mode=control_mode,
                        controlnet_cond_scale=cond_scale,
                        controlnet_blocks_repeat=controlnet_blocks_repeat,
                    )

                    # Process masks if collecting
                    if collect_masks and concept_sim_maps is not None:
                        _, constructed_attention_bias = _process_masks_and_debug(
                            concept_sim_maps=concept_sim_maps,
                            text_ids=text_ids,
                            step_idx=step_idx,
                            debug_save_path=debug_save_path,
                            height=height,
                            width=width,
                            latents=latents,
                            noise_pred=noise_pred,
                            t=t,
                            prompt_embeds=prompt_embeds,
                        )

                        if step_idx == num_steps-1:
                            if collect_masks:
                                self.enable_lora()
                                # self.set_adapters(prev_adapters)
                            break

                    # Scheduler step and callbacks
                    latents, prompt_embeds = _scheduler_step_and_callbacks(
                        latents=latents,
                        noise_pred=noise_pred,
                        t=t,
                        step_idx=step_idx,
                        num_warmup_steps=num_warmup_steps,
                        callback_on_step_end=callback_on_step_end,
                        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                        prompt_embeds=prompt_embeds,
                        progress_bar=progress_bar,
                    )

            return latents, constructed_attention_bias

        # Phase 1: Collect concept similarity maps and set freefuse masks

        negative_text_ids = negative_text_ids if do_true_cfg else None
        _, phase1_attention_bias = _run_denoising_loop(
            latents=latents,
            timesteps=timesteps,
            num_steps=aggreate_lora_score_step,
            guidance=guidance,
            pooled_prompt_embeds=pooled_prompt_embeds,
            prompt_embeds=prompt_embeds,
            text_ids=text_ids,
            latent_image_ids=latent_image_ids,
            do_true_cfg=do_true_cfg,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_text_ids=negative_text_ids,
            negative_image_embeds=negative_image_embeds,
            true_cfg_scale=true_cfg_scale,
            num_warmup_steps=num_warmup_steps,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            collect_masks=True,
            concept_embeds_map=concept_embeds_map,
            debug_save_path=debug_save_path,
            height=height,
            width=width,
            # ControlNet params
            control_image=control_image,
            control_mode=control_mode,
            controlnet_keep=controlnet_keep,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            controlnet_blocks_repeat=controlnet_blocks_repeat,
        )

        # Prepare attention bias for Phase 2
        if use_attention_bias and phase1_attention_bias is not None:
            # Process attention_bias_blocks to determine which blocks should apply bias
            if attention_bias_blocks is not None:
                # Handle preset strings
                if isinstance(attention_bias_blocks, str):
                    if attention_bias_blocks == "double_stream_only":
                        attention_bias_blocks = [f"transformer_blocks.{i}" for i in range(19)]
                    elif attention_bias_blocks == "single_stream_only":
                        attention_bias_blocks = [f"single_transformer_blocks.{i}" for i in range(38)]
                    elif attention_bias_blocks == "last_half_double":
                        attention_bias_blocks = [f"transformer_blocks.{i}" for i in range(10, 19)]
                    else:
                        attention_bias_blocks = [attention_bias_blocks]  # Single block name
            
            # Add attention bias to joint_attention_kwargs for Phase 2
            if self._joint_attention_kwargs is None:
                self._joint_attention_kwargs = {}
            self._joint_attention_kwargs['attention_mask'] = phase1_attention_bias
            self._joint_attention_kwargs['attention_bias_blocks'] = attention_bias_blocks
            print(f"Attention bias added to joint_attention_kwargs for Phase 2 (blocks: {attention_bias_blocks or 'all'})")

        # Phase 2: Regenerate with freefuse masks applied
        latents, _ = _run_denoising_loop(
            latents=initial_latents,
            timesteps=timesteps,
            num_steps=num_inference_steps,
            guidance=guidance,
            pooled_prompt_embeds=pooled_prompt_embeds,
            prompt_embeds=prompt_embeds,
            text_ids=text_ids,
            latent_image_ids=latent_image_ids,
            do_true_cfg=do_true_cfg,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_text_ids=negative_text_ids,
            negative_image_embeds=negative_image_embeds,
            true_cfg_scale=true_cfg_scale,
            num_warmup_steps=num_warmup_steps,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            collect_masks=False,
            concept_embeds_map=None,
            debug_save_path=None,
            height=height,
            width=width,
            # ControlNet params
            control_image=control_image,
            control_mode=control_mode,
            controlnet_keep=controlnet_keep,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            controlnet_blocks_repeat=controlnet_blocks_repeat,
        )

        self._current_timestep = None

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)
