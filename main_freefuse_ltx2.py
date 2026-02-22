import os

import torch

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ltx2 import LTX2LatentUpsamplePipeline
from diffusers.pipelines.ltx2.export_utils import encode_video
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES

from src.pipeline.freefuse_ltx2_pipeline import FreeFuseLTX2Pipeline


def main():
    device = "cuda:0"
    dtype = torch.bfloat16
    width = 768
    height = 512
    num_frames = 121
    frame_rate = 24.0

    stage1_steps = 40
    stage2_steps = 3
    use_torchao_quantization = False
    torchao_group_size = 128

    pipeline_load_kwargs = {"torch_dtype": dtype}
    if use_torchao_quantization:
        try:
            from diffusers import PipelineQuantizationConfig, TorchAoConfig
            from torchao.quantization import Int8WeightOnlyConfig
        except Exception as exc:
            raise RuntimeError(
                "TorchAO quantization is enabled but required deps are unavailable. "
                "Install compatible `diffusers` and `torchao`, or set "
                "`use_torchao_quantization=False`."
            ) from exc

        pipeline_quant_config = PipelineQuantizationConfig(
            quant_mapping={
                "transformer": TorchAoConfig(
                    Int8WeightOnlyConfig(group_size=torchao_group_size)
                )
            }
        )
        pipeline_load_kwargs["quantization_config"] = pipeline_quant_config
        # pipeline_load_kwargs["device_map"] = "cuda"

    pipe = FreeFuseLTX2Pipeline.from_pretrained(
        "Lightricks/LTX-2",
        **pipeline_load_kwargs,
    )

    # ------------------------------------------------------------------
    # Load multiple character LoRAs
    # ------------------------------------------------------------------
    lora_items = [
        ("loras/rapstangled_ltx2.safetensors", "rapstangled", 1.0),
        ("loras/tifa_ltx2.safetensors", "tifa", 1.0),
    ]
    if lora_items:
        for lora_path, adapter_name, _ in lora_items:
            if not os.path.isfile(lora_path):
                raise FileNotFoundError(f"Missing LoRA: {lora_path}")
            pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
        pipe.set_adapters([x[1] for x in lora_items], [x[2] for x in lora_items])
        # Swap transformer to FreeFuse variant only after adapters are set.
        pipe.setup_freefuse_attention_processors()
        pipe.convert_lora_layers(include_connectors=False)
        pipe.enable_sequential_cpu_offload(device=device)

    rapstangled_subject_prompt = "rapstangled2010 in her signature purple dress"
    tifa_subject_prompt = "Tifa Lockhart with long dark hair and crimson eyes"

    prompt = (
        "An animated cinematic medium two-shot in a mountain cabin kitchen at night, "
        "with a crackling campfire glowing through open wooden doors and cool misty blue mountains outside. "
        f"{rapstangled_subject_prompt} sits beside the fire, warm orange light flickering across her face, "
        "twirling a strand of hair before speaking with a soft smile: "
        "\"This is the first lora I have trained.\" "
        f"{tifa_subject_prompt} is already in motion in the same scene, bending to pick up a dropped receipt, "
        "rising smoothly and turning toward her with a playful deadpan smirk as steam drifts from a nearby coffee mug. "
        "She flicks the receipt lightly and replies with dry comedic rhythm about not needing a receipt for a doughnut. "
        "They exchange eye contact, react to each other, and share a brief laugh. "
        "The camera tracks in a slow continuous front-left to front-right arc, with natural body motion, expressive faces, "
        "clean lip sync, and coherent dialogue timing between both characters."
    )
    negative_prompt = (
        "shaky, glitchy, low quality, worst quality, deformed, distorted, motion smear, static"
    )

    # Map adapter_name -> concept text used for token routing.
    # Adapter names must match the names passed to `load_lora_weights`.
    freefuse_concept_map = {
        "rapstangled": rapstangled_subject_prompt,
        "tifa": tifa_subject_prompt,
    }
    debug_root = "debug_freefuse_ltx2"
    stage1_debug_path = os.path.join(debug_root, "stage1")
    stage2_debug_path = os.path.join(debug_root, "stage2")
    os.makedirs(stage1_debug_path, exist_ok=True)
    os.makedirs(stage2_debug_path, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage 1: FreeFuse generation in latent space
    # ------------------------------------------------------------------
    video_latent, audio_latent = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        frame_rate=frame_rate,
        num_inference_steps=stage1_steps,
        guidance_scale=4.0,
        noise_scale=0.0,
        output_type="latent",
        return_dict=False,
        # FreeFuse controls
        freefuse_enabled=bool(freefuse_concept_map),
        freefuse_concept_map=freefuse_concept_map if freefuse_concept_map else None,
        freefuse_top_k_ratio=0.1,
        freefuse_phase1_step=10,
        freefuse_attention_bias_scale=4.0,
        freefuse_attention_bias_positive_scale=2.0,
        freefuse_use_av_cross_attention_bias=False,
        freefuse_debug_save_path=stage1_debug_path,
        freefuse_debug_collect_per_step=True,
    )

    # ------------------------------------------------------------------
    # Latent upsample (official LTX2 stage transition)
    # ------------------------------------------------------------------
    latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
        "Lightricks/LTX-2",
        subfolder="latent_upsampler",
        torch_dtype=dtype,
    )
    upsample_pipe = LTX2LatentUpsamplePipeline(
        vae=pipe.vae,
        latent_upsampler=latent_upsampler,
    )
    upsample_pipe.enable_sequential_cpu_offload(device=device)
    upscaled_video_latent = upsample_pipe(
        latents=video_latent,
        output_type="latent",
        return_dict=False,
    )[0]

    # ------------------------------------------------------------------
    # Stage 2: distilled refinement with official sigma schedule
    # ------------------------------------------------------------------
    pipe.load_lora_weights(
        "Lightricks/LTX-2",
        adapter_name="stage_2_distilled",
        weight_name="ltx-2-19b-distilled-lora-384.safetensors",
    )
    stage2_adapter_names = [x[1] for x in lora_items] + ["stage_2_distilled"]
    stage2_adapter_scales = [x[2] for x in lora_items] + [1.0]
    pipe.set_adapters(stage2_adapter_names, stage2_adapter_scales)

    pipe.vae.enable_tiling()
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        use_dynamic_shifting=False,
        shift_terminal=None,
    )

    output = pipe(
        latents=upscaled_video_latent,
        audio_latents=audio_latent,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=stage2_steps,
        noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
        sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
        guidance_scale=1.0,
        output_type="np",
        return_dict=True,
        # Keep FreeFuse enabled in stage 2, distilled adapter remains unmasked.
        freefuse_enabled=bool(freefuse_concept_map),
        freefuse_concept_map=freefuse_concept_map if freefuse_concept_map else None,
        freefuse_top_k_ratio=0.1,
        freefuse_phase1_step=2,
        freefuse_attention_bias_scale=4.0,
        freefuse_attention_bias_positive_scale=2.0,
        freefuse_use_av_cross_attention_bias=False,
        freefuse_debug_save_path=stage2_debug_path,
        freefuse_debug_collect_per_step=True,
    )

    video = output.frames[0]
    audio = output.audio[0].float().cpu()
    encode_video(
        video,
        fps=frame_rate,
        audio=audio,
        audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
        output_path="freefuse_ltx2_sample.mp4",
    )

    print("Saved freefuse_ltx2_sample.mp4")
    print(f"Saved FreeFuse debug outputs to {debug_root}")


if __name__ == "__main__":
    main()
