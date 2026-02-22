import os

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ltx2 import LTX2LatentUpsamplePipeline, LTX2Pipeline
from diffusers.pipelines.ltx2.export_utils import encode_video
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES

DEVICE = "cuda:0"
DTYPE = torch.bfloat16
WIDTH = 768
HEIGHT = 512
FRAME_RATE = 24.0

SCENE_PROMPT = (
    "An animated cinematic medium shot in a mountain cabin kitchen at night, with warm "
    "firelight from open wooden doors and cool misty blue mountains outside. The camera "
    "moves slowly and steadily with natural body motion, expressive face details, and "
    "coherent cinematic lighting."
)
NEGATIVE_PROMPT = (
    "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, "
    "motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, "
    "transition, static"
)

LORA_TESTS = [
    {
        "adapter_name": "rapstangled",
        "lora_path": "loras/rapstangled_ltx2.safetensors",
        "subject_prompt": "rapstangled2010 in her signature purple dress",
        "output_path": "ltx2_rapstangled_sample.mp4",
    },
    {
        "adapter_name": "tifa",
        "lora_path": "loras/tifa_ltx2.safetensors",
        "subject_prompt": "Tifa Lockhart with long dark hair and crimson eyes",
        "output_path": "ltx2_tifa_sample.mp4",
    },
]


def run_single_lora_test(lora_test: dict) -> None:
    lora_path = lora_test["lora_path"]
    adapter_name = lora_test["adapter_name"]
    output_path = lora_test["output_path"]
    prompt = f"{lora_test['subject_prompt']}. {SCENE_PROMPT}"

    if not os.path.isfile(lora_path):
        raise FileNotFoundError(f"Missing LoRA file: {lora_path}")

    pipe = LTX2Pipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=DTYPE)
    pipe.enable_sequential_cpu_offload(device=DEVICE)

    # Stage 1 uses the subject LoRA only.
    pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
    pipe.set_adapters(adapter_name, 1.0)

    video_latent, audio_latent = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        width=WIDTH,
        height=HEIGHT,
        num_frames=121,
        frame_rate=FRAME_RATE,
        num_inference_steps=40,
        sigmas=None,
        guidance_scale=4.0,
        output_type="latent",
        return_dict=False,
    )

    latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
        "Lightricks/LTX-2",
        subfolder="latent_upsampler",
        torch_dtype=DTYPE,
    )
    upsample_pipe = LTX2LatentUpsamplePipeline(
        vae=pipe.vae,
        latent_upsampler=latent_upsampler,
    )
    upsample_pipe.enable_sequential_cpu_offload(device=DEVICE)
    upscaled_video_latent = upsample_pipe(
        latents=video_latent,
        output_type="latent",
        return_dict=False,
    )[0]

    # Stage 2 combines subject LoRA + official distilled LoRA.
    pipe.load_lora_weights(
        "Lightricks/LTX-2",
        adapter_name="stage_2_distilled",
        weight_name="ltx-2-19b-distilled-lora-384.safetensors",
    )
    pipe.set_adapters([adapter_name, "stage_2_distilled"], [1.0, 1.0])

    pipe.vae.enable_tiling()
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        use_dynamic_shifting=False,
        shift_terminal=None,
    )

    video, audio = pipe(
        latents=upscaled_video_latent,
        audio_latents=audio_latent,
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=3,
        noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
        sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
        guidance_scale=1.0,
        output_type="np",
        return_dict=False,
    )

    encode_video(
        video[0],
        fps=FRAME_RATE,
        audio=audio[0].float().cpu(),
        audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
        output_path=output_path,
    )
    print(f"Saved {output_path}")


def main() -> None:
    for lora_test in LORA_TESTS:
        run_single_lora_test(lora_test)


if __name__ == "__main__":
    main()
