#!/usr/bin/env python
"""
ComfyUI-only comparison: Flux1.dev, Z-Image, SDXL (baseline vs FreeFuse).

This script runs entirely through ComfyUI APIs for each model:
1) Baseline: load model + two LoRAs, generate image (no FreeFuse).
2) FreeFuse: Phase1 collect masks + Phase2 apply masks/bias and generate image.
3) Save per-model comparison and a combined stacked comparison.

Run from repo root:
  python freefuse_comfyui/tests/test_zimage_freefuse_compare.py
"""

import sys
import os
import gc
import argparse
from typing import Dict, Tuple, List, Optional


def _find_comfyui_dir(start_dir: str) -> str:
    cur = os.path.abspath(start_dir)
    for _ in range(10):
        if os.path.isdir(os.path.join(cur, "comfy")) and os.path.isfile(os.path.join(cur, "main.py")):
            return cur
        sibling = os.path.join(cur, "ComfyUI")
        if os.path.isdir(sibling) and os.path.isdir(os.path.join(sibling, "comfy")):
            return sibling
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    raise FileNotFoundError("Could not locate ComfyUI directory")


# Path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
freefuse_comfyui_dir = os.path.dirname(script_dir)
freefuse_root = os.path.dirname(freefuse_comfyui_dir)
comfyui_dir = _find_comfyui_dir(freefuse_root)

if comfyui_dir not in sys.path:
    sys.path.insert(0, comfyui_dir)
if freefuse_root not in sys.path:
    sys.path.insert(0, freefuse_root)

os.chdir(comfyui_dir)

import torch
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont

torch.set_grad_enabled(False)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import folder_paths
import comfy.sample
import comfy.model_management


MODEL_ORDER = ["flux", "z_image", "sdxl"]

MODEL_DEFAULTS = {
    "flux": {
        "steps": 28,
        "cfg": 1.0,
        "seed": 42,
        "sampler": "euler",
        "scheduler": "simple",
        "collect_step": 5,
        "collect_block": 18,
        "collect_region": "output_early ★ (recommended)",
        "collect_tf_index": 3,
        "top_k_ratio": 0.3,
        "temperature": 0.0,
        "bias_scale": 5.0,
        "positive_bias_scale": 1.0,
        # "bias_blocks": "double_stream_only",
        "bias_blocks": "all",
        "bidirectional": True,
        "use_morphological_cleaning": True,
    },
    "z_image": {
        "steps": 12,
        "cfg": 1.0,
        "seed": 42,
        "sampler": "euler",
        "scheduler": "simple",
        "collect_step": 3,
        "collect_block": 18,
        "collect_region": "output_early ★ (recommended)",
        "collect_tf_index": 3,
        "top_k_ratio": 0.1,
        "temperature": 0.0,
        "bias_scale": 4.0,
        "positive_bias_scale": 2.0,
        # "bias_blocks": "last_half",
        "bias_blocks": "all",
        "bidirectional": True,
        "use_morphological_cleaning": True,
    },
    "sdxl": {
        "steps": 30,
        "cfg": 7.0,
        "seed": 77,
        "sampler": "euler",
        "scheduler": "normal",
        "collect_step": 10,
        "collect_block": 18,
        "collect_region": "output_early ★ (recommended)",
        "collect_tf_index": 3,
        "top_k_ratio": 0.1,
        "temperature": 0.0,
        "bias_scale": 5.0,
        "positive_bias_scale": 7.0,
        "bias_blocks": "all",
        "bidirectional": False,
        "use_morphological_cleaning": True,
    },
}


def _get_latent_shape(model, width, height, fallback_channels=16, fallback_downscale=8):
    """Resolve latent shape from model if possible."""
    latent_format = getattr(getattr(model, "model", None), "latent_format", None)
    if latent_format is not None:
        latent_channels = getattr(latent_format, "latent_channels", fallback_channels)
        downscale = getattr(latent_format, "spacial_downscale_ratio", fallback_downscale)
    else:
        latent_channels = fallback_channels
        downscale = fallback_downscale
    latent_h, latent_w = height // downscale, width // downscale
    return latent_channels, latent_h, latent_w


def _reset_comfy_memory(tag: str = ""):
    """Force-release ComfyUI cached models between runs to avoid OOM."""
    try:
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if tag:
        print(f"[Memory] Reset complete: {tag}")


def _is_flux2_model(model) -> bool:
    model_cfg = getattr(getattr(model, "model", None), "model_config", None)
    unet_cfg = getattr(model_cfg, "unet_config", {}) if model_cfg is not None else {}
    if unet_cfg.get("image_model") == "flux2":
        return True
    latent_format = getattr(getattr(model, "model", None), "latent_format", None)
    if latent_format is not None and latent_format.__class__.__name__.lower() == "flux2":
        return True
    model_cls = getattr(getattr(model, "model", None), "__class__", None)
    if model_cls is not None and model_cls.__name__.lower() == "flux2":
        return True
    return False


def _get_flux_context_dim(model) -> Optional[int]:
    dm = getattr(getattr(model, "model", None), "diffusion_model", None)
    params = getattr(dm, "params", None)
    return getattr(params, "context_in_dim", None)


def _pick_flux2_text_encoder(clip_names: List[str], context_in_dim: Optional[int], unet_name: Optional[str]) -> Optional[str]:
    def pick_by_patterns(patterns: List[str]) -> Optional[str]:
        for pat in patterns:
            for name in clip_names:
                if pat in name.lower():
                    return name
        return None

    unet_lower = (unet_name or "").lower()
    if "mistral" in unet_lower:
        name = pick_by_patterns(["mistral3", "mistral", "24b"])
        if name:
            return name
    if "qwen" in unet_lower or "klein" in unet_lower:
        name = pick_by_patterns(["qwen3_8b", "qwen3-8b", "qwen3_4b", "qwen3-4b", "qwen3", "klein"])
        if name:
            return name

    if context_in_dim == 7680:
        return pick_by_patterns(["qwen3_4b", "qwen3-4b", "qwen3", "klein"])
    if context_in_dim == 12288:
        return pick_by_patterns(["qwen3_8b", "qwen3-8b", "klein8b", "qwen3"])
    if context_in_dim == 15360:
        return pick_by_patterns(["mistral3", "mistral", "24b"])

    return pick_by_patterns(["flux2", "mistral", "qwen", "klein"])


def _pick_flux_unet() -> str:
    diffusion_models = folder_paths.get_filename_list("diffusion_models")
    flux_models = [n for n in diffusion_models if "flux" in n.lower()]
    if not flux_models:
        raise FileNotFoundError("No Flux model found in diffusion_models")

    flux1_models = [n for n in flux_models if "flux1" in n.lower() or "flux-1" in n.lower() or "fluxdev" in n.lower()]
    flux2_models = [n for n in flux_models if "flux2" in n.lower() or "klein" in n.lower()]

    return flux1_models[0] if flux1_models else (flux2_models[0] if flux2_models else flux_models[0])


def load_flux_models():
    print("\n[Loading Flux Models]")

    from nodes import UNETLoader, DualCLIPLoader, CLIPLoader, VAELoader

    unet_name = _pick_flux_unet()
    model, = UNETLoader().load_unet(unet_name, "default")
    print(f"  ✅ UNET loaded: {unet_name}")

    clip_names = folder_paths.get_filename_list("text_encoders")
    is_flux2 = _is_flux2_model(model)
    if is_flux2:
        context_in_dim = _get_flux_context_dim(model)
        clip_name = _pick_flux2_text_encoder(clip_names, context_in_dim, unet_name)
        if not clip_name:
            raise FileNotFoundError(
                "Flux2 detected but no suitable text encoder found. "
                f"Available encoders: {clip_names}"
            )
        clip, = CLIPLoader().load_clip(clip_name, type="flux2")
        print(f"  ✅ CLIP loaded (flux2): {clip_name} (context_in_dim={context_in_dim})")
        flux_variant = "flux2"
    else:
        clip_l = next((n for n in clip_names if "clip_l" in n.lower()), None)
        t5 = next((n for n in clip_names if "t5" in n.lower()), None)
        if not clip_l or not t5:
            raise FileNotFoundError(f"Flux CLIP-L/T5 not found. Available: {clip_names}")
        clip, = DualCLIPLoader().load_clip(clip_l, t5, "flux")
        print(f"  ✅ CLIP loaded: {clip_l}, {t5}")
        flux_variant = "flux"

    vae_name = None
    for name in folder_paths.get_filename_list("vae"):
        if "ae" in name.lower() or "flux" in name.lower():
            vae_name = name
            break
    if not vae_name:
        raise FileNotFoundError("No VAE found for Flux")
    vae, = VAELoader().load_vae(vae_name)
    print(f"  ✅ VAE loaded: {vae_name}")

    return model, clip, vae, flux_variant


def load_zimage_models():
    print("\n[Loading Z-Image-Turbo Models]")
    unet_name = None
    for name in folder_paths.get_filename_list("diffusion_models"):
        if "z_image" in name.lower() or "zimage" in name.lower() or "lumina" in name.lower():
            unet_name = name
            break
    if not unet_name:
        raise FileNotFoundError("No Z-Image-Turbo model found in diffusion_models")

    clip_name = None
    for name in folder_paths.get_filename_list("text_encoders"):
        if "qwen" in name.lower():
            clip_name = name
            break

    vae_name = None
    for name in folder_paths.get_filename_list("vae"):
        if "ae" in name.lower() or "sdxl" in name.lower():
            vae_name = name
            break

    from nodes import UNETLoader, CLIPLoader, VAELoader
    unet_loader = UNETLoader()
    model, = unet_loader.load_unet(unet_name, "default")
    print(f"  ✅ UNET loaded: {unet_name}")

    if not clip_name:
        raise FileNotFoundError("Qwen3 text encoder not found in text_encoders/")
    clip, = CLIPLoader().load_clip(clip_name, type="lumina2")
    print(f"  ✅ CLIP loaded: {clip_name} (lumina2 type)")

    vae_loader = VAELoader()
    vae, = vae_loader.load_vae(vae_name)
    print(f"  ✅ VAE loaded: {vae_name}")
    return model, clip, vae


def load_sdxl_models():
    print("\n[Loading SDXL Models]")
    checkpoint_name = None
    for name in folder_paths.get_filename_list("checkpoints"):
        if "sdxl" in name.lower() or "xl" in name.lower():
            checkpoint_name = name
            break
    if not checkpoint_name:
        raise FileNotFoundError("No SDXL checkpoint found")

    from nodes import CheckpointLoaderSimple
    ckpt_loader = CheckpointLoaderSimple()
    model, clip, vae = ckpt_loader.load_checkpoint(checkpoint_name)
    print(f"  ✅ Checkpoint loaded: {checkpoint_name}")
    return model, clip, vae


def _find_loras_for_model(model_type: str) -> Tuple[str, str, str, str, float]:
    lora_list = folder_paths.get_filename_list("loras")
    if model_type == "z_image":
        lora_a = next((l for l in lora_list if "jinx" in l.lower() and "zit" in l.lower()), None)
        lora_b = next((l for l in lora_list if "skeletor" in l.lower() and "zit" in l.lower()), None)
        adapter_a, adapter_b = "jinx", "skeleton"
        strength = 0.8
    elif model_type == "flux":
        lora_a = next((l for l in lora_list if "harry" in l.lower() and "flux" in l.lower()), None)
        lora_b = next((l for l in lora_list if "daiyu" in l.lower() and "flux" in l.lower()), None)
        adapter_a, adapter_b = "harry", "daiyu"
        strength = 1.0
    else:
        lora_a = next((l for l in lora_list if "harry" in l.lower() and "xl" in l.lower()), None)
        lora_b = next((l for l in lora_list if "daiyu" in l.lower() and "xl" in l.lower()), None)
        adapter_a, adapter_b = "harry", "daiyu"
        strength = 1.0

    if not lora_a or not lora_b:
        raise FileNotFoundError(f"LoRAs not found for {model_type}. Available: {lora_list}")

    return lora_a, lora_b, adapter_a, adapter_b, strength


def load_loras(model, clip, lora_a, lora_b, adapter_a, adapter_b, strength):
    from freefuse_comfyui.nodes.lora_loader import FreeFuseLoRALoader
    loader = FreeFuseLoRALoader()

    model, clip, freefuse_data = loader.load_lora(
        model, clip, lora_a, adapter_a, strength, strength, None
    )
    print(f"  ✅ LoRA loaded: {adapter_a} ({lora_a}), strength={strength}")

    model, clip, freefuse_data = loader.load_lora(
        model, clip, lora_b, adapter_b, strength, strength, freefuse_data
    )
    print(f"  ✅ LoRA loaded: {adapter_b} ({lora_b}), strength={strength}")
    return model, clip, freefuse_data


def _default_prompt_config(model_type: str) -> Tuple[str, str, Dict[str, str], str, str, str]:
    if model_type == "z_image":
        prompt = (
            "A picture of two characters, a starry night scene with northern lights in background: "
            "The first character is Jinx_Arcane, a young woman with long blue hair in a loose braid "
            "and bright blue eyes, wearing a cropped halter top, gloves, striped pants with belts, "
            "and visible tattoos and the second character is Skeletor in purple hooded cloak flexing "
            "muscular blue arms triumphantly, skull face grinning menacingly, cartoon animation style, "
            "Masters of the Universe character, vibrant purple and blue color scheme"
        )
        negative_prompt = ""
        concept_map = {
            "jinx": (
                "Jinx_Arcane, a young woman with long blue hair in a loose braid and bright blue eyes, "
                "wearing a cropped halter top, gloves, striped pants with belts, and visible tattoos"
            ),
            "skeleton": (
                "Skeletor in purple hooded cloak flexing muscular blue arms triumphantly, "
                "skull face grinning menacingly, cartoon animation style, "
                "Masters of the Universe character, vibrant purple and blue color scheme"
            ),
        }
        background_text = "a starry night scene with northern lights"
        adapter_a, adapter_b = "jinx", "skeleton"
    elif model_type == "sdxl":
        prompt = (
            "Realistic photography, an European man wearing Hogwarts uniform hugging an Asian woman "
            "wearing Chinese traditional clothing warmly, both faces close together, autumn leaves "
            "blurred in the background."
        )
        negative_prompt = "low quality, blurry, deformed, line art"
        concept_map = {
            "harry": "an European man wearing Hogwarts uniform",
            "daiyu": "an Asian woman wearing Chinese traditional clothing",
        }
        background_text = "autumn leaves blurred in the background"
        adapter_a, adapter_b = "harry", "daiyu"
    else:
        prompt = (
            "Realistic photography, harry potter, an European photorealistic style teenage wizard boy "
            "with messy black hair, round wire-frame glasses, and bright green eyes, wearing a white shirt, "
            "burgundy and gold striped tie, and dark robes hugging daiyu_lin, a young East Asian photorealistic "
            "style woman in traditional Chinese hanfu dress, elaborate black updo hairstyle adorned with delicate "
            "white floral hairpins and ornaments, dangling red tassel earrings, soft pink and red color palette, "
            "gentle smile with knowing expression, autumn leaves blurred in the background, high quality, detailed"
        )
        negative_prompt = "low quality, blurry, deformed, ugly, bad anatomy"
        concept_map = {
            "harry": (
                "harry potter, an European photorealistic style teenage wizard boy with messy black hair, "
                "round wire-frame glasses, and bright green eyes, wearing a white shirt, burgundy and gold "
                "striped tie, and dark robes"
            ),
            "daiyu": (
                "daiyu_lin, a young East Asian photorealistic style woman in traditional Chinese hanfu dress, "
                "elaborate black updo hairstyle adorned with delicate white floral hairpins and ornaments, "
                "dangling red tassel earrings, soft pink and red color palette, gentle smile with knowing expression"
            ),
        }
        background_text = "autumn leaves blurred in the background"
        adapter_a, adapter_b = "harry", "daiyu"

    return prompt, negative_prompt, concept_map, background_text, adapter_a, adapter_b


def setup_concepts_and_conditioning(
    model_type: str,
    clip,
    freefuse_data,
    prompt: str,
    negative_prompt: str,
    concept_map: Dict[str, str],
    background_text: str,
    adapter_a: str,
    adapter_b: str,
    flux_variant: Optional[str],
    guidance: float,
):
    from freefuse_comfyui.nodes.concept_map import FreeFuseConceptMap, FreeFuseTokenPositions

    concept_mapper = FreeFuseConceptMap()
    freefuse_data, = concept_mapper.create_map(
        adapter_name_1=adapter_a,
        concept_text_1=concept_map[adapter_a],
        adapter_name_2=adapter_b,
        concept_text_2=concept_map[adapter_b],
        enable_background=True,
        background_text=background_text,
        freefuse_data=freefuse_data,
    )

    token_pos = FreeFuseTokenPositions()
    freefuse_data, = token_pos.compute_positions(
        clip=clip,
        prompt=prompt,
        freefuse_data=freefuse_data,
        filter_meaningless=True,
        filter_single_char=True,
    )

    if model_type == "z_image":
        system_prompt = (
            "You are an assistant designed to generate superior images with the superior "
            "degree of image-text alignment based on textual prompts or user prompts."
        )
        lumina_prompt = f"{system_prompt} <Prompt Start> {prompt}"
        tokens = clip.tokenize(lumina_prompt)
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        neg_lumina_prompt = (
            f"{system_prompt} <Prompt Start> {negative_prompt}" if negative_prompt else f"{system_prompt} <Prompt Start> "
        )
        neg_tokens = clip.tokenize(neg_lumina_prompt)
        neg_conditioning = clip.encode_from_tokens_scheduled(neg_tokens)
    elif model_type == "flux" and flux_variant == "flux2":
        tokens = clip.tokenize(prompt)
        conditioning = clip.encode_from_tokens_scheduled(tokens, add_dict={"guidance": guidance})

        neg_tokens = clip.tokenize("")
        neg_conditioning = clip.encode_from_tokens_scheduled(neg_tokens, add_dict={"guidance": guidance})
    elif model_type == "flux":
        from comfy_extras.nodes_flux import CLIPTextEncodeFlux
        encoder = CLIPTextEncodeFlux()
        conditioning, = encoder.encode(clip, prompt, prompt, guidance)
        neg_conditioning, = encoder.encode(clip, "", "", guidance)
    else:
        tokens = clip.tokenize(prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        conditioning = [[cond, {"pooled_output": pooled}]]

        neg_tokens = clip.tokenize(negative_prompt)
        neg_cond, neg_pooled = clip.encode_from_tokens(neg_tokens, return_pooled=True)
        neg_conditioning = [[neg_cond, {"pooled_output": neg_pooled}]]

    return freefuse_data, conditioning, neg_conditioning


def decode_to_pil(vae, samples) -> Image.Image:
    with torch.inference_mode():
        decoded = vae.decode(samples)

    if decoded.dim() == 4:
        image = decoded[0]
    else:
        image = decoded
    if image.shape[0] in [1, 3, 4]:
        image = image.permute(1, 2, 0)
    image = image.detach().cpu().numpy()
    image = (image * 255).clip(0, 255).astype(np.uint8)
    if image.shape[-1] == 1:
        image = image.squeeze(-1)
    return Image.fromarray(image)


def _format_model_name(model_name: str) -> str:
    return model_name.replace("_", " ").strip().upper()


def _load_bold_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def create_comparison_image(
    left: Image.Image,
    right: Image.Image,
    model_name: str,
    left_label: Optional[str] = None,
    right_label: Optional[str] = None,
):
    w, h = left.size
    label_height = max(128, h // 8)
    composite = Image.new("RGB", (w * 2, h + label_height), color=(18, 22, 28))
    composite.paste(left, (0, label_height))
    composite.paste(right, (w, label_height))

    draw = ImageDraw.Draw(composite)
    model_tag = _format_model_name(model_name)
    left_text = left_label or f"{model_tag} WITHOUT FREEFUSE"
    right_text = right_label or f"{model_tag} WITH FREEFUSE"

    def fit_font(text: str, max_width: int) -> ImageFont.ImageFont:
        for size in range(84, 23, -2):
            font = _load_bold_font(size)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            if text_w <= max_width:
                return font
        return _load_bold_font(24)

    left_font = fit_font(left_text, int(w * 0.92))
    right_font = fit_font(right_text, int(w * 0.92))
    stroke_left = max(2, getattr(left_font, "size", 24) // 14)
    stroke_right = max(2, getattr(right_font, "size", 24) // 14)

    def draw_label(center_x: int, text: str, font: ImageFont.ImageFont, color, stroke_w: int):
        center_y = label_height // 2 + 2
        draw.text(
            (center_x + 3, center_y + 3),
            text,
            fill=(0, 0, 0),
            font=font,
            anchor="mm",
        )
        draw.text(
            (center_x, center_y),
            text,
            fill=color,
            font=font,
            anchor="mm",
            stroke_width=stroke_w,
            stroke_fill=(255, 255, 255),
        )

    draw_label(w // 2, left_text, left_font, (235, 46, 46), stroke_left)
    draw_label(w + w // 2, right_text, right_font, (24, 215, 196), stroke_right)
    return composite


def stack_rows(rows: List[Image.Image], padding: int = 20, title: Optional[str] = None) -> Image.Image:
    if not rows:
        raise ValueError("No rows to stack")

    widths = [img.size[0] for img in rows]
    heights = [img.size[1] for img in rows]
    max_w = max(widths)
    total_h = sum(heights) + padding * (len(rows) - 1)

    title_height = 0
    if title:
        title_height = 60
        total_h += title_height

    out = Image.new("RGB", (max_w, total_h), color="white")
    y = 0

    if title:
        draw = ImageDraw.Draw(out)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        except Exception:
            font = ImageFont.load_default()
        draw.text((max_w // 2, 20), title, fill="black", font=font, anchor="mt")
        y += title_height

    for img in rows:
        x = (max_w - img.size[0]) // 2
        out.paste(img, (x, y))
        y += img.size[1] + padding

    return out


def run_model_compare(model_type: str, width: int, height: int, out_dir: str) -> Tuple[Image.Image, Image.Image, Image.Image]:
    _reset_comfy_memory(f"before {model_type}")
    defaults = MODEL_DEFAULTS[model_type]

    if model_type == "flux":
        model, clip, vae, flux_variant = load_flux_models()
    elif model_type == "z_image":
        model, clip, vae = load_zimage_models()
        flux_variant = None
    else:
        model, clip, vae = load_sdxl_models()
        flux_variant = None

    lora_a, lora_b, adapter_a, adapter_b, strength = _find_loras_for_model(model_type)
    model, clip, freefuse_data = load_loras(model, clip, lora_a, lora_b, adapter_a, adapter_b, strength)
    if model_type == "flux":
        freefuse_data["flux_variant"] = flux_variant

    prompt, negative_prompt, concept_map, background_text, adapter_a, adapter_b = _default_prompt_config(model_type)
    conditioning_guidance = 3.5 if model_type == "flux" else defaults["cfg"]
    freefuse_data, conditioning, neg_conditioning = setup_concepts_and_conditioning(
        model_type,
        clip,
        freefuse_data,
        prompt,
        negative_prompt,
        concept_map,
        background_text,
        adapter_a,
        adapter_b,
        flux_variant,
        conditioning_guidance,
    )

    fallback_channels = 16 if model_type in ["flux", "z_image"] else 4
    latent_channels, latent_h, latent_w = _get_latent_shape(
        model, width, height, fallback_channels=fallback_channels
    )
    latent = torch.zeros([1, latent_channels, latent_h, latent_w], device="cpu")
    latent_dict = {"samples": latent}

    model_out_dir = os.path.join(out_dir, model_type)
    os.makedirs(model_out_dir, exist_ok=True)

    # Baseline
    print(f"\n[{model_type}] Baseline generating...")
    noise = comfy.sample.prepare_noise(latent, defaults["seed"], None)
    with torch.inference_mode():
        samples = comfy.sample.sample(
            model,
            noise,
            defaults["steps"],
            defaults["cfg"],
            defaults["sampler"],
            defaults["scheduler"],
            conditioning,
            neg_conditioning,
            latent,
            denoise=1.0,
            disable_noise=False,
            start_step=0,
            last_step=defaults["steps"],
            force_full_denoise=True,
            noise_mask=None,
            callback=None,
            seed=defaults["seed"],
        )
    baseline_img = decode_to_pil(vae, samples)
    baseline_path = os.path.join(model_out_dir, "baseline.png")
    baseline_img.save(baseline_path)
    print(f"[{model_type}] Baseline saved: {baseline_path}")

    # FreeFuse
    print(f"\n[{model_type}] FreeFuse Phase 1 collecting masks...")
    from freefuse_comfyui.nodes.sampler import FreeFusePhase1Sampler
    from freefuse_comfyui.nodes.mask_applicator import FreeFuseMaskApplicator

    sampler = FreeFusePhase1Sampler()
    with torch.inference_mode():
        model_phase1, masks_output, preview = sampler.collect_masks(
            model=model,
            conditioning=conditioning,
            neg_conditioning=neg_conditioning,
            latent=latent_dict,
            freefuse_data=freefuse_data,
            seed=defaults["seed"],
            steps=defaults["steps"],
            collect_step=defaults["collect_step"],
            cfg=defaults["cfg"],
            sampler_name=defaults["sampler"],
            scheduler=defaults["scheduler"],
            collect_block=defaults["collect_block"],
            collect_region=defaults["collect_region"],
            collect_tf_index=defaults["collect_tf_index"],
            temperature=defaults["temperature"],
            top_k_ratio=defaults["top_k_ratio"],
            disable_lora_phase1=True,
            bg_scale=0.95,
            use_morphological_cleaning=defaults["use_morphological_cleaning"],
            balance_iterations=15,
        )

    preview_np = (preview[0].numpy() * 255).astype(np.uint8)
    preview_img = Image.fromarray(preview_np)
    preview_path = os.path.join(model_out_dir, "mask_preview.png")
    preview_img.save(preview_path)

    applicator = FreeFuseMaskApplicator()
    model_masked, = applicator.apply_masks(
        model=model_phase1,
        masks=masks_output,
        freefuse_data=freefuse_data,
        enable_token_masking=True,
        latent=latent_dict,
        enable_attention_bias=True,
        bias_scale=defaults["bias_scale"],
        positive_bias_scale=defaults["positive_bias_scale"],
        bidirectional=defaults["bidirectional"],
        use_positive_bias=True,
        bias_blocks=defaults["bias_blocks"],
    )

    print(f"[{model_type}] FreeFuse Phase 2 generating...")
    noise = comfy.sample.prepare_noise(latent, defaults["seed"], None)
    with torch.inference_mode():
        samples_ff = comfy.sample.sample(
            model_masked,
            noise,
            defaults["steps"],
            defaults["cfg"],
            defaults["sampler"],
            defaults["scheduler"],
            conditioning,
            neg_conditioning,
            latent,
            denoise=1.0,
            disable_noise=False,
            start_step=0,
            last_step=defaults["steps"],
            force_full_denoise=True,
            noise_mask=None,
            callback=None,
            seed=defaults["seed"],
        )
    freefuse_img = decode_to_pil(vae, samples_ff)
    freefuse_path = os.path.join(model_out_dir, "freefuse.png")
    freefuse_img.save(freefuse_path)
    print(f"[{model_type}] FreeFuse saved: {freefuse_path}")

    compare = create_comparison_image(baseline_img, freefuse_img, model_name=model_type)
    compare_path = os.path.join(model_out_dir, "compare.png")
    compare.save(compare_path)
    print(f"[{model_type}] Compare saved: {compare_path}")

    # Cleanup
    del model, clip, vae
    _reset_comfy_memory(f"after {model_type}")

    return baseline_img, freefuse_img, compare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="all", help="Comma-separated: flux,z_image,sdxl or 'all'")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--out_dir", default="test_outputs_multimodel_compare")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.models.strip().lower() == "all":
        models = MODEL_ORDER
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]

    rows = []
    for model_type in models:
        if model_type not in MODEL_DEFAULTS:
            print(f"Skipping unknown model type: {model_type}")
            continue
        _, _, compare = run_model_compare(model_type, args.width, args.height, args.out_dir)
        rows.append(compare)

    if not rows:
        raise RuntimeError("No valid models were run")

    combined = stack_rows(rows, padding=30, title="FreeFuse Comparison")
    combined_path = os.path.join(args.out_dir, "compare_all.png")
    combined.save(combined_path)
    print(f"[Compare] Combined saved: {combined_path}")


if __name__ == "__main__":
    main()
