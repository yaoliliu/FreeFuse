#!/usr/bin/env python
"""
Attention Bias Effect Test

Test different attention bias configurations to verify the effect is working.
Generates images with different bias scales for comparison.

Usage:
    python freefuse_comfyui/tests/test_attn_bias.py --model flux
    python freefuse_comfyui/tests/test_attn_bias.py --model sdxl
    python freefuse_comfyui/tests/test_attn_bias.py --model flux --bias-scale 50 --positive-scale 10
"""

import sys
import os
import gc
import time
import argparse
from dataclasses import dataclass
from typing import Optional, List, Tuple


def _find_comfyui_dir(start_dir: str) -> str:
    """Find the ComfyUI root directory."""
    cur = os.path.abspath(start_dir)
    for _ in range(10):
        if os.path.isdir(os.path.join(cur, "comfy")) and os.path.isfile(os.path.join(cur, "main.py")):
            return cur
        sibling_comfyui = os.path.join(cur, "ComfyUI")
        if os.path.isdir(sibling_comfyui) and os.path.isdir(os.path.join(sibling_comfyui, "comfy")):
            return sibling_comfyui
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    raise FileNotFoundError("Could not locate ComfyUI directory")


# Setup paths
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
from PIL import Image
import numpy as np

torch.set_grad_enabled(False)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import folder_paths
import comfy.sd
import comfy.utils
import comfy.samplers
import comfy.sample
import comfy.model_management


def _get_latent_shape(model, width, height, fallback_channels=16, fallback_downscale=8):
    """Resolve latent shape from model if possible (Flux2 uses different latent format)."""
    latent_format = getattr(getattr(model, "model", None), "latent_format", None)
    if latent_format is not None:
        latent_channels = getattr(latent_format, "latent_channels", fallback_channels)
        downscale = getattr(latent_format, "spacial_downscale_ratio", fallback_downscale)
    else:
        latent_channels = fallback_channels
        downscale = fallback_downscale
    latent_h, latent_w = height // downscale, width // downscale
    return latent_channels, latent_h, latent_w


def _is_flux2_model(model) -> bool:
    """Best-effort detection of Flux2 model."""
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


@dataclass
class BiasTestConfig:
    """Configuration for a bias test."""
    name: str
    bias_scale: float
    positive_bias_scale: float
    bidirectional: bool
    use_positive_bias: bool
    bias_blocks: str  # "all", "double", "single", or specific like "0-9"
    description: str = ""


# Default test configurations
DEFAULT_BIAS_TESTS = [
    BiasTestConfig(
        name="no_bias",
        bias_scale=0.0,
        positive_bias_scale=0.0,
        bidirectional=True,
        use_positive_bias=False,
        bias_blocks="all",
        description="No attention bias (baseline)"
    ),
    BiasTestConfig(
        name="default_bias",
        bias_scale=5.0,
        positive_bias_scale=1.0,
        bidirectional=True,
        use_positive_bias=True,
        bias_blocks="all",
        description="Default bias (neg=5, pos=1)"
    ),
    BiasTestConfig(
        name="strong_neg_bias",
        bias_scale=20.0,
        positive_bias_scale=1.0,
        bidirectional=True,
        use_positive_bias=True,
        bias_blocks="all",
        description="Strong negative bias (neg=20, pos=1)"
    ),
    BiasTestConfig(
        name="extreme_neg_bias",
        bias_scale=50.0,
        positive_bias_scale=1.0,
        bidirectional=True,
        use_positive_bias=True,
        bias_blocks="all",
        description="Extreme negative bias (neg=50, pos=1)"
    ),
    BiasTestConfig(
        name="strong_pos_bias",
        bias_scale=5.0,
        positive_bias_scale=10.0,
        bidirectional=True,
        use_positive_bias=True,
        bias_blocks="all",
        description="Strong positive bias (neg=5, pos=10)"
    ),
    BiasTestConfig(
        name="extreme_both_bias",
        bias_scale=50.0,
        positive_bias_scale=20.0,
        bidirectional=True,
        use_positive_bias=True,
        bias_blocks="all",
        description="Extreme both (neg=50, pos=20)"
    ),
    BiasTestConfig(
        name="neg_only_no_pos",
        bias_scale=20.0,
        positive_bias_scale=0.0,
        bidirectional=True,
        use_positive_bias=False,
        bias_blocks="all",
        description="Negative only, no positive (neg=20)"
    ),
    BiasTestConfig(
        name="unidirectional",
        bias_scale=20.0,
        positive_bias_scale=5.0,
        bidirectional=False,
        use_positive_bias=True,
        bias_blocks="all",
        description="Unidirectional bias"
    ),
]


def load_flux_models():
    """Load Flux model and return components."""
    print("\n[Loading Flux Models]")
    
    from nodes import UNETLoader, DualCLIPLoader, CLIPLoader, VAELoader
    
    diffusion_models = folder_paths.get_filename_list("diffusion_models")
    flux_models = [n for n in diffusion_models if "flux" in n.lower()]
    if not flux_models:
        raise FileNotFoundError("No Flux model found")

    flux1_models = [n for n in flux_models if "flux1" in n.lower() or "flux-1" in n.lower() or "fluxdev" in n.lower()]
    flux2_models = [n for n in flux_models if "flux2" in n.lower() or "klein" in n.lower()]

    unet_name = flux1_models[0] if flux1_models else (flux2_models[0] if flux2_models else flux_models[0])
    
    clip_names = folder_paths.get_filename_list("text_encoders")
    clip_l = next((n for n in clip_names if "clip_l" in n.lower()), None)
    t5 = next((n for n in clip_names if "t5" in n.lower()), None)
    
    vae_name = None
    for name in folder_paths.get_filename_list("vae"):
        if "ae" in name.lower() or "flux" in name.lower():
            vae_name = name
            break
    
    unet_loader = UNETLoader()
    model, = unet_loader.load_unet(unet_name, "default")
    print(f"  ✅ UNET: {unet_name}")
    print(f"  Flux candidates: {flux_models}")
    
    is_flux2 = _is_flux2_model(model)
    if is_flux2:
        context_in_dim = _get_flux_context_dim(model)
        clip_name = _pick_flux2_text_encoder(clip_names, context_in_dim, unet_name)
        if not clip_name:
            raise FileNotFoundError(
                "Flux2 detected but no suitable text encoder found. "
                f"Available encoders: {clip_names}"
            )
        clip_loader = CLIPLoader()
        clip, = clip_loader.load_clip(clip_name, type="flux2")
        print(f"  ✅ CLIP (flux2): {clip_name} (context_in_dim={context_in_dim})")
    else:
        clip_loader = DualCLIPLoader()
        clip, = clip_loader.load_clip(clip_l, t5, "flux")
        print(f"  ✅ CLIP: {clip_l}, {t5}")
    
    vae_loader = VAELoader()
    vae, = vae_loader.load_vae(vae_name)
    print(f"  ✅ VAE: {vae_name}")
    
    return model, clip, vae, "flux"


def load_sdxl_models():
    """Load SDXL model and return components."""
    print("\n[Loading SDXL Models]")
    
    from nodes import CheckpointLoaderSimple
    
    checkpoint_name = None
    for name in folder_paths.get_filename_list("checkpoints"):
        if "sdxl" in name.lower() or "xl" in name.lower():
            checkpoint_name = name
            break
    if not checkpoint_name:
        raise FileNotFoundError("No SDXL checkpoint found")
    
    ckpt_loader = CheckpointLoaderSimple()
    model, clip, vae = ckpt_loader.load_checkpoint(checkpoint_name)
    print(f"  ✅ Checkpoint: {checkpoint_name}")
    
    return model, clip, vae, "sdxl"


def load_loras(model, clip, model_type: str):
    """Load LoRAs in bypass mode."""
    print("\n[Loading LoRAs]")
    
    from freefuse_comfyui.nodes.lora_loader import FreeFuseLoRALoader
    
    lora_list = folder_paths.get_filename_list("loras")
    
    if model_type == "flux":
        harry_lora = next((l for l in lora_list if "harry" in l.lower() and "flux" in l.lower()), None)
        daiyu_lora = next((l for l in lora_list if "daiyu" in l.lower() and "flux" in l.lower()), None)
    else:
        harry_lora = next((l for l in lora_list if "harry" in l.lower() and "xl" in l.lower()), None)
        daiyu_lora = next((l for l in lora_list if "daiyu" in l.lower() and "xl" in l.lower()), None)
    
    if not harry_lora or not daiyu_lora:
        raise FileNotFoundError(f"LoRAs not found for {model_type}")
    
    loader = FreeFuseLoRALoader()
    
    model, clip, freefuse_data = loader.load_lora(
        model, clip, harry_lora, "harry", 1.0, 1.0, None
    )
    print(f"  ✅ harry: {harry_lora}")

    if model_type == "flux":
        freefuse_data["flux_variant"] = "flux2" if _is_flux2_model(model) else "flux"
    
    model, clip, freefuse_data = loader.load_lora(
        model, clip, daiyu_lora, "daiyu", 1.0, 1.0, freefuse_data
    )
    print(f"  ✅ daiyu: {daiyu_lora}")
    
    return model, clip, freefuse_data


def setup_conditioning(clip, freefuse_data, model_type: str):
    """Set up concept map and conditioning."""
    prompt = "Realistic photography, an European man wearing Hogwarts uniform chatting with an Asian woman wearing Chinese traditional clothing, autumn leaves blurred in the background, high quality, detailed"
    negative_prompt = "low quality, blurry, deformed, ugly, bad anatomy"
    
    concept_map = {
        "harry": "an European man wearing Hogwarts uniform",
        "daiyu": "an Asian woman wearing Chinese traditional clothing",
    }
    background_text = "autumn leaves blurred in the background"
    
    from freefuse_comfyui.nodes.concept_map import FreeFuseConceptMap, FreeFuseTokenPositions
    
    concept_mapper = FreeFuseConceptMap()
    freefuse_data, = concept_mapper.create_map(
        adapter_name_1="harry",
        concept_text_1=concept_map["harry"],
        adapter_name_2="daiyu",
        concept_text_2=concept_map["daiyu"],
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
    
    # Create conditioning
    if model_type == "flux" and freefuse_data.get("flux_variant") == "flux2":
        tokens = clip.tokenize(prompt)
        conditioning = clip.encode_from_tokens_scheduled(tokens, add_dict={"guidance": 3.5})

        neg_tokens = clip.tokenize("")
        neg_conditioning = clip.encode_from_tokens_scheduled(neg_tokens, add_dict={"guidance": 3.5})
    elif model_type == "flux":
        from comfy_extras.nodes_flux import CLIPTextEncodeFlux
        encoder = CLIPTextEncodeFlux()
        conditioning, = encoder.encode(clip, prompt, prompt, 3.5)
        neg_conditioning, = encoder.encode(clip, "", "", 3.5)
    else:
        tokens = clip.tokenize(prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        conditioning = [[cond, {"pooled_output": pooled}]]
        
        neg_tokens = clip.tokenize(negative_prompt)
        neg_cond, neg_pooled = clip.encode_from_tokens(neg_tokens, return_pooled=True)
        neg_conditioning = [[neg_cond, {"pooled_output": neg_pooled}]]
    
    return freefuse_data, conditioning, neg_conditioning


def run_bias_test(
    config: BiasTestConfig,
    model, clip, vae, freefuse_data,
    conditioning, neg_conditioning,
    model_type: str,
    output_dir: str,
    width: int = 1024,
    height: int = 1024,
    seed: int = 42,
    steps: int = None,
    cfg: float = None,
):
    """Run a single bias test and return results."""
    
    if steps is None:
        steps = 28 if model_type == "flux" else 30
    if cfg is None:
        cfg = 1.0 if model_type == "flux" else 7.0
    
    print(f"\n{'='*60}")
    print(f"Test: {config.name}")
    print(f"Description: {config.description}")
    print(f"{'='*60}")
    print(f"  bias_scale={config.bias_scale}, positive_scale={config.positive_bias_scale}")
    print(f"  bidirectional={config.bidirectional}, use_positive={config.use_positive_bias}")
    print(f"  bias_blocks={config.bias_blocks}")
    
    from freefuse_comfyui.nodes.sampler import FreeFusePhase1Sampler
    from freefuse_comfyui.nodes.mask_applicator import FreeFuseMaskApplicator
    
    try:
        # Create latent (respect model latent format when available)
        fallback_channels = 16 if model_type == "flux" else 4
        latent_channels, latent_h, latent_w = _get_latent_shape(
            model, width, height, fallback_channels=fallback_channels
        )
        latent = torch.zeros([1, latent_channels, latent_h, latent_w], device="cpu")
        latent_dict = {"samples": latent}
        
        # Phase 1 - Collect masks
        sampler = FreeFusePhase1Sampler()
        
        start_time = time.time()
        with torch.inference_mode():
            model_phase1, masks_output, preview = sampler.collect_masks(
                model=model,
                conditioning=conditioning,
                neg_conditioning=neg_conditioning,
                latent=latent_dict,
                freefuse_data=freefuse_data,
                seed=seed,
                steps=steps,
                collect_step=5 if model_type == "flux" else 10,
                cfg=cfg,
                sampler_name="euler",
                scheduler="simple" if model_type == "flux" else "normal",
                collect_block=18,
                temperature=0,
                top_k_ratio=0.3,
                disable_lora_phase1=True,
                bg_scale=0.95,
                use_morphological_cleaning=False,
                balance_iterations=15,
            )
        phase1_time = time.time() - start_time
        print(f"  Phase 1: {phase1_time:.1f}s")
        
        # Save mask preview
        preview_np = (preview[0].numpy() * 255).astype(np.uint8)
        preview_img = Image.fromarray(preview_np)
        mask_path = os.path.join(output_dir, f"{model_type}_{config.name}_mask.png")
        preview_img.save(mask_path)
        
        # Apply masks with specified bias config
        applicator = FreeFuseMaskApplicator()
        model_masked, = applicator.apply_masks(
            model=model_phase1,
            masks=masks_output,
            freefuse_data=freefuse_data,
            enable_token_masking=True,
            latent=latent_dict,
            enable_attention_bias=config.bias_scale > 0 or config.positive_bias_scale > 0,
            bias_scale=config.bias_scale,
            positive_bias_scale=config.positive_bias_scale,
            bidirectional=config.bidirectional,
            use_positive_bias=config.use_positive_bias,
            bias_blocks=config.bias_blocks,
        )
        
        # Phase 2 - Generate
        noise = comfy.sample.prepare_noise(latent, seed, None)
        
        start_time = time.time()
        with torch.inference_mode():
            samples = comfy.sample.sample(
                model_masked,
                noise,
                steps,
                cfg,
                "euler",
                "simple" if model_type == "flux" else "normal",
                conditioning,
                neg_conditioning,
                latent,
                denoise=1.0,
                disable_noise=False,
                start_step=0,
                last_step=steps,
                force_full_denoise=True,
                noise_mask=None,
                callback=None,
                seed=seed,
            )
        phase2_time = time.time() - start_time
        print(f"  Phase 2: {phase2_time:.1f}s")
        
        # Decode
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
        
        output_path = os.path.join(output_dir, f"{model_type}_{config.name}_output.png")
        Image.fromarray(image).save(output_path)
        print(f"  ✅ Saved: {output_path}")
        
        return True, None
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_custom_test(
    model_type: str,
    bias_scale: float,
    positive_scale: float,
    bidirectional: bool = True,
    use_positive: bool = True,
    bias_blocks: str = "all",
    width: int = 1024,
    height: int = 1024,
    seed: int = 42,
):
    """Run a single custom test with specified parameters."""
    
    output_dir = f"test_outputs_bias_{model_type}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    if model_type == "flux":
        model, clip, vae, mt = load_flux_models()
    else:
        model, clip, vae, mt = load_sdxl_models()
    
    model, clip, freefuse_data = load_loras(model, clip, model_type)
    freefuse_data, conditioning, neg_conditioning = setup_conditioning(clip, freefuse_data, model_type)
    
    config = BiasTestConfig(
        name=f"custom_neg{bias_scale}_pos{positive_scale}",
        bias_scale=bias_scale,
        positive_bias_scale=positive_scale,
        bidirectional=bidirectional,
        use_positive_bias=use_positive,
        bias_blocks=bias_blocks,
        description=f"Custom test: neg={bias_scale}, pos={positive_scale}"
    )
    
    success, error = run_bias_test(
        config, model, clip, vae, freefuse_data,
        conditioning, neg_conditioning,
        model_type, output_dir,
        width=width, height=height, seed=seed
    )
    
    return success


def run_all_tests(model_type: str, tests: List[BiasTestConfig] = None):
    """Run all bias tests for a model type."""
    
    if tests is None:
        tests = DEFAULT_BIAS_TESTS
    
    output_dir = f"test_outputs_bias_{model_type}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'#'*60}")
    print(f"# Attention Bias Tests - {model_type.upper()}")
    print(f"{'#'*60}")
    
    # Load models once
    if model_type == "flux":
        model, clip, vae, mt = load_flux_models()
    else:
        model, clip, vae, mt = load_sdxl_models()
    
    model, clip, freefuse_data = load_loras(model, clip, model_type)
    freefuse_data, conditioning, neg_conditioning = setup_conditioning(clip, freefuse_data, model_type)
    
    # Run tests
    results = []
    for config in tests:
        success, error = run_bias_test(
            config, model, clip, vae, freefuse_data,
            conditioning, neg_conditioning,
            model_type, output_dir
        )
        results.append((config.name, success, error))
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, s, _ in results if s)
    print(f"Total: {len(results)}, Passed: {passed}, Failed: {len(results) - passed}")
    print()
    
    for name, success, error in results:
        status = "✅" if success else "❌"
        print(f"  {status} {name}")
        if error:
            print(f"      Error: {error}")
    
    print(f"\nOutputs saved to: {output_dir}/")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Attention Bias Effect Test")
    parser.add_argument("--model", type=str, choices=["flux", "sdxl"], default="flux",
                       help="Model type to test")
    parser.add_argument("--all", action="store_true",
                       help="Run all predefined tests")
    
    # Custom test parameters
    parser.add_argument("--bias-scale", type=float, default=None,
                       help="Negative bias scale (custom test)")
    parser.add_argument("--positive-scale", type=float, default=None,
                       help="Positive bias scale (custom test)")
    parser.add_argument("--no-bidirectional", action="store_true",
                       help="Disable bidirectional bias")
    parser.add_argument("--no-positive", action="store_true",
                       help="Disable positive bias")
    parser.add_argument("--bias-blocks", type=str, default="all",
                       help="Which blocks to apply bias to")
    
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=768)  # 4:3 aspect ratio
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    if args.all or (args.bias_scale is None and args.positive_scale is None):
        # Run all predefined tests
        run_all_tests(args.model)
    else:
        # Run custom test
        bias_scale = args.bias_scale if args.bias_scale is not None else 5.0
        positive_scale = args.positive_scale if args.positive_scale is not None else 1.0
        
        run_custom_test(
            model_type=args.model,
            bias_scale=bias_scale,
            positive_scale=positive_scale,
            bidirectional=not args.no_bidirectional,
            use_positive=not args.no_positive,
            bias_blocks=args.bias_blocks,
            width=args.width,
            height=args.height,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
