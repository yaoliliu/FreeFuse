#!/usr/bin/env python
"""
FreeFuse Parameter Test Suite

Comprehensive tests for various parameters on both Flux and SDXL models:
1. Different aspect ratios (portrait, landscape, square)
2. Different collect_block values
3. Different similarity map parameters (temperature, top_k_ratio)
4. Different mask generation parameters (bg_scale, morphological_cleaning, iterations)
5. LoRA enable/disable in Phase 1
6. Different attention bias scales

Run from repository root:
    python freefuse_comfyui/tests/test_parameters.py --flux
    python freefuse_comfyui/tests/test_parameters.py --sdxl
    python freefuse_comfyui/tests/test_parameters.py --all
"""

import sys
import os
import gc
import time
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple


def _find_comfyui_dir(start_dir: str) -> str:
    """Find the ComfyUI root directory by walking up from start_dir."""
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


@dataclass
class TestConfig:
    """Configuration for a single test run."""
    name: str
    width: int
    height: int
    collect_block: int
    temperature: float  # 0 = auto
    top_k_ratio: float
    disable_lora_phase1: bool
    bg_scale: float
    use_morphological_cleaning: bool
    balance_iterations: int
    bias_scale: float
    positive_bias_scale: float
    description: str = ""


# Test configurations for different scenarios
ASPECT_RATIO_TESTS = [
    TestConfig(
        name="square_1024",
        width=1024, height=1024,
        collect_block=18, temperature=0, top_k_ratio=0.3,
        disable_lora_phase1=True, bg_scale=0.95,
        use_morphological_cleaning=False, balance_iterations=15,
        bias_scale=5.0, positive_bias_scale=1.0,
        description="Standard 1:1 square (baseline)"
    ),
    TestConfig(
        name="portrait_768x1024",
        width=768, height=1024,
        collect_block=18, temperature=0, top_k_ratio=0.3,
        disable_lora_phase1=True, bg_scale=0.95,
        use_morphological_cleaning=False, balance_iterations=15,
        bias_scale=5.0, positive_bias_scale=1.0,
        description="Portrait 3:4 aspect ratio"
    ),
    TestConfig(
        name="landscape_1024x768",
        width=1024, height=768,
        collect_block=18, temperature=0, top_k_ratio=0.3,
        disable_lora_phase1=True, bg_scale=0.95,
        use_morphological_cleaning=False, balance_iterations=15,
        bias_scale=5.0, positive_bias_scale=1.0,
        description="Landscape 4:3 aspect ratio"
    ),
    TestConfig(
        name="wide_1280x768",
        width=1280, height=768,
        collect_block=18, temperature=0, top_k_ratio=0.3,
        disable_lora_phase1=True, bg_scale=0.95,
        use_morphological_cleaning=False, balance_iterations=15,
        bias_scale=5.0, positive_bias_scale=1.0,
        description="Wide 5:3 aspect ratio"
    ),
    TestConfig(
        name="tall_768x1280",
        width=768, height=1280,
        collect_block=18, temperature=0, top_k_ratio=0.3,
        disable_lora_phase1=True, bg_scale=0.95,
        use_morphological_cleaning=False, balance_iterations=15,
        bias_scale=5.0, positive_bias_scale=1.0,
        description="Tall 3:5 aspect ratio"
    ),
]

BLOCK_TESTS = [
    TestConfig(
        name="block_0",
        width=1024, height=1024,
        collect_block=0, temperature=0, top_k_ratio=0.3,
        disable_lora_phase1=True, bg_scale=0.95,
        use_morphological_cleaning=False, balance_iterations=15,
        bias_scale=5.0, positive_bias_scale=1.0,
        description="First block (early features)"
    ),
    TestConfig(
        name="block_9",
        width=1024, height=1024,
        collect_block=9, temperature=0, top_k_ratio=0.3,
        disable_lora_phase1=True, bg_scale=0.95,
        use_morphological_cleaning=False, balance_iterations=15,
        bias_scale=5.0, positive_bias_scale=1.0,
        description="Middle-early block"
    ),
    TestConfig(
        name="block_18",
        width=1024, height=1024,
        collect_block=18, temperature=0, top_k_ratio=0.3,
        disable_lora_phase1=True, bg_scale=0.95,
        use_morphological_cleaning=False, balance_iterations=15,
        bias_scale=5.0, positive_bias_scale=1.0,
        description="Default block (baseline)"
    ),
    TestConfig(
        name="block_38",
        width=1024, height=1024,
        collect_block=38, temperature=0, top_k_ratio=0.3,
        disable_lora_phase1=True, bg_scale=0.95,
        use_morphological_cleaning=False, balance_iterations=15,
        bias_scale=5.0, positive_bias_scale=1.0,
        description="Single stream block (Flux only)"
    ),
]

SIMMAP_PARAM_TESTS = [
    TestConfig(
        name="temp_100",
        width=1024, height=1024,
        collect_block=18, temperature=100.0, top_k_ratio=0.3,
        disable_lora_phase1=True, bg_scale=0.95,
        use_morphological_cleaning=False, balance_iterations=15,
        bias_scale=5.0, positive_bias_scale=1.0,
        description="Very low temperature (sharper attention)"
    ),
    TestConfig(
        name="temp_10000",
        width=1024, height=1024,
        collect_block=18, temperature=10000.0, top_k_ratio=0.3,
        disable_lora_phase1=True, bg_scale=0.95,
        use_morphological_cleaning=False, balance_iterations=15,
        bias_scale=5.0, positive_bias_scale=1.0,
        description="Very high temperature (smoother attention)"
    ),
    TestConfig(
        name="topk_0.05",
        width=1024, height=1024,
        collect_block=18, temperature=0, top_k_ratio=0.05,
        disable_lora_phase1=True, bg_scale=0.95,
        use_morphological_cleaning=False, balance_iterations=15,
        bias_scale=5.0, positive_bias_scale=1.0,
        description="Very low top_k ratio (sparse selection)"
    ),
    TestConfig(
        name="topk_0.8",
        width=1024, height=1024,
        collect_block=18, temperature=0, top_k_ratio=0.8,
        disable_lora_phase1=True, bg_scale=0.95,
        use_morphological_cleaning=False, balance_iterations=15,
        bias_scale=5.0, positive_bias_scale=1.0,
        description="Very high top_k ratio (dense selection)"
    ),
]

MASK_PARAM_TESTS = [
    TestConfig(
        name="bg_scale_0.5",
        width=1024, height=1024,
        collect_block=18, temperature=0, top_k_ratio=0.3,
        disable_lora_phase1=True, bg_scale=0.5,
        use_morphological_cleaning=False, balance_iterations=15,
        bias_scale=5.0, positive_bias_scale=1.0,
        description="Low background scale (less background)"
    ),
    TestConfig(
        name="bg_scale_1.5",
        width=1024, height=1024,
        collect_block=18, temperature=0, top_k_ratio=0.3,
        disable_lora_phase1=True, bg_scale=1.5,
        use_morphological_cleaning=False, balance_iterations=15,
        bias_scale=5.0, positive_bias_scale=1.0,
        description="High background scale (more background)"
    ),
    TestConfig(
        name="morph_cleaning_on",
        width=1024, height=1024,
        collect_block=18, temperature=0, top_k_ratio=0.3,
        disable_lora_phase1=True, bg_scale=0.95,
        use_morphological_cleaning=True, balance_iterations=15,
        bias_scale=5.0, positive_bias_scale=1.0,
        description="Morphological cleaning enabled"
    ),
    TestConfig(
        name="balance_iter_3",
        width=1024, height=1024,
        collect_block=18, temperature=0, top_k_ratio=0.3,
        disable_lora_phase1=True, bg_scale=0.95,
        use_morphological_cleaning=False, balance_iterations=3,
        bias_scale=5.0, positive_bias_scale=1.0,
        description="Few balance iterations (faster, less balanced)"
    ),
    TestConfig(
        name="balance_iter_50",
        width=1024, height=1024,
        collect_block=18, temperature=0, top_k_ratio=0.3,
        disable_lora_phase1=True, bg_scale=0.95,
        use_morphological_cleaning=False, balance_iterations=50,
        bias_scale=5.0, positive_bias_scale=1.0,
        description="Many balance iterations (slower, more balanced)"
    ),
]

LORA_PHASE1_TESTS = [
    TestConfig(
        name="lora_disabled_phase1",
        width=1024, height=1024,
        collect_block=18, temperature=0, top_k_ratio=0.3,
        disable_lora_phase1=True, bg_scale=0.95,
        use_morphological_cleaning=False, balance_iterations=15,
        bias_scale=5.0, positive_bias_scale=1.0,
        description="LoRA disabled in Phase 1 (default, cleaner attention)"
    ),
    TestConfig(
        name="lora_enabled_phase1",
        width=1024, height=1024,
        collect_block=18, temperature=0, top_k_ratio=0.3,
        disable_lora_phase1=False, bg_scale=0.95,
        use_morphological_cleaning=False, balance_iterations=15,
        bias_scale=5.0, positive_bias_scale=1.0,
        description="LoRA enabled in Phase 1 (may have LoRA influence on attention)"
    ),
]

ATTN_BIAS_TESTS = [
    TestConfig(
        name="bias_scale_0",
        width=1024, height=1024,
        collect_block=18, temperature=0, top_k_ratio=0.3,
        disable_lora_phase1=True, bg_scale=0.95,
        use_morphological_cleaning=False, balance_iterations=15,
        bias_scale=0.0, positive_bias_scale=0.0,
        description="No attention bias (baseline without bias)"
    ),
    TestConfig(
        name="bias_scale_20",
        width=1024, height=1024,
        collect_block=18, temperature=0, top_k_ratio=0.3,
        disable_lora_phase1=True, bg_scale=0.95,
        use_morphological_cleaning=False, balance_iterations=15,
        bias_scale=20.0, positive_bias_scale=1.0,
        description="Very high negative bias (strong suppression)"
    ),
    TestConfig(
        name="pos_bias_scale_5",
        width=1024, height=1024,
        collect_block=18, temperature=0, top_k_ratio=0.3,
        disable_lora_phase1=True, bg_scale=0.95,
        use_morphological_cleaning=False, balance_iterations=15,
        bias_scale=5.0, positive_bias_scale=5.0,
        description="High positive bias (strong enhancement)"
    ),
]


def load_flux_models():
    """Load Flux model and return components."""
    print("\n[Loading Flux Models]")
    
    # Find model files
    unet_name = None
    for name in folder_paths.get_filename_list("diffusion_models"):
        if "flux" in name.lower():
            unet_name = name
            break
    
    if not unet_name:
        raise FileNotFoundError("No Flux model found in diffusion_models")
    
    clip_names = []
    for name in folder_paths.get_filename_list("text_encoders"):
        if "clip" in name.lower() or "t5" in name.lower():
            clip_names.append(name)
    
    vae_name = None
    for name in folder_paths.get_filename_list("vae"):
        if "ae" in name.lower() or "flux" in name.lower():
            vae_name = name
            break
    
    # Load models
    from nodes import UNETLoader, DualCLIPLoader, VAELoader
    
    unet_loader = UNETLoader()
    model, = unet_loader.load_unet(unet_name, "default")
    print(f"  ✅ UNET loaded: {unet_name}")
    
    clip_loader = DualCLIPLoader()
    clip_l = next((n for n in clip_names if "clip_l" in n.lower()), None)
    t5 = next((n for n in clip_names if "t5" in n.lower()), None)
    clip, = clip_loader.load_clip(clip_l, t5, "flux")
    print(f"  ✅ CLIP loaded: {clip_l}, {t5}")
    
    vae_loader = VAELoader()
    vae, = vae_loader.load_vae(vae_name)
    print(f"  ✅ VAE loaded: {vae_name}")
    
    return model, clip, vae, "flux"


def load_sdxl_models():
    """Load SDXL model and return components."""
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
        print(f"  Warning: LoRAs not found for {model_type}")
        print(f"  Available: {lora_list}")
        return model, clip, {"adapters": []}
    
    loader = FreeFuseLoRALoader()
    
    model, clip, freefuse_data = loader.load_lora(
        model, clip, harry_lora, "harry", 1.0, 1.0, None
    )
    print(f"  ✅ LoRA loaded: harry ({harry_lora})")
    
    model, clip, freefuse_data = loader.load_lora(
        model, clip, daiyu_lora, "daiyu", 1.0, 1.0, freefuse_data
    )
    print(f"  ✅ LoRA loaded: daiyu ({daiyu_lora})")
    
    return model, clip, freefuse_data


def setup_concepts_and_conditioning(clip, freefuse_data, model_type: str):
    """Set up concept map and conditioning."""
    # prompt = "Realistic photography, an European man wearing Hogwarts uniform hugging an Asian woman wearing Chinese traditional clothing warmly, both faces close together, autumn leaves blurred in the background, high quality, detailed"
    prompt = "Realistic photography, harry potter, an European photorealistic style teenage wizard boy with messy black hair, round wire-frame glasses, and bright green eyes, wearing a white shirt, burgundy and gold striped tie, and dark robes hugging daiyu_lin, a young East Asian photorealistic style woman in traditional Chinese hanfu dress, elaborate black updo hairstyle adorned with delicate white floral hairpins and ornaments, dangling red tassel earrings, soft pink and red color palette, gentle smile with knowing expression, autumn leaves blurred in the background, high quality, detailed"
    negative_prompt = "low quality, blurry, deformed, ugly, bad anatomy"
    
    concept_map = {
        "harry": "harry potter, an European photorealistic style teenage wizard boy with messy black hair, round wire-frame glasses, and bright green eyes, wearing a white shirt, burgundy and gold striped tie, and dark robes",
        "daiyu": "daiyu_lin, a young East Asian photorealistic style woman in traditional Chinese hanfu dress, elaborate black updo hairstyle adorned with delicate white floral hairpins and ornaments, dangling red tassel earrings, soft pink and red color palette, gentle smile with knowing expression",
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
    if model_type == "flux":
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


def run_single_test(
    config: TestConfig,
    model, clip, vae, freefuse_data, 
    conditioning, neg_conditioning,
    model_type: str,
    output_dir: str,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run a single test configuration and return results."""
    
    print(f"\n{'='*70}")
    print(f"Test: {config.name}")
    print(f"Description: {config.description}")
    print(f"{'='*70}")
    print(f"  Width: {config.width}, Height: {config.height}")
    print(f"  Block: {config.collect_block}, Temp: {config.temperature}, TopK: {config.top_k_ratio}")
    print(f"  LoRA Phase1: {'disabled' if config.disable_lora_phase1 else 'enabled'}")
    print(f"  BG Scale: {config.bg_scale}, Morph: {config.use_morphological_cleaning}, Iters: {config.balance_iterations}")
    print(f"  Bias: neg={config.bias_scale}, pos={config.positive_bias_scale}")
    
    from freefuse_comfyui.nodes.sampler import FreeFusePhase1Sampler
    from freefuse_comfyui.nodes.mask_applicator import FreeFuseMaskApplicator
    
    results = {
        "name": config.name,
        "model_type": model_type,
        "success": False,
        "error": None,
        "phase1_time": 0,
        "phase2_time": 0,
        "masks": {},
    }
    
    try:
        # Create latent
        if model_type == "flux":
            latent_h, latent_w = config.height // 8, config.width // 8
            latent = torch.zeros([1, 16, latent_h, latent_w], device="cpu")
        else:
            latent_h, latent_w = config.height // 8, config.width // 8
            latent = torch.zeros([1, 4, latent_h, latent_w], device="cpu")
        
        latent_dict = {"samples": latent}
        
        # Phase 1
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
                steps=28 if model_type == "flux" else 30,
                collect_step=5 if model_type == "flux" else 10,
                cfg=1.0 if model_type == "flux" else 7.0,
                sampler_name="euler",
                scheduler="simple" if model_type == "flux" else "normal",
                collect_block=config.collect_block,
                temperature=config.temperature,
                top_k_ratio=config.top_k_ratio,
                disable_lora_phase1=config.disable_lora_phase1,
                bg_scale=config.bg_scale,
                use_morphological_cleaning=config.use_morphological_cleaning,
                balance_iterations=config.balance_iterations,
            )
        results["phase1_time"] = time.time() - start_time
        
        masks = masks_output.get("masks", {})
        results["masks"] = {name: f"{mask.shape}, coverage={mask.sum()/mask.numel()*100:.1f}%" 
                          for name, mask in masks.items()}
        
        print(f"  Phase 1: {results['phase1_time']:.1f}s, masks: {list(masks.keys())}")
        for name, info in results["masks"].items():
            print(f"    {name}: {info}")
        
        # Save mask preview
        preview_np = (preview[0].numpy() * 255).astype(np.uint8)
        preview_img = Image.fromarray(preview_np)
        preview_path = os.path.join(output_dir, f"{model_type}_{config.name}_mask.png")
        preview_img.save(preview_path)
        print(f"  Saved mask: {preview_path}")
        
        # Apply masks
        applicator = FreeFuseMaskApplicator()
        model_masked, = applicator.apply_masks(
            model=model_phase1,
            masks=masks_output,
            freefuse_data=freefuse_data,
            enable_token_masking=True,
            latent=latent_dict,
            enable_attention_bias=True,
            bias_scale=config.bias_scale,
            positive_bias_scale=config.positive_bias_scale,
            bidirectional=True,
            use_positive_bias=True,
            bias_blocks="all",
        )
        
        # Phase 2 - Generate
        noise = comfy.sample.prepare_noise(latent, seed, None)
        
        start_time = time.time()
        with torch.inference_mode():
            samples = comfy.sample.sample(
                model_masked,
                noise,
                28 if model_type == "flux" else 30,
                1.0 if model_type == "flux" else 7.0,
                "euler",
                "simple" if model_type == "flux" else "normal",
                conditioning,
                neg_conditioning,
                latent,
                denoise=1.0,
                disable_noise=False,
                start_step=0,
                last_step=28 if model_type == "flux" else 30,
                force_full_denoise=True,
                noise_mask=None,
                callback=None,
                seed=seed,
            )
        results["phase2_time"] = time.time() - start_time
        print(f"  Phase 2: {results['phase2_time']:.1f}s")
        
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
        print(f"  Saved output: {output_path}")
        
        results["success"] = True
        
    except Exception as e:
        results["error"] = str(e)
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results


def run_test_suite(model_type: str, test_categories: List[str] = None):
    """Run full test suite for a model type."""
    
    print(f"\n{'#'*70}")
    print(f"# FreeFuse Parameter Test Suite - {model_type.upper()}")
    print(f"{'#'*70}")
    
    # Create output directory
    output_dir = f"test_outputs_{model_type}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    if model_type == "flux":
        model, clip, vae, mt = load_flux_models()
    else:
        model, clip, vae, mt = load_sdxl_models()
    
    # Load LoRAs
    model, clip, freefuse_data = load_loras(model, clip, model_type)
    
    # Setup concepts and conditioning
    freefuse_data, conditioning, neg_conditioning = setup_concepts_and_conditioning(
        clip, freefuse_data, model_type
    )
    
    # Select test categories
    all_tests = {
        "aspect": ASPECT_RATIO_TESTS,
        "block": BLOCK_TESTS,
        "simmap": SIMMAP_PARAM_TESTS,
        "mask": MASK_PARAM_TESTS,
        "lora": LORA_PHASE1_TESTS,
        "bias": ATTN_BIAS_TESTS,
    }
    
    if test_categories is None:
        test_categories = list(all_tests.keys())
    
    # Run tests
    all_results = []
    
    for category in test_categories:
        if category not in all_tests:
            print(f"Warning: Unknown test category '{category}'")
            continue
        
        tests = all_tests[category]
        print(f"\n{'='*70}")
        print(f"Test Category: {category.upper()}")
        print(f"{'='*70}")
        
        for config in tests:
            # Skip single-stream block tests for SDXL
            if model_type == "sdxl" and config.collect_block > 20:
                print(f"\nSkipping {config.name} (block {config.collect_block} not available in SDXL)")
                continue
            
            result = run_single_test(
                config, model, clip, vae, freefuse_data,
                conditioning, neg_conditioning,
                model_type, output_dir
            )
            all_results.append(result)
    
    # Print summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    
    passed = sum(1 for r in all_results if r["success"])
    failed = len(all_results) - passed
    
    print(f"Total: {len(all_results)}, Passed: {passed}, Failed: {failed}")
    print()
    
    for result in all_results:
        status = "✅" if result["success"] else "❌"
        print(f"{status} {result['name']}: Phase1={result['phase1_time']:.1f}s, Phase2={result['phase2_time']:.1f}s")
        if result["error"]:
            print(f"   Error: {result['error']}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="FreeFuse Parameter Test Suite")
    parser.add_argument("--flux", action="store_true", help="Run Flux tests")
    parser.add_argument("--sdxl", action="store_true", help="Run SDXL tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--category", type=str, nargs="+", 
                       choices=["aspect", "block", "simmap", "mask", "lora", "bias"],
                       help="Specific test categories to run")
    parser.add_argument("--quick", action="store_true", 
                       help="Run only aspect ratio tests (quick validation)")
    
    args = parser.parse_args()
    
    # Default to all if nothing specified
    if not args.flux and not args.sdxl and not args.all:
        args.all = True
    
    categories = args.category
    if args.quick:
        categories = ["aspect"]
    
    results = {}
    
    if args.flux or args.all:
        try:
            results["flux"] = run_test_suite("flux", categories)
        except Exception as e:
            print(f"Flux tests failed: {e}")
            import traceback
            traceback.print_exc()
    
    if args.sdxl or args.all:
        try:
            results["sdxl"] = run_test_suite("sdxl", categories)
        except Exception as e:
            print(f"SDXL tests failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print(f"\n{'#'*70}")
    print("# FINAL SUMMARY")
    print(f"{'#'*70}")
    
    for model_type, model_results in results.items():
        if model_results:
            passed = sum(1 for r in model_results if r["success"])
            print(f"{model_type.upper()}: {passed}/{len(model_results)} passed")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
