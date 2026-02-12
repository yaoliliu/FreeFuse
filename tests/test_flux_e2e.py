#!/usr/bin/env python
"""
FreeFuse ComfyUI End-to-End Test

Complete test that generates an actual image using the FreeFuse pipeline:
1. Load Flux model and LoRAs
2. Phase 1: Collect attention and generate masks
3. Phase 2: Generate image with masked LoRA application

Run from ComfyUI directory:
    python custom_nodes/freefuse_comfyui/tests/test_flux_e2e.py --fix-test
    python custom_nodes/freefuse_comfyui/tests/test_flux_e2e.py --quick
    python custom_nodes/freefuse_comfyui/tests/test_flux_e2e.py --all
"""

import sys
import os
import gc
import time


def _find_comfyui_dir(start_dir: str) -> str:
    """Find the ComfyUI root directory by walking up from start_dir."""
    cur = os.path.abspath(start_dir)
    for _ in range(10):
        if os.path.isdir(os.path.join(cur, "comfy")) and os.path.isfile(os.path.join(cur, "main.py")):
            return cur
        # Also check sibling ComfyUI directory
        sibling_comfyui = os.path.join(cur, "ComfyUI")
        if os.path.isdir(sibling_comfyui) and os.path.isdir(os.path.join(sibling_comfyui, "comfy")):
            return sibling_comfyui
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    raise FileNotFoundError("Could not locate ComfyUI directory (expected comfy/ and main.py)")

# Determine paths
script_dir = os.path.dirname(os.path.abspath(__file__))
freefuse_comfyui_dir = os.path.dirname(script_dir)

# Robustly find ComfyUI root (works from both repo-root and ComfyUI/custom_nodes symlink)
# For freefuse_comfyui as sibling of ComfyUI, we need to go up to repo root first
freefuse_root = os.path.dirname(freefuse_comfyui_dir)
comfyui_dir = _find_comfyui_dir(freefuse_root)

# Add ComfyUI to path first
if comfyui_dir not in sys.path:
    sys.path.insert(0, comfyui_dir)
# Add freefuse_comfyui parent to path
if freefuse_root not in sys.path:
    sys.path.insert(0, freefuse_root)

# Change to ComfyUI directory for folder_paths to work
os.chdir(comfyui_dir)

import torch
import logging
from PIL import Image
import numpy as np

# These tests are pure inference; ensure autograd is disabled globally.
torch.set_grad_enabled(False)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import ComfyUI modules
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


def run_freefuse_e2e_test():
    """Run complete FreeFuse pipeline test."""
    print("\n" + "="*70)
    print("FreeFuse ComfyUI End-to-End Test")
    print("="*70)
    
    # Configuration
    prompt = "Realistic photography, an European man wearing Hogwarts uniform hugging an Asian woman wearing Chinese traditional clothing warmly, both faces close together, autumn leaves blurred in the background, high quality, detailed"
    
    concept_map = {
        "harry": "an European man wearing Hogwarts uniform",
        "daiyu": "an Asian woman wearing Chinese traditional clothing",
    }
    
    # Background concept (tokens after all concepts)
    background_text = "autumn leaves blurred in the background"
    
    width, height = 1024, 1024
    seed = 42
    total_steps = 28  # Full sigma schedule for both phases
    collect_step = 5  # Collect attention at step 5, then stop Phase 1
    guidance = 3.5
    
    # ============================================================
    # Step 1: Load Models
    # ============================================================
    print("\n[Step 1] Loading models...")
    
    # Load UNET
    unet_path = folder_paths.get_full_path("diffusion_models", "flux1-dev-fp8.safetensors")
    model = comfy.sd.load_diffusion_model(unet_path)
    print(f"  ✅ UNET loaded: {type(model.model).__name__}")
    
    # Load CLIP
    clip_path = folder_paths.get_full_path("text_encoders", "clip_l.safetensors")
    t5_path = folder_paths.get_full_path("text_encoders", "t5xxl_fp8_e4m3fn.safetensors")
    clip = comfy.sd.load_clip(
        ckpt_paths=[clip_path, t5_path],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=comfy.sd.CLIPType.FLUX,
    )
    print(f"  ✅ CLIP loaded")
    
    # Load VAE
    vae_path = folder_paths.get_full_path("vae", "ae.safetensors")
    vae_sd = comfy.utils.load_torch_file(vae_path)
    vae = comfy.sd.VAE(sd=vae_sd)
    print(f"  ✅ VAE loaded")
    
    # ============================================================
    # Step 2: Load LoRAs in bypass mode
    # ============================================================
    print("\n[Step 2] Loading LoRAs in bypass mode...")
    
    from freefuse_comfyui.nodes.lora_loader import FreeFuseLoRALoader
    loader = FreeFuseLoRALoader()
    
    # Load harry potter LoRA
    model, clip, freefuse_data = loader.load_lora(
        model=model,
        clip=clip,
        lora_name="harry_potter_flux.safetensors",
        adapter_name="harry",
        strength_model=1.0,
        strength_clip=1.0,
        freefuse_data=None,
    )
    print(f"  ✅ LoRA 'harry' loaded")
    
    # Load daiyu LoRA
    model, clip, freefuse_data = loader.load_lora(
        model=model,
        clip=clip,
        lora_name="daiyu_lin_flux.safetensors",
        adapter_name="daiyu",
        strength_model=1.0,
        strength_clip=1.0,
        freefuse_data=freefuse_data,
    )
    print(f"  ✅ LoRA 'daiyu' loaded")
    print(f"  Adapters: {[a['name'] for a in freefuse_data.get('adapters', [])]}")
    
    # ============================================================
    # Step 3: Create concept map and compute token positions
    # ============================================================
    print("\n[Step 3] Computing token positions...")
    
    from freefuse_comfyui.nodes.concept_map import FreeFuseConceptMap, FreeFuseTokenPositions
    
    # Create concept map
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
    print(f"  Concepts: {freefuse_data.get('concepts', {})}")
    
    # Compute token positions
    token_pos = FreeFuseTokenPositions()
    freefuse_data, = token_pos.compute_positions(
        clip=clip,
        prompt=prompt,
        freefuse_data=freefuse_data,
        filter_meaningless=True,
        filter_single_char=True,
    )
    
    token_pos_maps = freefuse_data.get("token_pos_maps", {})
    print(f"  Token positions:")
    for name, pos in token_pos_maps.items():
        print(f"    {name}: {pos}")
    
    # ============================================================
    # Step 4: Create conditioning
    # ============================================================
    print("\n[Step 4] Creating conditioning...")
    
    tokens = clip.tokenize(prompt)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    
    # Apply FluxGuidance manually
    conditioning = [[cond, {"pooled_output": pooled, "guidance": guidance}]]
    
    # Empty negative
    neg_tokens = clip.tokenize("")
    neg_cond, neg_pooled = clip.encode_from_tokens(neg_tokens, return_pooled=True)
    neg_conditioning = [[neg_cond, {"pooled_output": neg_pooled, "guidance": guidance}]]
    
    print(f"  ✅ Conditioning created")
    print(f"     cond shape: {cond.shape}")
    
    # ============================================================
    # Step 5: Phase 1 - Collect masks
    # ============================================================
    print("\n[Step 5] Phase 1: Collecting attention masks...")
    
    from freefuse_comfyui.nodes.sampler import FreeFusePhase1Sampler
    
    # Create latent
    latent_channels, latent_h, latent_w = _get_latent_shape(model, width, height)
    latent = torch.zeros([1, latent_channels, latent_h, latent_w], device="cpu")
    latent_dict = {"samples": latent}
    
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
            steps=total_steps,  # Full sigma schedule
            collect_step=collect_step,  # Stop early after this step
            cfg=1.0,  # Flux uses guidance embedding, not CFG
            sampler_name="euler",
            scheduler="simple",
            collect_block=18,
        )
    phase1_time = time.time() - start_time
    
    masks = masks_output.get("masks", {})
    print(f"  ✅ Phase 1 complete ({phase1_time:.1f}s)")
    print(f"  Generated {len(masks)} masks:")
    for name, mask in masks.items():
        coverage = mask.sum() / mask.numel() * 100
        print(f"    {name}: {mask.shape}, coverage={coverage:.1f}%")
    
    # Save mask preview
    preview_np = (preview[0].numpy() * 255).astype(np.uint8)
    preview_img = Image.fromarray(preview_np)
    preview_path = "freefuse_mask_preview.png"
    preview_img.save(preview_path)
    print(f"  Saved mask preview to {preview_path}")
    
    # ============================================================
    # Step 6: Apply masks to model
    # ============================================================
    print("\n[Step 6] Applying masks to model...")
    
    from freefuse_comfyui.nodes.mask_applicator import FreeFuseMaskApplicator
    
    applicator = FreeFuseMaskApplicator()
    model_masked, = applicator.apply_masks(
        model=model_phase1,
        masks=masks_output,
        freefuse_data=freefuse_data,
        enable_token_masking=True,
        latent=latent_dict,
        # Test with attention bias on all blocks (double + single)
        enable_attention_bias=True,
        bias_scale=5.0,
        positive_bias_scale=1.0,
        bidirectional=True,
        use_positive_bias=True,
        bias_blocks="all",  # Apply to both double and single stream blocks
    )
    print(f"  ✅ Masks applied to model")
    
    # ============================================================
    # Step 7: Phase 2 - Full generation
    # ============================================================
    print("\n[Step 7] Phase 2: Full generation...")
    
    # Create fresh latent for Phase 2
    noise = comfy.sample.prepare_noise(latent, seed, None)
    
    start_time = time.time()
    with torch.inference_mode():
        samples = comfy.sample.sample(
            model_masked,
            noise,
            total_steps,
            1.0,  # cfg=1.0 for Flux
            "euler",
            "simple",
            conditioning,
            neg_conditioning,
            latent,
            denoise=1.0,
            disable_noise=False,
            start_step=0,
            last_step=total_steps,
            force_full_denoise=True,
            noise_mask=None,
            callback=None,
            seed=seed,
        )
    phase2_time = time.time() - start_time
    print(f"  ✅ Phase 2 complete ({phase2_time:.1f}s)")
    
    # ============================================================
    # Step 8: Decode and save
    # ============================================================
    print("\n[Step 8] Decoding and saving...")
    
    with torch.inference_mode():
        decoded = vae.decode(samples)
    
    # Convert to image
    if decoded.dim() == 4:
        image = decoded[0]  # (C, H, W) or (H, W, C)
    else:
        image = decoded
    
    if image.shape[0] in [1, 3, 4]:  # (C, H, W)
        image = image.permute(1, 2, 0)  # -> (H, W, C)
    
    image = image.detach().cpu().numpy()
    image = (image * 255).clip(0, 255).astype(np.uint8)
    
    if image.shape[-1] == 1:
        image = image.squeeze(-1)
    
    output_path = "freefuse_output.png"
    Image.fromarray(image).save(output_path)
    print(f"  ✅ Saved output to {output_path}")
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    print(f"  Phase 1 time: {phase1_time:.1f}s")
    print(f"  Phase 2 time: {phase2_time:.1f}s")
    print(f"  Total time: {phase1_time + phase2_time:.1f}s")
    print(f"  Output: {output_path}")
    print(f"  Mask preview: {preview_path}")
    
    return True


def run_baseline_test():
    """Run baseline test without FreeFuse for comparison."""
    print("\n" + "="*70)
    print("Baseline Test (No FreeFuse)")
    print("="*70)
    
    prompt = "Realistic photography, an European man wearing Hogwarts uniform hugging an Asian woman wearing Chinese traditional clothing warmly, both faces close together, autumn leaves blurred in the background, high quality, detailed"
    
    width, height = 1024, 1024
    seed = 42
    steps = 28
    guidance = 3.5
    
    # Load models
    print("\n[Step 1] Loading models...")
    
    unet_path = folder_paths.get_full_path("diffusion_models", "flux1-dev-fp8.safetensors")
    model = comfy.sd.load_diffusion_model(unet_path)
    
    clip_path = folder_paths.get_full_path("text_encoders", "clip_l.safetensors")
    t5_path = folder_paths.get_full_path("text_encoders", "t5xxl_fp8_e4m3fn.safetensors")
    clip = comfy.sd.load_clip(
        ckpt_paths=[clip_path, t5_path],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=comfy.sd.CLIPType.FLUX,
    )
    
    vae_path = folder_paths.get_full_path("vae", "ae.safetensors")
    vae_sd = comfy.utils.load_torch_file(vae_path)
    vae = comfy.sd.VAE(sd=vae_sd)
    
    # Load LoRAs (standard mode, merged)
    print("\n[Step 2] Loading LoRAs (standard mode)...")
    
    lora1_path = folder_paths.get_full_path("loras", "harry_potter_flux.safetensors")
    lora1 = comfy.utils.load_torch_file(lora1_path, safe_load=True)
    model, clip = comfy.sd.load_lora_for_models(model, clip, lora1, 1.0, 1.0)
    
    lora2_path = folder_paths.get_full_path("loras", "daiyu_lin_flux.safetensors")
    lora2 = comfy.utils.load_torch_file(lora2_path, safe_load=True)
    model, clip = comfy.sd.load_lora_for_models(model, clip, lora2, 1.0, 1.0)
    
    print(f"  ✅ LoRAs loaded (merged)")
    
    # Create conditioning
    print("\n[Step 3] Creating conditioning...")
    
    tokens = clip.tokenize(prompt)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    conditioning = [[cond, {"pooled_output": pooled, "guidance": guidance}]]
    
    neg_tokens = clip.tokenize("")
    neg_cond, neg_pooled = clip.encode_from_tokens(neg_tokens, return_pooled=True)
    neg_conditioning = [[neg_cond, {"pooled_output": neg_pooled, "guidance": guidance}]]
    
    # Generate
    print("\n[Step 4] Generating...")
    
    latent_channels, latent_h, latent_w = _get_latent_shape(model, width, height)
    latent = torch.zeros([1, latent_channels, latent_h, latent_w], device="cpu")
    noise = comfy.sample.prepare_noise(latent, seed, None)
    
    start_time = time.time()
    with torch.inference_mode():
        samples = comfy.sample.sample(
            model,
            noise,
            steps,
            1.0,
            "euler",
            "simple",
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
    gen_time = time.time() - start_time
    
    # Decode
    print("\n[Step 5] Decoding...")
    
    with torch.inference_mode():
        decoded = vae.decode(samples)
    
    if decoded.dim() == 4:
        image = decoded[0]
    else:
        image = decoded
    
    if image.shape[0] in [1, 3, 4]:
        image = image.permute(1, 2, 0)
    
    image = image.cpu().numpy()
    image = (image * 255).clip(0, 255).astype(np.uint8)
    
    if image.shape[-1] == 1:
        image = image.squeeze(-1)
    
    output_path = "baseline_output.png"
    Image.fromarray(image).save(output_path)
    
    print(f"\n✅ Baseline complete")
    print(f"  Time: {gen_time:.1f}s")
    print(f"  Output: {output_path}")
    
    return True


def run_bypass_lora_fix_test():
    """
    Test the bypass LoRA loading fix for Flux fused QKV weights.
    
    This test verifies that our fixed loader correctly handles tuple keys
    that ComfyUI's original implementation fails to load.
    """
    print("\n" + "="*70)
    print("Bypass LoRA Fix Test (Flux QKV Weights)")
    print("="*70)
    
    import comfy.lora
    import comfy.lora_convert
    import comfy.weight_adapter
    
    # Build Flux key mapping
    mmdit_config = {
        'depth': 19,
        'depth_single_blocks': 38,
        'hidden_size': 3072,
    }
    diffusers_keys = comfy.utils.flux_to_diffusers(mmdit_config, output_prefix='diffusion_model.')
    
    # Build key_map (simulating what model_lora_keys_unet does for Flux)
    key_map = {}
    for k in diffusers_keys:
        if k.endswith('.weight'):
            to = diffusers_keys[k]
            key_map['transformer.{}'.format(k[:-len('.weight')])] = to
    
    # Load LoRA
    print("\n[Step 1] Loading LoRA file...")
    lora_path = folder_paths.get_full_path("loras", "harry_potter_flux.safetensors")
    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
    lora_converted = comfy.lora_convert.convert_lora(lora)
    
    # Load with key_map
    loaded = comfy.lora.load_lora(lora_converted, key_map, log_missing=False)
    print(f"  Total loaded entries: {len(loaded)}")
    
    # Separate adapters
    bypass_patches = {}
    for key, patch_data in loaded.items():
        if isinstance(patch_data, comfy.weight_adapter.WeightAdapterBase):
            bypass_patches[key] = patch_data
    
    print(f"  Bypass adapters: {len(bypass_patches)}")
    
    # Count tuple vs string keys
    tuple_keys = [k for k in bypass_patches.keys() if isinstance(k, tuple)]
    string_keys = [k for k in bypass_patches.keys() if isinstance(k, str)]
    print(f"  Tuple keys (fused QKV): {len(tuple_keys)}")
    print(f"  String keys (regular): {len(string_keys)}")
    
    # Build model state dict keys
    model_sd_keys = set()
    for i in range(19):
        model_sd_keys.add(f'diffusion_model.double_blocks.{i}.img_attn.qkv.weight')
        model_sd_keys.add(f'diffusion_model.double_blocks.{i}.txt_attn.qkv.weight')
        model_sd_keys.add(f'diffusion_model.double_blocks.{i}.img_attn.proj.weight')
        model_sd_keys.add(f'diffusion_model.double_blocks.{i}.txt_attn.proj.weight')
        model_sd_keys.add(f'diffusion_model.double_blocks.{i}.img_mod.lin.weight')
        model_sd_keys.add(f'diffusion_model.double_blocks.{i}.txt_mod.lin.weight')
        model_sd_keys.add(f'diffusion_model.double_blocks.{i}.img_mlp.0.weight')
        model_sd_keys.add(f'diffusion_model.double_blocks.{i}.img_mlp.2.weight')
        model_sd_keys.add(f'diffusion_model.double_blocks.{i}.txt_mlp.0.weight')
        model_sd_keys.add(f'diffusion_model.double_blocks.{i}.txt_mlp.2.weight')
    for i in range(38):
        model_sd_keys.add(f'diffusion_model.single_blocks.{i}.linear1.weight')
        model_sd_keys.add(f'diffusion_model.single_blocks.{i}.linear2.weight')
        model_sd_keys.add(f'diffusion_model.single_blocks.{i}.modulation.lin.weight')
    
    # Test original ComfyUI logic (BUGGY)
    print("\n[Step 2] Testing original ComfyUI logic...")
    original_loaded = 0
    original_not_loaded = 0
    for key in bypass_patches:
        if key in model_sd_keys:  # BUG: tuple never matches string set!
            original_loaded += 1
        else:
            original_not_loaded += 1
    
    print(f"  Original loaded: {original_loaded} / {len(bypass_patches)}")
    print(f"  Original NOT loaded (BUG): {original_not_loaded}")
    
    # Test fixed logic
    print("\n[Step 3] Testing fixed logic...")
    from freefuse_comfyui.freefuse_core.bypass_lora_loader import OffsetBypassInjectionManager
    
    manager = OffsetBypassInjectionManager()
    fixed_loaded = 0
    fixed_not_loaded = 0
    
    for key, adapter in bypass_patches.items():
        # Parse tuple key - THIS IS THE FIX
        offset = None
        if isinstance(key, str):
            actual_key = key
        else:
            actual_key = key[0]
            offset = key[1]
        
        if actual_key in model_sd_keys:
            manager.add_adapter(actual_key, adapter, strength=1.0, offset=offset)
            fixed_loaded += 1
        else:
            fixed_not_loaded += 1
    
    print(f"  Fixed loaded: {fixed_loaded} / {len(bypass_patches)}")
    print(f"  Fixed NOT loaded: {fixed_not_loaded}")
    print(f"  Total adapters in manager: {manager.get_total_adapter_count()}")
    print(f"  Unique modules: {len(manager.adapters)}")
    
    # Verify QKV grouping
    print("\n[Step 4] Verifying QKV grouping...")
    qkv_modules = [k for k in manager.adapters.keys() if 'qkv' in k]
    print(f"  QKV modules: {len(qkv_modules)}")
    
    if qkv_modules:
        sample_module = qkv_modules[0]
        adapters_in_module = manager.adapters[sample_module]
        print(f"  Sample module: {sample_module}")
        print(f"  Adapters in module: {len(adapters_in_module)}")
        for _, _, offset, _ in adapters_in_module:
            print(f"    offset: {offset}")
    
    # Summary
    print("\n" + "="*70)
    print("BYPASS LORA FIX TEST RESULTS")
    print("="*70)
    
    improvement = fixed_loaded - original_loaded
    success = improvement == len(tuple_keys)
    
    print(f"  Original ComfyUI: {original_loaded} / {len(bypass_patches)} loaded")
    print(f"  Fixed version: {fixed_loaded} / {len(bypass_patches)} loaded")
    print(f"  Improvement: +{improvement} adapters")
    print(f"  Expected improvement: +{len(tuple_keys)} (all tuple keys)")
    
    if success:
        print(f"\n  ✅ TEST PASSED: All {len(tuple_keys)} QKV adapters now load correctly!")
    else:
        print(f"\n  ❌ TEST FAILED: Expected +{len(tuple_keys)}, got +{improvement}")
    
    return success


def run_quick_inference_test():
    """
    Quick inference test to verify LoRA effects are applied correctly.
    
    Generates a small image to verify the fix works in practice.
    """
    print("\n" + "="*70)
    print("Quick Inference Test")
    print("="*70)
    
    prompt = "an European man wearing Hogwarts uniform, portrait, detailed"
    width, height = 512, 512  # Small size for quick test
    seed = 42
    steps = 8  # Minimal steps
    guidance = 3.5
    
    # Load models
    print("\n[Step 1] Loading models...")
    
    unet_path = folder_paths.get_full_path("diffusion_models", "flux1-dev-fp8.safetensors")
    model = comfy.sd.load_diffusion_model(unet_path)
    print(f"  ✅ UNET loaded")
    
    clip_path = folder_paths.get_full_path("text_encoders", "clip_l.safetensors")
    t5_path = folder_paths.get_full_path("text_encoders", "t5xxl_fp8_e4m3fn.safetensors")
    clip = comfy.sd.load_clip(
        ckpt_paths=[clip_path, t5_path],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=comfy.sd.CLIPType.FLUX,
    )
    print(f"  ✅ CLIP loaded")
    
    vae_path = folder_paths.get_full_path("vae", "ae.safetensors")
    vae_sd = comfy.utils.load_torch_file(vae_path)
    vae = comfy.sd.VAE(sd=vae_sd)
    print(f"  ✅ VAE loaded")
    
    # Load LoRA with fixed bypass loader
    print("\n[Step 2] Loading LoRA with fixed bypass loader...")
    
    from freefuse_comfyui.freefuse_core.bypass_lora_loader import load_bypass_lora_for_models_fixed
    
    lora_path = folder_paths.get_full_path("loras", "harry_potter_flux.safetensors")
    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
    
    model_lora, clip_lora = load_bypass_lora_for_models_fixed(
        model, clip, lora, 1.0, 1.0, adapter_name="harry"
    )
    print(f"  ✅ LoRA loaded in bypass mode")
    
    # Check injections were set (ModelPatcher stores them on `injections`)
    injections = getattr(model_lora, "injections", {})
    print(f"  Injections: {list(injections.keys())}")
    
    # Create conditioning
    print("\n[Step 3] Creating conditioning...")
    
    tokens = clip_lora.tokenize(prompt)
    cond, pooled = clip_lora.encode_from_tokens(tokens, return_pooled=True)
    conditioning = [[cond, {"pooled_output": pooled, "guidance": guidance}]]
    
    neg_tokens = clip_lora.tokenize("")
    neg_cond, neg_pooled = clip_lora.encode_from_tokens(neg_tokens, return_pooled=True)
    neg_conditioning = [[neg_cond, {"pooled_output": neg_pooled, "guidance": guidance}]]
    
    print(f"  ✅ Conditioning created")
    
    # Generate
    print("\n[Step 4] Generating (quick test)...")
    
    latent_channels, latent_h, latent_w = _get_latent_shape(model, width, height)
    latent = torch.zeros([1, latent_channels, latent_h, latent_w], device="cpu")
    noise = comfy.sample.prepare_noise(latent, seed, None)
    
    start_time = time.time()
    with torch.inference_mode():
        samples = comfy.sample.sample(
            model_lora,
            noise,
            steps,
            1.0,
            "euler",
            "simple",
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
    gen_time = time.time() - start_time
    print(f"  ✅ Generation complete ({gen_time:.1f}s)")
    
    # Decode
    print("\n[Step 5] Decoding...")
    
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
    
    output_path = "quick_inference_test.png"
    Image.fromarray(image).save(output_path)
    
    print(f"\n✅ Quick inference test complete")
    print(f"  Time: {gen_time:.1f}s")
    print(f"  Output: {output_path}")
    
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="Run baseline test only")
    parser.add_argument("--freefuse", action="store_true", help="Run FreeFuse test only")
    parser.add_argument("--fix-test", action="store_true", help="Run bypass LoRA fix test only")
    parser.add_argument("--quick", action="store_true", help="Run quick inference test only")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    args = parser.parse_args()
    
    try:
        if args.baseline:
            run_baseline_test()
        elif args.freefuse:
            run_freefuse_e2e_test()
        elif args.fix_test:
            run_bypass_lora_fix_test()
        elif args.quick:
            run_quick_inference_test()
        elif args.all:
            # Run all tests
            print("\n" + "#"*70)
            print("# RUNNING ALL TESTS")
            print("#"*70)
            
            # Test 1: Bypass LoRA fix
            success1 = run_bypass_lora_fix_test()
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Test 2: Quick inference
            print("\n" + "-"*70)
            success2 = run_quick_inference_test()
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Test 3: FreeFuse E2E
            print("\n" + "-"*70)
            success3 = run_freefuse_e2e_test()
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Test 4: Baseline comparison
            print("\n" + "-"*70)
            success4 = run_baseline_test()
            
            # Summary
            print("\n" + "#"*70)
            print("# ALL TESTS COMPLETE")
            print("#"*70)
            print(f"  Bypass LoRA Fix Test: {'✅ PASSED' if success1 else '❌ FAILED'}")
            print(f"  Quick Inference Test: {'✅ PASSED' if success2 else '❌ FAILED'}")
            print(f"  FreeFuse E2E Test: {'✅ PASSED' if success3 else '❌ FAILED'}")
            print(f"  Baseline Test: {'✅ PASSED' if success4 else '❌ FAILED'}")
            
        else:
            # Default: run FreeFuse E2E and baseline
            run_freefuse_e2e_test()
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("\n" + "-"*70)
            
            run_baseline_test()
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
