#!/usr/bin/env python
"""
FreeFuse ComfyUI SDXL End-to-End Test

Complete test that generates an actual image using the FreeFuse pipeline with SDXL:
1. Load SDXL model and LoRAs
2. Phase 1: Collect attention and generate masks
3. Phase 2: Generate image with masked LoRA application

Run from ComfyUI directory:
    python custom_nodes/freefuse_comfyui/tests/test_sdxl_e2e.py --fix-test
    python custom_nodes/freefuse_comfyui/tests/test_sdxl_e2e.py --quick
    python custom_nodes/freefuse_comfyui/tests/test_sdxl_e2e.py --all
    
Or from repository root:
    python freefuse_comfyui/tests/test_sdxl_e2e.py --all
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


def run_freefuse_sdxl_e2e_test():
    """Run complete FreeFuse SDXL pipeline test."""
    print("\n" + "="*70)
    print("FreeFuse ComfyUI SDXL End-to-End Test")
    print("="*70)
    
    # Configuration
    prompt = "Realistic photography, an European man wearing Hogwarts uniform hugging an Asian woman wearing Chinese traditional clothing warmly, both faces close together, autumn leaves blurred in the background, high quality, detailed"
    negative_prompt = "low quality, blurry, deformed, ugly, bad anatomy"
    
    concept_map = {
        "harry": "an European man wearing Hogwarts uniform",
        "daiyu": "an Asian woman wearing Chinese traditional clothing",
    }
    
    # Background concept (tokens after all concepts)
    background_text = "autumn leaves blurred in the background"
    
    width, height = 1024, 1024
    seed = 42
    total_steps = 30  # SDXL typically uses more steps
    collect_step = 10  # Collect attention at step 10
    guidance = 7.0  # SDXL uses higher CFG
    
    # ============================================================
    # Step 1: Load Models
    # ============================================================
    print("\n[Step 1] Loading SDXL models...")
    
    # Find SDXL checkpoint
    sdxl_checkpoint_name = None
    checkpoint_list = folder_paths.get_filename_list("checkpoints")
    for ckpt in checkpoint_list:
        ckpt_lower = ckpt.lower()
        if "sdxl" in ckpt_lower or "xl" in ckpt_lower:
            sdxl_checkpoint_name = ckpt
            break
    
    if sdxl_checkpoint_name is None:
        # Try diffusion_models folder for SDXL unet
        diffusion_models = folder_paths.get_filename_list("diffusion_models")
        for dm in diffusion_models:
            if "sdxl" in dm.lower():
                print(f"  Found SDXL in diffusion_models: {dm}")
                break
        
        print("\n[ERROR] No SDXL checkpoint found!")
        print("Please download an SDXL checkpoint to ComfyUI/models/checkpoints/")
        print("Suggested: stable-diffusion-xl-base-1.0 from HuggingFace")
        print("\nExample commands:")
        print("  cd ComfyUI/models/checkpoints")
        print('  curl -LJO "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"')
        return False
    
    print(f"  Using checkpoint: {sdxl_checkpoint_name}")
    
    # Load SDXL checkpoint (includes UNet, CLIP, VAE)
    checkpoint_path = folder_paths.get_full_path("checkpoints", sdxl_checkpoint_name)
    
    # Use ComfyUI's checkpoint loader
    from nodes import CheckpointLoaderSimple
    ckpt_loader = CheckpointLoaderSimple()
    model, clip, vae = ckpt_loader.load_checkpoint(sdxl_checkpoint_name)
    
    print(f"  ✅ SDXL Model loaded: {type(model.model).__name__}")
    print(f"  ✅ CLIP loaded")
    print(f"  ✅ VAE loaded")
    
    # ============================================================
    # Step 2: Load LoRAs in bypass mode
    # ============================================================
    print("\n[Step 2] Loading SDXL LoRAs in bypass mode...")
    
    from freefuse_comfyui.nodes.lora_loader import FreeFuseLoRALoader
    loader = FreeFuseLoRALoader()
    
    # Check for XL LoRAs
    lora_list = folder_paths.get_filename_list("loras")
    harry_lora = None
    daiyu_lora = None
    
    for lora in lora_list:
        lora_lower = lora.lower()
        if "harry" in lora_lower and "xl" in lora_lower:
            harry_lora = lora
        if "daiyu" in lora_lower and "xl" in lora_lower:
            daiyu_lora = lora
    
    if not harry_lora or not daiyu_lora:
        print("\n[WARNING] XL LoRAs not found, looking for any available LoRAs...")
        for lora in lora_list:
            print(f"  Available: {lora}")
        
        # Try to find any XL loras
        for lora in lora_list:
            if "xl" in lora.lower():
                if not harry_lora:
                    harry_lora = lora
                elif not daiyu_lora:
                    daiyu_lora = lora
        
        if not harry_lora:
            harry_lora = "harry_potter_xl.safetensors"
        if not daiyu_lora:
            daiyu_lora = "daiyu_lin_xl.safetensors"
    
    print(f"  Using LoRAs: {harry_lora}, {daiyu_lora}")
    
    # Load harry potter XL LoRA
    try:
        model, clip, freefuse_data = loader.load_lora(
            model=model,
            clip=clip,
            lora_name=harry_lora,
            adapter_name="harry",
            strength_model=1.0,
            strength_clip=1.0,
            freefuse_data=None,
        )
        print(f"  ✅ LoRA 'harry' loaded: {harry_lora}")
    except Exception as e:
        print(f"  ❌ Failed to load harry LoRA: {e}")
        return False
    
    # Load daiyu XL LoRA
    try:
        model, clip, freefuse_data = loader.load_lora(
            model=model,
            clip=clip,
            lora_name=daiyu_lora,
            adapter_name="daiyu",
            strength_model=1.0,
            strength_clip=1.0,
            freefuse_data=freefuse_data,
        )
        print(f"  ✅ LoRA 'daiyu' loaded: {daiyu_lora}")
    except Exception as e:
        print(f"  ❌ Failed to load daiyu LoRA: {e}")
        return False
    
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
    
    # SDXL uses dual CLIP encoding
    tokens = clip.tokenize(prompt)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    
    # SDXL conditioning format
    conditioning = [[cond, {"pooled_output": pooled}]]
    
    # Negative conditioning
    neg_tokens = clip.tokenize(negative_prompt)
    neg_cond, neg_pooled = clip.encode_from_tokens(neg_tokens, return_pooled=True)
    neg_conditioning = [[neg_cond, {"pooled_output": neg_pooled}]]
    
    print(f"  ✅ Conditioning created")
    print(f"     cond shape: {cond.shape}")
    
    # ============================================================
    # Step 5: Phase 1 - Collect masks
    # ============================================================
    print("\n[Step 5] Phase 1: Collecting attention masks...")
    
    from freefuse_comfyui.nodes.sampler import FreeFusePhase1Sampler
    
    # Create latent - SDXL uses 4 channels
    latent_h, latent_w = height // 8, width // 8
    latent = torch.zeros([1, 4, latent_h, latent_w], device="cpu")
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
            steps=total_steps,
            collect_step=collect_step,
            cfg=guidance,
            sampler_name="euler",
            scheduler="normal",  # SDXL commonly uses normal scheduler
            collect_block=0,  # For SDXL, use output block - this is handled internally
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
    preview_path = "freefuse_sdxl_mask_preview.png"
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
        # SDXL attention bias settings
        enable_attention_bias=True,
        bias_scale=5.0,
        positive_bias_scale=1.0,
        bidirectional=False,  # SDXL uses separate cross-attention
        use_positive_bias=True,
        bias_blocks="all",
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
            guidance,  # SDXL uses CFG scale
            "euler",
            "normal",
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
    
    output_path = "freefuse_sdxl_output.png"
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


def run_sdxl_baseline_test():
    """Run baseline SDXL test without FreeFuse for comparison."""
    print("\n" + "="*70)
    print("SDXL Baseline Test (No FreeFuse)")
    print("="*70)
    
    prompt = "Realistic photography, an European man wearing Hogwarts uniform hugging an Asian woman wearing Chinese traditional clothing warmly, both faces close together, autumn leaves blurred in the background, high quality, detailed"
    negative_prompt = "low quality, blurry, deformed, ugly, bad anatomy"
    
    width, height = 1024, 1024
    seed = 42
    steps = 30
    guidance = 7.0
    
    # Load models
    print("\n[Step 1] Loading SDXL models...")
    
    # Find SDXL checkpoint
    sdxl_checkpoint_name = None
    checkpoint_list = folder_paths.get_filename_list("checkpoints")
    for ckpt in checkpoint_list:
        ckpt_lower = ckpt.lower()
        if "sdxl" in ckpt_lower or "xl" in ckpt_lower:
            sdxl_checkpoint_name = ckpt
            break
    
    if sdxl_checkpoint_name is None:
        print("\n[ERROR] No SDXL checkpoint found!")
        return False
    
    print(f"  Using checkpoint: {sdxl_checkpoint_name}")
    
    from nodes import CheckpointLoaderSimple
    ckpt_loader = CheckpointLoaderSimple()
    model, clip, vae = ckpt_loader.load_checkpoint(sdxl_checkpoint_name)
    print(f"  ✅ SDXL Model loaded")
    
    # Load LoRAs (standard mode, merged)
    print("\n[Step 2] Loading LoRAs (standard mode)...")
    
    # Find XL LoRAs
    lora_list = folder_paths.get_filename_list("loras")
    harry_lora = None
    daiyu_lora = None
    
    for lora in lora_list:
        lora_lower = lora.lower()
        if "harry" in lora_lower and "xl" in lora_lower:
            harry_lora = lora
        if "daiyu" in lora_lower and "xl" in lora_lower:
            daiyu_lora = lora
    
    if harry_lora:
        lora1_path = folder_paths.get_full_path("loras", harry_lora)
        lora1 = comfy.utils.load_torch_file(lora1_path, safe_load=True)
        model, clip = comfy.sd.load_lora_for_models(model, clip, lora1, 1.0, 1.0)
        print(f"  ✅ Loaded: {harry_lora}")
    
    if daiyu_lora:
        lora2_path = folder_paths.get_full_path("loras", daiyu_lora)
        lora2 = comfy.utils.load_torch_file(lora2_path, safe_load=True)
        model, clip = comfy.sd.load_lora_for_models(model, clip, lora2, 1.0, 1.0)
        print(f"  ✅ Loaded: {daiyu_lora}")
    
    # Create conditioning
    print("\n[Step 3] Creating conditioning...")
    
    tokens = clip.tokenize(prompt)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    conditioning = [[cond, {"pooled_output": pooled}]]
    
    neg_tokens = clip.tokenize(negative_prompt)
    neg_cond, neg_pooled = clip.encode_from_tokens(neg_tokens, return_pooled=True)
    neg_conditioning = [[neg_cond, {"pooled_output": neg_pooled}]]
    
    # Generate
    print("\n[Step 4] Generating...")
    
    latent_h, latent_w = height // 8, width // 8
    latent = torch.zeros([1, 4, latent_h, latent_w], device="cpu")
    noise = comfy.sample.prepare_noise(latent, seed, None)
    
    start_time = time.time()
    with torch.inference_mode():
        samples = comfy.sample.sample(
            model,
            noise,
            steps,
            guidance,
            "euler",
            "normal",
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
    
    output_path = "baseline_sdxl_output.png"
    Image.fromarray(image).save(output_path)
    
    print(f"\n✅ SDXL Baseline complete")
    print(f"  Time: {gen_time:.1f}s")
    print(f"  Output: {output_path}")
    
    return True


def run_sdxl_lora_test():
    """
    Test SDXL LoRA loading and verify keys are correctly mapped.
    """
    print("\n" + "="*70)
    print("SDXL LoRA Loading Test")
    print("="*70)
    
    import comfy.lora
    import comfy.lora_convert
    
    # Find XL LoRA
    lora_list = folder_paths.get_filename_list("loras")
    xl_lora = None
    for lora in lora_list:
        if "xl" in lora.lower():
            xl_lora = lora
            break
    
    if not xl_lora:
        print("  No XL LoRA found, skipping test")
        return True
    
    print(f"  Testing with: {xl_lora}")
    
    # Load LoRA file
    lora_path = folder_paths.get_full_path("loras", xl_lora)
    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
    
    # Check LoRA structure
    print(f"\n  LoRA keys sample:")
    lora_keys = list(lora.keys())[:10]
    for k in lora_keys:
        print(f"    {k}")
    
    # Convert LoRA
    lora_converted = comfy.lora_convert.convert_lora(lora)
    
    print(f"\n  Converted LoRA keys sample:")
    conv_keys = list(lora_converted.keys())[:10]
    for k in conv_keys:
        print(f"    {k}")
    
    # Check for SDXL-specific keys
    unet_keys = [k for k in lora_converted.keys() if "unet" in k.lower() or "model" in k.lower()]
    te_keys = [k for k in lora_converted.keys() if "text" in k.lower() or "clip" in k.lower() or "encoder" in k.lower()]
    
    print(f"\n  UNet-related keys: {len(unet_keys)}")
    print(f"  Text encoder keys: {len(te_keys)}")
    print(f"  Total keys: {len(lora_converted)}")
    
    # Verify it's likely SDXL
    has_sdxl_patterns = any(
        "down_blocks" in k or "up_blocks" in k or "mid_block" in k
        for k in lora_converted.keys()
    )
    
    if has_sdxl_patterns:
        print(f"\n  ✅ LoRA appears to be SDXL format (has UNet block patterns)")
    else:
        print(f"\n  ⚠️ LoRA may not be SDXL format")
    
    return True


def run_quick_sdxl_inference_test():
    """
    Quick SDXL inference test with minimal steps.
    """
    print("\n" + "="*70)
    print("Quick SDXL Inference Test")
    print("="*70)
    
    prompt = "an European man wearing Hogwarts uniform, portrait, detailed"
    negative_prompt = "low quality, blurry"
    width, height = 512, 512  # Smaller for quick test
    seed = 42
    steps = 8  # Minimal steps
    guidance = 7.0
    
    # Load models
    print("\n[Step 1] Loading SDXL model...")
    
    sdxl_checkpoint_name = None
    checkpoint_list = folder_paths.get_filename_list("checkpoints")
    for ckpt in checkpoint_list:
        if "sdxl" in ckpt.lower() or "xl" in ckpt.lower():
            sdxl_checkpoint_name = ckpt
            break
    
    if not sdxl_checkpoint_name:
        print("  No SDXL checkpoint found, skipping")
        return True
    
    from nodes import CheckpointLoaderSimple
    ckpt_loader = CheckpointLoaderSimple()
    model, clip, vae = ckpt_loader.load_checkpoint(sdxl_checkpoint_name)
    print(f"  ✅ Model loaded")
    
    # Load single LoRA
    print("\n[Step 2] Loading LoRA...")
    
    lora_list = folder_paths.get_filename_list("loras")
    xl_lora = None
    for lora in lora_list:
        if "xl" in lora.lower() and "harry" in lora.lower():
            xl_lora = lora
            break
    
    if xl_lora:
        lora_path = folder_paths.get_full_path("loras", xl_lora)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        model, clip = comfy.sd.load_lora_for_models(model, clip, lora, 1.0, 1.0)
        print(f"  ✅ LoRA loaded: {xl_lora}")
    
    # Create conditioning
    print("\n[Step 3] Creating conditioning...")
    
    tokens = clip.tokenize(prompt)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    conditioning = [[cond, {"pooled_output": pooled}]]
    
    neg_tokens = clip.tokenize(negative_prompt)
    neg_cond, neg_pooled = clip.encode_from_tokens(neg_tokens, return_pooled=True)
    neg_conditioning = [[neg_cond, {"pooled_output": neg_pooled}]]
    
    # Generate
    print("\n[Step 4] Generating (quick test)...")
    
    latent_h, latent_w = height // 8, width // 8
    latent = torch.zeros([1, 4, latent_h, latent_w], device="cpu")
    noise = comfy.sample.prepare_noise(latent, seed, None)
    
    start_time = time.time()
    with torch.inference_mode():
        samples = comfy.sample.sample(
            model,
            noise,
            steps,
            guidance,
            "euler",
            "normal",
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
    
    output_path = "quick_sdxl_inference_test.png"
    Image.fromarray(image).save(output_path)
    
    print(f"\n✅ Quick SDXL inference test complete")
    print(f"  Time: {gen_time:.1f}s")
    print(f"  Output: {output_path}")
    
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="Run SDXL baseline test only")
    parser.add_argument("--freefuse", action="store_true", help="Run FreeFuse SDXL test only")
    parser.add_argument("--lora-test", action="store_true", help="Run SDXL LoRA loading test only")
    parser.add_argument("--quick", action="store_true", help="Run quick SDXL inference test only")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    args = parser.parse_args()
    
    try:
        if args.baseline:
            run_sdxl_baseline_test()
        elif args.freefuse:
            run_freefuse_sdxl_e2e_test()
        elif args.lora_test:
            run_sdxl_lora_test()
        elif args.quick:
            run_quick_sdxl_inference_test()
        elif args.all:
            # Run all SDXL tests
            print("\n" + "#"*70)
            print("# RUNNING ALL SDXL TESTS")
            print("#"*70)
            
            # Test 1: LoRA loading test
            success1 = run_sdxl_lora_test()
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Test 2: Quick inference
            print("\n" + "-"*70)
            success2 = run_quick_sdxl_inference_test()
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Test 3: FreeFuse E2E
            print("\n" + "-"*70)
            success3 = run_freefuse_sdxl_e2e_test()
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Test 4: Baseline comparison
            print("\n" + "-"*70)
            success4 = run_sdxl_baseline_test()
            
            # Summary
            print("\n" + "#"*70)
            print("# ALL SDXL TESTS COMPLETE")
            print("#"*70)
            print(f"  SDXL LoRA Test: {'✅ PASSED' if success1 else '❌ FAILED'}")
            print(f"  Quick Inference Test: {'✅ PASSED' if success2 else '❌ FAILED'}")
            print(f"  FreeFuse E2E Test: {'✅ PASSED' if success3 else '❌ FAILED'}")
            print(f"  Baseline Test: {'✅ PASSED' if success4 else '❌ FAILED'}")
            
        else:
            # Default: run FreeFuse E2E and baseline
            run_freefuse_sdxl_e2e_test()
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("\n" + "-"*70)
            
            run_sdxl_baseline_test()
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
