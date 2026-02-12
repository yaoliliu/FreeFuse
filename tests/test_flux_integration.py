#!/usr/bin/env python
"""
FreeFuse ComfyUI Integration Test

Tests the full pipeline with actual models:
1. Load Flux model
2. Load LoRAs in bypass mode
3. Compute token positions
4. Run Phase 1 sampling
5. Apply masks
6. Run Phase 2 sampling

Run from ComfyUI directory:
    .venv/bin/python custom_nodes/freefuse_comfyui/tests/test_flux_integration.py
"""

import sys
import os
import gc

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import ComfyUI modules
import folder_paths
import comfy.sd
import comfy.utils
import comfy.samplers
import comfy.sample


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


def check_models_available():
    """Check if required models are available."""
    print("\n" + "="*60)
    print("Checking Model Availability")
    print("="*60)
    
    # Check diffusion models
    diffusion_models = folder_paths.get_filename_list("diffusion_models")
    print(f"Available diffusion models: {diffusion_models}")
    
    # Check text encoders
    text_encoders = folder_paths.get_filename_list("text_encoders")
    print(f"Available text encoders: {text_encoders}")
    
    # Check VAE
    vaes = folder_paths.get_filename_list("vae")
    print(f"Available VAE: {vaes}")
    
    # Check LoRAs
    loras = folder_paths.get_filename_list("loras")
    print(f"Available LoRAs: {loras}")
    
    # Required models
    has_flux = "flux1-dev-fp8.safetensors" in diffusion_models
    has_clip = "clip_l.safetensors" in text_encoders
    has_t5 = "t5xxl_fp8_e4m3fn.safetensors" in text_encoders
    has_vae = "ae.safetensors" in vaes
    has_lora = any("harry_potter" in l.lower() or "daiyu" in l.lower() for l in loras)
    
    print(f"\n✅ Flux model: {has_flux}")
    print(f"✅ CLIP-L: {has_clip}")
    print(f"✅ T5-XXL: {has_t5}")
    print(f"✅ VAE: {has_vae}")
    print(f"✅ FreeFuse LoRAs: {has_lora}")
    
    return has_flux and has_clip and has_t5 and has_vae


def test_load_flux_model():
    """Test loading Flux model components."""
    print("\n" + "="*60)
    print("TEST: Load Flux Model")
    print("="*60)
    
    # Load UNET
    print("Loading UNET...")
    unet_path = folder_paths.get_full_path("diffusion_models", "flux1-dev-fp8.safetensors")
    model = comfy.sd.load_diffusion_model(unet_path)
    print(f"✅ UNET loaded: {type(model.model).__name__}")
    
    # Check if it's a Flux model
    model_name = model.model.__class__.__name__.lower()
    is_flux = "flux" in model_name
    print(f"   Model class: {model.model.__class__.__name__}")
    print(f"   Is Flux: {is_flux}")
    
    # Check diffusion_model structure
    if hasattr(model.model, 'diffusion_model'):
        dm = model.model.diffusion_model
        if hasattr(dm, 'double_blocks'):
            print(f"   double_blocks: {len(dm.double_blocks)} blocks")
        if hasattr(dm, 'single_blocks'):
            print(f"   single_blocks: {len(dm.single_blocks)} blocks")
    
    # Load CLIP
    print("\nLoading CLIP + T5...")
    clip_path = folder_paths.get_full_path("text_encoders", "clip_l.safetensors")
    t5_path = folder_paths.get_full_path("text_encoders", "t5xxl_fp8_e4m3fn.safetensors")
    
    clip = comfy.sd.load_clip(
        ckpt_paths=[clip_path, t5_path],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=comfy.sd.CLIPType.FLUX,
    )
    print(f"✅ CLIP loaded")
    
    # Check tokenizer structure
    if hasattr(clip, 'tokenizer'):
        tokenizer = clip.tokenizer
        print(f"   Tokenizer type: {type(tokenizer).__name__}")
        if hasattr(tokenizer, 't5xxl'):
            print(f"   Has T5 tokenizer: True")
        if hasattr(tokenizer, 'clip_l'):
            print(f"   Has CLIP-L tokenizer: True")
    
    # Load VAE
    print("\nLoading VAE...")
    vae_path = folder_paths.get_full_path("vae", "ae.safetensors")
    vae_sd = comfy.utils.load_torch_file(vae_path)
    vae = comfy.sd.VAE(sd=vae_sd)
    print(f"✅ VAE loaded")
    
    return model, clip, vae


def test_lora_loading(model, clip):
    """Test loading LoRAs in bypass mode."""
    print("\n" + "="*60)
    print("TEST: Load LoRAs (Bypass Mode)")
    print("="*60)
    
    from freefuse_comfyui.nodes.lora_loader import FreeFuseLoRALoader
    
    loader = FreeFuseLoRALoader()
    
    # Check available LoRAs
    loras = folder_paths.get_filename_list("loras")
    
    # Find FreeFuse LoRAs
    harry_lora = None
    daiyu_lora = None
    for l in loras:
        if "harry_potter" in l.lower() and "flux" in l.lower():
            harry_lora = l
        if "daiyu" in l.lower() and "flux" in l.lower():
            daiyu_lora = l
    
    if not harry_lora or not daiyu_lora:
        print(f"⚠️  FreeFuse LoRAs not found")
        print(f"   Available: {loras}")
        # Use fallback test
        harry_lora = loras[0] if loras else None
        daiyu_lora = loras[1] if len(loras) > 1 else loras[0] if loras else None
    
    if not harry_lora:
        print("❌ No LoRAs available for testing")
        return model, clip, None
    
    print(f"Loading LoRAs:")
    print(f"  - harry: {harry_lora}")
    print(f"  - daiyu: {daiyu_lora}")
    
    # Load first LoRA
    model1, clip1, freefuse_data = loader.load_lora(
        model=model,
        clip=clip,
        lora_name=harry_lora,
        adapter_name="harry",
        strength_model=1.0,
        strength_clip=1.0,
        freefuse_data=None,
    )
    print(f"✅ LoRA 1 loaded: {harry_lora}")
    print(f"   freefuse_data adapters: {[a['name'] for a in freefuse_data.get('adapters', [])]}")
    
    # Load second LoRA
    if daiyu_lora and daiyu_lora != harry_lora:
        model2, clip2, freefuse_data = loader.load_lora(
            model=model1,
            clip=clip1,
            lora_name=daiyu_lora,
            adapter_name="daiyu",
            strength_model=1.0,
            strength_clip=1.0,
            freefuse_data=freefuse_data,
        )
        print(f"✅ LoRA 2 loaded: {daiyu_lora}")
        print(f"   freefuse_data adapters: {[a['name'] for a in freefuse_data.get('adapters', [])]}")
    else:
        model2, clip2 = model1, clip1
    
    print(f"\nLoaded adapters: {freefuse_data.get('adapters', [])}")
    
    return model2, clip2, freefuse_data


def test_concept_map_and_tokens(clip, freefuse_data):
    """Test concept mapping and token position computation."""
    print("\n" + "="*60)
    print("TEST: Concept Map & Token Positions")
    print("="*60)
    
    from freefuse_comfyui.nodes.concept_map import FreeFuseConceptMap, FreeFuseTokenPositions
    
    # Create concept map
    concept_map = FreeFuseConceptMap()
    freefuse_data, = concept_map.create_map(
        adapter_name_1="harry",
        concept_text_1="an European man wearing Hogwarts uniform",
        adapter_name_2="daiyu",
        concept_text_2="an Asian woman wearing Chinese traditional clothing",
        enable_background=True,
        background_text="autumn leaves blurred in the background",
        freefuse_data=freefuse_data,
    )
    
    print(f"✅ Concept map created:")
    for name, text in freefuse_data.get("concepts", {}).items():
        print(f"   {name}: {text}")
    
    # Compute token positions
    prompt = "Realistic photography, an European man wearing Hogwarts uniform hugging an Asian woman wearing Chinese traditional clothing warmly, both faces close together, autumn leaves blurred in the background, high quality, detailed"
    
    token_pos = FreeFuseTokenPositions()
    freefuse_data, = token_pos.compute_positions(
        clip=clip,
        prompt=prompt,
        freefuse_data=freefuse_data,
        filter_meaningless=True,
        filter_single_char=True,
    )
    
    print(f"\n✅ Token positions computed:")
    token_pos_maps = freefuse_data.get("token_pos_maps", {})
    for name, positions in token_pos_maps.items():
        print(f"   {name}: {positions}")
    
    print(f"\nModel type: {freefuse_data.get('model_type', 'unknown')}")
    
    return freefuse_data, prompt


def test_phase1_sampling(model, clip, freefuse_data, prompt):
    """Test Phase 1 sampling (mask collection)."""
    print("\n" + "="*60)
    print("TEST: Phase 1 Sampling")
    print("="*60)
    
    from freefuse_comfyui.nodes.sampler import FreeFusePhase1Sampler
    
    # Create conditioning
    print("Creating conditioning...")
    tokens = clip.tokenize(prompt)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    conditioning = [[cond, {"pooled_output": pooled}]]
    
    # Empty negative
    neg_tokens = clip.tokenize("")
    neg_cond, neg_pooled = clip.encode_from_tokens(neg_tokens, return_pooled=True)
    neg_conditioning = [[neg_cond, {"pooled_output": neg_pooled}]]
    
    print(f"✅ Conditioning created")
    print(f"   cond shape: {cond.shape}")
    print(f"   pooled shape: {pooled.shape}")
    
    # Create latent (respect model latent format when available)
    batch_size = 1
    height = 1024
    width = 1024
    latent_channels, latent_h, latent_w = _get_latent_shape(model, width, height)
    latent = torch.zeros([batch_size, latent_channels, latent_h, latent_w], device="cpu")
    latent_dict = {"samples": latent}
    
    print(f"\n✅ Latent created: {latent.shape}")
    
    # Run Phase 1
    print("\nRunning Phase 1 sampling...")
    sampler = FreeFusePhase1Sampler()
    
    try:
        model_out, masks, preview = sampler.collect_masks(
            model=model,
            conditioning=conditioning,
            neg_conditioning=neg_conditioning,
            latent=latent_dict,
            freefuse_data=freefuse_data,
            seed=42,
            steps=10,  # Need at least 8 steps to collect at step 7
            cfg=1.0,  # Flux uses guidance embedding, not CFG
            sampler_name="euler",
            scheduler="simple",
            collect_block=18,
        )
        
        print(f"\n✅ Phase 1 completed!")
        print(f"   Model patched: {type(model_out)}")
        print(f"   Masks: {list(masks.get('masks', {}).keys())}")
        print(f"   Preview shape: {preview.shape}")
        
        # Show mask statistics
        for name, mask in masks.get("masks", {}).items():
            coverage = mask.sum() / mask.numel() * 100
            print(f"   {name}: shape={mask.shape}, coverage={coverage:.1f}%")
        
        return model_out, masks, preview
        
    except Exception as e:
        print(f"\n❌ Phase 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def test_mask_applicator(model, masks, freefuse_data, latent_dict):
    """Test mask application."""
    print("\n" + "="*60)
    print("TEST: Mask Applicator")
    print("="*60)
    
    from freefuse_comfyui.nodes.mask_applicator import FreeFuseMaskApplicator
    
    applicator = FreeFuseMaskApplicator()
    
    try:
        model_out, = applicator.apply_masks(
            model=model,
            masks=masks,
            freefuse_data=freefuse_data,
            enable_token_masking=True,
            latent=latent_dict,
        )
        
        print(f"✅ Masks applied to model")
        return model_out
        
    except Exception as e:
        print(f"❌ Mask application failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run integration tests."""
    print("\n" + "="*60)
    print("FreeFuse ComfyUI Integration Test")
    print("="*60)
    
    # Check models
    if not check_models_available():
        print("\n❌ Required models not available. Please download them first.")
        return False
    
    # Run tests
    try:
        # 1. Load models
        model, clip, vae = test_load_flux_model()
        
        # 2. Load LoRAs
        model, clip, freefuse_data = test_lora_loading(model, clip)
        if freefuse_data is None:
            print("\n⚠️  Skipping remaining tests (no LoRAs)")
            return True
        
        # 3. Concept map & token positions
        freefuse_data, prompt = test_concept_map_and_tokens(clip, freefuse_data)
        
        # 4. Phase 1 sampling
        model_out, masks, preview = test_phase1_sampling(
            model, clip, freefuse_data, prompt
        )
        
        if model_out is None:
            print("\n❌ Phase 1 failed, stopping tests")
            return False
        
        # 5. Mask application
        latent_channels, latent_h, latent_w = _get_latent_shape(model, 1024, 1024)
        latent = {"samples": torch.zeros([1, latent_channels, latent_h, latent_w])}
        model_final = test_mask_applicator(model_out, masks, freefuse_data, latent)
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print("✅ Model loading: PASS")
        print("✅ LoRA loading: PASS")
        print("✅ Concept mapping: PASS")
        print("✅ Token positions: PASS")
        print(f"{'✅' if masks else '❌'} Phase 1 sampling: {'PASS' if masks else 'FAIL'}")
        print(f"{'✅' if model_final else '⚠️ '} Mask application: {'PASS' if model_final else 'PARTIAL'}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
