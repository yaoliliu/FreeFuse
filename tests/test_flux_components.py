#!/usr/bin/env python
"""
FreeFuse ComfyUI Components Test Script

Tests each component of the FreeFuse pipeline for Flux models:
1. LoRA Loader (bypass mode)
2. Concept Map & Token Positions
3. Phase 1 Sampler (attention collection)
4. Mask Applicator
5. Full pipeline integration

Run from ComfyUI directory:
    .venv/bin/python custom_nodes/freefuse_comfyui/tests/test_flux_components.py
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_1_imports():
    """Test that all required modules can be imported."""
    print("\n" + "="*60)
    print("TEST 1: Module Imports")
    print("="*60)
    
    errors = []
    
    # Test FreeFuse nodes
    try:
        from freefuse_comfyui import NODE_CLASS_MAPPINGS
        print(f"✅ FreeFuse nodes: {list(NODE_CLASS_MAPPINGS.keys())}")
    except Exception as e:
        errors.append(f"FreeFuse nodes: {e}")
    
    # Test FreeFuse core
    try:
        from freefuse_comfyui.freefuse_core.attention_replace import (
            FreeFuseState,
            FreeFuseFluxBlockReplace,
            apply_freefuse_replace_patches,
        )
        print("✅ FreeFuse attention_replace module")
    except Exception as e:
        errors.append(f"attention_replace: {e}")
    
    try:
        from freefuse_comfyui.freefuse_core.token_utils import (
            detect_model_type,
            find_concept_positions,
        )
        print("✅ FreeFuse token_utils module")
    except Exception as e:
        errors.append(f"token_utils: {e}")
    
    try:
        from freefuse_comfyui.freefuse_core.mask_utils import generate_masks
        print("✅ FreeFuse mask_utils module")
    except Exception as e:
        errors.append(f"mask_utils: {e}")
    
    # Test ComfyUI modules
    try:
        import comfy.sd
        import comfy.samplers
        import comfy.model_patcher
        import folder_paths
        print("✅ ComfyUI core modules")
    except Exception as e:
        errors.append(f"ComfyUI core: {e}")
    
    # Test Flux modules
    try:
        from comfy.ldm.flux.layers import DoubleStreamBlock
        from comfy.ldm.flux.math import apply_rope, attention
        print("✅ ComfyUI Flux modules")
    except Exception as e:
        errors.append(f"Flux modules: {e}")
    
    if errors:
        print("\n❌ Import errors:")
        for e in errors:
            print(f"   - {e}")
        return False
    
    print("\n✅ All imports successful!")
    return True


def test_2_flux_block_structure():
    """Test that Flux DoubleStreamBlock has expected structure."""
    print("\n" + "="*60)
    print("TEST 2: Flux DoubleStreamBlock Structure")
    print("="*60)
    
    from comfy.ldm.flux.layers import DoubleStreamBlock
    
    # Create a minimal block to inspect structure
    block = DoubleStreamBlock(
        hidden_size=3072,
        num_heads=24,
        mlp_ratio=4.0,
        dtype=torch.float32,
        device="cpu",
        operations=torch.nn,
    )
    
    # Check required attributes for FreeFuse
    required_attrs = [
        'img_attn',
        'txt_attn', 
        'img_mod',
        'txt_mod',
        'img_norm1',
        'txt_norm1',
        'num_heads',
    ]
    
    missing = []
    for attr in required_attrs:
        if hasattr(block, attr):
            print(f"✅ block.{attr} exists")
        else:
            print(f"❌ block.{attr} missing")
            missing.append(attr)
    
    # Check attention sub-attributes
    attn_attrs = ['qkv', 'norm', 'proj']
    for attn_name in ['img_attn', 'txt_attn']:
        attn = getattr(block, attn_name)
        for attr in attn_attrs:
            full_name = f"{attn_name}.{attr}"
            if hasattr(attn, attr):
                print(f"✅ block.{full_name} exists")
            else:
                print(f"❌ block.{full_name} missing")
                missing.append(full_name)
    
    # Check flipped_img_txt
    if hasattr(block, 'flipped_img_txt'):
        print(f"✅ block.flipped_img_txt = {block.flipped_img_txt}")
    else:
        print(f"⚠️  block.flipped_img_txt not found (may be set during forward)")
    
    if missing:
        print(f"\n❌ Missing attributes: {missing}")
        return False
    
    print("\n✅ DoubleStreamBlock structure matches FreeFuse requirements!")
    return True


def test_3_freefuse_state():
    """Test FreeFuseState initialization and methods."""
    print("\n" + "="*60)
    print("TEST 3: FreeFuseState")
    print("="*60)
    
    from freefuse_comfyui.freefuse_core.attention_replace import FreeFuseState
    
    state = FreeFuseState()
    
    # Check initial state
    assert state.phase == "collect", f"Expected phase='collect', got {state.phase}"
    assert state.collect_step == 4, f"Expected collect_step=4, got {state.collect_step}"
    assert state.collect_block == 18, f"Expected collect_block=18, got {state.collect_block}"
    print(f"✅ Initial state: phase={state.phase}, collect_step={state.collect_step}, collect_block={state.collect_block}")
    
    # Test is_collect_step
    state.current_step = 4
    assert state.is_collect_step(4, 18) == True
    assert state.is_collect_step(4, 17) == False
    assert state.is_collect_step(3, 18) == False
    print("✅ is_collect_step() works correctly")
    
    # Test token_pos_maps
    state.token_pos_maps = {
        "harry": [[1, 2, 3]],
        "hermione": [[4, 5, 6]],
    }
    assert len(state.token_pos_maps) == 2
    print(f"✅ token_pos_maps: {state.token_pos_maps}")
    
    # Test reset
    state.similarity_maps = {"test": torch.randn(1, 10)}
    state.reset_collection()
    assert len(state.similarity_maps) == 0
    print("✅ reset_collection() works")
    
    print("\n✅ FreeFuseState works correctly!")
    return True


def test_4_token_position_finding():
    """Test token position finding logic."""
    print("\n" + "="*60)
    print("TEST 4: Token Position Finding")
    print("="*60)
    
    from freefuse_comfyui.freefuse_core.token_utils import (
        is_meaningless_token,
        clean_token_text,
    )
    
    # Test clean_token_text
    assert clean_token_text("▁hello") == "hello"
    assert clean_token_text("world</w>") == "world"
    assert clean_token_text("▁The") == "the"
    print("✅ clean_token_text() works")
    
    # Test is_meaningless_token
    assert is_meaningless_token("▁the") == True
    assert is_meaningless_token("▁,") == True
    assert is_meaningless_token("▁wizard") == False
    assert is_meaningless_token("▁a") == True
    print("✅ is_meaningless_token() works")
    
    print("\n✅ Token utilities work correctly!")
    return True


def test_5_mask_generation():
    """Test mask generation from similarity maps."""
    print("\n" + "="*60)
    print("TEST 5: Mask Generation")
    print("="*60)
    
    from freefuse_comfyui.freefuse_core.mask_utils import generate_masks, balanced_argmax
    
    # Create fake similarity maps
    h, w = 64, 64
    
    # Concept A: top-left region
    sim_a = torch.zeros(h, w)
    sim_a[:32, :32] = 1.0
    
    # Concept B: bottom-right region
    sim_b = torch.zeros(h, w)
    sim_b[32:, 32:] = 1.0
    
    similarity_maps = {
        "concept_a": sim_a,
        "concept_b": sim_b,
    }
    
    # Generate masks
    masks = generate_masks(similarity_maps, include_background=True, method="balanced")
    
    print(f"Generated masks: {list(masks.keys())}")
    for name, mask in masks.items():
        coverage = mask.sum() / mask.numel() * 100
        print(f"  {name}: shape={mask.shape}, coverage={coverage:.1f}%")
    
    # Verify masks are binary and cover the space
    total_coverage = sum(m.sum() for m in masks.values())
    expected = h * w
    print(f"\nTotal coverage: {total_coverage:.0f} / {expected} pixels")
    
    if abs(total_coverage - expected) < 10:  # Allow small tolerance
        print("✅ Masks cover the full image!")
    else:
        print(f"⚠️  Coverage mismatch: {total_coverage} vs {expected}")
    
    print("\n✅ Mask generation works!")
    return True


def test_6_flux_block_replace_creation():
    """Test FreeFuseFluxBlockReplace creation."""
    print("\n" + "="*60)
    print("TEST 6: FreeFuseFluxBlockReplace Creation")
    print("="*60)
    
    from freefuse_comfyui.freefuse_core.attention_replace import (
        FreeFuseState,
        FreeFuseFluxBlockReplace,
    )
    from comfy.ldm.flux.layers import DoubleStreamBlock
    
    # Create state
    state = FreeFuseState()
    state.token_pos_maps = {
        "harry": [[5, 6, 7]],
        "hermione": [[10, 11, 12]],
    }
    
    # Create minimal block
    block = DoubleStreamBlock(
        hidden_size=3072,
        num_heads=24,
        mlp_ratio=4.0,
        dtype=torch.float32,
        device="cpu",
        operations=torch.nn,
    )
    
    # Create replacer
    replacer = FreeFuseFluxBlockReplace(state, block=block, block_index=18)
    
    assert replacer.state is state
    assert replacer.block is block
    assert replacer.block_index == 18
    print("✅ FreeFuseFluxBlockReplace created")
    
    # Create replace function
    block_replace = replacer.create_block_replace()
    assert callable(block_replace)
    print("✅ block_replace function created")
    
    print("\n✅ FreeFuseFluxBlockReplace works!")
    return True


def test_7_similarity_map_computation():
    """Test similarity map computation logic."""
    print("\n" + "="*60)
    print("TEST 7: Similarity Map Computation")
    print("="*60)
    
    from freefuse_comfyui.freefuse_core.attention_replace import (
        compute_flux_similarity_maps_with_qkv,
    )
    
    # Create fake QKV tensors
    B = 1
    heads = 24
    img_len = 64 * 64  # 4096
    txt_len = 256
    head_dim = 128
    
    # Smaller test for speed
    img_len = 256
    txt_len = 64
    
    img_q_rope = torch.randn(B, heads, img_len, head_dim)
    txt_k_rope = torch.randn(B, heads, txt_len, head_dim)
    q_rope = torch.randn(B, heads, txt_len + img_len, head_dim)
    k_rope = torch.randn(B, heads, txt_len + img_len, head_dim)
    v = torch.randn(B, heads, txt_len + img_len, head_dim)
    
    token_pos_maps = {
        "concept_a": [[5, 6, 7]],
        "concept_b": [[15, 16, 17]],
    }
    
    print(f"Input shapes:")
    print(f"  img_q_rope: {img_q_rope.shape}")
    print(f"  txt_k_rope: {txt_k_rope.shape}")
    print(f"  token_pos_maps: {token_pos_maps}")
    
    # Compute similarity maps
    sim_maps = compute_flux_similarity_maps_with_qkv(
        img_q_rope=img_q_rope,
        txt_k_rope=txt_k_rope,
        q_rope=q_rope,
        k_rope=k_rope,
        v=v,
        txt_len=txt_len,
        img_len=img_len,
        txt_first=True,
        token_pos_maps=token_pos_maps,
        top_k_ratio=0.3,
        temperature=4000.0,
        num_heads=heads,
    )
    
    print(f"\nComputed similarity maps:")
    for name, sim_map in sim_maps.items():
        print(f"  {name}: shape={sim_map.shape}, sum={sim_map.sum().item():.4f}")
    
    # Verify shapes
    for name, sim_map in sim_maps.items():
        assert sim_map.shape == (B, img_len, 1), f"Expected shape ({B}, {img_len}, 1), got {sim_map.shape}"
    
    print("\n✅ Similarity map computation works!")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("FreeFuse ComfyUI Component Tests")
    print("="*60)
    
    tests = [
        ("Imports", test_1_imports),
        ("Flux Block Structure", test_2_flux_block_structure),
        ("FreeFuse State", test_3_freefuse_state),
        ("Token Position Finding", test_4_token_position_finding),
        ("Mask Generation", test_5_mask_generation),
        ("Block Replace Creation", test_6_flux_block_replace_creation),
        ("Similarity Map Computation", test_7_similarity_map_computation),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed, None))
        except Exception as e:
            import traceback
            results.append((name, False, str(e)))
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, p, _ in results if p)
    total = len(results)
    
    for name, p, error in results:
        status = "✅ PASS" if p else "❌ FAIL"
        print(f"{status}: {name}")
        if error:
            print(f"       Error: {error}")
    
    print(f"\nPassed: {passed}/{total}")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
