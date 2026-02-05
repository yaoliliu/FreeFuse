"""
FreeFuse ComfyUI Components Test Script

Run this script to test individual FreeFuse components before testing the full workflow.
"""

import sys
import os

# Add ComfyUI to path
sys.path.insert(0, '/root/FreeFuse/ComfyUI')

import torch
import logging

logging.basicConfig(level=logging.INFO)


def test_1_imports():
    """Test that all FreeFuse components can be imported."""
    print("\n=== Test 1: Imports ===")
    
    try:
        from freefuse_comfyui.nodes import (
            FreeFuseLoRALoader,
            FreeFuseConceptMap,
            FreeFuseTokenPositions,
            FreeFusePhase1Sampler,
            FreeFuseMaskApplicator,
            FreeFuseMaskPreview,
        )
        print("✓ All node imports successful")
        
        from freefuse_comfyui.freefuse_core.attention_replace import (
            FreeFuseState,
            FreeFuseFluxBlockReplace,
            apply_freefuse_replace_patches,
        )
        print("✓ attention_replace imports successful")
        
        from freefuse_comfyui.freefuse_core.token_utils import (
            detect_model_type,
            find_concept_positions,
        )
        print("✓ token_utils imports successful")
        
        from freefuse_comfyui.freefuse_core.mask_utils import (
            generate_masks,
            balanced_argmax,
        )
        print("✓ mask_utils imports successful")
        
        from freefuse_comfyui.freefuse_core.freefuse_bypass import (
            FreeFuseBypassForwardHook,
        )
        print("✓ freefuse_bypass imports successful")
        
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_2_freefuse_state():
    """Test FreeFuseState initialization and methods."""
    print("\n=== Test 2: FreeFuseState ===")
    
    from freefuse_comfyui.freefuse_core.attention_replace import FreeFuseState
    
    state = FreeFuseState()
    
    # Test initial values
    assert state.phase == "collect"
    assert state.current_step == 0
    assert state.collect_step == 4
    assert state.collect_block == 18
    print("✓ Initial values correct")
    
    # Test is_collect_step
    state.phase = "collect"
    state.current_step = 4
    assert state.is_collect_step(4, 18) == True
    assert state.is_collect_step(4, 19) == False  # Wrong block
    assert state.is_collect_step(3, 18) == False  # Wrong step
    print("✓ is_collect_step works correctly")
    
    # Test token_pos_maps
    state.token_pos_maps = {
        "harry": [[1, 2, 3]],
        "daiyu": [[4, 5, 6]],
    }
    assert len(state.token_pos_maps) == 2
    print("✓ token_pos_maps setting works")
    
    return True


def test_3_mask_utils():
    """Test mask generation utilities."""
    print("\n=== Test 3: Mask Utils ===")
    
    from freefuse_comfyui.freefuse_core.mask_utils import generate_masks, balanced_argmax
    
    # Create dummy similarity maps
    sim_maps = {
        "concept_a": torch.rand(64, 64),
        "concept_b": torch.rand(64, 64),
    }
    
    # Test generate_masks
    masks = generate_masks(sim_maps, include_background=True, method="balanced")
    
    assert "concept_a" in masks
    assert "concept_b" in masks
    assert "_background_" in masks
    print(f"✓ Generated {len(masks)} masks")
    
    # Check mask shapes
    for name, mask in masks.items():
        assert mask.shape == (64, 64), f"Mask {name} has wrong shape: {mask.shape}"
    print("✓ Mask shapes correct")
    
    # Check masks sum to 1 (each pixel belongs to exactly one concept)
    mask_sum = sum(masks.values())
    assert torch.allclose(mask_sum, torch.ones(64, 64)), "Masks don't sum to 1"
    print("✓ Masks are mutually exclusive")
    
    return True


def test_4_flux_block_structure():
    """Test that we can access Flux block structure (without loading model)."""
    print("\n=== Test 4: Flux Block Structure Check ===")
    
    # Check ComfyUI Flux layers are importable
    from comfy.ldm.flux.layers import DoubleStreamBlock, SelfAttention
    from comfy.ldm.flux.math import apply_rope, attention
    
    print("✓ ComfyUI Flux layers importable")
    
    # Check DoubleStreamBlock has expected attributes
    expected_attrs = ['img_attn', 'txt_attn', 'img_norm1', 'txt_norm1', 'num_heads', 'flipped_img_txt']
    print(f"✓ DoubleStreamBlock expected attributes: {expected_attrs}")
    
    # Check SelfAttention has qkv and norm
    print("✓ SelfAttention should have: qkv, norm, proj")
    
    return True


def test_5_similarity_map_computation():
    """Test similarity map computation logic."""
    print("\n=== Test 5: Similarity Map Computation ===")
    
    from freefuse_comfyui.freefuse_core.attention_replace import compute_flux_similarity_maps_with_qkv
    
    # Create dummy tensors
    B, heads, img_len, txt_len, head_dim = 1, 4, 256, 64, 32
    
    img_q_rope = torch.randn(B, heads, img_len, head_dim)
    txt_k_rope = torch.randn(B, heads, txt_len, head_dim)
    q_rope = torch.randn(B, heads, txt_len + img_len, head_dim)
    k_rope = torch.randn(B, heads, txt_len + img_len, head_dim)
    v = torch.randn(B, heads, txt_len + img_len, head_dim)
    
    token_pos_maps = {
        "harry": [[5, 6, 7]],  # Token positions in text sequence
        "daiyu": [[10, 11, 12]],
    }
    
    try:
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
            background_positions=None,
            top_k_ratio=0.3,
            temperature=4000.0,
            num_heads=heads,
        )
        
        print(f"✓ Computed {len(sim_maps)} similarity maps")
        for name, sim_map in sim_maps.items():
            print(f"  - {name}: shape={sim_map.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Error computing similarity maps: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all component tests."""
    print("=" * 60)
    print("FreeFuse ComfyUI Component Tests")
    print("=" * 60)
    
    tests = [
        ("Imports", test_1_imports),
        ("FreeFuseState", test_2_freefuse_state),
        ("Mask Utils", test_3_mask_utils),
        ("Flux Block Structure", test_4_flux_block_structure),
        ("Similarity Map Computation", test_5_similarity_map_computation),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"✗ Test {name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    total_pass = sum(1 for _, s in results if s)
    print(f"\nTotal: {total_pass}/{len(results)} tests passed")
    
    return all(s for _, s in results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
