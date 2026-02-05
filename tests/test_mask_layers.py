#!/usr/bin/env python
"""
Test script to verify which layers have spatial masks applied.

This test validates that the mask application logic matches the diffusers implementation:
- Flux: 
  - single_transformer_blocks: to_q/k/v, proj_mlp, proj_out (with text)
  - transformer_blocks: to_q/k/v, to_out, ff (img only, NOT ff_context or add_*_proj)
- SDXL:
  - attn1: all layers
  - attn2: to_q, to_out only (NOT to_k, to_v - they process text tokens)
  - ff.net

Run with:
    python freefuse_comfyui/tests/test_mask_layers.py
"""

import os
import sys

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
freefuse_comfyui_dir = os.path.dirname(script_dir)
freefuse_root = os.path.dirname(freefuse_comfyui_dir)

# Find ComfyUI
def _find_comfyui_dir(start_dir: str) -> str:
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

comfyui_dir = _find_comfyui_dir(freefuse_root)

if comfyui_dir not in sys.path:
    sys.path.insert(0, comfyui_dir)
if freefuse_root not in sys.path:
    sys.path.insert(0, freefuse_root)

os.chdir(comfyui_dir)

print(f"ComfyUI dir: {comfyui_dir}")
print(f"FreeFuse root: {freefuse_root}")

# Now import the module we want to test
from freefuse_comfyui.freefuse_core.bypass_lora_loader import MultiAdapterBypassForwardHook

import torch.nn as nn


class DummyModule(nn.Module):
    """Dummy module for testing."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(10, 10))
    
    def forward(self, x):
        return x


import torch


def test_flux_layer_detection():
    """Test that Flux layers are correctly categorized for mask application."""
    print("\n" + "="*70)
    print("Testing Flux Layer Detection")
    print("="*70)
    
    # Flux layer patterns to test
    test_cases = [
        # Single stream blocks - should have mask with text
        ("diffusion_model.single_transformer_blocks.0.attn.to_q", True, "img_with_text"),
        ("diffusion_model.single_transformer_blocks.0.attn.to_k", True, "img_with_text"),
        ("diffusion_model.single_transformer_blocks.0.attn.to_v", True, "img_with_text"),
        ("diffusion_model.single_transformer_blocks.5.proj_mlp", True, "img_with_text"),
        ("diffusion_model.single_transformer_blocks.10.proj_out", True, "img_with_text"),
        ("diffusion_model.single_blocks.0.linear1", True, "img_with_text"),
        
        # Double stream blocks - image path should have mask (img only)
        ("diffusion_model.transformer_blocks.0.attn.to_q", True, "img_only"),
        ("diffusion_model.transformer_blocks.0.attn.to_k", True, "img_only"),
        ("diffusion_model.transformer_blocks.0.attn.to_v", True, "img_only"),
        ("diffusion_model.transformer_blocks.0.attn.to_out.0.weight", True, "img_only"),
        ("diffusion_model.transformer_blocks.5.ff.net.0.proj", True, "img_only"),
        ("diffusion_model.double_blocks.0.img_attn.qkv", True, "img_only"),
        ("diffusion_model.double_blocks.0.img_mlp.0", True, "img_only"),
        
        # Double stream blocks - context path should NOT have mask
        ("diffusion_model.transformer_blocks.0.ff_context.net.0.proj", False, None),
        ("diffusion_model.transformer_blocks.0.attn.add_q_proj", False, None),
        ("diffusion_model.transformer_blocks.0.attn.add_k_proj", False, None),
        ("diffusion_model.transformer_blocks.0.attn.add_v_proj", False, None),
        ("diffusion_model.transformer_blocks.0.attn.to_add_out.0", False, None),
        ("diffusion_model.double_blocks.0.txt_attn.qkv", False, None),
    ]
    
    passed = 0
    failed = 0
    
    for module_key, expected_apply, expected_type in test_cases:
        dummy = DummyModule()
        hook = MultiAdapterBypassForwardHook(dummy, module_key=module_key)
        
        should_apply = hook._should_apply_spatial_mask()
        mask_type = hook._get_mask_type()
        
        # Check result
        apply_match = should_apply == expected_apply
        type_match = mask_type == expected_type
        
        if apply_match and type_match:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1
        
        print(f"{status}: {module_key}")
        if not apply_match:
            print(f"       Expected apply={expected_apply}, got {should_apply}")
        if not type_match:
            print(f"       Expected type={expected_type}, got {mask_type}")
    
    print(f"\nFlux: {passed}/{passed+failed} tests passed")
    return failed == 0


def test_sdxl_layer_detection():
    """Test that SDXL layers are correctly categorized for mask application."""
    print("\n" + "="*70)
    print("Testing SDXL Layer Detection")
    print("="*70)
    
    # SDXL layer patterns to test
    test_cases = [
        # Self-attention (attn1) - all should have mask
        ("diffusion_model.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q", True, "img_only"),
        ("diffusion_model.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k", True, "img_only"),
        ("diffusion_model.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v", True, "img_only"),
        ("diffusion_model.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.0", True, "img_only"),
        ("diffusion_model.mid_block.attentions.0.transformer_blocks.0.attn1.to_q", True, "img_only"),
        ("diffusion_model.up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k", True, "img_only"),
        
        # Cross-attention (attn2) - to_q, to_out should have mask; to_k, to_v should NOT
        ("diffusion_model.down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_q", True, "img_only"),
        ("diffusion_model.down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_out.0", True, "img_only"),
        ("diffusion_model.down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_k", False, None),  # Process text!
        ("diffusion_model.down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_v", False, None),  # Process text!
        ("diffusion_model.mid_block.attentions.0.transformer_blocks.0.attn2.to_k", False, None),
        ("diffusion_model.up_blocks.0.attentions.0.transformer_blocks.0.attn2.to_v", False, None),
        
        # FeedForward - should have mask
        ("diffusion_model.down_blocks.2.attentions.0.transformer_blocks.0.ff.net.0.proj", True, "img_only"),
        ("diffusion_model.mid_block.attentions.0.transformer_blocks.0.ff.net.2", True, "img_only"),
        ("diffusion_model.up_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj", True, "img_only"),
        
        # Resnet blocks - should NOT have mask (not attention/ff)
        ("diffusion_model.down_blocks.0.resnets.0.conv1", False, None),
        ("diffusion_model.up_blocks.2.resnets.0.conv2", False, None),
    ]
    
    passed = 0
    failed = 0
    
    for module_key, expected_apply, expected_type in test_cases:
        dummy = DummyModule()
        hook = MultiAdapterBypassForwardHook(dummy, module_key=module_key)
        
        should_apply = hook._should_apply_spatial_mask()
        mask_type = hook._get_mask_type()
        
        # Check result
        apply_match = should_apply == expected_apply
        type_match = mask_type == expected_type
        
        if apply_match and type_match:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1
        
        print(f"{status}: {module_key}")
        if not apply_match:
            print(f"       Expected apply={expected_apply}, got {should_apply}")
        if not type_match:
            print(f"       Expected type={expected_type}, got {mask_type}")
    
    print(f"\nSDXL: {passed}/{passed+failed} tests passed")
    return failed == 0


def test_mask_construction():
    """Test that masks are constructed correctly for different mask types."""
    print("\n" + "="*70)
    print("Testing Mask Construction")
    print("="*70)
    
    # Create a simple 4x4 mask
    mask = torch.tensor([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.2, 0.3, 0.4, 0.5],
        [0.6, 0.7, 0.8, 0.9],
    ])
    
    # Test 1: img_with_text mask (Flux single stream)
    print("\nTest 1: img_with_text mask (Flux single stream)")
    dummy = DummyModule()
    hook = MultiAdapterBypassForwardHook(dummy, module_key="diffusion_model.single_transformer_blocks.0.attn.to_q")
    hook.set_masks({"adapter1": mask}, latent_size=(4, 4), txt_len=10)
    
    # seq_len = txt_len(10) + img_len(16) = 26
    result = hook._get_mask_for_sequence("adapter1", seq_len=26, device=torch.device("cpu"), dtype=torch.float32)
    
    if result is not None:
        print(f"  Mask shape: {result.shape}")
        print(f"  Text portion (first 10): {result[:10].tolist()}")
        print(f"  Img portion (11-26): min={result[10:].min():.2f}, max={result[10:].max():.2f}")
        
        # Text portion should be all 1.0
        assert torch.allclose(result[:10], torch.ones(10)), "Text portion should be 1.0"
        # Img portion should be the mask values
        assert result[10:].min() < 1.0, "Img portion should have mask values < 1.0"
        print("  ✅ PASS: img_with_text mask correct")
    else:
        print("  ❌ FAIL: Mask is None")
    
    # Test 2: img_only mask (Flux double stream / SDXL)
    print("\nTest 2: img_only mask (Flux double stream / SDXL)")
    dummy2 = DummyModule()
    hook2 = MultiAdapterBypassForwardHook(dummy2, module_key="diffusion_model.transformer_blocks.0.attn.to_q")
    hook2.set_masks({"adapter1": mask}, latent_size=(4, 4), txt_len=10)
    
    # seq_len = img_len only = 16
    result2 = hook2._get_mask_for_sequence("adapter1", seq_len=16, device=torch.device("cpu"), dtype=torch.float32)
    
    if result2 is not None:
        print(f"  Mask shape: {result2.shape}")
        print(f"  Values: min={result2.min():.2f}, max={result2.max():.2f}")
        
        # Should be exactly img_len elements
        assert result2.shape[0] == 16, f"Expected 16 elements, got {result2.shape[0]}"
        # Should have mask values (not all 1.0)
        assert result2.min() < 1.0, "Should have mask values < 1.0"
        print("  ✅ PASS: img_only mask correct")
    else:
        print("  ❌ FAIL: Mask is None")
    
    # Test 3: Layer that should NOT have mask applied
    print("\nTest 3: Layer that should NOT have mask applied")
    dummy3 = DummyModule()
    hook3 = MultiAdapterBypassForwardHook(dummy3, module_key="diffusion_model.transformer_blocks.0.ff_context.net.0.proj")
    hook3.set_masks({"adapter1": mask}, latent_size=(4, 4), txt_len=10)
    
    should_apply = hook3._should_apply_spatial_mask()
    print(f"  Should apply mask: {should_apply}")
    assert not should_apply, "ff_context should not have mask applied"
    print("  ✅ PASS: ff_context correctly excluded")
    
    print("\n✅ All mask construction tests passed!")


def main():
    print("="*70)
    print("FreeFuse Mask Layer Application Test")
    print("="*70)
    print("\nThis test validates that spatial masks are applied to the correct layers,")
    print("matching the diffusers implementation.")
    
    all_passed = True
    
    try:
        if not test_flux_layer_detection():
            all_passed = False
    except Exception as e:
        print(f"❌ Flux test error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        if not test_sdxl_layer_detection():
            all_passed = False
    except Exception as e:
        print(f"❌ SDXL test error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        test_mask_construction()
    except Exception as e:
        print(f"❌ Mask construction test error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
