#!/usr/bin/env python
"""
Test script to diagnose mask resizing issues for non-square images.

Flux packs latent tokens: (H, W) -> (H/2) * (W/2) sequence length
This test verifies the mask resizing logic works correctly for different aspect ratios.
"""

import os
import sys

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
freefuse_comfyui_dir = os.path.dirname(script_dir)
freefuse_root = os.path.dirname(freefuse_comfyui_dir)

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

import torch
import torch.nn as nn
import torch.nn.functional as F

from freefuse_comfyui.freefuse_core.bypass_lora_loader import MultiAdapterBypassForwardHook


class DummyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(10, 10))
    
    def forward(self, x):
        return x


def test_flux_packing_logic():
    """Test how Flux packs image tokens and how masks should be resized."""
    print("\n" + "="*70)
    print("Testing Flux Packing Logic for Different Aspect Ratios")
    print("="*70)
    
    # Test different image sizes
    # Image size -> Latent size (÷8) -> Packed seq len (÷2 each dim)
    test_cases = [
        # (image_h, image_w, latent_h, latent_w, packed_img_len)
        (1024, 1024, 128, 128, 64*64),      # 1:1
        (768, 1344, 96, 168, 48*84),        # Portrait-ish (768x1344)
        (1344, 768, 168, 96, 84*48),        # Landscape-ish
        (832, 1216, 104, 152, 52*76),       # Another common size
        (1216, 832, 152, 104, 76*52),       # Flipped
        (512, 512, 64, 64, 32*32),          # Small square
        (640, 1536, 80, 192, 40*96),        # Very tall
        (1536, 640, 192, 80, 96*40),        # Very wide
    ]
    
    print("\nFlux packing: (H, W) -> (H/2, W/2) -> H/2 * W/2 tokens\n")
    print(f"{'Image Size':<15} {'Latent Size':<15} {'Packed Seq Len':<15} {'Expected':<15} {'Match'}")
    print("-"*70)
    
    for img_h, img_w, lat_h, lat_w, expected_len in test_cases:
        packed_h = lat_h // 2
        packed_w = lat_w // 2
        actual_len = packed_h * packed_w
        match = "✅" if actual_len == expected_len else "❌"
        print(f"{img_h}x{img_w:<10} {lat_h}x{lat_w:<10} {actual_len:<15} {expected_len:<15} {match}")


def test_mask_resize_for_flux():
    """Test mask resizing for Flux with different aspect ratios."""
    print("\n" + "="*70)
    print("Testing Mask Resize for Flux Double Stream (img_only)")
    print("="*70)
    
    # Create test masks with different sizes
    test_cases = [
        # (mask_h, mask_w, latent_h, latent_w, txt_len)
        # 1:1 case - should work fine
        (128, 128, 128, 128, 512),
        # Non-1:1 cases - these might have issues
        (96, 168, 96, 168, 512),   # 768x1344 -> latent 96x168 -> packed 48x84
        (168, 96, 168, 96, 512),   # 1344x768 -> latent 168x96 -> packed 84x48
        (104, 152, 104, 152, 512), # 832x1216
        (80, 192, 80, 192, 512),   # Very tall image
    ]
    
    for mask_h, mask_w, lat_h, lat_w, txt_len in test_cases:
        print(f"\n--- Mask: {mask_h}x{mask_w}, Latent: {lat_h}x{lat_w} ---")
        
        # Create a gradient mask to visualize resize behavior
        mask = torch.zeros(mask_h, mask_w)
        # Left half = 1.0 (first concept), Right half = 0.0 (second concept)
        mask[:, :mask_w//2] = 1.0
        
        # For Flux double stream (img_only), seq_len = packed img tokens
        packed_h = lat_h // 2
        packed_w = lat_w // 2
        img_seq_len = packed_h * packed_w
        
        print(f"  Packed dims: {packed_h}x{packed_w} = {img_seq_len} tokens")
        print(f"  Mask elements: {mask.numel()}")
        
        # Simulate what happens in _get_mask_for_sequence
        dummy = DummyModule()
        hook = MultiAdapterBypassForwardHook(dummy, module_key="diffusion_model.transformer_blocks.0.attn.to_q")
        hook.set_masks({"test": mask}, latent_size=(lat_h, lat_w), txt_len=txt_len)
        
        # For img_only type, seq_len = img_seq_len
        result = hook._get_mask_for_sequence("test", seq_len=img_seq_len, 
                                             device=torch.device("cpu"), dtype=torch.float32)
        
        if result is not None:
            print(f"  Result mask shape: {result.shape}")
            print(f"  Result mask range: [{result.min():.3f}, {result.max():.3f}]")
            
            # Check if spatial structure is preserved
            # Reshape to 2D to visualize
            if result.numel() == packed_h * packed_w:
                result_2d = result.view(packed_h, packed_w)
                left_mean = result_2d[:, :packed_w//2].mean().item()
                right_mean = result_2d[:, packed_w//2:].mean().item()
                print(f"  Left half mean: {left_mean:.3f} (expected ~1.0)")
                print(f"  Right half mean: {right_mean:.3f} (expected ~0.0)")
                
                if left_mean > 0.8 and right_mean < 0.2:
                    print(f"  ✅ Spatial structure preserved!")
                else:
                    print(f"  ❌ Spatial structure may be corrupted!")
            else:
                print(f"  ⚠️ Cannot reshape: {result.numel()} != {packed_h * packed_w}")
        else:
            print(f"  ❌ Result is None!")


def test_mask_resize_mismatch():
    """Test what happens when mask size doesn't match expected sequence length."""
    print("\n" + "="*70)
    print("Testing Mask Size Mismatch Handling")
    print("="*70)
    
    # Case: Mask is at full latent resolution, but Flux expects packed resolution
    # This is likely the bug!
    
    print("\n--- POTENTIAL BUG: Mask at latent res, Flux expects packed res ---")
    
    # Original mask at latent resolution (128x128 for 1024x1024 image)
    lat_h, lat_w = 96, 168  # Non-square latent
    mask = torch.zeros(lat_h, lat_w)
    mask[:, :lat_w//2] = 1.0  # Left = 1, Right = 0
    
    # Flux packed resolution
    packed_h, packed_w = lat_h // 2, lat_w // 2
    img_seq_len = packed_h * packed_w
    
    print(f"  Mask shape: {mask.shape} = {mask.numel()} elements")
    print(f"  Expected img_seq_len: {img_seq_len} ({packed_h}x{packed_w})")
    
    # What the current code does
    dummy = DummyModule()
    hook = MultiAdapterBypassForwardHook(dummy, module_key="diffusion_model.transformer_blocks.0.attn.to_q")
    hook.set_masks({"test": mask}, latent_size=(lat_h, lat_w), txt_len=512)
    
    result = hook._get_mask_for_sequence("test", seq_len=img_seq_len,
                                         device=torch.device("cpu"), dtype=torch.float32)
    
    if result is not None:
        print(f"  Result shape: {result.shape}")
        
        # Try to understand the spatial mapping
        result_2d = result.view(packed_h, packed_w)
        
        # The mask was left=1, right=0
        # After packing, the left half should still be ~1, right ~0
        left_cols = packed_w // 2
        left_mean = result_2d[:, :left_cols].mean().item()
        right_mean = result_2d[:, left_cols:].mean().item()
        
        print(f"  After resize to {packed_h}x{packed_w}:")
        print(f"    Left half mean: {left_mean:.4f}")
        print(f"    Right half mean: {right_mean:.4f}")
        
        # Check the resize method being used
        # The issue might be in HOW we resize
        print(f"\n  Let's trace the resize logic:")
        
        mask_flat = mask.view(-1)
        print(f"    Original mask_flat: {mask_flat.shape}")
        print(f"    Target img_len: {img_seq_len}")
        print(f"    Mismatch: {mask_flat.shape[0]} vs {img_seq_len}")
        
        # Current resize approach: find new_h, new_w that multiply to img_len
        ratio = lat_w / lat_h
        new_h = int((img_seq_len / ratio) ** 0.5)
        new_w = img_seq_len // new_h if new_h > 0 else img_seq_len
        
        while new_h > 1 and new_h * new_w != img_seq_len:
            new_h -= 1
            new_w = img_seq_len // new_h
        
        print(f"    Computed resize target: {new_h}x{new_w} = {new_h*new_w}")
        print(f"    Actual packed dims: {packed_h}x{packed_w} = {packed_h*packed_w}")
        
        if (new_h, new_w) != (packed_h, packed_w):
            print(f"    ❌ MISMATCH! Resize target doesn't match packed dimensions!")
            print(f"    This corrupts the spatial structure!")


def test_correct_flux_mask_handling():
    """Test the CORRECT way to handle Flux masks."""
    print("\n" + "="*70)
    print("Testing CORRECT Flux Mask Handling")
    print("="*70)
    
    # For non-1:1 images, the correct approach is:
    # 1. Mask should be resized to PACKED dimensions (lat_h/2, lat_w/2)
    # 2. Then flattened in the correct order
    
    lat_h, lat_w = 96, 168
    packed_h, packed_w = lat_h // 2, lat_w // 2  # 48, 84
    
    # Create mask at latent resolution
    mask = torch.zeros(lat_h, lat_w)
    mask[:, :lat_w//2] = 1.0
    
    print(f"Original mask: {lat_h}x{lat_w}")
    print(f"Packed dims: {packed_h}x{packed_w}")
    
    # CORRECT: Resize directly to packed dimensions
    mask_2d = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    mask_correct = F.interpolate(mask_2d, size=(packed_h, packed_w), mode='bilinear', align_corners=False)
    mask_correct = mask_correct.squeeze()  # (packed_h, packed_w)
    
    print(f"\nCorrect resize to {packed_h}x{packed_w}:")
    left_mean = mask_correct[:, :packed_w//2].mean().item()
    right_mean = mask_correct[:, packed_w//2:].mean().item()
    print(f"  Left half mean: {left_mean:.4f} (expected ~1.0)")
    print(f"  Right half mean: {right_mean:.4f} (expected ~0.0)")
    
    if left_mean > 0.9 and right_mean < 0.1:
        print(f"  ✅ Spatial structure preserved with correct resize!")
    
    # Now flatten correctly
    mask_flat_correct = mask_correct.view(-1)
    print(f"  Flattened shape: {mask_flat_correct.shape}")


def main():
    print("="*70)
    print("FreeFuse Mask Resize Diagnostic")
    print("="*70)
    
    test_flux_packing_logic()
    test_mask_resize_for_flux()
    test_mask_resize_mismatch()
    test_correct_flux_mask_handling()
    
    print("\n" + "="*70)
    print("DIAGNOSIS COMPLETE")
    print("="*70)
    print("""
Key findings:
1. Flux packs latent tokens: (H, W) -> (H/2, W/2) sequence
2. Masks should be resized to PACKED dimensions, not arbitrary dimensions
3. The current resize logic may compute wrong target dimensions for non-1:1

Fix needed: In _get_mask_for_sequence, when resizing for Flux:
- For img_only: resize mask directly to (lat_h/2, lat_w/2) then flatten
- For img_with_text: same, but prepend txt portion
""")


if __name__ == "__main__":
    main()
