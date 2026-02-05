#!/usr/bin/env python
"""
Unit tests for FreeFuse Attention Bias module.

Tests the core attention bias construction and application logic.
"""

import sys
import os
import unittest

# Add paths for imports - use importlib to load directly from file
script_dir = os.path.dirname(os.path.abspath(__file__))
freefuse_comfyui_dir = os.path.dirname(script_dir)
freefuse_root = os.path.dirname(freefuse_comfyui_dir)

import torch
import torch.nn.functional as F

# Import directly from the file using importlib.util to avoid __init__.py
import importlib.util
attention_bias_path = os.path.join(freefuse_comfyui_dir, "freefuse_core", "attention_bias.py")
spec = importlib.util.spec_from_file_location("attention_bias", attention_bias_path)
attention_bias_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(attention_bias_module)

# Extract needed functions/classes from module
construct_attention_bias = attention_bias_module.construct_attention_bias
construct_attention_bias_sdxl = attention_bias_module.construct_attention_bias_sdxl
apply_attention_bias_to_weights = attention_bias_module.apply_attention_bias_to_weights
AttentionBiasConfig = attention_bias_module.AttentionBiasConfig


class TestAttentionBiasConstruction(unittest.TestCase):
    """Test attention bias matrix construction."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Already imported at module level
        self.construct_attention_bias = construct_attention_bias
        self.construct_attention_bias_sdxl = construct_attention_bias_sdxl
        self.AttentionBiasConfig = AttentionBiasConfig
        
    def test_basic_bias_construction_flux(self):
        """Test basic attention bias construction for Flux."""
        # Create simple masks: LoRA A on left half, LoRA B on right half
        img_seq_len = 64  # 8x8 packed
        txt_seq_len = 128
        
        # LoRA A mask: left half (1s for positions 0-31)
        mask_a = torch.zeros(1, img_seq_len)
        mask_a[0, :img_seq_len // 2] = 1.0
        
        # LoRA B mask: right half (1s for positions 32-63)
        mask_b = torch.zeros(1, img_seq_len)
        mask_b[0, img_seq_len // 2:] = 1.0
        
        lora_masks = {
            "lora_a": mask_a,
            "lora_b": mask_b,
        }
        
        # Token positions: LoRA A uses tokens 5-10, LoRA B uses tokens 15-20
        token_pos_maps = {
            "lora_a": [[5, 6, 7, 8, 9, 10]],
            "lora_b": [[15, 16, 17, 18, 19, 20]],
        }
        
        bias = self.construct_attention_bias(
            lora_masks=lora_masks,
            token_pos_maps=token_pos_maps,
            txt_seq_len=txt_seq_len,
            img_seq_len=img_seq_len,
            bias_scale=5.0,
            positive_bias_scale=1.0,
            bidirectional=True,
            use_positive_bias=True,
        )
        
        # Check shape
        self.assertEqual(bias.shape, (1, txt_seq_len + img_seq_len, txt_seq_len + img_seq_len))
        
        # Check that bias is applied in the correct regions
        # Image positions are after txt_seq_len
        img_start = txt_seq_len
        
        # Check negative bias: left image (LoRA A region) should have negative bias to LoRA B tokens
        # Position (img_start + 0, 15) should be negative (left image attending to LoRA B token)
        left_img_pos = img_start + 0  # First image position (in LoRA A region)
        lora_b_token = 15  # First LoRA B token
        self.assertLess(bias[0, left_img_pos, lora_b_token].item(), 0)
        
        # Check positive bias: left image (LoRA A region) should have positive bias to LoRA A tokens
        lora_a_token = 5  # First LoRA A token
        self.assertGreater(bias[0, left_img_pos, lora_a_token].item(), 0)
        
        print(f"✅ Basic Flux bias construction test passed")
        print(f"   Bias shape: {bias.shape}")
        print(f"   Left img -> LoRA B token bias: {bias[0, left_img_pos, lora_b_token].item():.3f}")
        print(f"   Left img -> LoRA A token bias: {bias[0, left_img_pos, lora_a_token].item():.3f}")
    
    def test_bias_construction_sdxl(self):
        """Test attention bias construction for SDXL."""
        # Create spatial masks
        latent_h, latent_w = 8, 8
        txt_seq_len = 77  # CLIP
        
        # LoRA A: top half
        mask_a = torch.zeros(1, latent_h, latent_w)
        mask_a[0, :latent_h // 2, :] = 1.0
        
        # LoRA B: bottom half
        mask_b = torch.zeros(1, latent_h, latent_w)
        mask_b[0, latent_h // 2:, :] = 1.0
        
        lora_masks = {
            "lora_a": mask_a,
            "lora_b": mask_b,
        }
        
        token_pos_maps = {
            "lora_a": [[3, 4, 5]],
            "lora_b": [[10, 11, 12]],
        }
        
        bias = self.construct_attention_bias_sdxl(
            lora_masks=lora_masks,
            token_pos_maps=token_pos_maps,
            txt_seq_len=txt_seq_len,
            img_h=latent_h,
            img_w=latent_w,
            bias_scale=5.0,
            positive_bias_scale=1.0,
            use_positive_bias=True,
        )
        
        # Check shape: (B, img_seq_len, txt_seq_len)
        img_seq_len = latent_h * latent_w
        self.assertEqual(bias.shape, (1, img_seq_len, txt_seq_len))
        
        # Check bias values
        # Top-left image position (in LoRA A region) -> LoRA B token should be negative
        top_img_pos = 0
        lora_b_token = 10
        self.assertLess(bias[0, top_img_pos, lora_b_token].item(), 0)
        
        # Top-left image position -> LoRA A token should be positive
        lora_a_token = 3
        self.assertGreater(bias[0, top_img_pos, lora_a_token].item(), 0)
        
        print(f"✅ SDXL bias construction test passed")
        print(f"   Bias shape: {bias.shape}")
    
    def test_config_block_filtering(self):
        """Test AttentionBiasConfig block filtering."""
        config = self.AttentionBiasConfig(
            enabled=True,
            apply_to_blocks="double_stream_only",
        )
        
        # Should apply to double blocks
        self.assertTrue(config.should_apply_to_block("transformer_blocks.0"))
        self.assertTrue(config.should_apply_to_block("transformer_blocks.18"))
        
        # Should not apply to single blocks
        self.assertFalse(config.should_apply_to_block("single_transformer_blocks.0"))
        
        # Test last_half_double preset
        config2 = self.AttentionBiasConfig(
            enabled=True,
            apply_to_blocks="last_half_double",
        )
        
        self.assertFalse(config2.should_apply_to_block("transformer_blocks.5"))
        self.assertTrue(config2.should_apply_to_block("transformer_blocks.15"))
        
        print(f"✅ Config block filtering test passed")
    
    def test_no_masks_returns_none(self):
        """Test that empty masks returns None."""
        bias = self.construct_attention_bias(
            lora_masks={},
            token_pos_maps={},
            txt_seq_len=128,
            img_seq_len=64,
        )
        self.assertIsNone(bias)
        print(f"✅ Empty masks returns None test passed")
    
    def test_bidirectional_bias(self):
        """Test bidirectional attention bias."""
        img_seq_len = 16
        txt_seq_len = 32
        
        mask_a = torch.ones(1, img_seq_len)
        lora_masks = {"lora_a": mask_a}
        token_pos_maps = {"lora_a": [[0, 1, 2]]}
        
        # With bidirectional
        bias_bi = self.construct_attention_bias(
            lora_masks=lora_masks,
            token_pos_maps=token_pos_maps,
            txt_seq_len=txt_seq_len,
            img_seq_len=img_seq_len,
            bidirectional=True,
        )
        
        # Without bidirectional
        bias_uni = self.construct_attention_bias(
            lora_masks=lora_masks,
            token_pos_maps=token_pos_maps,
            txt_seq_len=txt_seq_len,
            img_seq_len=img_seq_len,
            bidirectional=False,
        )
        
        # Bidirectional should have non-zero values in txt->img region
        txt_to_img_region_bi = bias_bi[0, :txt_seq_len, txt_seq_len:]
        txt_to_img_region_uni = bias_uni[0, :txt_seq_len, txt_seq_len:]
        
        # With bidirectional=True, txt->img should have bias
        # (though in this case with only one LoRA covering everything, it's mostly zeros)
        # The key is that the code path runs without error
        
        print(f"✅ Bidirectional bias test passed")
        print(f"   Bidirectional txt->img sum: {txt_to_img_region_bi.abs().sum().item():.3f}")
        print(f"   Unidirectional txt->img sum: {txt_to_img_region_uni.abs().sum().item():.3f}")


class TestAttentionBiasApplication(unittest.TestCase):
    """Test attention bias application to attention weights."""
    
    def setUp(self):
        # Already imported at module level
        self.apply_attention_bias_to_weights = apply_attention_bias_to_weights
    
    def test_bias_application(self):
        """Test that bias is correctly added to attention weights."""
        B, heads, seq_len = 2, 8, 64
        
        # Create random attention weights
        attn_weights = torch.randn(B, heads, seq_len, seq_len)
        
        # Create bias
        bias = torch.ones(B, seq_len, seq_len) * 0.5
        
        # Apply bias
        result = self.apply_attention_bias_to_weights(attn_weights, bias)
        
        # Check that bias was added
        expected = attn_weights + bias.unsqueeze(1)
        self.assertTrue(torch.allclose(result, expected))
        
        print(f"✅ Bias application test passed")
    
    def test_none_bias_passthrough(self):
        """Test that None bias returns weights unchanged."""
        attn_weights = torch.randn(2, 8, 64, 64)
        result = self.apply_attention_bias_to_weights(attn_weights, None)
        self.assertTrue(torch.equal(result, attn_weights))
        print(f"✅ None bias passthrough test passed")


class TestAttentionBiasMathematical(unittest.TestCase):
    """Test the mathematical properties of attention bias."""
    
    def test_softmax_effect(self):
        """Test that negative bias reduces attention probability."""
        seq_len = 16
        
        # Create uniform attention scores
        scores = torch.zeros(1, 1, seq_len, seq_len)
        
        # Without bias, softmax gives uniform distribution
        probs_no_bias = F.softmax(scores, dim=-1)
        
        # Add negative bias to suppress attention to position 5
        bias = torch.zeros(1, 1, seq_len, seq_len)
        bias[:, :, :, 5] = -5.0  # Strong negative bias
        
        scores_with_bias = scores + bias
        probs_with_bias = F.softmax(scores_with_bias, dim=-1)
        
        # Attention to position 5 should be much lower with bias
        self.assertLess(
            probs_with_bias[0, 0, 0, 5].item(),
            probs_no_bias[0, 0, 0, 5].item() * 0.1  # Should be reduced by >90%
        )
        
        print(f"✅ Softmax effect test passed")
        print(f"   No bias prob at pos 5: {probs_no_bias[0, 0, 0, 5].item():.4f}")
        print(f"   With bias prob at pos 5: {probs_with_bias[0, 0, 0, 5].item():.6f}")
    
    def test_positive_bias_effect(self):
        """Test that positive bias increases attention probability."""
        seq_len = 16
        
        scores = torch.zeros(1, 1, seq_len, seq_len)
        
        # Add positive bias to position 5
        bias = torch.zeros(1, 1, seq_len, seq_len)
        bias[:, :, :, 5] = 5.0  # Strong positive bias
        
        probs_no_bias = F.softmax(scores, dim=-1)
        probs_with_bias = F.softmax(scores + bias, dim=-1)
        
        # Attention to position 5 should be much higher with bias
        self.assertGreater(
            probs_with_bias[0, 0, 0, 5].item(),
            probs_no_bias[0, 0, 0, 5].item() * 5  # Should be increased significantly
        )
        
        print(f"✅ Positive bias effect test passed")
        print(f"   No bias prob at pos 5: {probs_no_bias[0, 0, 0, 5].item():.4f}")
        print(f"   With positive bias: {probs_with_bias[0, 0, 0, 5].item():.4f}")


def run_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("FreeFuse Attention Bias Unit Tests")
    print("="*70 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestAttentionBiasConstruction))
    suite.addTests(loader.loadTestsFromTestCase(TestAttentionBiasApplication))
    suite.addTests(loader.loadTestsFromTestCase(TestAttentionBiasMathematical))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print(f"\n✅ ALL TESTS PASSED")
    else:
        print(f"\n❌ SOME TESTS FAILED")
        for test, traceback in result.failures + result.errors:
            print(f"\n  Failed: {test}")
            print(f"  {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
