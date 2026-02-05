#!/usr/bin/env python3
"""
Test script for FreeFuse token position utilities.

This script tests the token position finding logic without requiring
the full ComfyUI environment. It uses the transformers library directly.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_t5_tokenizer():
    """Test T5 tokenizer (Flux model) token position finding."""
    print("\n" + "=" * 60)
    print("Testing T5 Tokenizer (Flux)")
    print("=" * 60)
    
    try:
        from transformers import T5TokenizerFast
    except ImportError:
        print("ERROR: transformers library not installed")
        return False
    
    # Load T5 tokenizer (same as Flux uses)
    try:
        tokenizer = T5TokenizerFast.from_pretrained("google/t5-v1_1-xxl")
    except Exception as e:
        print(f"Could not load T5-XXL tokenizer, trying smaller model: {e}")
        tokenizer = T5TokenizerFast.from_pretrained("t5-small")
    
    # Test prompt
    prompt = "Harry Potter and Ron Weasley standing in front of Hogwarts castle"
    concepts = {
        "char1": "Harry Potter",
        "char2": "Ron Weasley",
    }
    
    print(f"\nPrompt: {prompt}")
    print(f"Concepts: {concepts}")
    
    # Import our function
    from freefuse_comfyui.freefuse_core.token_utils import find_concept_positions_t5
    
    # Find positions
    pos_maps = find_concept_positions_t5(
        tokenizer=tokenizer,
        prompts=prompt,
        concepts=concepts,
        filter_meaningless=True,
        filter_single_char=True,
    )
    
    print("\nResults:")
    for name, positions_list in pos_maps.items():
        positions = positions_list[0]  # First (and only) prompt
        print(f"  {name}: positions = {positions}")
        
        # Decode tokens at those positions for verification
        encoded = tokenizer(prompt, return_tensors=None)
        input_ids = encoded['input_ids']
        tokens_at_pos = [tokenizer.decode([input_ids[p]]) for p in positions if p < len(input_ids)]
        print(f"    tokens: {tokens_at_pos}")
    
    # Verify results
    success = True
    for name, positions_list in pos_maps.items():
        if not positions_list[0]:
            print(f"  WARNING: No positions found for {name}")
            success = False
    
    return success


def test_clip_tokenizer():
    """Test CLIP tokenizer (SDXL model) token position finding."""
    print("\n" + "=" * 60)
    print("Testing CLIP Tokenizer (SDXL)")
    print("=" * 60)
    
    try:
        from transformers import CLIPTokenizer
    except ImportError:
        print("ERROR: transformers library not installed")
        return False
    
    # Load CLIP tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    
    # Test prompt
    prompt = "a woman with red hair and a man in a blue suit walking together"
    concepts = {
        "lora_a": "a woman with red hair",
        "lora_b": "a man in a blue suit",
    }
    
    print(f"\nPrompt: {prompt}")
    print(f"Concepts: {concepts}")
    
    # Import our function
    from freefuse_comfyui.freefuse_core.token_utils import find_concept_positions_clip
    
    # Find positions
    pos_maps = find_concept_positions_clip(
        tokenizer=tokenizer,
        prompts=prompt,
        concepts=concepts,
        filter_meaningless=True,
        filter_single_char=True,
    )
    
    print("\nResults:")
    for name, positions_list in pos_maps.items():
        positions = positions_list[0]  # First (and only) prompt
        print(f"  {name}: positions = {positions}")
        
        # Decode tokens at those positions for verification
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        tokens_at_pos = [tokenizer.decode([input_ids[p]]) for p in positions if p < len(input_ids)]
        print(f"    tokens: {tokens_at_pos}")
    
    # Verify results
    success = True
    for name, positions_list in pos_maps.items():
        if not positions_list[0]:
            print(f"  WARNING: No positions found for {name}")
            success = False
    
    return success


def test_multiple_prompts():
    """Test with multiple prompts (batch processing)."""
    print("\n" + "=" * 60)
    print("Testing Multiple Prompts (Batch)")
    print("=" * 60)
    
    try:
        from transformers import CLIPTokenizer
    except ImportError:
        print("ERROR: transformers library not installed")
        return False
    
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    
    prompts = [
        "Harry Potter at Hogwarts",
        "Harry Potter in the library",
    ]
    concepts = {"char1": "Harry Potter"}
    
    print(f"\nPrompts: {prompts}")
    print(f"Concepts: {concepts}")
    
    from freefuse_comfyui.freefuse_core.token_utils import find_concept_positions_clip
    
    pos_maps = find_concept_positions_clip(
        tokenizer=tokenizer,
        prompts=prompts,
        concepts=concepts,
    )
    
    print("\nResults:")
    for name, positions_list in pos_maps.items():
        print(f"  {name}:")
        for i, positions in enumerate(positions_list):
            print(f"    prompt {i}: positions = {positions}")
    
    # Verify we got positions for both prompts
    success = len(pos_maps["char1"]) == 2
    return success


def test_edge_cases():
    """Test edge cases: empty concepts, no matches, etc."""
    print("\n" + "=" * 60)
    print("Testing Edge Cases")
    print("=" * 60)
    
    try:
        from transformers import CLIPTokenizer
    except ImportError:
        print("ERROR: transformers library not installed")
        return False
    
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    
    from freefuse_comfyui.freefuse_core.token_utils import find_concept_positions_clip
    
    # Test 1: Concept not in prompt
    print("\n1. Concept not in prompt:")
    prompt = "a beautiful landscape"
    concepts = {"char1": "Harry Potter"}
    pos_maps = find_concept_positions_clip(tokenizer, prompt, concepts)
    print(f"  Result: {pos_maps}")
    
    # Test 2: Empty concept text
    print("\n2. Empty concepts:")
    pos_maps = find_concept_positions_clip(tokenizer, prompt, {})
    print(f"  Result: {pos_maps}")
    
    # Test 3: Concept appears multiple times
    print("\n3. Concept appears multiple times:")
    prompt = "a cat and another cat playing with a cat toy"
    concepts = {"animal": "cat"}
    pos_maps = find_concept_positions_clip(tokenizer, prompt, concepts)
    print(f"  Prompt: {prompt}")
    print(f"  Result: {pos_maps}")
    
    return True


if __name__ == "__main__":
    print("FreeFuse Token Position Utilities - Test Suite")
    print("=" * 60)
    
    all_passed = True
    
    # Run tests
    tests = [
        ("T5 Tokenizer (Flux)", test_t5_tokenizer),
        ("CLIP Tokenizer (SDXL)", test_clip_tokenizer),
        ("Multiple Prompts", test_multiple_prompts),
        ("Edge Cases", test_edge_cases),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
            all_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed!"))
    sys.exit(0 if all_passed else 1)
