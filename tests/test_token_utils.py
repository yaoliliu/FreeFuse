#!/usr/bin/env python3
"""
Test script for FreeFuse token position utilities.

This script tests the Z-Image (Qwen3) token position finding logic
against the reference implementation in main_freefuse_z_image.py.
"""

import sys
import os
import importlib.util

# Add workspace root (FreeFuse/) to path for imports
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, WORKSPACE_ROOT)

# Direct import of token_utils to bypass freefuse_comfyui/__init__.py
# (which imports ComfyUI-specific modules like folder_paths)
_token_utils_path = os.path.join(
    WORKSPACE_ROOT, "freefuse_comfyui", "freefuse_core", "token_utils.py"
)
_spec = importlib.util.spec_from_file_location("token_utils", _token_utils_path)
token_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(token_utils)


# =====================================================================
# Reference implementation from main_freefuse_z_image.py (ground truth)
# =====================================================================

def reference_find_concept_positions(
    tokenizer,
    prompts,
    concepts,
    filter_meaningless=True,
    filter_single_char=True,
    max_sequence_length=512,
):
    """
    Reference implementation copied from main_freefuse_z_image.py.
    This is the known-correct version.
    """
    STOPWORDS = {
        'a', 'an', 'the', 'and', 'or', 'but', 'nor', 'so', 'yet',
        'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from',
        'as', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'under', 'over',
        'it', 'its', 'this', 'that', 'these', 'those', 'their', 'his',
        'her', 'my', 'your', 'our',
        'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'has', 'have', 'had', 'having',
        'which', 'who', 'whom', 'whose', 'where', 'when', 'while',
    }
    PUNCTUATION = {',', '.', '!', '?', ';', ':', '"', "'", '(', ')',
                   '[', ']', '{', '}', '-', '–', '—', '/', '\\', '...', '..'}
    MEANINGLESS = STOPWORDS | PUNCTUATION

    def _is_meaningless(tok_text, check_single):
        cleaned = tok_text.strip().lower()
        if not cleaned:
            return True
        if check_single and len(cleaned) == 1:
            return True
        return cleaned in MEANINGLESS

    if isinstance(prompts, str):
        prompts = [prompts]

    # ── build template-wrapped texts & tokenize (mirrors _encode_prompt) ──
    prompt_data_list = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        wrapped_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        tok_out = tokenizer(
            wrapped_text,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        token_ids = tok_out.input_ids[0].tolist()
        attn_mask = tok_out.attention_mask[0].tolist()

        token_texts = [tokenizer.decode([tid]) for tid in token_ids]

        active_indices = [idx for idx, m in enumerate(attn_mask) if m == 1]
        active_token_texts = [token_texts[idx] for idx in active_indices]

        concat_text = ""
        token_spans = []
        for tt in active_token_texts:
            start = len(concat_text)
            concat_text += tt
            token_spans.append((start, len(concat_text)))

        prompt_data_list.append({
            "raw": prompt,
            "wrapped": wrapped_text,
            "active_indices": active_indices,
            "active_token_texts": active_token_texts,
            "concat_text": concat_text,
            "token_spans": token_spans,
        })

    # ── find positions ──
    concept_pos_map = {}
    for concept_name, concept_text in concepts.items():
        concept_pos_map[concept_name] = []

        for pd in prompt_data_list:
            positions = []
            positions_with_text = []

            search_start = 0
            while True:
                idx = pd["concat_text"].find(concept_text, search_start)
                if idx == -1:
                    break
                c_start, c_end = idx, idx + len(concept_text)

                for tok_i, (ts, te) in enumerate(pd["token_spans"]):
                    if te > c_start and ts < c_end and tok_i not in positions:
                        positions.append(tok_i)
                        positions_with_text.append(
                            (tok_i, pd["active_token_texts"][tok_i])
                        )
                search_start = idx + 1

            # Filter meaningless tokens
            if filter_meaningless and positions_with_text:
                filtered = [
                    p for p, t in positions_with_text
                    if not _is_meaningless(t, filter_single_char)
                ]
                if not filtered:
                    non_punct = [
                        p for p, t in positions_with_text
                        if t.strip() not in PUNCTUATION
                    ]
                    filtered = non_punct[:1] if non_punct else [positions_with_text[0][0]]
                positions = filtered

            positions.sort()
            concept_pos_map[concept_name].append(positions)

    return concept_pos_map


# =====================================================================
# Mock clip object for testing token_utils without ComfyUI
# =====================================================================

class MockClip:
    """Mock ComfyUI CLIP object that forces the fallback path in token_utils."""
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def tokenize(self, prompt):
        """Always raise to trigger the fallback path in find_concept_positions_qwen3."""
        raise RuntimeError("Mock: clip.tokenize not available in test")


# =====================================================================
# Tests
# =====================================================================

def test_qwen3_concept_positions():
    """
    Test Z-Image (Qwen3) concept position finding against the reference
    implementation from main_freefuse_z_image.py.

    Uses the exact same prompt and concepts from main_freefuse_z_image.py.
    """
    print("\n" + "=" * 70)
    print("TEST: Qwen3 concept positions vs reference (main_freefuse_z_image.py)")
    print("=" * 70)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

    find_concept_positions_qwen3 = token_utils.find_concept_positions_qwen3
    is_meaningless_token = token_utils.is_meaningless_token
    clean_token_text = token_utils.clean_token_text

    # ── Use the exact prompt/concepts from main_freefuse_z_image.py ──
    prompt = (
        "A picture of two characters, a starry night scene with northern lights "
        "in background: The first character is Jinx_Arcane, a young woman with "
        "long blue hair in a loose braid and bright blue eyes, wearing a cropped "
        "halter top, gloves, striped pants with belts, and visible tattoos and "
        "the second character is Skeletor in purple hooded cloak flexing muscular "
        "blue arms triumphantly, skull face grinning menacingly, cartoon animation "
        "style, Masters of the Universe character, vibrant purple and blue color scheme"
    )

    concept_map = {
        "jinx": (
            "Jinx_Arcane, a young woman with long blue hair in a loose braid "
            "and bright blue eyes, wearing a cropped halter top, gloves, striped "
            "pants with belts, and visible tattoos"
        ),
        "skeleton": (
            "Skeletor in purple hooded cloak flexing muscular blue arms "
            "triumphantly, skull face grinning menacingly, cartoon animation "
            "style, Masters of the Universe character, vibrant purple and blue "
            "color scheme"
        ),
    }

    background_concept = "a starry night scene with northern lights"

    all_concepts = {**concept_map, "__bg__": background_concept}

    # ── Run reference implementation ──
    print("\n--- Reference implementation (main_freefuse_z_image.py) ---")
    ref_results = reference_find_concept_positions(
        tokenizer, prompt, all_concepts,
        filter_meaningless=True, filter_single_char=True,
    )
    for name, pos_list in ref_results.items():
        positions = pos_list[0]
        print(f"  {name}: positions = {positions}")

    # ── Run token_utils implementation (with mock clip) ──
    print("\n--- token_utils implementation (find_concept_positions_qwen3) ---")
    mock_clip = MockClip(tokenizer)
    tu_results = find_concept_positions_qwen3(
        clip=mock_clip,
        tokenizer=tokenizer,
        prompts=prompt,
        concepts=all_concepts,
        filter_meaningless=True,
        filter_single_char=True,
    )
    for name, pos_list in tu_results.items():
        positions = pos_list[0]
        print(f"  {name}: positions = {positions}")

    # ── Compare ──
    print("\n--- Comparison ---")
    success = True
    for name in all_concepts:
        ref_pos = ref_results[name][0]
        tu_pos = tu_results[name][0]
        match = ref_pos == tu_pos
        status = "✓ MATCH" if match else "✗ MISMATCH"
        print(f"  {status}: {name}")
        if not match:
            success = False
            print(f"    reference : {ref_pos}")
            print(f"    token_utils: {tu_pos}")
            # Show which positions differ
            only_in_ref = set(ref_pos) - set(tu_pos)
            only_in_tu = set(tu_pos) - set(ref_pos)
            if only_in_ref:
                print(f"    MISSING from token_utils: {sorted(only_in_ref)}")
            if only_in_tu:
                print(f"    EXTRA in token_utils: {sorted(only_in_tu)}")

    return success


def test_qwen3_unfiltered_positions():
    """
    Test that the raw (unfiltered) positions match between reference
    and token_utils, before any meaningless-token filtering.
    """
    print("\n" + "=" * 70)
    print("TEST: Qwen3 UNFILTERED positions (no stopword/single-char filter)")
    print("=" * 70)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

    find_concept_positions_qwen3 = token_utils.find_concept_positions_qwen3

    prompt = (
        "A picture of two characters, a starry night scene with northern lights "
        "in background: The first character is Jinx_Arcane, a young woman with "
        "long blue hair in a loose braid and bright blue eyes, wearing a cropped "
        "halter top, gloves, striped pants with belts, and visible tattoos and "
        "the second character is Skeletor in purple hooded cloak flexing muscular "
        "blue arms triumphantly, skull face grinning menacingly, cartoon animation "
        "style, Masters of the Universe character, vibrant purple and blue color scheme"
    )

    concept_map = {
        "jinx": (
            "Jinx_Arcane, a young woman with long blue hair in a loose braid "
            "and bright blue eyes, wearing a cropped halter top, gloves, striped "
            "pants with belts, and visible tattoos"
        ),
        "skeleton": (
            "Skeletor in purple hooded cloak flexing muscular blue arms "
            "triumphantly, skull face grinning menacingly, cartoon animation "
            "style, Masters of the Universe character, vibrant purple and blue "
            "color scheme"
        ),
    }

    # Run both WITHOUT filtering
    ref_results = reference_find_concept_positions(
        tokenizer, prompt, concept_map,
        filter_meaningless=False, filter_single_char=False,
    )
    mock_clip = MockClip(tokenizer)
    tu_results = find_concept_positions_qwen3(
        clip=mock_clip, tokenizer=tokenizer,
        prompts=prompt, concepts=concept_map,
        filter_meaningless=False, filter_single_char=False,
    )

    print("\n--- Comparison (no filtering) ---")
    success = True
    for name in concept_map:
        ref_pos = ref_results[name][0]
        tu_pos = tu_results[name][0]
        match = ref_pos == tu_pos
        status = "✓ MATCH" if match else "✗ MISMATCH"
        print(f"  {status}: {name}")
        print(f"    reference : {ref_pos}")
        print(f"    token_utils: {tu_pos}")
        if not match:
            success = False

    return success


def test_clean_token_text_underscore_bug():
    """
    Diagnose the clean_token_text bug: stripping '_' causes tokens
    like '_A' (part of 'Jinx_Arcane') to become single-char 'a' and
    get filtered out.
    """
    print("\n" + "=" * 70)
    print("TEST: clean_token_text underscore stripping bug")
    print("=" * 70)

    clean_token_text = token_utils.clean_token_text
    is_meaningless_token = token_utils.is_meaningless_token

    # Simulate Qwen3 tokens for "Jinx_Arcane"
    # Qwen3 tokenizes "Jinx_Arcane" as: J | inx | _A | rc | ane
    test_tokens = {
        "J":    "single char in both → filtered by both (OK)",
        "inx":  "should be KEPT by both",
        "_A":   "part of _Arcane — reference KEEPS (len('_a')==2), token_utils DROPS (strips _ → 'a' → single char)",
        "rc":   "should be KEPT by both",
        "ane":  "should be KEPT by both",
    }

    print("\nToken analysis:")
    success = True
    for tok, description in test_tokens.items():
        # Reference method
        ref_cleaned = tok.strip().lower()
        ref_meaningless = (not ref_cleaned) or (len(ref_cleaned) == 1)

        # token_utils method
        tu_cleaned = clean_token_text(tok)
        tu_meaningless = is_meaningless_token(tok, check_single_char=True)

        agree = ref_meaningless == tu_meaningless
        status = "✓" if agree else "✗ BUG"

        print(f"  {status} token={repr(tok):6s}  "
              f"ref_clean={repr(ref_cleaned):6s} ref_filter={ref_meaningless!s:5s}  |  "
              f"tu_clean={repr(tu_cleaned):6s} tu_filter={tu_meaningless!s:5s}  "
              f"  -- {description}")

        if not agree:
            success = False

    if not success:
        print("\n  ⚠ BUG CONFIRMED: clean_token_text strips '_' characters,")
        print("    turning tokens like '_A' into 'a' (single char → filtered).")
        print("    The reference uses simple strip().lower() which preserves '_A' as '_a'.")

    return success


def test_qwen3_attention_mask_filtering():
    """
    Test whether token_utils properly handles attention mask filtering.

    The reference pads to max_length and uses attention_mask to get only
    active tokens. token_utils (fallback path) doesn't pad, so positions
    should still be equivalent. This test verifies that.
    """
    print("\n" + "=" * 70)
    print("TEST: Attention mask filtering (padding behavior)")
    print("=" * 70)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

    prompt = "Jinx_Arcane standing in a field"
    max_seq_len = 512

    # Reference: pad to max_length, then filter by attention_mask
    messages = [{"role": "user", "content": prompt}]
    wrapped = tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=True,
    )
    tok_padded = tokenizer(
        wrapped, padding="max_length", max_length=max_seq_len,
        truncation=True, return_tensors="pt",
    )
    padded_ids = tok_padded.input_ids[0].tolist()
    attn_mask = tok_padded.attention_mask[0].tolist()
    active_ids_ref = [padded_ids[i] for i, m in enumerate(attn_mask) if m == 1]

    # token_utils fallback: no padding
    tok_unpadded = tokenizer(wrapped, add_special_tokens=True)
    unpadded_ids = tok_unpadded['input_ids']
    if hasattr(unpadded_ids, 'tolist'):
        unpadded_ids = unpadded_ids.tolist()

    print(f"  Padded + masked active tokens: {len(active_ids_ref)}")
    print(f"  Unpadded tokens:               {len(unpadded_ids)}")
    print(f"  IDs match: {active_ids_ref == unpadded_ids}")

    # The concat_text and token_spans should be identical
    ref_texts = [tokenizer.decode([tid]) for tid in active_ids_ref]
    tu_texts = [tokenizer.decode([tid]) for tid in unpadded_ids]
    texts_match = ref_texts == tu_texts
    print(f"  Decoded texts match: {texts_match}")

    if not texts_match:
        # Show first difference
        for i, (r, t) in enumerate(zip(ref_texts, tu_texts)):
            if r != t:
                print(f"    First diff at pos {i}: ref={repr(r)}, tu={repr(t)}")
                break

    return active_ids_ref == unpadded_ids


def test_qwen3_detailed_token_dump():
    """
    Detailed dump of token positions for both methods to aid debugging.
    Shows exactly which tokens are found and which are kept/filtered.
    """
    print("\n" + "=" * 70)
    print("TEST: Detailed token dump for 'jinx' concept")
    print("=" * 70)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

    find_concept_positions_qwen3 = token_utils.find_concept_positions_qwen3
    is_meaningless_token = token_utils.is_meaningless_token
    clean_token_text = token_utils.clean_token_text

    prompt = (
        "A picture of two characters: Jinx_Arcane, a young woman with "
        "long blue hair and Skeletor in purple cloak"
    )
    concept_text = "Jinx_Arcane, a young woman with long blue hair"

    # Get reference positions (unfiltered)
    ref_unfiltered = reference_find_concept_positions(
        tokenizer, prompt, {"jinx": concept_text},
        filter_meaningless=False, filter_single_char=False,
    )["jinx"][0]

    ref_filtered = reference_find_concept_positions(
        tokenizer, prompt, {"jinx": concept_text},
        filter_meaningless=True, filter_single_char=True,
    )["jinx"][0]

    # Decode reference tokens
    messages = [{"role": "user", "content": prompt}]
    wrapped = tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=True,
    )
    tok_out = tokenizer(
        wrapped, padding="max_length", max_length=512,
        truncation=True, return_tensors="pt",
    )
    all_ids = tok_out.input_ids[0].tolist()
    mask = tok_out.attention_mask[0].tolist()
    active_ids = [all_ids[i] for i, m in enumerate(mask) if m == 1]
    active_texts = [tokenizer.decode([tid]) for tid in active_ids]

    print(f"\nPrompt: {prompt}")
    print(f"Concept: {concept_text}")
    print(f"\nReference unfiltered positions: {ref_unfiltered}")
    print(f"Reference filtered positions:   {ref_filtered}")

    print("\nDetailed token analysis at concept positions:")
    print(f"  {'Pos':>4s}  {'Token':>10s}  {'ref_clean':>12s}  {'ref_filt':>8s}  "
          f"{'tu_clean':>12s}  {'tu_filt':>8s}  {'agree':>5s}")
    print("  " + "-" * 75)

    all_agree = True
    for pos in ref_unfiltered:
        tok_text = active_texts[pos]
        # Reference filtering (full _is_meaningless check)
        ref_clean = tok_text.strip().lower()
        ref_is_meaningless = (not ref_clean) or (len(ref_clean) == 1) or (ref_clean in {
            'a', 'an', 'the', 'and', 'or', 'but', 'nor', 'so', 'yet',
            'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'under', 'over',
            'it', 'its', 'this', 'that', 'these', 'those', 'their', 'his',
            'her', 'my', 'your', 'our',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'has', 'have', 'had', 'having',
            'which', 'who', 'whom', 'whose', 'where', 'when', 'while',
            ',', '.', '!', '?', ';', ':', '"', "'", '(', ')',
            '[', ']', '{', '}', '-', '–', '—', '/', '\\', '...', '..',
        })

        # token_utils filtering
        tu_clean = clean_token_text(tok_text)
        tu_is_meaningless = is_meaningless_token(tok_text, check_single_char=True)

        agree = ref_is_meaningless == tu_is_meaningless
        if not agree:
            all_agree = False

        flag = "✓" if agree else "✗ BUG"
        print(f"  {pos:4d}  {repr(tok_text):>10s}  {repr(ref_clean):>12s}  "
              f"{'DROP' if ref_is_meaningless else 'KEEP':>8s}  "
              f"{repr(tu_clean):>12s}  "
              f"{'DROP' if tu_is_meaningless else 'KEEP':>8s}  {flag}")

    # Now run token_utils
    mock_clip = MockClip(tokenizer)
    tu_filtered = find_concept_positions_qwen3(
        clip=mock_clip, tokenizer=tokenizer,
        prompts=prompt, concepts={"jinx": concept_text},
        filter_meaningless=True, filter_single_char=True,
    )["jinx"][0]

    print(f"\ntoken_utils filtered positions: {tu_filtered}")
    print(f"Reference filtered positions:   {ref_filtered}")
    match = ref_filtered == tu_filtered
    print(f"Match: {match}")

    if not match:
        missing = set(ref_filtered) - set(tu_filtered)
        extra = set(tu_filtered) - set(ref_filtered)
        if missing:
            missing_tokens = [(p, active_texts[p]) for p in sorted(missing)]
            print(f"  MISSING from token_utils: {missing_tokens}")
        if extra:
            extra_tokens = [(p, active_texts[p]) for p in sorted(extra)]
            print(f"  EXTRA in token_utils: {extra_tokens}")

    return all_agree and match


# =====================================================================
# Part 2: Embedding ↔ Position Alignment Tests
# =====================================================================
# These tests check whether the token positions from find_concept_positions_qwen3
# actually correspond to the correct positions in the prompt_embeds tensor.

# Use the constant from token_utils (the module under test)
LUMINA2_SYSTEM_PROMPT = token_utils.LUMINA2_SYSTEM_PROMPT

LLAMA_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"


def _find_token_indices(token_texts, concept_text):
    """Helper: find token indices whose spans overlap with concept_text."""
    concat = ""
    spans = []
    for tt in token_texts:
        s = len(concat)
        concat += tt
        spans.append((s, len(concat)))
    idx = concat.find(concept_text)
    if idx == -1:
        return []
    c_start, c_end = idx, idx + len(concept_text)
    return [i for i, (s, e) in enumerate(spans) if e > c_start and s < c_end]


def test_comfyui_system_prompt_alignment():
    """
    Test that passing system_prompt=LUMINA2_SYSTEM_PROMPT to
    find_concept_positions_qwen3 makes token positions align with the
    ComfyUI embedding path (CLIPTextEncodeLumina2 + ZImageTokenizer).

    In ComfyUI the actual tokenized text is:
      <|im_start|>user\n{system_prompt} <Prompt Start> {user_prompt}<|im_end|>\n<|im_start|>assistant\n

    When system_prompt is passed, find_concept_positions_qwen3 constructs:
      tokenize_prompt = f"{system_prompt} <Prompt Start> {user_prompt}"
    and passes that to clip.tokenize() / fallback, so positions match.
    """
    print("\n" + "=" * 70)
    print("TEST: ComfyUI system prompt alignment (Bug 2 fix)")
    print("=" * 70)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

    find_concept_positions_qwen3 = token_utils.find_concept_positions_qwen3

    user_prompt = (
        "A picture of two characters, a starry night scene with northern lights "
        "in background: Jinx_Arcane, a young woman with long blue hair"
    )
    concept = "Jinx_Arcane"

    # ── Ground truth: ComfyUI embedding path ──
    comfyui_prompt = f"{LUMINA2_SYSTEM_PROMPT} <Prompt Start> {user_prompt}"
    comfyui_wrapped = LLAMA_TEMPLATE.format(comfyui_prompt)
    comfyui_ids = tokenizer(comfyui_wrapped, add_special_tokens=True)['input_ids']
    if hasattr(comfyui_ids, 'tolist'):
        comfyui_ids = comfyui_ids.tolist()
    comfyui_texts = [tokenizer.decode([tid]) for tid in comfyui_ids]
    comfyui_positions = _find_token_indices(comfyui_texts, concept)

    # ── token_utils WITH system_prompt (should now align) ──
    mock_clip = MockClip(tokenizer)
    tu_results = find_concept_positions_qwen3(
        clip=mock_clip, tokenizer=tokenizer,
        prompts=user_prompt,
        concepts={"jinx": concept},
        filter_meaningless=False, filter_single_char=False,
        system_prompt=LUMINA2_SYSTEM_PROMPT,
    )
    tu_positions = tu_results["jinx"][0]

    # ── token_utils WITHOUT system_prompt (should be misaligned) ──
    tu_no_sp_results = find_concept_positions_qwen3(
        clip=mock_clip, tokenizer=tokenizer,
        prompts=user_prompt,
        concepts={"jinx": concept},
        filter_meaningless=False, filter_single_char=False,
        system_prompt=None,
    )
    tu_no_sp_positions = tu_no_sp_results["jinx"][0]

    print(f"\n  '{concept}' positions:")
    print(f"    ComfyUI embedding (ground truth): {comfyui_positions}")
    print(f"    token_utils WITH system_prompt:    {tu_positions}")
    print(f"    token_utils WITHOUT system_prompt: {tu_no_sp_positions}")

    aligned_with_sp = comfyui_positions == tu_positions
    misaligned_without_sp = comfyui_positions != tu_no_sp_positions

    print(f"\n  With system_prompt → ComfyUI:    {'✓ ALIGNED' if aligned_with_sp else '✗ MISALIGNED'}")
    print(f"  Without system_prompt → ComfyUI: {'✓ correctly MISALIGNED (as expected)' if misaligned_without_sp else '✗ unexpectedly aligned'}")

    if not aligned_with_sp:
        shift = (comfyui_positions[0] - tu_positions[0]) if comfyui_positions and tu_positions else "?"
        print(f"\n  ✗ STILL MISALIGNED by {shift} tokens despite system_prompt fix!")

    success = aligned_with_sp and misaligned_without_sp
    if success:
        print(f"\n  ✓ Bug 2 fix verified: system_prompt parameter correctly aligns positions")
    return success


def test_diffusers_pipeline_alignment():
    """
    Test alignment between the diffusers pipeline (main_freefuse_z_image.py)
    and token_utils WITHOUT system_prompt.

    The diffusers pipeline's _encode_prompt uses:
      tokenizer.apply_chat_template(messages, tokenize=False, ...)
    which wraps as: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n

    token_utils without system_prompt uses the same wrapping, so they SHOULD align.
    """
    print("\n" + "=" * 70)
    print("TEST: Diffusers pipeline alignment (no system prompt)")
    print("=" * 70)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

    find_concept_positions_qwen3 = token_utils.find_concept_positions_qwen3

    user_prompt = (
        "A picture of two characters: Jinx_Arcane, a young woman with long blue hair "
        "and Skeletor in purple cloak"
    )
    concept_text = "Jinx_Arcane, a young woman with long blue hair"
    max_seq_len = 512

    # ── Diffusers pipeline _encode_prompt path ──
    messages = [{"role": "user", "content": user_prompt}]
    pipeline_wrapped = tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=True,
    )
    pipeline_tok = tokenizer(
        pipeline_wrapped, padding="max_length", max_length=max_seq_len,
        truncation=True, return_tensors="pt",
    )
    pipeline_ids = pipeline_tok.input_ids[0].tolist()
    pipeline_mask = pipeline_tok.attention_mask[0].tolist()
    pipeline_active_ids = [pipeline_ids[i] for i, m in enumerate(pipeline_mask) if m == 1]
    pipeline_active_texts = [tokenizer.decode([tid]) for tid in pipeline_active_ids]

    # Ground truth positions in the pipeline embeddings
    pipeline_positions = _find_token_indices(pipeline_active_texts, concept_text)

    # ── token_utils WITHOUT system_prompt (should match diffusers) ──
    mock_clip = MockClip(tokenizer)
    tu_results = find_concept_positions_qwen3(
        clip=mock_clip, tokenizer=tokenizer,
        prompts=user_prompt,
        concepts={"jinx": concept_text},
        filter_meaningless=False, filter_single_char=False,
        system_prompt=None,
    )
    tu_positions = tu_results["jinx"][0]

    print(f"\n  Pipeline active tokens: {len(pipeline_active_ids)}")
    print(f"  Pipeline positions for 'jinx': {pipeline_positions}")
    print(f"  token_utils positions (no SP): {tu_positions}")

    aligned = pipeline_positions == tu_positions
    print(f"\n  Alignment: {'✓ ALIGNED' if aligned else '✗ MISALIGNED'}")

    if aligned:
        # Verify: tokens at those positions reconstruct the concept text
        concept_tokens_text = "".join(
            pipeline_active_texts[pos] for pos in pipeline_positions
            if pos < len(pipeline_active_texts)
        )
        contains_concept = concept_text in concept_tokens_text or \
                           concept_text.replace(" ", "") in concept_tokens_text.replace(" ", "")
        print(f"  Concatenated text at positions: {repr(concept_tokens_text)}")
        print(f"  Contains concept text: {contains_concept}")
        print(f"\n  ✓ token_utils (without system_prompt) aligns with diffusers pipeline")
        return contains_concept

    return False


def test_full_alignment_summary():
    """
    Summary test showing all three paths and their alignment status.
    
    Verifies:
      - token_utils WITH system_prompt → aligns with ComfyUI embedding
      - token_utils WITHOUT system_prompt → aligns with Diffusers pipeline
    """
    print("\n" + "=" * 70)
    print("TEST: Full alignment summary — all three paths")
    print("=" * 70)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

    find_concept_positions_qwen3 = token_utils.find_concept_positions_qwen3

    user_prompt = "A starry night: Jinx_Arcane, a young woman with long blue hair"
    concept = "Jinx_Arcane"

    # ── Path 1: ComfyUI embedding (CLIPTextEncodeLumina2 + ZImageTokenizer) ──
    comfyui_text = f"{LUMINA2_SYSTEM_PROMPT} <Prompt Start> {user_prompt}"
    comfyui_wrapped = LLAMA_TEMPLATE.format(comfyui_text)
    comfyui_ids = tokenizer(comfyui_wrapped, add_special_tokens=True)['input_ids']
    if hasattr(comfyui_ids, 'tolist'):
        comfyui_ids = comfyui_ids.tolist()
    comfyui_texts = [tokenizer.decode([tid]) for tid in comfyui_ids]
    comfyui_positions = _find_token_indices(comfyui_texts, concept)

    # ── Path 2: Diffusers pipeline (apply_chat_template, no system prompt) ──
    messages = [{"role": "user", "content": user_prompt}]
    diffusers_wrapped = tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=True,
    )
    diffusers_tok = tokenizer(
        diffusers_wrapped, padding="max_length", max_length=512,
        truncation=True, return_tensors="pt",
    )
    d_ids = diffusers_tok.input_ids[0].tolist()
    d_mask = diffusers_tok.attention_mask[0].tolist()
    diffusers_active_ids = [d_ids[i] for i, m in enumerate(d_mask) if m == 1]
    diffusers_texts = [tokenizer.decode([tid]) for tid in diffusers_active_ids]
    diffusers_positions = _find_token_indices(diffusers_texts, concept)

    # ── Path 3a: token_utils WITH system_prompt (for ComfyUI) ──
    mock_clip = MockClip(tokenizer)
    tu_sp_results = find_concept_positions_qwen3(
        clip=mock_clip, tokenizer=tokenizer,
        prompts=user_prompt,
        concepts={"jinx": concept},
        filter_meaningless=False, filter_single_char=False,
        system_prompt=LUMINA2_SYSTEM_PROMPT,
    )
    tu_sp_positions = tu_sp_results["jinx"][0]

    # ── Path 3b: token_utils WITHOUT system_prompt (for Diffusers) ──
    tu_no_sp_results = find_concept_positions_qwen3(
        clip=mock_clip, tokenizer=tokenizer,
        prompts=user_prompt,
        concepts={"jinx": concept},
        filter_meaningless=False, filter_single_char=False,
        system_prompt=None,
    )
    tu_no_sp_positions = tu_no_sp_results["jinx"][0]

    print(f"\n  User prompt: {user_prompt}")
    print(f"  Concept: {concept}")
    print(f"\n  Token counts:")
    print(f"    ComfyUI embedding:               {len(comfyui_ids)} tokens")
    print(f"    Diffusers pipeline:               {len(diffusers_active_ids)} tokens")
    print(f"\n  '{concept}' positions:")
    print(f"    ComfyUI embedding:               {comfyui_positions}")
    print(f"    Diffusers pipeline:               {diffusers_positions}")
    print(f"    token_utils (with system_prompt): {tu_sp_positions}")
    print(f"    token_utils (no system_prompt):   {tu_no_sp_positions}")

    # Check alignments
    comfyui_aligned = comfyui_positions == tu_sp_positions
    diffusers_aligned = diffusers_positions == tu_no_sp_positions

    print(f"\n  Alignment check:")
    print(f"    token_utils+SP ↔ ComfyUI:    {'✓ ALIGNED' if comfyui_aligned else '✗ MISALIGNED'}")
    print(f"    token_utils    ↔ Diffusers:  {'✓ ALIGNED' if diffusers_aligned else '✗ MISALIGNED'}")

    success = comfyui_aligned and diffusers_aligned
    if success:
        print(f"\n  ✓ token_utils correctly serves both ComfyUI and Diffusers paths")
        print(f"    - Pass system_prompt=LUMINA2_SYSTEM_PROMPT for ComfyUI workflows")
        print(f"    - Omit system_prompt for Diffusers pipelines")
    else:
        if not comfyui_aligned:
            shift = (comfyui_positions[0] - tu_sp_positions[0]) if comfyui_positions and tu_sp_positions else "?"
            print(f"\n  ✗ ComfyUI still misaligned by {shift} tokens!")
        if not diffusers_aligned:
            print(f"\n  ✗ Diffusers unexpectedly misaligned!")

    return success


if __name__ == "__main__":
    print("FreeFuse Token Position Utilities - Z-Image Debug Test Suite")
    print("=" * 70)
    print("Part 1: Token filtering bugs (clean_token_text)")
    print("Part 2: Embedding ↔ position alignment (system prompt offset)")
    print()

    all_passed = True

    tests = [
        ("clean_token_text underscore bug", test_clean_token_text_underscore_bug),
        ("Qwen3 attention mask filtering", test_qwen3_attention_mask_filtering),
        ("Qwen3 unfiltered positions", test_qwen3_unfiltered_positions),
        ("Qwen3 concept positions vs ref", test_qwen3_concept_positions),
        ("Detailed token dump (jinx)", test_qwen3_detailed_token_dump),
        ("ComfyUI system prompt alignment", test_comfyui_system_prompt_alignment),
        ("Diffusers pipeline alignment", test_diffusers_pipeline_alignment),
        ("Full alignment summary", test_full_alignment_summary),
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
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    print("\n" + ("All tests passed!" if all_passed else "Some tests FAILED!"))
    sys.exit(0 if all_passed else 1)
