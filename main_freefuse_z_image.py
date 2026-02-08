"""
Main script for FreeFuse on Z-Image (Turbo).

This script demonstrates:
  1. Loading Z-Image-Turbo pipeline
  2. Loading multiple LoRA adapters
  3. Swapping in FreeFuse transformer + attention processors
  4. Converting PEFT LoRA layers to FreeFuseLinear
  5. Finding concept token positions with Qwen3 tokenizer (chat-template aware)
  6. Running the FreeFuse two-phase pipeline with attention bias & BG exclusion
"""
import os
import torch
from peft.tuners.lora.layer import LoraLayer, Linear

from src.pipeline.freefuse_z_image_pipeline import FreeFuseZImagePipeline
from src.attn_processor.freefuse_z_image_attn_processor import FreeFuseZImageAttnProcessor
from src.models.freefuse_transformer_z_image import ZImageTransformer2DModel, ZImageTransformerBlock
from src.tuner.freefuse_lora_layer import FreeFuseLinear

# Original diffusers classes (for class-swap)
from diffusers.models.transformers.transformer_z_image import (
    ZImageTransformer2DModel as OrigZImageTransformer2DModel,
)


# ── Qwen3-aware concept position finder ────────────────────────────────

def find_concept_positions(
    pipe,
    prompts,
    concepts,
    filter_meaningless=True,
    filter_single_char=True,
    max_sequence_length=512,
):
    """
    Find token positions for each concept inside the *chat-template-wrapped*
    prompt, matching the exact tokenization that ``encode_prompt`` produces.

    The Qwen3 tokenizer may not support ``return_offsets_mapping`` when a chat
    template is applied, so we use a character-search + per-token decoding
    strategy instead.

    Args:
        pipe: Pipeline (must have ``.tokenizer``).
        prompts: ``str`` or ``List[str]``
        concepts: ``dict``  –  ``{adapter_name: concept_text}``
        filter_meaningless: drop stop-words / punctuation tokens
        filter_single_char: drop single-character tokens
        max_sequence_length: tokenizer max length

    Returns:
        ``{adapter_name: [[positions_for_prompt_0], [positions_for_prompt_1], ...]}``
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

    tokenizer = pipe.tokenizer

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

        # Decode each token individually to build char offsets
        # (works regardless of whether return_offsets_mapping is supported)
        token_texts = [tokenizer.decode([tid]) for tid in token_ids]

        # Only keep tokens where attention_mask == 1 (non-padding)
        # After encoding, the pipeline keeps only unpadded tokens via:
        #   embeddings_list.append(prompt_embeds[i][prompt_masks[i]])
        # So the *effective* positions are indices into the masked subset.
        active_indices = [idx for idx, m in enumerate(attn_mask) if m == 1]
        active_token_texts = [token_texts[idx] for idx in active_indices]

        # Reconstruct the concatenated decoded string and track spans
        concat_text = ""
        token_spans = []  # (start_in_concat, end_in_concat) per active token
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

            # Find concept_text inside concat_text (case-sensitive)
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

            # If nothing found in concat_text, try a looser search in the
            # raw prompt (concept might span special tokens differently).
            if not positions:
                print(f"[warn] concept '{concept_name}' not found via concat "
                      f"decode; falling back to raw-prompt search.")
                # Tokenize the concept alone and do subsequence search
                concept_ids = tokenizer.encode(concept_text, add_special_tokens=False)
                all_ids = [
                    pd["active_indices"]  # we still need original ids
                ]
                # (leave positions empty – user should adjust concept text)

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


def find_eos_index(pipe, prompt, max_sequence_length=512):
    """
    Find the EOS token position in the Qwen3 tokenized prompt
    (after chat template, with attention-mask filtering applied).

    Returns the index *within the active (non-padding) token sequence*,
    or ``None`` if not found.
    """
    tokenizer = pipe.tokenizer
    messages = [{"role": "user", "content": prompt}]
    wrapped = tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=True,
    )
    tok_out = tokenizer(
        wrapped, padding="max_length",
        max_length=max_sequence_length,
        truncation=True, return_tensors="pt",
    )
    ids = tok_out.input_ids[0]
    mask = tok_out.attention_mask[0].bool()
    active_ids = ids[mask]

    eos_id = tokenizer.eos_token_id
    eos_pos = (active_ids == eos_id).nonzero(as_tuple=True)[0]
    if len(eos_pos) > 0:
        return eos_pos[0].item()
    print(f"[warn] EOS token (id={eos_id}) not found in prompt")
    return None


# ── Main ────────────────────────────────────────────────────────────────

def main():
    device = "cuda"
    dtype = torch.bfloat16
    model_id = "Tongyi-MAI/Z-Image-Turbo"

    # ── 1. Load pipeline ──
    pipe = FreeFuseZImagePipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.to(device)

    # ── 2. Load LoRA adapters ──
    # TODO: replace with Z-Image-compatible LoRA files once available
    pipe.load_lora_weights("loras/Jinx_Arcane_zit.safetensors", adapter_name="jinx")
    pipe.load_lora_weights("loras/Vi_Arcane_zit.safetensors", adapter_name="vi")
    pipe.set_adapters(["jinx", "vi"], [0.8, 0.8])

    # ── 3. Swap transformer class ──
    pipe.transformer.__class__ = ZImageTransformer2DModel

    # ── 4. Set FreeFuse attention processors on every block ──
    current_processors = pipe.transformer.attn_processors
    processor_dict = {}
    for name in current_processors.keys():
        processor_dict[name] = FreeFuseZImageAttnProcessor()
    pipe.transformer.set_attn_processor(processor_dict)

    # ── 5. Replace PEFT LoRA Linear layers with FreeFuseLinear ──
    for module in pipe.transformer.modules():
        if isinstance(module, LoraLayer) and isinstance(module, Linear):
            FreeFuseLinear.init_from_lora_linear(module)

    # ── 6. Build prompts & concept map ──
    concept_map = {
        "jinx": "Jinx_Arcane, a young woman with long blue hair in a loose braid and bright blue eyes, wearing a cropped halter top, gloves, striped pants with belts, and visible tattoos",
        "vi": "Vi_Arcane, a young woman with short pink hair in an undercut swept to one side and blue eyes, wearing a red jacket over a fitted top, small 'VI' tattoo under her eye, nose ring"
    }

    prompt = "A picture of two characters, a starry night scene with northern lights in background: Jinx_Arcane, a young woman with long blue hair in a loose braid and bright blue eyes, wearing a cropped halter top, gloves, striped pants with belts, and visible tattoos and Vi_Arcane, a young woman with short pink hair in an undercut swept to one side and blue eyes, wearing a red jacket over a fitted top, small 'VI' tattoo under her eye, nose ring"
    negative_prompt = ""

    # ── 7. Find concept positions (Qwen3-aware) ──
    freefuse_token_pos_maps = find_concept_positions(pipe, prompt, concept_map)
    print("[Info] Concept token position maps:")
    for k, v in freefuse_token_pos_maps.items():
        print(f"  {k}: {v}")

    # Background via concept text (option A) or EOS (option B)
    background_concept = "a starry night scene with northern lights"
    background_token_positions = find_concept_positions(
        pipe, prompt, {"__bg__": background_concept}
    )["__bg__"][0]
    print(f"[Info] Background token positions: {background_token_positions}")
    eos_token_index = None  # set to find_eos_index(pipe, prompt) for option B

    # ── 8. Determine which block to collect sim maps from ──
    # Use a block in the second half (deep features)
    n_layers = len(pipe.transformer.layers)
    sim_map_block_idx = 18  # use block 18
    print(f"[Info] sim_map_block_idx = {sim_map_block_idx}  (out of {n_layers} layers)")

    # ── 9. Generate ──
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=1024,
        width=1024,
        num_inference_steps=12,      # turbo schedule
        guidance_scale=0.0,
        generator=generator,
        sim_map_block_idx=sim_map_block_idx,
        aggreate_lora_score_step=3,
        use_attention_bias=True,
        attention_bias_scale=3.0,
        attention_bias_positive=True,
        attention_bias_positive_scale=1.0,
        attention_bias_bidirectional=True,
        attention_bias_blocks="last_half",
        debug_save_path="debug_z_image",
        joint_attention_kwargs={
            "freefuse_token_pos_maps": freefuse_token_pos_maps,
            "eos_token_index": eos_token_index,
            "background_token_positions": background_token_positions,
            "top_k_ratio": 0.1,
        },
    ).images[0]

    out_path = "z_image_freefuse_output.png"
    image.save(out_path)
    print(f"[Done] Image saved to {out_path}")


if __name__ == "__main__":
    main()
