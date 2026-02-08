#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FreeFuse Z-Image (ZIT) Block Comparison Script

This script tests which transformer block produces the best concept-similarity
maps for spatial mask extraction in the Z-Image single-stream architecture.

For each of the 30 main transformer layers (``pipe.transformer.layers[i]``),
we run Phase 1 (collect sim maps) + Phase 2 (generate with masks & bias),
save per-block debug masks & result images, and finally build a comparison grid.

Key differences from sdxl_block_comparison.py:
- Z-Image uses single-stream self-attention (not cross-attn like SDXL)
- Uses FreeFuseZImageAttnProcessor + FreeFuseZImagePipeline
- Concept positions found via Qwen3 tokenizer with chat template
- Pipeline param ``sim_map_block_idx`` selects the layer for sim-map collection
"""

import os
import sys
import glob
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image, ImageDraw, ImageFont

from peft.tuners.lora.layer import LoraLayer, Linear as LoraLinear

from src.pipeline.freefuse_z_image_pipeline import FreeFuseZImagePipeline
from src.attn_processor.freefuse_z_image_attn_processor import FreeFuseZImageAttnProcessor
from src.models.freefuse_transformer_z_image import ZImageTransformer2DModel, ZImageTransformerBlock
from src.tuner.freefuse_lora_layer import FreeFuseLinear

from diffusers.models.transformers.transformer_z_image import (
    ZImageTransformer2DModel as OrigZImageTransformer2DModel,
)


# ──────────────────────────────────────────────────────────────
# Qwen3-aware concept position finder (same as main script)
# ──────────────────────────────────────────────────────────────

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

    prompt_data_list = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        wrapped_text = tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=True,
        )
        tok_out = tokenizer(
            wrapped_text, padding="max_length",
            max_length=max_sequence_length, truncation=True,
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
            "raw": prompt, "wrapped": wrapped_text,
            "active_indices": active_indices,
            "active_token_texts": active_token_texts,
            "concat_text": concat_text,
            "token_spans": token_spans,
        })

    concept_pos_map = {}
    for concept_name, concept_text in concepts.items():
        concept_pos_map[concept_name] = []
        for pd in prompt_data_list:
            positions, positions_with_text = [], []
            search_start = 0
            while True:
                idx = pd["concat_text"].find(concept_text, search_start)
                if idx == -1:
                    break
                c_start, c_end = idx, idx + len(concept_text)
                for tok_i, (ts, te) in enumerate(pd["token_spans"]):
                    if te > c_start and ts < c_end and tok_i not in positions:
                        positions.append(tok_i)
                        positions_with_text.append((tok_i, pd["active_token_texts"][tok_i]))
                search_start = idx + 1
            if not positions:
                print(f"[warn] concept '{concept_name}' not found via concat decode")

            if filter_meaningless and positions_with_text:
                filtered = [p for p, t in positions_with_text if not _is_meaningless(t, filter_single_char)]
                if not filtered:
                    non_punct = [p for p, t in positions_with_text if t.strip() not in PUNCTUATION]
                    filtered = non_punct[:1] if non_punct else [positions_with_text[0][0]]
                positions = filtered

            positions.sort()
            concept_pos_map[concept_name].append(positions)

    return concept_pos_map


def find_eos_index(pipe, prompt, max_sequence_length=512):
    tokenizer = pipe.tokenizer
    messages = [{"role": "user", "content": prompt}]
    wrapped = tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=True,
    )
    tok_out = tokenizer(
        wrapped, padding="max_length",
        max_length=max_sequence_length, truncation=True,
        return_tensors="pt",
    )
    ids = tok_out.input_ids[0]
    mask = tok_out.attention_mask[0].bool()
    active_ids = ids[mask]
    eos_id = tokenizer.eos_token_id
    eos_pos = (active_ids == eos_id).nonzero(as_tuple=True)[0]
    if len(eos_pos) > 0:
        return eos_pos[0].item()
    return None


# ──────────────────────────────────────────────────────────────
# Setup helpers
# ──────────────────────────────────────────────────────────────

def upgrade_lora_to_freefuse(transformer):
    """Upgrade all LoraLinear layers in the transformer to FreeFuseLinear."""
    upgraded = 0
    for name, module in transformer.named_modules():
        if isinstance(module, LoraLayer) and isinstance(module, LoraLinear):
            FreeFuseLinear.init_from_lora_linear(module)
            upgraded += 1
    return upgraded


def setup_freefuse_processors(transformer):
    """Replace every attention processor with FreeFuseZImageAttnProcessor."""
    procs = transformer.attn_processors
    new_procs = {name: FreeFuseZImageAttnProcessor() for name in procs}
    transformer.set_attn_processor(new_procs)
    return len(new_procs)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    # ═══════════════ Configuration ═══════════════
    model_id = "Tongyi-MAI/Z-Image-Turbo"

    lora_configs = [
        {
            "path": "loras/Jinx_Arcane_zit.safetensors",
            "adapter_name": "jinx",
            "concept_text": (
                "Jinx_Arcane, a young woman with long blue hair in a loose braid "
                "and bright blue eyes, wearing a cropped halter top, gloves, "
                "striped pants with belts, and visible tattoos"
            ),
            "weight": 0.8,
        },
        {
            "path": "loras/Vi_Arcane_zit.safetensors",
            "adapter_name": "vi",
            "concept_text": (
                "Vi_Arcane, a young woman with short pink hair in an undercut "
                "swept to one side and blue eyes, wearing a red jacket over a "
                "fitted top, small 'VI' tattoo under her eye, nose ring"
            ),
            "weight": 0.8,
        },
    ]

    prompt = (
        "A picture of two characters, a starry night scene with northern lights "
        "in background: Jinx_Arcane, a young woman with long blue hair in a loose "
        "braid and bright blue eyes, wearing a cropped halter top, gloves, striped "
        "pants with belts, and visible tattoos and Vi_Arcane, a young woman with "
        "short pink hair in an undercut swept to one side and blue eyes, wearing a "
        "red jacket over a fitted top, small 'VI' tattoo under her eye, nose ring"
    )
    negative_prompt = ""
    background_concept = "a starry night scene with northern lights"

    height = 1024
    width = 1024
    num_inference_steps = 12       # turbo schedule
    guidance_scale = 3.5
    seed = 42
    aggreate_lora_score_step = 3

    # Attention bias
    use_attention_bias = True
    attention_bias_scale = 3.0
    attention_bias_positive = True
    attention_bias_positive_scale = 1.0
    attention_bias_bidirectional = True
    attention_bias_blocks = "last_half"

    # Which blocks to test (None → auto-discover all main layers)
    block_indices = None  # will be set to range(n_layers)

    output_base_dir = "outputs/zit_block_comparison"
    os.makedirs(output_base_dir, exist_ok=True)

    device = "cuda"
    dtype = torch.bfloat16

    # ═══════════════ Load pipeline ═══════════════
    print("Loading Z-Image pipeline …")
    pipe = FreeFuseZImagePipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.to(device)

    # ═══════════════ Load LoRAs ═══════════════
    print("Loading LoRAs …")
    adapter_names = []
    concepts = {}
    for lora in lora_configs:
        if os.path.exists(lora["path"]):
            pipe.load_lora_weights(lora["path"], adapter_name=lora["adapter_name"])
            adapter_names.append(lora["adapter_name"])
            concepts[lora["adapter_name"]] = lora["concept_text"]
            print(f"  ✓ {lora['adapter_name']}  ({lora['path']})")
        else:
            print(f"  ✗ {lora['adapter_name']}  — file not found: {lora['path']}")

    if len(adapter_names) < 2:
        print("Error: need at least 2 LoRAs loaded.")
        return

    weights = [l["weight"] for l in lora_configs if l["adapter_name"] in adapter_names]
    pipe.set_adapters(adapter_names, weights)

    # ═══════════════ Swap in FreeFuse transformer + processors ═══════════════
    print("Swapping transformer class & setting FreeFuse processors …")
    pipe.transformer.__class__ = ZImageTransformer2DModel
    n_procs = setup_freefuse_processors(pipe.transformer)
    print(f"  {n_procs} attention processors set.")

    n_upgraded = upgrade_lora_to_freefuse(pipe.transformer)
    print(f"  {n_upgraded} LoRA layers upgraded to FreeFuseLinear.")

    # ═══════════════ Discover blocks ═══════════════
    n_layers = len(pipe.transformer.layers)
    if block_indices is None:
        block_indices = list(range(n_layers))
    print(f"\nTotal main transformer layers: {n_layers}")
    print(f"Blocks to test: {len(block_indices)}")

    # ═══════════════ Concept positions ═══════════════
    freefuse_token_pos_maps = find_concept_positions(pipe, prompt, concepts)
    print("\nConcept token positions:")
    for k, v in freefuse_token_pos_maps.items():
        print(f"  {k}: {v[0][:8]}… ({len(v[0])} tokens)")

    background_token_positions = find_concept_positions(
        pipe, prompt, {"__bg__": background_concept}
    )["__bg__"][0]
    print(f"Background positions: {background_token_positions[:8]}… ({len(background_token_positions)} tokens)")

    # ═══════════════ Test each block ═══════════════
    print(f"\n{'=' * 60}")
    print("Testing block configurations …")
    print(f"{'=' * 60}")

    results = []  # (block_idx, status, output_dir)

    for test_i, blk_idx in enumerate(block_indices):
        output_dir = os.path.join(output_base_dir, f"block_{blk_idx:02d}")
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n[{test_i + 1}/{len(block_indices)}]  sim_map_block_idx = {blk_idx}")

        # Reset FreeFuse state on transformer between runs
        pipe.transformer.reset_freefuse()

        generator = torch.Generator(device).manual_seed(seed)

        try:
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                sim_map_block_idx=blk_idx,
                aggreate_lora_score_step=aggreate_lora_score_step,
                use_attention_bias=use_attention_bias,
                attention_bias_scale=attention_bias_scale,
                attention_bias_positive=attention_bias_positive,
                attention_bias_positive_scale=attention_bias_positive_scale,
                attention_bias_bidirectional=attention_bias_bidirectional,
                attention_bias_blocks=attention_bias_blocks,
                debug_save_path=output_dir,
                joint_attention_kwargs={
                    "freefuse_token_pos_maps": freefuse_token_pos_maps,
                    "eos_token_index": None,
                    "background_token_positions": background_token_positions,
                    "top_k_ratio": 0.1,
                },
            ).images[0]

            image.save(os.path.join(output_dir, "result.png"))
            print(f"  ✓ Saved → {output_dir}")
            results.append((blk_idx, "success", output_dir))

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ✗ Error: {e}")
            results.append((blk_idx, f"error: {e}", output_dir))

    # ═══════════════ Comparison grid ═══════════════
    print(f"\n{'=' * 60}")
    print("Creating comprehensive comparison grid …")

    concept_names = list(concepts.keys())
    step_tag = f"phase1_{aggreate_lora_score_step}"

    # Collect successful results
    comparison_data = []
    for blk_idx, status, output_dir in results:
        if status != "success":
            continue
        result_path = os.path.join(output_dir, "result.png")
        if not os.path.exists(result_path):
            continue

        result_img = Image.open(result_path)
        mask_images = {}
        sim_images = {}
        for cname in concept_names:
            # Look for the clean LoRA mask
            for pattern in [
                os.path.join(output_dir, f"lora_mask_{cname}_{step_tag}_clean.png"),
                os.path.join(output_dir, f"lora_mask_{cname}_{step_tag}.png"),
            ]:
                if os.path.exists(pattern):
                    mask_images[cname] = Image.open(pattern)
                    break

            # Look for concept sim map (clean viridis preferred)
            for pattern in [
                os.path.join(output_dir, f"concept_sim_map_{cname}_{step_tag}_clean_viridis.png"),
                os.path.join(output_dir, f"concept_sim_map_{cname}_{step_tag}.png"),
            ]:
                if os.path.exists(pattern):
                    sim_images[cname] = Image.open(pattern)
                    break

        comparison_data.append({
            "block_idx": blk_idx,
            "result_img": result_img,
            "mask_images": mask_images,
            "sim_images": sim_images,
        })

    if not comparison_data:
        print("No successful results to build grid.")
        return

    # ── grid layout ──
    num_blocks = len(comparison_data)
    num_concepts = len(concept_names)

    thumb_size = 256
    label_width = 160
    padding = 5
    header_height = 50
    row_height = thumb_size + padding

    # Columns: label | sim_map_c1 | sim_map_c2 | mask_c1 | mask_c2 | result
    col_labels = []
    for cn in concept_names:
        col_labels.append(f"SimMap: {cn}")
    for cn in concept_names:
        col_labels.append(f"Mask: {cn}")
    col_labels.append("Result")

    num_img_cols = len(col_labels)
    total_width = label_width + num_img_cols * (thumb_size + padding) + padding
    total_height = header_height + num_blocks * row_height + padding

    grid_img = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(grid_img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
        header_font = font

    # Column headers
    draw.text((padding, 14), "Block", fill="black", font=header_font)
    x_off = label_width + padding
    for col_label in col_labels:
        draw.text((x_off + 10, 14), col_label, fill="black", font=header_font)
        x_off += thumb_size + padding
    draw.line([(0, header_height - 3), (total_width, header_height - 3)], fill="gray", width=2)

    # Rows
    for row_idx, data in enumerate(comparison_data):
        y = header_height + row_idx * row_height
        blk_label = f"layers.{data['block_idx']}"
        draw.text((padding, y + thumb_size // 2 - 8), blk_label, fill="black", font=font)
        draw.line([(label_width, y), (label_width, y + thumb_size)], fill="lightgray", width=1)

        x = label_width + padding

        # Sim maps
        for cn in concept_names:
            if cn in data["sim_images"]:
                img = data["sim_images"][cn].resize((thumb_size, thumb_size), Image.LANCZOS).convert("RGB")
                grid_img.paste(img, (x, y))
            else:
                draw.rectangle([(x, y), (x + thumb_size, y + thumb_size)], outline="lightgray", fill="#f0f0f0")
                draw.text((x + 30, y + thumb_size // 2), "N/A", fill="gray", font=font)
            x += thumb_size + padding

        # Masks
        for cn in concept_names:
            if cn in data["mask_images"]:
                img = data["mask_images"][cn].resize((thumb_size, thumb_size), Image.LANCZOS).convert("RGB")
                grid_img.paste(img, (x, y))
            else:
                draw.rectangle([(x, y), (x + thumb_size, y + thumb_size)], outline="lightgray", fill="#f0f0f0")
                draw.text((x + 30, y + thumb_size // 2), "N/A", fill="gray", font=font)
            x += thumb_size + padding

        # Result
        res = data["result_img"].resize((thumb_size, thumb_size), Image.LANCZOS).convert("RGB")
        grid_img.paste(res, (x, y))

        if row_idx < num_blocks - 1:
            line_y = y + thumb_size + padding // 2
            draw.line([(0, line_y), (total_width, line_y)], fill="#e0e0e0", width=1)

    grid_path = os.path.join(output_base_dir, "comprehensive_comparison.png")
    grid_img.save(grid_path, quality=95)
    print(f"Saved comprehensive comparison grid → {grid_path}")
    print(f"  Size: {total_width}×{total_height}  |  {num_blocks} blocks × {num_img_cols} cols")

    # ── Simple result-only grid ──
    print("\nCreating simple comparison grid (results only) …")
    grid_cols = min(6, num_blocks)
    grid_rows = (num_blocks + grid_cols - 1) // grid_cols
    label_h = 30
    gw = grid_cols * thumb_size + (grid_cols + 1) * padding
    gh = grid_rows * (thumb_size + label_h) + (grid_rows + 1) * padding
    simple_grid = Image.new("RGB", (gw, gh), "white")
    sdraw = ImageDraw.Draw(simple_grid)

    for idx, data in enumerate(comparison_data):
        row = idx // grid_cols
        col = idx % grid_cols
        x = padding + col * (thumb_size + padding)
        y = padding + row * (thumb_size + label_h + padding)
        img = data["result_img"].resize((thumb_size, thumb_size), Image.LANCZOS)
        simple_grid.paste(img, (x, y))
        sdraw.text((x + 5, y + thumb_size + 5), f"block {data['block_idx']}", fill="black", font=font)

    simple_path = os.path.join(output_base_dir, "comparison_grid.png")
    simple_grid.save(simple_path, quality=95)
    print(f"Saved simple comparison grid → {simple_path}")

    # ═══════════════ Summary ═══════════════
    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"{'=' * 60}")
    for blk_idx, status, output_dir in results:
        icon = "✓" if status == "success" else "✗"
        print(f"  {icon} block {blk_idx:2d}: {status}")

    success_count = sum(1 for _, s, _ in results if s == "success")
    print(f"\n{success_count}/{len(results)} blocks succeeded.")
    print(f"All outputs saved to: {output_base_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
