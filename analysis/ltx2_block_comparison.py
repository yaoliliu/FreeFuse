#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FreeFuse LTX2 block comparison script.

Purpose:
- sweep `transformer_blocks.{i}.attn2` + `transformer_blocks.{i}.audio_attn2`
- run FreeFuse once per block pair
- save per-block keyframe + FreeFuse debug outputs
- parse debug summaries into simple metrics
- generate comparison grids for quick visual inspection
"""

import csv
import glob
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Add project root to import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.freefuse_ltx2_pipeline import FreeFuseLTX2Pipeline


def _safe_float(x: Optional[float], default: float = 0.0) -> float:
    if x is None:
        return default
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Optional[float], default: int = 0) -> int:
    if x is None:
        return default
    try:
        return int(float(x))
    except Exception:
        return default


def _to_uint8_image(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        max_val = float(arr.max()) if arr.size > 0 else 0.0
        min_val = float(arr.min()) if arr.size > 0 else 0.0
        if max_val <= 1.0 and min_val >= 0.0:
            arr = (arr * 255.0).clip(0, 255)
        else:
            arr = arr.clip(0, 255)
        return arr.astype(np.uint8)
    return arr.astype(np.uint8)


def _extract_keyframe_from_output(output) -> Image.Image:
    """
    Convert pipeline output to one representative frame image.
    """
    frames = output.frames if hasattr(output, "frames") else output[0]

    if isinstance(frames, list):
        first_item = frames[0]
    else:
        first_item = frames

    # np format: (T, H, W, C)
    if isinstance(first_item, np.ndarray):
        if first_item.ndim == 4:
            frame = first_item[0]
        elif first_item.ndim == 3:
            frame = first_item
        else:
            raise ValueError(f"Unexpected ndarray frame shape: {first_item.shape}")
        return Image.fromarray(_to_uint8_image(frame)).convert("RGB")

    # PIL format: list[PIL.Image]
    if isinstance(first_item, list) and first_item and isinstance(first_item[0], Image.Image):
        return first_item[0].convert("RGB")
    if isinstance(first_item, Image.Image):
        return first_item.convert("RGB")

    raise TypeError(f"Unsupported output frame type: {type(first_item)}")


def _find_latest_phase1_dir(block_dir: str) -> Optional[str]:
    phase1_dirs = [d for d in glob.glob(os.path.join(block_dir, "phase1_step_*")) if os.path.isdir(d)]
    if not phase1_dirs:
        return None
    phase1_dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return phase1_dirs[0]


def _parse_summary_file(summary_path: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not os.path.isfile(summary_path):
        return out
    with open(summary_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            # parse numeric values where possible
            try:
                if "." in v or "e" in v.lower():
                    out[k] = float(v)
                else:
                    out[k] = float(int(v))
            except Exception:
                # keep only numeric keys in this dict
                continue
    return out


def _collect_debug_images(phase1_dir: str, concept_names: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    sim_paths: Dict[str, str] = {}
    mask_paths: Dict[str, str] = {}

    for concept in concept_names:
        sim_candidates = [
            os.path.join(phase1_dir, f"video_sim_map_{concept}_frames.png"),
            os.path.join(phase1_dir, f"video_sim_map_{concept}_token_curve.png"),
            os.path.join(phase1_dir, f"video_sim_map_{concept}_temporal_curve.png"),
        ]
        mask_candidates = [
            os.path.join(phase1_dir, f"video_mask_{concept}_frames.png"),
            os.path.join(phase1_dir, f"video_mask_{concept}_token_curve.png"),
            os.path.join(phase1_dir, f"video_mask_{concept}_temporal_curve.png"),
        ]

        for p in sim_candidates:
            if os.path.isfile(p):
                sim_paths[concept] = p
                break
        for p in mask_candidates:
            if os.path.isfile(p):
                mask_paths[concept] = p
                break

    return sim_paths, mask_paths


def _compute_metrics(summary: Dict[str, float], concept_names: List[str]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    video_seq_len = _safe_int(summary.get("video_seq_len"), 0)
    audio_seq_len = _safe_int(summary.get("audio_seq_len"), 0)

    video_sums = [_safe_float(summary.get(f"video_mask_{c}_sum"), 0.0) for c in concept_names]
    audio_sums = [_safe_float(summary.get(f"audio_mask_{c}_sum"), 0.0) for c in concept_names]

    def _mask_stats(mask_sums: List[float], seq_len: int, prefix: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        total = float(sum(mask_sums))
        out[f"{prefix}_mask_total"] = total
        out[f"{prefix}_foreground_ratio"] = total / float(seq_len) if seq_len > 0 else 0.0

        max_sum = max(mask_sums) if mask_sums else 0.0
        min_sum = min(mask_sums) if mask_sums else 0.0
        out[f"{prefix}_balance"] = (min_sum / max_sum) if max_sum > 1e-8 else 0.0

        if total > 1e-8 and len(mask_sums) > 1:
            probs = [x / total for x in mask_sums if x > 0]
            entropy = -sum(p * math.log(p + 1e-12) for p in probs)
            out[f"{prefix}_entropy_norm"] = entropy / math.log(len(mask_sums))
        else:
            out[f"{prefix}_entropy_norm"] = 0.0
        return out

    metrics.update(_mask_stats(video_sums, video_seq_len, "video"))
    metrics.update(_mask_stats(audio_sums, audio_seq_len, "audio"))

    # Heuristic ranking signal: video mask quality is primary.
    metrics["composite_score"] = (
        metrics.get("video_entropy_norm", 0.0)
        * metrics.get("video_foreground_ratio", 0.0)
        * (0.5 + 0.5 * metrics.get("video_balance", 0.0))
    )
    return metrics


def _build_comprehensive_grid(
    rows: List[Dict],
    concept_names: List[str],
    output_path: str,
    thumb_size: int = 256,
) -> None:
    if not rows:
        return

    label_width = 220
    padding = 6
    header_height = 54
    row_height = thumb_size + padding

    col_labels = []
    for c in concept_names:
        col_labels.append(f"Sim: {c}")
    for c in concept_names:
        col_labels.append(f"Mask: {c}")
    col_labels.append("Keyframe")

    n_img_cols = len(col_labels)
    total_width = label_width + n_img_cols * (thumb_size + padding) + padding
    total_height = header_height + len(rows) * row_height + padding

    canvas = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        head_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
        head_font = font

    draw.text((padding, 16), "Block", fill="black", font=head_font)
    x = label_width + padding
    for lab in col_labels:
        draw.text((x + 8, 16), lab, fill="black", font=head_font)
        x += thumb_size + padding
    draw.line([(0, header_height - 2), (total_width, header_height - 2)], fill="gray", width=2)

    for row_idx, row in enumerate(rows):
        y = header_height + row_idx * row_height
        block_label = f"block {row['block_idx']:02d}"
        score = _safe_float(row.get("composite_score"), 0.0)
        draw.text((padding, y + 12), block_label, fill="black", font=font)
        draw.text((padding, y + 34), f"score={score:.4f}", fill="#1f4d8f", font=font)

        x = label_width + padding
        for c in concept_names:
            img = row["sim_images"].get(c)
            if img is None:
                draw.rectangle([(x, y), (x + thumb_size, y + thumb_size)], outline="lightgray", fill="#f0f0f0")
                draw.text((x + 24, y + thumb_size // 2 - 8), "N/A", fill="gray", font=font)
            else:
                canvas.paste(img.resize((thumb_size, thumb_size), Image.LANCZOS).convert("RGB"), (x, y))
            x += thumb_size + padding

        for c in concept_names:
            img = row["mask_images"].get(c)
            if img is None:
                draw.rectangle([(x, y), (x + thumb_size, y + thumb_size)], outline="lightgray", fill="#f0f0f0")
                draw.text((x + 24, y + thumb_size // 2 - 8), "N/A", fill="gray", font=font)
            else:
                canvas.paste(img.resize((thumb_size, thumb_size), Image.LANCZOS).convert("RGB"), (x, y))
            x += thumb_size + padding

        key = row.get("keyframe")
        if key is None:
            draw.rectangle([(x, y), (x + thumb_size, y + thumb_size)], outline="lightgray", fill="#f0f0f0")
            draw.text((x + 24, y + thumb_size // 2 - 8), "N/A", fill="gray", font=font)
        else:
            canvas.paste(key.resize((thumb_size, thumb_size), Image.LANCZOS).convert("RGB"), (x, y))

        if row_idx < len(rows) - 1:
            line_y = y + thumb_size + padding // 2
            draw.line([(0, line_y), (total_width, line_y)], fill="#e0e0e0", width=1)

    canvas.save(output_path, quality=95)


def _build_keyframe_grid(rows: List[Dict], output_path: str, thumb_size: int = 256, max_cols: int = 6) -> None:
    if not rows:
        return

    rows = sorted(rows, key=lambda x: int(x["block_idx"]))
    cols = min(max_cols, max(1, len(rows)))
    rnum = (len(rows) + cols - 1) // cols
    label_h = 30
    pad = 6
    width = cols * thumb_size + (cols + 1) * pad
    height = rnum * (thumb_size + label_h) + (rnum + 1) * pad

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for idx, row in enumerate(rows):
        rr = idx // cols
        cc = idx % cols
        x = pad + cc * (thumb_size + pad)
        y = pad + rr * (thumb_size + label_h + pad)

        key = row.get("keyframe")
        if key is not None:
            canvas.paste(key.resize((thumb_size, thumb_size), Image.LANCZOS).convert("RGB"), (x, y))
        else:
            draw.rectangle([(x, y), (x + thumb_size, y + thumb_size)], outline="lightgray", fill="#f0f0f0")
            draw.text((x + 24, y + thumb_size // 2 - 8), "N/A", fill="gray", font=font)

        draw.text((x + 4, y + thumb_size + 5), f"block {int(row['block_idx']):02d}", fill="black", font=font)

    canvas.save(output_path, quality=95)


def main():
    # -------------------------- Config --------------------------
    model_id = "Lightricks/LTX-2"
    output_base_dir = "outputs/ltx2_block_comparison"
    os.makedirs(output_base_dir, exist_ok=True)

    lora_items = [
        ("loras/rapstangled_ltx2.safetensors", "rapstangled", 1.0, "rapstangled2010 in her signature purple dress"),
        ("loras/tifa_ltx2.safetensors", "tifa", 1.0, "Tifa Lockhart with long dark hair and crimson eyes"),
    ]

    prompt = (
        "An animated cinematic medium two-shot in a mountain cabin kitchen at night, "
        "with a crackling campfire glowing through open wooden doors and cool misty blue mountains outside. "
        "rapstangled2010 in her signature purple dress sits beside the fire, warm orange light flickering across her face. "
        "Tifa Lockhart with long dark hair and crimson eyes picks up a dropped receipt, turns toward her with a playful smirk. "
        "They exchange eye contact, react to each other, and share a brief laugh. "
        "The camera tracks in a slow front-left to front-right arc with coherent dialogue timing."
    )
    negative_prompt = "shaky, glitchy, low quality, worst quality, deformed, distorted, motion smear, static"

    width = 768
    height = 512
    num_frames = 121
    frame_rate = 24.0
    num_inference_steps = 40
    guidance_scale = 4.0
    noise_scale = 0.0

    # FreeFuse controls
    freefuse_top_k_ratio = 0.1
    freefuse_phase1_step = 10
    freefuse_attention_bias_scale = 4.0
    freefuse_attention_bias_positive_scale = 2.0
    freefuse_use_av_cross_attention_bias = False

    # Block sweep config
    block_indices: Optional[List[int]] = None  # None => all transformer blocks
    skip_if_result_exists = True
    seed = 42
    max_sequence_length = 1024

    # Runtime
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    use_sequential_cpu_offload = bool(torch.cuda.is_available())

    # -------------------------- Load pipeline --------------------------
    print("Loading FreeFuseLTX2Pipeline ...")
    pipe = FreeFuseLTX2Pipeline.from_pretrained(model_id, torch_dtype=dtype)

    # load + activate adapters before class-swap
    print("Loading LoRAs ...")
    adapter_names: List[str] = []
    adapter_scales: List[float] = []
    concept_map: Dict[str, str] = {}
    for lora_path, adapter_name, scale, concept_text in lora_items:
        if not os.path.isfile(lora_path):
            raise FileNotFoundError(f"Missing LoRA file: {lora_path}")
        pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
        adapter_names.append(adapter_name)
        adapter_scales.append(float(scale))
        concept_map[adapter_name] = concept_text
        print(f"  loaded {adapter_name} <- {lora_path}")

    pipe.set_adapters(adapter_names, adapter_scales)
    pipe.setup_freefuse_attention_processors()
    pipe.convert_lora_layers(include_connectors=False)

    if use_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(device=device)
    else:
        pipe.to(device)

    # -------------------------- Discover blocks --------------------------
    total_blocks = len(pipe.transformer.transformer_blocks)
    if block_indices is None:
        block_indices = list(range(total_blocks))
    else:
        block_indices = [int(i) for i in block_indices if 0 <= int(i) < total_blocks]

    if not block_indices:
        raise ValueError("No valid block indices to test.")

    print(f"Total transformer blocks: {total_blocks}")
    print(f"Testing blocks: {block_indices[0]}..{block_indices[-1]} ({len(block_indices)} blocks)")

    # -------------------------- Token maps --------------------------
    token_pos_maps = pipe.find_concept_positions(
        prompt=prompt,
        concept_map=concept_map,
        max_sequence_length=max_sequence_length,
    )
    eos_index = pipe.find_eos_token_index(prompt, max_sequence_length=max_sequence_length)

    print("Concept token positions:")
    for k, v in token_pos_maps.items():
        pos = v[0] if v else []
        print(f"  {k}: {pos[:10]} ... ({len(pos)} tokens)")
    print(f"EOS index: {eos_index}")

    # -------------------------- Sweep --------------------------
    raw_results: List[Dict] = []
    concept_names = list(concept_map.keys())

    for i, block_idx in enumerate(block_indices):
        block_name = f"transformer_blocks.{block_idx}.attn2"
        audio_block_name = f"transformer_blocks.{block_idx}.audio_attn2"
        block_dir = os.path.join(output_base_dir, f"block_{block_idx:02d}")
        os.makedirs(block_dir, exist_ok=True)

        keyframe_path = os.path.join(block_dir, "result_keyframe.png")
        print(f"\n[{i + 1}/{len(block_indices)}] {block_name} + {audio_block_name}")

        if skip_if_result_exists and os.path.isfile(keyframe_path):
            print("  skip: existing result_keyframe.png")
            phase1_dir = _find_latest_phase1_dir(block_dir)
            summary = _parse_summary_file(os.path.join(phase1_dir, "summary.txt")) if phase1_dir else {}
            metrics = _compute_metrics(summary, concept_names)
            raw_results.append(
                {
                    "block_idx": block_idx,
                    "status": "success",
                    "block_dir": block_dir,
                    "phase1_dir": phase1_dir,
                    "summary": summary,
                    "metrics": metrics,
                }
            )
            continue

        try:
            generator = torch.Generator(device=device).manual_seed(seed)

            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                noise_scale=noise_scale,
                output_type="np",
                return_dict=True,
                max_sequence_length=max_sequence_length,
                freefuse_enabled=True,
                freefuse_token_pos_maps=token_pos_maps,
                freefuse_eos_token_index=eos_index,
                freefuse_top_k_ratio=freefuse_top_k_ratio,
                freefuse_phase1_step=freefuse_phase1_step,
                freefuse_video_collect_block=block_name,
                freefuse_audio_collect_block=audio_block_name,
                freefuse_attention_bias_scale=freefuse_attention_bias_scale,
                freefuse_attention_bias_positive_scale=freefuse_attention_bias_positive_scale,
                freefuse_use_av_cross_attention_bias=freefuse_use_av_cross_attention_bias,
                freefuse_debug_save_path=block_dir,
                freefuse_debug_collect_per_step=False,
                generator=generator,
            )

            keyframe = _extract_keyframe_from_output(output)
            keyframe.save(keyframe_path)
            print(f"  saved keyframe -> {keyframe_path}")

            phase1_dir = _find_latest_phase1_dir(block_dir)
            summary = _parse_summary_file(os.path.join(phase1_dir, "summary.txt")) if phase1_dir else {}
            metrics = _compute_metrics(summary, concept_names)
            raw_results.append(
                {
                    "block_idx": block_idx,
                    "status": "success",
                    "block_dir": block_dir,
                    "phase1_dir": phase1_dir,
                    "summary": summary,
                    "metrics": metrics,
                }
            )
        except Exception as exc:
            traceback.print_exc()
            print(f"  error: {exc}")
            raw_results.append(
                {
                    "block_idx": block_idx,
                    "status": f"error: {exc}",
                    "block_dir": block_dir,
                    "phase1_dir": None,
                    "summary": {},
                    "metrics": {},
                }
            )

    # -------------------------- Assemble rows for grids --------------------------
    rows_for_grid: List[Dict] = []
    for item in raw_results:
        if item["status"] != "success":
            continue

        keyframe_path = os.path.join(item["block_dir"], "result_keyframe.png")
        keyframe = Image.open(keyframe_path).convert("RGB") if os.path.isfile(keyframe_path) else None

        sim_images: Dict[str, Image.Image] = {}
        mask_images: Dict[str, Image.Image] = {}
        phase1_dir = item.get("phase1_dir")
        if phase1_dir:
            sim_paths, mask_paths = _collect_debug_images(phase1_dir, concept_names)
            for c, p in sim_paths.items():
                if os.path.isfile(p):
                    sim_images[c] = Image.open(p).convert("RGB")
            for c, p in mask_paths.items():
                if os.path.isfile(p):
                    mask_images[c] = Image.open(p).convert("RGB")

        rows_for_grid.append(
            {
                "block_idx": item["block_idx"],
                "keyframe": keyframe,
                "sim_images": sim_images,
                "mask_images": mask_images,
                "composite_score": _safe_float(item["metrics"].get("composite_score"), 0.0),
            }
        )

    rows_for_grid.sort(key=lambda x: int(x["block_idx"]))

    # -------------------------- Save tables --------------------------
    csv_path = os.path.join(output_base_dir, "metrics.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "block_idx",
                "status",
                "composite_score",
                "video_entropy_norm",
                "video_foreground_ratio",
                "video_balance",
                "audio_entropy_norm",
                "audio_foreground_ratio",
                "audio_balance",
                "video_seq_len",
                "audio_seq_len",
            ]
        )
        for item in sorted(raw_results, key=lambda x: int(x["block_idx"])):
            m = item.get("metrics", {})
            s = item.get("summary", {})
            writer.writerow(
                [
                    item["block_idx"],
                    item["status"],
                    _safe_float(m.get("composite_score"), 0.0),
                    _safe_float(m.get("video_entropy_norm"), 0.0),
                    _safe_float(m.get("video_foreground_ratio"), 0.0),
                    _safe_float(m.get("video_balance"), 0.0),
                    _safe_float(m.get("audio_entropy_norm"), 0.0),
                    _safe_float(m.get("audio_foreground_ratio"), 0.0),
                    _safe_float(m.get("audio_balance"), 0.0),
                    _safe_int(s.get("video_seq_len"), 0),
                    _safe_int(s.get("audio_seq_len"), 0),
                ]
            )
    print(f"\nSaved metrics table -> {csv_path}")

    # -------------------------- Build grids --------------------------
    comp_grid_path = os.path.join(output_base_dir, "comprehensive_comparison.png")
    _build_comprehensive_grid(rows_for_grid, concept_names, comp_grid_path)
    print(f"Saved comprehensive grid -> {comp_grid_path}")

    keyframe_grid_path = os.path.join(output_base_dir, "keyframe_grid.png")
    _build_keyframe_grid(rows_for_grid, keyframe_grid_path)
    print(f"Saved keyframe grid -> {keyframe_grid_path}")

    # -------------------------- Ranking summary --------------------------
    success_items = [x for x in raw_results if x["status"] == "success"]
    success_items.sort(key=lambda x: _safe_float(x.get("metrics", {}).get("composite_score"), 0.0), reverse=True)
    rank_path = os.path.join(output_base_dir, "ranking.txt")
    with open(rank_path, "w", encoding="utf-8") as f:
        f.write("LTX2 FreeFuse block ranking (by composite_score)\n")
        f.write("=" * 72 + "\n")
        for rank, item in enumerate(success_items, start=1):
            m = item.get("metrics", {})
            f.write(
                f"{rank:02d}. block {item['block_idx']:02d} | "
                f"score={_safe_float(m.get('composite_score')):.6f} | "
                f"video_balance={_safe_float(m.get('video_balance')):.4f} | "
                f"video_fg={_safe_float(m.get('video_foreground_ratio')):.4f} | "
                f"video_entropy={_safe_float(m.get('video_entropy_norm')):.4f}\n"
            )
    print(f"Saved ranking -> {rank_path}")

    # -------------------------- Console summary --------------------------
    print("\nSummary:")
    print("=" * 72)
    success_count = 0
    for item in sorted(raw_results, key=lambda x: int(x["block_idx"])):
        ok = item["status"] == "success"
        success_count += int(ok)
        icon = "✓" if ok else "✗"
        score = _safe_float(item.get("metrics", {}).get("composite_score"), 0.0)
        print(f"  {icon} block {item['block_idx']:02d}: {item['status']}  score={score:.6f}")
    print(f"\n{success_count}/{len(raw_results)} blocks succeeded.")
    if success_items:
        best = success_items[0]
        print(f"Best block by heuristic score: {best['block_idx']}")
    print(f"All outputs saved to: {output_base_dir}")


if __name__ == "__main__":
    main()

