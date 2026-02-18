#!/usr/bin/env python3
"""
FreeFuse ComfyUI API demo script.

Usage examples:
1) Submit directly:
   python export_api.py \
     --workflow workflow_api.json \
     --main-prompt "your main prompt" \
     --background-prompt "your background text" \
     --subject-prompt "harry=harry potter, ..." \
     --subject-prompt "daiyu=daiyu_lin, ..." \
     --lora "harry=harry_potter_flux.safetensors" \
     --lora "daiyu=daiyu_lin_flux.safetensors" \
     --negative-prompt "" \
     --seed 42

2) Save modified API graph without submitting:
   python export_api.py \
     --workflow workflow_api.json \
     --main-prompt "your main prompt" \
     --save modified_api.json \
     --dry-run

3) Run with built-in defaults (workflow/prompt/subjects/loras/seed):
   python export_api.py --wait
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent

DEFAULT_WORKFLOW = THIS_DIR / "workflows" / "flux_freefuse_api.json"
DEFAULT_MAIN_PROMPT = (
    "Realistic photography, harry potter, an European photorealistic style teenage wizard boy "
    "with messy black hair, round wire-frame glasses, and bright green eyes, wearing a white "
    "shirt, burgundy and gold striped tie, and dark robes hugging daiyu_lin, a young East "
    "Asian photorealistic style woman in traditional Chinese hanfu dress, elaborate black updo "
    "hairstyle adorned with delicate white floral hairpins and ornaments, dangling red tassel "
    "earrings, soft pink and red color palette, gentle smile with knowing expression, autumn "
    "leaves blurred in the background, high quality, detailed"
)
DEFAULT_BACKGROUND_PROMPT = "autumn leaves blurred in the background"
DEFAULT_SUBJECT_PROMPTS = [
    (
        "harry=harry potter, an European photorealistic style teenage wizard boy with messy black "
        "hair, round wire-frame glasses, and bright green eyes, wearing a white shirt, burgundy "
        "and gold striped tie, and dark robes"
    ),
    (
        "daiyu=daiyu_lin, a young East Asian photorealistic style woman in traditional Chinese "
        "hanfu dress, elaborate black updo hairstyle adorned with delicate white floral hairpins "
        "and ornaments, dangling red tassel earrings, soft pink and red color palette, gentle "
        "smile with knowing expression"
    ),
]
DEFAULT_LORA_OVERRIDES = [
    "harry=harry_potter_flux.safetensors",
    "daiyu=daiyu_lin_flux.safetensors",
]
DEFAULT_NEGATIVE_PROMPT = ""
DEFAULT_SEED = 8888


def _parse_kv_mapping(items: list[str], flag_name: str, value_name: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(
                f"Invalid {flag_name} format: '{raw}'. "
                f"Expected: KEY={value_name}"
            )
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(
                f"Invalid {flag_name} format: '{raw}'. "
                "Key cannot be empty."
            )
        parsed[key] = value
    return parsed


def _parse_subject_prompts(items: list[str]) -> dict[str, str]:
    return _parse_kv_mapping(
        items=items,
        flag_name="--subject-prompt",
        value_name="CONCEPT_TEXT",
    )


def _parse_lora_overrides(items: list[str]) -> dict[str, str]:
    return _parse_kv_mapping(
        items=items,
        flag_name="--lora",
        value_name="LORA_FILENAME",
    )


def _load_api_graph(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "prompt" in data and isinstance(data["prompt"], dict):
        return data["prompt"]

    if isinstance(data, dict):
        return data

    raise ValueError(f"Invalid JSON in {path}: expected object at top level")


def _node_title(node: dict[str, Any]) -> str:
    return str(node.get("_meta", {}).get("title", "")).strip().lower()


def _apply_prompt_overrides(
    graph: dict[str, Any],
    main_prompt: str | None,
    negative_prompt: str | None,
) -> dict[str, int]:
    stats = {
        "freefuse_token_positions_prompt": 0,
        "clip_positive": 0,
        "clip_negative": 0,
        "string_concat_b": 0,
        "primitive_value": 0,
    }

    for node in graph.values():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type", ""))
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        title = _node_title(node)

        if main_prompt is not None and class_type == "FreeFuseTokenPositions" and "prompt" in inputs:
            inputs["prompt"] = main_prompt
            stats["freefuse_token_positions_prompt"] += 1

        if class_type == "CLIPTextEncodeFlux":
            is_negative = "negative" in title
            if main_prompt is not None and not is_negative:
                if "clip_l" in inputs:
                    inputs["clip_l"] = main_prompt
                if "t5xxl" in inputs:
                    inputs["t5xxl"] = main_prompt
                stats["clip_positive"] += 1
            if negative_prompt is not None and is_negative:
                if "clip_l" in inputs:
                    inputs["clip_l"] = negative_prompt
                if "t5xxl" in inputs:
                    inputs["t5xxl"] = negative_prompt
                stats["clip_negative"] += 1

        if class_type == "CLIPTextEncode":
            is_negative = "negative" in title
            if main_prompt is not None and not is_negative and "text" in inputs:
                inputs["text"] = main_prompt
                stats["clip_positive"] += 1
            if negative_prompt is not None and is_negative and "text" in inputs:
                inputs["text"] = negative_prompt
                stats["clip_negative"] += 1

        # Z-Image workflow commonly wraps user prompt via StringConcatenate.string_b.
        if main_prompt is not None and class_type == "StringConcatenate" and "string_b" in inputs:
            if "prompt" in title or "wrapper" in title:
                inputs["string_b"] = main_prompt
                stats["string_concat_b"] += 1

        # In some workflows primitives are still materialized in API graph.
        if main_prompt is not None and class_type in {"PrimitiveString", "PrimitiveStringMultiline"}:
            if "value" in inputs and "prompt" in title:
                inputs["value"] = main_prompt
                stats["primitive_value"] += 1

    return stats


def _apply_concept_overrides(
    graph: dict[str, Any],
    subject_prompts: dict[str, str],
    background_prompt: str | None,
) -> dict[str, int]:
    stats = {
        "concept_nodes_found": 0,
        "subject_concepts_updated": 0,
        "background_updated": 0,
    }

    for node in graph.values():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type", ""))
        if class_type not in {"FreeFuseConceptMap", "FreeFuseConceptMapSimple"}:
            continue

        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        stats["concept_nodes_found"] += 1

        for i in range(1, 5):
            adapter_key = f"adapter_name_{i}"
            concept_key = f"concept_text_{i}"
            if adapter_key not in inputs or concept_key not in inputs:
                continue
            adapter_name = str(inputs.get(adapter_key, "")).strip()
            if adapter_name and adapter_name in subject_prompts:
                inputs[concept_key] = subject_prompts[adapter_name]
                stats["subject_concepts_updated"] += 1

        if background_prompt is not None and "background_text" in inputs:
            inputs["background_text"] = background_prompt
            stats["background_updated"] += 1

    return stats


def _apply_lora_overrides(
    graph: dict[str, Any],
    lora_overrides: dict[str, str],
) -> tuple[dict[str, int], list[str]]:
    stats = {
        "lora_nodes_found": 0,
        "lora_updated": 0,
    }
    matched_adapters: set[str] = set()

    if not lora_overrides:
        return stats, []

    for node in graph.values():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type", ""))
        if class_type not in {"FreeFuseLoRALoader", "FreeFuseLoRALoaderSimple"}:
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        stats["lora_nodes_found"] += 1

        adapter_name = str(inputs.get("adapter_name", "")).strip()
        if adapter_name in lora_overrides and "lora_name" in inputs:
            inputs["lora_name"] = lora_overrides[adapter_name]
            matched_adapters.add(adapter_name)
            stats["lora_updated"] += 1

    unmatched = sorted(set(lora_overrides.keys()) - matched_adapters)
    return stats, unmatched


def _apply_seed_override(graph: dict[str, Any], seed: int | None) -> int:
    if seed is None:
        return 0

    updated = 0
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        if "seed" in inputs:
            inputs["seed"] = int(seed)
            updated += 1
    return updated


def _queue_prompt(server: str, graph: dict[str, Any], client_id: str | None) -> dict[str, Any]:
    payload: dict[str, Any] = {"prompt": graph}
    if client_id:
        payload["client_id"] = client_id

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://{server}/prompt",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_history(server: str, prompt_id: str) -> dict[str, Any]:
    with urllib.request.urlopen(f"http://{server}/history/{prompt_id}") as resp:
        return json.loads(resp.read().decode("utf-8"))


def _wait_until_done(server: str, prompt_id: str, timeout_sec: float, poll_sec: float) -> dict[str, Any]:
    start = time.time()
    while True:
        history = _get_history(server, prompt_id)
        if prompt_id in history:
            return history[prompt_id]
        if time.time() - start > timeout_sec:
            raise TimeoutError(f"Timeout waiting for prompt {prompt_id} (>{timeout_sec}s)")
        time.sleep(poll_sec)


def _print_output_summary(history_entry: dict[str, Any]) -> None:
    outputs = history_entry.get("outputs", {})
    image_count = 0
    print("Execution finished. Output summary:")
    for node_id, node_out in outputs.items():
        if not isinstance(node_out, dict):
            continue
        images = node_out.get("images", [])
        if not images:
            continue
        for image in images:
            filename = image.get("filename", "")
            subfolder = image.get("subfolder", "")
            folder_type = image.get("type", "")
            image_count += 1
            print(f"- node={node_id} file={filename} subfolder={subfolder} type={folder_type}")
    if image_count == 0:
        print("- no image outputs found in history entry")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit FreeFuse workflow API JSON to ComfyUI.")
    parser.add_argument(
        "--no-defaults",
        action="store_true",
        help="Disable built-in runnable defaults; only use explicitly passed arguments.",
    )
    parser.add_argument(
        "--workflow",
        type=Path,
        default=None,
        help=(
            "Path to API JSON (File -> Export(API) result). "
            f"Default: {DEFAULT_WORKFLOW}"
        ),
    )
    parser.add_argument(
        "--server",
        type=str,
        default="127.0.0.1:8188",
        help="ComfyUI server host:port (default: 127.0.0.1:8188).",
    )
    parser.add_argument(
        "--main-prompt",
        "--prompt",
        dest="main_prompt",
        type=str,
        default=None,
        help="Main prompt. Will be synchronized to relevant FreeFuse/CLIP nodes.",
    )
    parser.add_argument(
        "--background-prompt",
        type=str,
        default=None,
        help=(
            "Background text override for FreeFuseConceptMap.background_text. "
            f"Default: {DEFAULT_BACKGROUND_PROMPT!r}"
        ),
    )
    parser.add_argument(
        "--subject-prompt",
        action="append",
        default=None,
        metavar="ADAPTER=TEXT",
        help=(
            "Subject concept override by adapter name. "
            "Repeat for multiple subjects, e.g. --subject-prompt \"harry=...\". "
            f"Default adapters: {[x.split('=', 1)[0] for x in DEFAULT_SUBJECT_PROMPTS]}"
        ),
    )
    parser.add_argument(
        "--lora",
        action="append",
        default=None,
        metavar="ADAPTER=LORA_FILE",
        help=(
            "LoRA override by adapter name for FreeFuseLoRALoader nodes. "
            "Repeat for multiple adapters, e.g. --lora \"harry=my_harry.safetensors\". "
            f"Default adapters: {[x.split('=', 1)[0] for x in DEFAULT_LORA_OVERRIDES]}"
        ),
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help=(
            "Negative prompt (applies to negative CLIP nodes). "
            f"Default: {DEFAULT_NEGATIVE_PROMPT!r}"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"Override every node input named 'seed'. Default: {DEFAULT_SEED}",
    )
    parser.add_argument(
        "--client-id",
        type=str,
        default=None,
        help="Optional client_id for ComfyUI queue payload.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional output path to save modified API graph JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not submit; only apply overrides and optionally save.",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait until job appears in /history and print output summary.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Wait timeout in seconds when --wait is enabled (default: 600).",
    )
    parser.add_argument(
        "--poll",
        type=float,
        default=1.0,
        help="Polling interval in seconds for --wait (default: 1).",
    )
    return parser.parse_args()


def _populate_runnable_defaults(args: argparse.Namespace) -> None:
    if args.no_defaults:
        return

    if args.workflow is None:
        args.workflow = DEFAULT_WORKFLOW
    if args.main_prompt is None:
        args.main_prompt = DEFAULT_MAIN_PROMPT
    if args.background_prompt is None:
        args.background_prompt = DEFAULT_BACKGROUND_PROMPT
    if args.subject_prompt is None:
        args.subject_prompt = list(DEFAULT_SUBJECT_PROMPTS)
    if args.lora is None:
        args.lora = list(DEFAULT_LORA_OVERRIDES)
    if args.negative_prompt is None:
        args.negative_prompt = DEFAULT_NEGATIVE_PROMPT
    if args.seed is None:
        args.seed = DEFAULT_SEED


def main() -> int:
    args = _parse_args()
    _populate_runnable_defaults(args)

    if args.workflow is None:
        print("Missing --workflow. Provide a path or remove --no-defaults.")
        return 1
    if not args.workflow.exists():
        print(f"Workflow file not found: {args.workflow}")
        return 1

    graph = _load_api_graph(args.workflow)
    try:
        subject_prompts = _parse_subject_prompts(args.subject_prompt or [])
        lora_overrides = _parse_lora_overrides(args.lora or [])
    except ValueError as e:
        print(str(e))
        return 1

    concept_stats = _apply_concept_overrides(
        graph=graph,
        subject_prompts=subject_prompts,
        background_prompt=args.background_prompt,
    )
    lora_stats, unmatched_lora_adapters = _apply_lora_overrides(
        graph=graph,
        lora_overrides=lora_overrides,
    )
    prompt_stats = _apply_prompt_overrides(graph, args.main_prompt, args.negative_prompt)
    seed_updates = _apply_seed_override(graph, args.seed)

    print("Applied overrides:")
    print(f"- concept overrides: {concept_stats}")
    print(f"- lora overrides: {lora_stats}")
    print(f"- prompt sync stats: {prompt_stats}")
    print(f"- seed fields updated: {seed_updates}")
    if unmatched_lora_adapters:
        print(
            f"Warning: these --lora adapters were not matched in workflow: {unmatched_lora_adapters}"
        )

    if (subject_prompts or args.background_prompt is not None) and args.main_prompt is None:
        print(
            "Warning: subject/background updated but main prompt not changed. "
            "FreeFuse requires subject concept_text entries to appear in main prompt."
        )
    if args.main_prompt is not None and subject_prompts:
        for adapter_name, concept_text in subject_prompts.items():
            if concept_text and concept_text not in args.main_prompt:
                print(
                    f"Warning: subject '{adapter_name}' concept_text not found in --main-prompt. "
                    "FreeFuseTokenPositions may fail validation."
                )

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        with args.save.open("w", encoding="utf-8") as f:
            json.dump(graph, f, ensure_ascii=False, indent=2)
        print(f"Saved modified graph to: {args.save}")

    if args.dry_run:
        print("Dry-run mode: skip submitting to ComfyUI.")
        return 0

    client_id = args.client_id or str(uuid.uuid4())
    try:
        result = _queue_prompt(args.server, graph, client_id=client_id)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"HTTP error when posting /prompt: {e.code}\n{body}")
        return 1
    except urllib.error.URLError as e:
        print(f"Failed to connect to ComfyUI server {args.server}: {e}")
        return 1

    prompt_id = result.get("prompt_id")
    if not prompt_id:
        print(f"Unexpected /prompt response: {result}")
        return 1

    print(f"Queued successfully. prompt_id={prompt_id}")
    if result.get("node_errors"):
        print(f"Node validation messages: {result['node_errors']}")

    if args.wait:
        try:
            history_entry = _wait_until_done(args.server, prompt_id, args.timeout, args.poll)
            _print_output_summary(history_entry)
        except TimeoutError as e:
            print(str(e))
            return 1
        except urllib.error.URLError as e:
            print(f"Failed while polling history: {e}")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
