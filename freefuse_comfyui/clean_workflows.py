#!/usr/bin/env python3
"""
Clean FreeFuse workflow JSON files for shipping.
Removes personal data like:
- Loaded LoRAs
- Concept texts
- Prompts
- Seeds
- Specific model selections
"""

import json
import os

WORKFLOW_FILES = [
    "FreeFuse-qwen-image-manual-mask.json",
    "FreeFuse-zimage-manual-mask.json",
    "FreeFuse-zimage-standard.json",
    "FreeFuse-qwen-image-ai-mask.json",
    "FreeFuse-zimage-ai-mask.json",
]

def clean_widget_value(node_type, widget_name, value):
    """Clean widget values based on node type and widget name."""
    # Seeds - always reset to default
    if widget_name == "seed":
        if isinstance(value, list):
            return [0, "fixed"]
        return [0, "fixed"]
    
    # Prompts/text - clear to empty or default
    if widget_name in ["text", "concept_text_1", "concept_text_2", "concept_text_3", 
                       "concept_text_4", "concept_text_5", "concept_text_6",
                       "user_text", "injection_text", "background_text", "location_text",
                       "adapter_name", "filename_prefix"]:
        if widget_name == "filename_prefix":
            return "FreeFuse"
        elif widget_name == "adapter_name":
            return "character1"
        elif widget_name == "user_text":
            return ""
        elif widget_name == "injection_text":
            return "You are an assistant designed to generate superior images with the superior degree of image-text alignment based on textual prompts or user prompts. <Prompt Start>"
        return ""
    
    # LoRA names - reset to None or first option
    if widget_name in ["lora_name_1", "lora_name_2", "lora_name_3", 
                       "lora_name_4", "lora_name_5", "lora_name_6",
                       "lora_name", "unet_name", "clip_name", "vae_name"]:
        if widget_name == "lora_name_1":
            return "None"
        return "None"
    
    # Strength values - reset to defaults
    if widget_name in ["strength_model_1", "strength_clip_1",
                       "strength_model_2", "strength_clip_2",
                       "strength_model_3", "strength_clip_3",
                       "strength_model_4", "strength_clip_4",
                       "strength_model_5", "strength_clip_5",
                       "strength_model_6", "strength_clip_6",
                       "strength_model"]:
        if widget_name in ["strength_model_1", "strength_clip_1"]:
            return 1.0
        return 0.0
    
    # Boolean flags - keep reasonable defaults
    if widget_name in ["enable_background", "enable_token_masking", "enable_attention_bias",
                       "bidirectional", "use_positive_bias", "filter_meaningless", 
                       "filter_single_char", "fill_holes", "remove_small_regions",
                       "smooth_boundaries", "apply_threshold", "use_morphological_cleaning",
                       "show_background", "show_legend", "low_vram_mode", "remove_small"]:
        return True if widget_name not in ["show_background", "apply_threshold", "smooth_boundaries"] else False
    
    # Numeric parameters - keep defaults or reset to reasonable values
    if widget_name in ["steps", "collect_step", "collect_block", "collect_block_end",
                       "collect_tf_index", "max_iter", "max_hole_size", "kernel_size",
                       "iterations", "min_region_size", "preview_size", "max_iterations"]:
        defaults = {
            "steps": 28,
            "collect_step": 5,
            "collect_block": 18,
            "collect_block_end": 18,
            "collect_tf_index": 3,
            "max_iter": 15,
            "max_hole_size": 50,
            "kernel_size": 3,
            "iterations": 1,
            "min_region_size": 100,
            "preview_size": 1024,
            "max_iterations": 12,
        }
        return defaults.get(widget_name, 0)
    
    # Float parameters
    if widget_name in ["cfg", "denoise", "temperature", "top_k_ratio", "balance_lr",
                       "gravity_weight", "spatial_weight", "momentum", "anisotropy",
                       "centroid_margin", "border_penalty", "bg_scale", "sensitivity",
                       "bias_scale", "positive_bias_scale", "blur_sigma", "threshold_value",
                       "shift", "balance_iterations", "preview_sensitivity"]:
        defaults = {
            "cfg": 3.5,
            "denoise": 1.0,
            "temperature": 4000.0,
            "top_k_ratio": 0.3,
            "balance_lr": 0.01,
            "gravity_weight": 0.00004,
            "spatial_weight": 0.00004,
            "momentum": 0.2,
            "anisotropy": 1.3,
            "centroid_margin": 0.0,
            "border_penalty": 0.0,
            "bg_scale": 0.95,
            "sensitivity": 5.0,
            "bias_scale": 5.0,
            "positive_bias_scale": 1.0,
            "blur_sigma": 1.0,
            "threshold_value": 0.5,
            "shift": 3.0,
            "balance_iterations": 15,
            "preview_sensitivity": 5.0,
        }
        return defaults.get(widget_name, 1.0)
    
    # Combo selections - keep defaults
    if widget_name in ["sampler_name", "scheduler", "argmax_method", "morph_operation",
                       "bias_blocks", "collect_region", "weight_dtype", "device", "type"]:
        defaults = {
            "sampler_name": "euler",
            "scheduler": "simple",
            "argmax_method": "stabilized",
            "morph_operation": "close",
            "bias_blocks": "double_stream_only",
            "collect_region": "output_early ★ (recommended)",
            "weight_dtype": "default",
            "device": "default",
            "type": "default",
        }
        return defaults.get(widget_name, "default")
    
    # Width/Height - keep reasonable defaults
    if widget_name in ["width", "height", "batch_size", "preview_size"]:
        defaults = {
            "width": 512,
            "height": 512,
            "batch_size": 1,
            "preview_size": 1024,
        }
        return defaults.get(widget_name, 512)
    
    # Keep the value as-is for everything else
    return value


def clean_node(node):
    """Clean a single node's widget values."""
    node_type = node.get("type", "")
    widgets = node.get("widgets_values", [])
    
    if not widgets:
        return
    
    # Get widget names based on node type
    widget_names = []
    
    # Define widget names for common node types
    if node_type == "FreeFuse6LoraLoader":
        widget_names = [
            "adapter_name", "location_text",
            "lora_name_1", "strength_model_1", "strength_clip_1", "concept_text_1",
            "lora_name_2", "strength_model_2", "strength_clip_2", "concept_text_2",
            "lora_name_3", "strength_model_3", "strength_clip_3", "concept_text_3",
            "lora_name_4", "strength_model_4", "strength_clip_4", "concept_text_4",
            "lora_name_5", "strength_model_5", "strength_clip_5", "concept_text_5",
            "lora_name_6", "strength_model_6", "strength_clip_6", "concept_text_6",
        ]
    elif node_type == "CLIPTextEncode":
        widget_names = ["text"]
    elif node_type == "KSampler":
        widget_names = ["seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"]
    elif node_type == "Seed Generator":
        widget_names = ["seed"]
    elif node_type == "EmptySD3LatentImage":
        widget_names = ["width", "height", "batch_size"]
    elif node_type == "FreeFuseTokenPositions":
        widget_names = ["injection_text", "user_text", "filter_meaningless", "filter_single_char"]
    elif node_type == "FreeFuseBackgroundLoader":
        widget_names = ["enable_background", "background_text"]
    elif node_type == "FreeFuseMaskApplicator":
        widget_names = [
            "enable_token_masking", "enable_attention_bias", "bias_scale",
            "positive_bias_scale", "bidirectional", "use_positive_bias", "bias_blocks"
        ]
    elif node_type == "FreeFuseQwenSimilarityExtractor":
        widget_names = [
            "seed", "steps", "collect_step", "collect_block", "cfg", "sampler_name",
            "scheduler", "temperature", "top_k_ratio", "preview_size", "low_vram_mode",
            "attention_head_index"
        ]
    elif node_type == "FreeFuseRawSimilarityOverlay":
        widget_names = [
            "preview_size", "sensitivity", "show_background", "show_legend",
            "argmax_method", "max_iter", "balance_lr", "gravity_weight", "spatial_weight",
            "momentum", "anisotropy", "centroid_margin", "border_penalty", "bg_scale",
            "use_morphological_cleaning"
        ]
    elif node_type == "FreeFuseMaskRefiner":
        widget_names = [
            "fill_holes", "max_hole_size", "morph_operation", "kernel_size", "iterations",
            "smooth_boundaries", "blur_sigma", "apply_threshold", "threshold_value",
            "remove_small_regions", "min_region_size"
        ]
    elif node_type == "SaveImage":
        widget_names = ["filename_prefix"]
    elif node_type == "ModelSamplingAuraFlow":
        widget_names = ["shift"]
    elif node_type == "UNETLoader":
        widget_names = ["unet_name", "weight_dtype"]
    elif node_type == "CLIPLoader":
        widget_names = ["clip_name", "type", "device"]
    elif node_type == "VAELoader":
        widget_names = ["vae_name"]
    elif node_type == "LoraLoaderModelOnly":
        widget_names = ["lora_name", "strength_model"]
    
    # Clean widgets
    for i, value in enumerate(widgets):
        widget_name = widget_names[i] if i < len(widget_names) else f"widget_{i}"
        widgets[i] = clean_widget_value(node_type, widget_name, value)


def clean_workflow(filepath):
    """Clean a workflow JSON file."""
    print(f"Cleaning: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        workflow = json.load(f)
    
    # Clean all nodes
    nodes = workflow.get("nodes", [])
    for node in nodes:
        clean_node(node)
    
    # Remove workflow ID and revision (will be regenerated on load)
    workflow["id"] = None
    workflow["revision"] = 0
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(workflow, f, separators=(',', ':'))
    
    print(f"  ✓ Cleaned {len(nodes)} nodes")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    for filename in WORKFLOW_FILES:
        filepath = os.path.join(script_dir, filename)
        if os.path.exists(filepath):
            clean_workflow(filepath)
        else:
            print(f"  ✗ Not found: {filepath}")
    
    print("\nDone! All workflows cleaned for shipping.")


if __name__ == "__main__":
    main()
