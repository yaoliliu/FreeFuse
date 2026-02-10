#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FreeFuse SDXL Example Script with Multiple Processor Options

This script demonstrates how to use FreeFuse with SDXL for multi-LoRA image generation
using various attention processor methods for concept-aware spatial masking.

Supported Processors:
1. FreeFuseCrossAttnSDXLAttnProcessor - Uses cross-attention weights directly
2. FreeFuseCrossAttnSelfConceptSDXLAttnProcessor - Cross-attn top-k + hidden_states inner product
3. FreeFuseCrossAttnSelfAttnSDXLAttnProcessor - Cross-attn top-k + Q-K dot product from self-attention

The SelfConcept and SelfAttn methods require passing _self_attn_cache via cross_attention_kwargs
to enable communication between attn1 (self-attention) and attn2 (cross-attention).
"""

import os
# Set GPU
import torch
from PIL import Image, ImageDraw, ImageFont


from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, AutoencoderKL
from peft.tuners.lora import Linear as LoraLinear

from src.pipeline.freefuse_sdxl_pipeline import FreeFuseSDXLPipeline, find_concept_positions_sdxl
from src.attn_processor.freefuse_sdxl_attn_processor import (
    FreeFuseSDXLAttnProcessor
)
from src.tuner.freefuse_lora_layer import FreeFuseLinear


# Processor type selection
# Default temperature differs by model type:
# - SDXL: 1000.0 (smaller hidden_states magnitude)
# - Flux: 4000.0 (larger hidden_states magnitude)
PROCESSOR_TYPES = {
    "freefuse": {
        "class": FreeFuseSDXLAttnProcessor,
        "requires_cache": True,
        "kwargs": {"top_k_ratio": 0.1, "temperature": 1000.0},
    },
}


def upgrade_lora_to_freefuse(unet):
    """
    Upgrade all LoraLinear layers in UNet to FreeFuseLinear.
    This allows spatial masking of LoRA outputs.
    """
    upgraded_count = 0
    for name, module in unet.named_modules():
        if isinstance(module, LoraLinear):
            FreeFuseLinear.init_from_lora_linear(module)
            upgraded_count += 1
    print(f"Upgraded {upgraded_count} LoRA layers to FreeFuseLinear")
    return upgraded_count


def setup_attention_processors(unet, processor_type="freefuse", cal_blocks=None):
    """
    Set up FreeFuse attention processors for the UNet.
    
    Args:
        unet: The UNet model
        processor_type: Which processor to use (see PROCESSOR_TYPES)
        cal_blocks: List of block patterns where to calculate concept similarity.
                   Default: ['mid_block'] (balances quality and speed)
                   
    Returns:
        bool: Whether this processor type requires _self_attn_cache
    """
    if cal_blocks is None:
        cal_blocks = ['mid_block']
    
    if processor_type not in PROCESSOR_TYPES:
        raise ValueError(f"Unknown processor type: {processor_type}. Available: {list(PROCESSOR_TYPES.keys())}")
    
    proc_config = PROCESSOR_TYPES[processor_type]
    processor_class = proc_config["class"]
    processor_kwargs = proc_config["kwargs"]
    requires_cache = proc_config["requires_cache"]
    
    attn_processors = {}
    cal_count = 0
    
    for name in unet.attn_processors.keys():
        proc = processor_class(**processor_kwargs)
        
        # Enable similarity calculation for specified blocks (cross-attention only)
        for block_pattern in cal_blocks:
            if block_pattern in name and 'attn2' in name:
                proc.cal_concept_sim_map = True
                cal_count += 1
                break
        
        attn_processors[name] = proc
    
    unet.set_attn_processor(attn_processors)
    print(f"Set up {len(attn_processors)} {processor_type} attention processors, {cal_count} calculating similarity")
    
    return requires_cache


def main():
    # ========== Configuration ==========
    # Model path (can be a local path or HuggingFace model ID)
    # model_path = "/media/shared_space/liuyl/FlowFuse/loras/waiIllustriousSDXL_v160.safetensors"
    model_path = "stabilityai/stable-diffusion-xl-base-1.0"
    
    lora_configs = [
        {
            "path": "loras/daiyu_lin_xl.safetensors",
            "adapter_name": "daiyu",
            "concept_text": "an Asian woman wearing Chinese traditional clothing",  # Description of what this LoRA represents
            "weight": 1.0,
        },
        {
            "path": "loras/harry_potter_xl.safetensors",  
            "adapter_name": "harry",
            "concept_text": "an European man wearing Hogwarts uniform",
            "weight": 1.0,
        },
    ]
    
    # Generation settings
    # prompt = "2 boys, shalnark is shooting conan"
    prompt = "Realistic photography, an European man wearing Hogwarts uniform hugging an Asian woman wearing Chinese traditional clothing warmly, both faces close together, autumn leaves blurred in the background."
    negative_prompt = "low quality, blurry, deformed, line art"
    height = 1024
    width = 1024
    num_inference_steps = 30
    guidance_scale = 7.0
    seed = 42
    
    # FreeFuse settings
    num_mask_collect_steps = 10  # Phase 1 steps
    # Processor type: "CrossAttn", "freefuse", or "freefuse"
    processor_type = "freefuse"
    # Best block discovered from experiments
    cal_sim_blocks = ['up_blocks.0.attentions.0.transformer_blocks.3.attn2']
    
    # Compare option: generate baseline image (no FreeFuse) for comparison
    compare = True
    
    
    # ========== Load Pipeline ==========
    print("Loading base model...")
    
    # Load from safetensors file
    if model_path.endswith('.safetensors'):
        pipe = StableDiffusionXLPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
    else:
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16"
        )
    
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    
    # ========== Load LoRAs ==========
    print("Loading LoRAs...")
    adapter_names = []
    concepts = {}
    
    for lora_config in lora_configs:
        if os.path.exists(lora_config["path"]):
            pipe.load_lora_weights(
                lora_config["path"],
                adapter_name=lora_config["adapter_name"],
            )
            adapter_names.append(lora_config["adapter_name"])
            concepts[lora_config["adapter_name"]] = lora_config["concept_text"]
            print(f"  Loaded: {lora_config['adapter_name']}")
        else:
            print(f"  Warning: LoRA not found: {lora_config['path']}")
    
    if len(adapter_names) < 2:
        print("\nError: Need at least 2 LoRAs for FreeFuse demonstration.")
        print("Please update lora_configs with valid LoRA paths.")
        return
    
    # Merge/activate all adapters
    if hasattr(pipe, 'set_adapters'):
        weights = [lora['weight'] for lora in lora_configs if lora['adapter_name'] in adapter_names]
        pipe.set_adapters(adapter_names, weights)
    
    # ========== Upgrade to FreeFuse ==========
    print(f"\nUpgrading to FreeFuse (using {processor_type} method)...")
    
    # Convert pipeline to FreeFuse
    pipe = FreeFuseSDXLPipeline.from_pipe(pipe)
    
    # Upgrade LoRA layers
    upgrade_lora_to_freefuse(pipe.unet)
    
    # Set up attention processors with selected type
    requires_cache = setup_attention_processors(
        pipe.unet, 
        processor_type=processor_type,
        cal_blocks=cal_sim_blocks
    )
    
    # ========== Find Concept Positions ==========
    print("\nFinding concept positions in prompt...")
    
    # Use tokenizer_2 if available (SDXL uses two tokenizers)
    tokenizer = getattr(pipe, 'tokenizer_2', None) or pipe.tokenizer
    
    freefuse_token_pos_maps = find_concept_positions_sdxl(
        tokenizer,
        prompt,
        concepts,
        filter_meaningless=True,
        filter_single_char=True,
    )
    
    for adapter_name, positions in freefuse_token_pos_maps.items():
        print(f"  {adapter_name}: positions {positions[0]}")
    
    # ========== Generate Image ==========
    # Output
    output_dir = f"./"
    os.makedirs(output_dir, exist_ok=True)

    image_no_freefuse = None
    if compare:
        print("\nGenerating without FreeFuse for comparison...")
        
        generator = torch.Generator("cuda").manual_seed(seed)
        
        # Reset cross_attention_kwargs for baseline
        cross_attention_kwargs_baseline = {}
        if requires_cache:
            cross_attention_kwargs_baseline["_self_attn_cache"] = {}
        
        image_no_freefuse = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            freefuse_concepts=concepts,
            freefuse_token_pos_maps=freefuse_token_pos_maps,
            num_mask_collect_steps=num_mask_collect_steps,
            use_freefuse=False,  # Disable FreeFuse
            cross_attention_kwargs=cross_attention_kwargs_baseline,
        ).images[0]
        
        output_path_no_ff = os.path.join(output_dir, "result_no_freefuse_sdxl.png")
        image_no_freefuse.save(output_path_no_ff)
        print(f"Saved comparison to: {output_path_no_ff}")

    print(f"\nGenerating image with FreeFuse ({processor_type} method)...")
    
    generator = torch.Generator("cuda").manual_seed(seed)
    
    # Prepare cross_attention_kwargs
    # For SelfConcept/SelfAttn processors, we need to pass a mutable cache dict
    cross_attention_kwargs = {}
    if requires_cache:
        cross_attention_kwargs["_self_attn_cache"] = {}
        print("  Using _self_attn_cache for attn1/attn2 communication")
    
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        # FreeFuse args
        freefuse_concepts=concepts,
        freefuse_token_pos_maps=freefuse_token_pos_maps,
        num_mask_collect_steps=num_mask_collect_steps,
        use_freefuse=True,
        # debug_dir=output_dir+"/debug",  # Save mask visualizations
        cross_attention_kwargs=cross_attention_kwargs,
    ).images[0]
    
    # Save result
    output_path = os.path.join(output_dir, "freefuse_sdxl.png")
    image.save(output_path)
    print(f"\nSaved result to: {output_path}")
    
    # ========== Generate without FreeFuse for comparison ==========
    if compare and image_no_freefuse is not None:
        # Create comparison composite image
        def create_comparison_image(before_image, after_image, 
                                    before_label="Before (No FreeFuse)", 
                                    after_label="After (FreeFuse)"):
            """创建水平拼接的对比图像，带有标签"""
            w, h = before_image.size
            label_height = 40
            composite = Image.new('RGB', (w * 2, h + label_height), color='white')
            
            composite.paste(before_image, (0, label_height))
            composite.paste(after_image, (w, label_height))
            
            draw = ImageDraw.Draw(composite)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            draw.text((w // 2, 10), before_label, fill='black', font=font, anchor='mt')
            draw.text((w + w // 2, 10), after_label, fill='black', font=font, anchor='mt')
            
            return composite
        
        composite = create_comparison_image(image_no_freefuse, image)
        composite_path = os.path.join(output_dir, "freefuse_sdxl_compare.png")
        composite.save(composite_path)
        print(f"Saved comparison composite to: {composite_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
