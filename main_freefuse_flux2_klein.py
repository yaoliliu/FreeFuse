#!/usr/bin/env python3
"""
FreeFuse for Flux2 Klein - Main Entry Script

This script demonstrates multi-concept image generation with FreeFuse on Flux2 Klein.
It supports:
- Loading multiple LoRAs
- Concept similarity map extraction
- Per-concept spatial masking
- Attention bias for cross-LoRA suppression

Usage:
    python main_freefuse_flux2_klein.py \
        --model_path "black-forest-labs/FLUX.2-klein-4B" \
        --lora_paths "path/to/lora1" "path/to/lora2" \
        --lora_names "concept1" "concept2" \
        --concept_tokens "<sks1>" "<tok2>" \
        --prompt "A photo of <sks1> and <tok2> together in a garden" \
        --output_path "output.png"
"""

import argparse
import os
import sys
from typing import List, Optional

import torch
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.freefuse_flux2_klein_pipeline import FreeFuseFlux2KleinPipeline
from src.tuner.freefuse_lora_layer import convert_peft_lora_to_freefuse_lora


def parse_args():
    parser = argparse.ArgumentParser(description="FreeFuse for Flux2 Klein")
    
    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default="black-forest-labs/FLUX.2-klein-4B",
        help="Path to Flux2 Klein model (4B or 9B)",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype",
    )
    
    # LoRA arguments
    parser.add_argument(
        "--lora_paths",
        type=str,
        nargs="+",
        default=[],
        help="Paths to LoRA weights",
    )
    parser.add_argument(
        "--lora_names",
        type=str,
        nargs="+",
        default=[],
        help="Names for each LoRA adapter",
    )
    parser.add_argument(
        "--concept_tokens",
        type=str,
        nargs="+",
        default=[],
        help="Concept tokens to track (e.g., '<sks1>', '<tok2>')",
    )
    parser.add_argument(
        "--lora_weights",
        type=float,
        nargs="+",
        default=[],
        help="Weight for each LoRA (default: 1.0 for all)",
    )
    
    # Generation arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default="A photo of <sks1> and <tok2> together",
        help="Generation prompt",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Output image height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Output image width",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    # FreeFuse arguments
    parser.add_argument(
        "--sim_map_extraction_step",
        type=int,
        default=4,
        help="Denoising step at which to extract similarity maps (0-indexed)",
    )
    parser.add_argument(
        "--sim_map_extraction_block",
        type=str,
        default=None,
        help="Block to extract sim maps from (e.g., 'transformer_blocks.4'). None = auto",
    )
    parser.add_argument(
        "--top_k_ratio",
        type=float,
        default=0.1,
        help="Ratio of top-k image tokens for concept attention",
    )
    parser.add_argument(
        "--exclude_background",
        action="store_true",
        default=True,
        help="Exclude background from concept masks",
    )
    parser.add_argument(
        "--suppress_strength",
        type=float,
        default=-1e4,
        help="Attention bias strength for cross-LoRA suppression",
    )
    
    # Output arguments
    parser.add_argument(
        "--output_path",
        type=str,
        default="freefuse_flux2_output.png",
        help="Output image path",
    )
    parser.add_argument(
        "--save_sim_maps",
        action="store_true",
        help="Save concept similarity maps as debug images",
    )
    parser.add_argument(
        "--debug_dir",
        type=str,
        default="debug_output",
        help="Directory for debug outputs",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        default=True,
        help="Generate comparison image showing before/after FreeFuse",
    )
    
    return parser.parse_args()

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


def main():
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Parse dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.torch_dtype]
    
    print(f"Loading Flux2 Klein from {args.model_path}...")
    
    # Load pipeline
    pipe = FreeFuseFlux2KleinPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
    )
    pipe.to(device)
    
    # Load LoRAs
    if args.lora_paths:
        assert len(args.lora_paths) == len(args.lora_names), \
            "Number of LoRA paths must match number of LoRA names"
        
        if args.lora_weights:
            assert len(args.lora_weights) == len(args.lora_paths), \
                "Number of LoRA weights must match number of LoRA paths"
        else:
            args.lora_weights = [1.0] * len(args.lora_paths)
        
        print(f"Loading {len(args.lora_paths)} LoRAs...")
        for lora_path, lora_name in zip(args.lora_paths, args.lora_names):
            print(f"  Loading {lora_name} from {lora_path}")
            pipe.load_lora_weights(lora_path, adapter_name=lora_name)
        
        # Set adapter weights
        pipe.set_adapters(args.lora_names, adapter_weights=args.lora_weights)
        
        # Convert to FreeFuseLinear for masked application
        print("Converting LoRA layers to FreeFuseLinear...")
        pipe.convert_lora_layers(args.lora_names)
    
    # Setup FreeFuse attention processors
    print("Setting up FreeFuse attention processors...")
    pipe.setup_freefuse_attention_processors()
    
    # Find concept token positions
    if args.concept_tokens:
        print("Finding concept token positions...")
        token_positions = pipe.find_concept_token_positions(args.prompt, args.concept_tokens)
        
        # Build freefuse_token_pos_maps: lora_name -> [[positions], ...]
        freefuse_token_pos_maps = {}
        for lora_name, concept_tok in zip(args.lora_names, args.concept_tokens):
            positions = token_positions.get(concept_tok, [])
            freefuse_token_pos_maps[lora_name] = [positions]  # Wrap in list for batch support
            print(f"  {lora_name} ({concept_tok}): positions {positions}")
        
        # Find EOS token for background
        eos_idx = pipe.find_eos_token_index(args.prompt)
        print(f"  EOS token index: {eos_idx}")
    else:
        freefuse_token_pos_maps = None
        eos_idx = None
    
    # Set random seed
    # Set random seed
    generator = torch.Generator(device=device).manual_seed(args.seed)
    
    # Generate comparison if enabled
    image_no_freefuse = None
    if args.compare:
        print("\nGenerating without FreeFuse for comparison...")
        
        # Reset generator with same seed
        generator = torch.Generator(device=device).manual_seed(args.seed)
        
        # Ensure FreeFuse state is clear
        pipe.transformer.clear_freefuse_state()
        pipe.transformer.enable_concept_sim_map_extraction(None)
        
        # Run baseline generation
        output_no_freefuse = pipe(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        )
        
        image_no_freefuse = output_no_freefuse.images[0]
        no_ff_path = args.output_path.replace(".png", "_no_freefuse.png")
        image_no_freefuse.save(no_ff_path)
        print(f"Baseline image saved to {no_ff_path}")

    # Enable sim map extraction on specified block for FreeFuse
    if freefuse_token_pos_maps:
        pipe.transformer.enable_concept_sim_map_extraction(args.sim_map_extraction_block)
        pipe.transformer.set_freefuse_token_pos_maps(freefuse_token_pos_maps)
        pipe.transformer.set_freefuse_background_info(eos_token_index=eos_idx)
    
    print(f"\nGenerating image with prompt: {args.prompt}")
    print(f"  Size: {args.width}x{args.height}")
    print(f"  Steps: {args.num_inference_steps}, Guidance: {args.guidance_scale}")
    
    # Setup generator again for FreeFuse
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Phase 1: Generate with sim map extraction
    if freefuse_token_pos_maps:
        print("\n[Phase 1] Generating with sim map extraction...")
        
        # Create a callback to capture sim maps at the extraction step
        captured_sim_maps = {}
        
        def extract_sim_maps_callback(pipeline, step_index, timestep, callback_kwargs):
            if step_index == args.sim_map_extraction_step:
                sim_maps = pipeline.transformer.get_concept_sim_maps()
                if sim_maps:
                    captured_sim_maps.update(sim_maps)
                    print(f"  Captured sim maps at step {step_index}: {list(sim_maps.keys())}")
            return callback_kwargs
        
        # Run phase 1
        output_phase1 = pipe(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            callback_on_step_end=extract_sim_maps_callback,
        )
        
        # Build masks from sim maps
        if captured_sim_maps and len(args.lora_names) > 0:
            print("\n[Phase 1.5] Building spatial masks from sim maps...")
            
            # Calculate latent dimensions
            lat_height = args.height // (pipe.vae_scale_factor * 2)
            lat_width = args.width // (pipe.vae_scale_factor * 2)
            
            lora_masks = pipe.sim_maps_to_masks(
                captured_sim_maps,
                height=lat_height,
                width=lat_width,
                exclude_background=args.exclude_background,
            )
            
            # Save debug sim maps if requested
            if args.save_sim_maps:
                os.makedirs(args.debug_dir, exist_ok=True)
                for lora_name, mask in lora_masks.items():
                    mask_np = mask[0, 0].cpu().numpy()
                    mask_img = Image.fromarray((mask_np * 255).astype("uint8"))
                    mask_path = os.path.join(args.debug_dir, f"mask_{lora_name}.png")
                    mask_img.save(mask_path)
                    print(f"  Saved mask to {mask_path}")
            
            # Build attention bias
            txt_len = pipe.tokenizer_max_length
            img_len = lat_height * lat_width
            
            attention_bias = pipe.build_attention_bias(
                lora_masks,
                freefuse_token_pos_maps,
                txt_len=txt_len,
                img_len=img_len,
                suppress_strength=args.suppress_strength,
            )
            
            # Set masks on transformer
            pipe.transformer.set_freefuse_masks(lora_masks)
            pipe.transformer.set_freefuse_attention_bias(attention_bias)
            
            print("\n[Phase 2] Generating with FreeFuse masks applied...")
            
            # Reset generator
            generator = torch.Generator(device=device).manual_seed(args.seed)
            
            # Disable sim map extraction for phase 2
            pipe.transformer.enable_concept_sim_map_extraction(None)
            
            # Run phase 2
            output = pipe(
                prompt=args.prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            )
        else:
            print("  No sim maps captured, using phase 1 output")
            output = output_phase1
    else:
        # No FreeFuse, just generate normally
        output = pipe(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        )
    
    # Save output
    image = output.images[0]
    image.save(args.output_path)
    print(f"\nSaved output to {args.output_path}")
    
    if args.compare and image_no_freefuse is not None:
        # Create comparison composite
        composite = create_comparison_image(image_no_freefuse, image)
        compare_path = args.output_path.replace(".png", "_compare.png")
        composite.save(compare_path)
        print(f"Comparison image saved to {compare_path}")
    
    # Cleanup
    pipe.transformer.clear_freefuse_state()
    
    print("Done!")


if __name__ == "__main__":
    main()
