#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FreeFuse SDXL Block Comparison Script - CrossAttn Method

This script tests the CrossAttn method for mask extraction and compares
results across different blocks to determine which block(s) produce the 
best concept separation using attention weights directly.

Key difference from test_block_comparison.py:
- Uses FreeFuseSDXLAttnProcessor instead of FreeFuseSDXLAttnProcessor
- Extracts attention maps directly from cross-attention weights (softmax(Q @ K.T))
"""

import os
# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from PIL import Image
import shutil

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)) # Add project root to path

from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from diffusers import AutoencoderKL
from peft.tuners.lora import Linear as LoraLinear



from src.pipeline.freefuse_sdxl_pipeline import FreeFuseSDXLPipeline, find_concept_positions_sdxl
from src.attn_processor.freefuse_sdxl_attn_processor import FreeFuseSDXLAttnProcessor
from src.tuner.freefuse_lora_layer import FreeFuseLinear


def upgrade_lora_to_freefuse(unet):
    """Upgrade all LoraLinear layers in UNet to FreeFuseLinear."""
    upgraded_count = 0
    for name, module in unet.named_modules():
        if isinstance(module, LoraLinear):
            FreeFuseLinear.init_from_lora_linear(module)
            upgraded_count += 1
    return upgraded_count


def setup_attention_processors(unet, cal_blocks):
    """Set up FreeFuse CrossAttn attention processors for specified blocks."""
    attn_processors = {}
    cal_count = 0
    
    for name in unet.attn_processors.keys():
        proc = FreeFuseSDXLAttnProcessor()
        
        # Enable similarity calculation for specified blocks (cross-attention only)
        for block_pattern in cal_blocks:
            if block_pattern in name and 'attn2' in name:
                proc.cal_concept_sim_map = True
                cal_count += 1
                break
        
        attn_processors[name] = proc
    
    unet.set_attn_processor(attn_processors)
    return cal_count


def main():
    # ========== Configuration ==========
    model_path = "/media/shared_space/liuyl/FlowFuse/loras/waiIllustriousSDXL_v160.safetensors"
    # model_path = "stabilityai/stable-diffusion-xl-base-1.0"
    
    lora_configs = [
        {
            "path": "loras/daiyu_lin_xl.safetensors",
            "adapter_name": "shalnark",
            "concept_text": "a boy with brown hair",
            "weight": 0.0,
        },
        {
            "path": "loras/harry_potter_xl.safetensors",  
            "adapter_name": "conan",
            "concept_text": "a girl with white hair",
            "weight": 0.0,
        },
    ]
    
    # prompt = "2 boys, shalnark is shooting conan"
    # prompt = "a boy and a girl"
    prompt = "a boy with brown hair and a girl with white hair"
    negative_prompt = "low quality, blurry, deformed, line art"
    height = 1024
    width = 1024
    num_inference_steps = 30
    guidance_scale = 7.0
    seed = 77
    num_mask_collect_steps = 10
    
    # Will be auto-discovered from UNet
    block_configs = None  # Placeholder, will be set after model loading
    
    output_base_dir = "outputs/sdxl_block_comparison"
    os.makedirs(output_base_dir, exist_ok=True)
    
    # ========== Load Pipeline ==========
    print("Loading base model...")
    
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
    
    # ========== Auto-discover all cross-attention (attn2) blocks ==========
    print("\nDiscovering cross-attention blocks...")
    all_attn2_blocks = []
    for name in pipe.unet.attn_processors.keys():
        if 'attn2' in name:
            # Extract the block pattern up to transformer_blocks.X.attn2
            pattern = name.replace('.processor', '')
            all_attn2_blocks.append(pattern)
    
    all_attn2_blocks = sorted(set(all_attn2_blocks))
    print(f"Found {len(all_attn2_blocks)} individual cross-attention blocks:")
    
    # Group by block type for display
    block_groups = {}
    for block in all_attn2_blocks:
        parts = block.split('.')
        if 'mid_block' in block:
            group = 'mid_block'
        else:
            group = f"{parts[0]}.{parts[1]}"
        if group not in block_groups:
            block_groups[group] = []
        block_groups[group].append(block)
    
    for group, blocks in block_groups.items():
        print(f"  {group}: {len(blocks)} blocks")
    
    # Create block configs - test each attn2 individually
    block_configs = [[block] for block in all_attn2_blocks]
    print(f"\nTotal configurations to test: {len(block_configs)}")
    
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
    
    if len(adapter_names) < 2:
        print("Error: Need at least 2 LoRAs")
        return
    
    if hasattr(pipe, 'set_adapters'):
        weights = [lora['weight'] for lora in lora_configs if lora['adapter_name'] in adapter_names]
        pipe.set_adapters(adapter_names, weights)
    
    # ========== Upgrade to FreeFuse ==========
    print("\nUpgrading to FreeFuse (CrossAttn method)...")
    pipe = FreeFuseSDXLPipeline.from_pipe(pipe)
    upgrade_lora_to_freefuse(pipe.unet)
    
    # Find concept positions (shared across all tests)
    tokenizer = getattr(pipe, 'tokenizer_2', None) or pipe.tokenizer
    freefuse_token_pos_maps = find_concept_positions_sdxl(
        tokenizer, prompt, concepts,
        filter_meaningless=True, filter_single_char=True,
    )
    print(f"\nToken positions:")
    for adapter_name, positions in freefuse_token_pos_maps.items():
        print(f"  {adapter_name}: {positions[0][:5]}... ({len(positions[0])} tokens)")
    
    # ========== Test Each Block Configuration ==========
    print(f"\n{'='*60}")
    print("Testing block configurations with CrossAttn method...")
    print(f"{'='*60}")
    
    results = []
    
    for i, cal_blocks in enumerate(block_configs):
        block_name = "_".join(cal_blocks).replace(".", "_")
        output_dir = os.path.join(output_base_dir, block_name)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n[{i+1}/{len(block_configs)}] Testing: {cal_blocks}")
        
        # Reset attention processors for this test
        cal_count = setup_attention_processors(pipe.unet, cal_blocks)
        print(f"  Configured {cal_count} CrossAttn attention processors for similarity calculation")
        
        # Generate with FreeFuse
        generator = torch.Generator("cuda").manual_seed(seed)
        
        try:
            image = pipe(
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
                use_freefuse=True,
                debug_dir=output_dir,
            ).images[0]
            
            image.save(os.path.join(output_dir, "result.png"))
            print(f"  ✓ Saved to: {output_dir}")
            results.append((block_name, "success", output_dir))
            
        except Exception as e:
            raise
            print(f"  ✗ Error: {e}")
            results.append((block_name, f"error: {e}", output_dir))
    
    # ========== Create Comprehensive Comparison Grid ==========
    print(f"\n{'='*60}")
    print("Creating comprehensive comparison grid (sim_maps + masks + results)...")
    
    from PIL import ImageDraw, ImageFont
    import glob
    
    # Collect all successful results with their sim maps and masks
    comparison_data = []
    concept_names = list(concepts.keys())  # e.g., ['shalnark', 'conan']

    def open_first_matching_image(path_patterns):
        """Open the first existing image from a list of path patterns."""
        for pattern in path_patterns:
            if '*' in pattern:
                matches = sorted(glob.glob(pattern))
                if matches:
                    # Use the last one for step-based patterns (e.g., step09 > step00)
                    return Image.open(matches[-1])
            elif os.path.exists(pattern):
                return Image.open(pattern)
        return None
    
    for block_name, status, output_dir in results:
        if status == "success":
            result_path = os.path.join(output_dir, "result.png")
            if os.path.exists(result_path):
                result_img = Image.open(result_path)
                
                # Find sim-map and mask images for each concept
                mask_images = {}
                sim_images = {}
                for concept_name in concept_names:
                    sim_img = open_first_matching_image([
                        os.path.join(output_dir, f"phase1_final_sim_{concept_name}.png"),
                        os.path.join(output_dir, f"final_sim_{concept_name}.png"),
                        os.path.join(output_dir, f"phase*_final_sim_{concept_name}.png"),
                        os.path.join(output_dir, f"phase1_step*_sim_{concept_name}.png"),
                        os.path.join(output_dir, f"sim_{concept_name}.png"),
                    ])
                    if sim_img is not None:
                        sim_images[concept_name] = sim_img

                    # Look for mask files like "mask_shalnark.png" or similar
                    mask_img = open_first_matching_image([
                        os.path.join(output_dir, f"mask_{concept_name}.png"),
                        os.path.join(output_dir, f"{concept_name}_mask.png"),
                        os.path.join(output_dir, f"mask_{concept_name}_*.png"),
                    ])
                    if mask_img is not None:
                        mask_images[concept_name] = mask_img
                
                comparison_data.append({
                    'block_name': block_name,
                    'result_img': result_img,
                    'sim_images': sim_images,
                    'mask_images': mask_images,
                })
    
    if comparison_data:
        # Calculate grid dimensions
        num_concepts = len(concept_names)
        num_blocks = len(comparison_data)
        
        thumb_size = 256  # Smaller to fit more
        label_width = 450  # Width for block name label (enough for long names)
        padding = 5
        header_height = 40  # Height for column headers
        row_height = thumb_size + padding
        
        # Number of columns: label + sim_maps + masks + result
        num_image_cols = 2 * num_concepts + 1
        total_width = label_width + num_image_cols * (thumb_size + padding) + padding
        total_height = header_height + num_blocks * row_height + padding
        
        grid_img = Image.new('RGB', (total_width, total_height), 'white')
        draw = ImageDraw.Draw(grid_img)
        
        # Try to load a font, fallback to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
            header_font = font
        
        # Draw column headers
        x_offset = label_width + padding
        draw.text((padding, 10), "Block Name", fill='black', font=header_font)
        for i, concept_name in enumerate(concept_names):
            x = x_offset + i * (thumb_size + padding) + thumb_size // 4
            draw.text((x, 10), f"SimMap: {concept_name}", fill='black', font=header_font)
        mask_offset = x_offset + num_concepts * (thumb_size + padding)
        for i, concept_name in enumerate(concept_names):
            x = mask_offset + i * (thumb_size + padding) + thumb_size // 4
            draw.text((x, 10), f"Mask: {concept_name}", fill='black', font=header_font)
        result_x = x_offset + (2 * num_concepts) * (thumb_size + padding) + thumb_size // 4
        draw.text((result_x, 10), "Result", fill='black', font=header_font)
        
        # Draw horizontal line under header
        draw.line([(0, header_height - 5), (total_width, header_height - 5)], fill='gray', width=2)
        
        # Draw each block row
        for row_idx, data in enumerate(comparison_data):
            y = header_height + row_idx * row_height
            
            # Draw block name (truncate if too long)
            block_label = data['block_name'].replace("_", ".")
            # No need to truncate with larger label_width
            if len(block_label) > 60:
                block_label = "..." + block_label[-57:]
            draw.text((padding, y + thumb_size // 2 - 10), block_label, fill='black', font=font)
            
            # Draw vertical separator
            draw.line([(label_width, y), (label_width, y + thumb_size)], fill='lightgray', width=1)
            
            # Draw sim-map images
            x = label_width + padding
            for concept_name in concept_names:
                if concept_name in data['sim_images']:
                    sim_img = data['sim_images'][concept_name]
                    sim_resized = sim_img.resize((thumb_size, thumb_size), Image.LANCZOS)
                    if sim_resized.mode != 'RGB':
                        sim_resized = sim_resized.convert('RGB')
                    grid_img.paste(sim_resized, (x, y))
                else:
                    # Draw placeholder for missing sim-map
                    draw.rectangle([(x, y), (x + thumb_size, y + thumb_size)], outline='lightgray', fill='#f0f0f0')
                    draw.text((x + thumb_size // 5, y + thumb_size // 2), "No SimMap", fill='gray', font=font)
                x += thumb_size + padding

            # Draw mask images
            for concept_name in concept_names:
                if concept_name in data['mask_images']:
                    mask_img = data['mask_images'][concept_name]
                    mask_resized = mask_img.resize((thumb_size, thumb_size), Image.LANCZOS)
                    # Convert to RGB if necessary (masks might be grayscale)
                    if mask_resized.mode != 'RGB':
                        mask_resized = mask_resized.convert('RGB')
                    grid_img.paste(mask_resized, (x, y))
                else:
                    # Draw placeholder for missing mask
                    draw.rectangle([(x, y), (x + thumb_size, y + thumb_size)], outline='lightgray', fill='#f0f0f0')
                    draw.text((x + thumb_size // 4, y + thumb_size // 2), "No Mask", fill='gray', font=font)
                x += thumb_size + padding
            
            # Draw result image
            result_resized = data['result_img'].resize((thumb_size, thumb_size), Image.LANCZOS)
            if result_resized.mode != 'RGB':
                result_resized = result_resized.convert('RGB')
            grid_img.paste(result_resized, (x, y))
            
            # Draw horizontal line separator (except for last row)
            if row_idx < num_blocks - 1:
                line_y = y + thumb_size + padding // 2
                draw.line([(0, line_y), (total_width, line_y)], fill='#e0e0e0', width=1)
        
        # Save the comprehensive comparison grid
        grid_path = os.path.join(output_base_dir, "comprehensive_comparison.png")
        grid_img.save(grid_path, quality=95)
        print(f"Saved comprehensive comparison grid to: {grid_path}")
        print(f"  Grid size: {total_width} x {total_height} pixels")
        print(f"  Showing {num_blocks} blocks x ({num_concepts} sim_maps + {num_concepts} masks + 1 result)")
    
    # Also create the original simple comparison grid (results only)
    print("\nCreating simple comparison grid (results only)...")
    comparison_images = [(d['block_name'], d['result_img']) for d in comparison_data]
    
    if comparison_images:
        # Create a side-by-side comparison image
        num_images = len(comparison_images)
        grid_cols = min(5, num_images)  # More columns for better layout
        grid_rows = (num_images + grid_cols - 1) // grid_cols
        
        thumb_size = 256
        padding = 10
        label_height = 30
        
        grid_width = grid_cols * thumb_size + (grid_cols + 1) * padding
        grid_height = grid_rows * (thumb_size + label_height) + (grid_rows + 1) * padding
        
        grid_img = Image.new('RGB', (grid_width, grid_height), 'white')
        draw = ImageDraw.Draw(grid_img)
        
        for idx, (block_name, img) in enumerate(comparison_images):
            row = idx // grid_cols
            col = idx % grid_cols
            
            x = padding + col * (thumb_size + padding)
            y = padding + row * (thumb_size + label_height + padding)
            
            # Resize image
            img_resized = img.resize((thumb_size, thumb_size), Image.LANCZOS)
            grid_img.paste(img_resized, (x, y))
            
            # Draw label (truncate if needed)
            label = block_name.replace("_", ".")
            if len(label) > 35:
                label = "..." + label[-32:]
            draw.text((x + 5, y + thumb_size + 5), label, fill='black', font=font if 'font' in dir() else None)
        
        grid_path = os.path.join(output_base_dir, "comparison_grid.png")
        grid_img.save(grid_path)
        print(f"Saved simple comparison grid to: {grid_path}")
    
    # ========== Summary ==========
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    for block_name, status, output_dir in results:
        status_icon = "✓" if status == "success" else "✗"
        print(f"  {status_icon} {block_name}: {status}")
    
    print(f"\nAll results saved to: {output_base_dir}")
    print("Compare sim-map / mask quality and final image quality in each folder.")
    print("Done!")


if __name__ == "__main__":
    main()
