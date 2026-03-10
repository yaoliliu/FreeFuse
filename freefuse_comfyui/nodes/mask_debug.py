"""
FreeFuse Mask Debug Nodes

Debug utilities for visualizing FreeFuse masks.
Includes bank grid view and detailed bank inspection.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Optional, List, Tuple


def ensure_cpu(tensor):
    """Ensure tensor is on CPU for numpy/PIL operations."""
    if tensor is None:
        return None
    if isinstance(tensor, torch.Tensor):
        # Forceer naar CPU, detach, maar behoud device info
        return tensor.detach().cpu()
    return tensor


class FreeFuseMaskBankDebug:
    """
    Debug node to preview all mask banks (0-56) from FreeFuse masks.
    Shows which banks are active and visualizes each mask.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("FREEFUSE_MASKS",),
                "freefuse_data": ("FREEFUSE_DATA",),
            },
            "optional": {
                "bank_start": ("INT", {"default": 0, "min": 0, "max": 56, "step": 1}),
                "bank_end": ("INT", {"default": 56, "min": 0, "max": 56, "step": 1}),
                "preview_size": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "show_similarity": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "DICT")
    RETURN_NAMES = ("bank_grid", "bank_info", "bank_data")
    FUNCTION = "debug_banks"
    CATEGORY = "FreeFuse/Debug"
    
    def debug_banks(self, masks, freefuse_data, bank_start=0, bank_end=56, 
                    preview_size=512, show_similarity=False):
        import torch
        import torch.nn.functional as F
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        
        mask_dict = masks.get("masks", {})
        sim_maps = masks.get("similarity_maps", {})
        token_pos_maps = freefuse_data.get("token_pos_maps", {})
        concepts = freefuse_data.get("concepts", {})
        
        # Create a grid of mask previews
        banks_to_show = min(bank_end - bank_start + 1, 57)  # Max 57 banks (0-56)
        grid_cols = 8  # 8 columns for nice display
        grid_rows = (banks_to_show + grid_cols - 1) // grid_cols
        
        # Each cell size
        cell_size = preview_size // 4
        
        # Create canvas
        canvas_width = grid_cols * cell_size
        canvas_height = grid_rows * cell_size
        canvas = Image.new('RGB', (canvas_width, canvas_height), color='black')
        draw = ImageDraw.Draw(canvas)
        
        # Try to load font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        bank_info_lines = []
        bank_data_dict = {}
        active_banks = []
        
        # For each bank in range
        for idx, bank_idx in enumerate(range(bank_start, bank_end + 1)):
            row = idx // grid_cols
            col = idx % grid_cols
            x = col * cell_size
            y = row * cell_size
            
            # Create cell
            cell = Image.new('RGB', (cell_size, cell_size), color='black')
            cell_draw = ImageDraw.Draw(cell)
            
            # Check if this bank index corresponds to a mask
            # Banks are indexed by adapter order
            adapter_names = list(mask_dict.keys())
            
            if bank_idx < len(adapter_names):
                adapter_name = adapter_names[bank_idx]
                mask = mask_dict[adapter_name]
                
                active_banks.append(bank_idx)
                
                # 👉 ALTIJD NAAR CPU VOOR OPSLAG
                mask_cpu = ensure_cpu(mask)
                bank_data_dict[f"bank_{bank_idx}"] = {
                    "adapter": adapter_name,
                    "concept": concepts.get(adapter_name, ""),
                    "mask": mask_cpu,
                }
                
                # Get concept text
                concept_text = concepts.get(adapter_name, "")
                if len(concept_text) > 30:
                    concept_text = concept_text[:27] + "..."
                
                # 👉 ALTIJD NAAR CPU VOOR VERWERKING
                mask_vis = ensure_cpu(mask)
                if mask_vis.dim() == 3:
                    mask_vis = mask_vis[0]  # Take first channel
                
                mask_vis = mask_vis.float()
                
                # Resize to cell size if needed
                if mask_vis.shape[0] != cell_size or mask_vis.shape[1] != cell_size:
                    mask_tensor = F.interpolate(
                        mask_vis.unsqueeze(0).unsqueeze(0),
                        size=(cell_size, cell_size),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).squeeze(0)
                else:
                    mask_tensor = mask_vis
                
                # 🔴 FIX: gebruik .cpu().numpy() in plaats van .numpy()
                mask_np = mask_tensor.cpu().numpy()
                mask_np = (mask_np * 255).clip(0, 255).astype(np.uint8)
                mask_img = Image.fromarray(mask_np, mode='L')
                
                # Colorize based on adapter index
                colors = [
                    (255, 100, 100),   # Red
                    (100, 255, 100),   # Green
                    (100, 100, 255),   # Blue
                    (255, 255, 100),   # Yellow
                    (255, 100, 255),   # Magenta
                    (100, 255, 255),   # Cyan
                ]
                color = colors[bank_idx % len(colors)]
                
                # Create colored mask
                colored = Image.new('RGB', mask_img.size, color=color)
                mask_img_rgb = Image.composite(colored, Image.new('RGB', mask_img.size, 'black'), mask_img)
                cell.paste(mask_img_rgb, (0, 0))
                
                # Add border
                cell_draw.rectangle([0, 0, cell_size-1, cell_size-1], outline=color, width=2)
                
                # Add text
                cell_draw.text((5, 5), f"B{bank_idx}: {adapter_name[:10]}", fill='white', font=font)
                cell_draw.text((5, 25), concept_text[:15], fill='white', font=font)
                
                # Add token positions if available
                if adapter_name in token_pos_maps:
                    positions_list = token_pos_maps[adapter_name]
                    if positions_list and positions_list[0]:
                        pos_str = str(positions_list[0])[:20]
                        cell_draw.text((5, 45), f"tokens:{pos_str}", fill='yellow', font=font)
                
                # Add to info
                bank_info_lines.append(f"Bank {bank_idx:02d}: {adapter_name} - {concept_text}")
                
                # Also show similarity map if requested
                if show_similarity and adapter_name in sim_maps:
                    sim = sim_maps[adapter_name]
                    
                    # 👉 ALTIJD NAAR CPU
                    sim_vis = ensure_cpu(sim)
                    if sim_vis.dim() == 3:
                        sim_vis = sim_vis[0]
                    
                    sim_vis = sim_vis.float()
                    
                    sim_resized = F.interpolate(
                        sim_vis.unsqueeze(0).unsqueeze(0),
                        size=(cell_size//2, cell_size//2),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).squeeze(0)
                    
                    # 🔴 FIX: gebruik .cpu().numpy() in plaats van .numpy()
                    sim_np = sim_resized.cpu().numpy()
                    sim_np = (sim_np * 255).clip(0, 255).astype(np.uint8)
                    sim_img = Image.fromarray(sim_np, mode='L')
                    
                    # Paste similarity in corner
                    cell.paste(sim_img, (cell_size - cell_size//2 - 5, cell_size - cell_size//2 - 5))
                    cell_draw.rectangle(
                        [cell_size - cell_size//2 - 5, cell_size - cell_size//2 - 5, cell_size-5, cell_size-5],
                        outline='cyan', width=1
                    )
            else:
                # Empty bank
                cell_draw.rectangle([0, 0, cell_size-1, cell_size-1], outline='gray', width=1)
                cell_draw.text((5, 5), f"B{bank_idx}: EMPTY", fill='gray', font=font)
                bank_info_lines.append(f"Bank {bank_idx:02d}: EMPTY")
            
            # Paste cell into canvas
            canvas.paste(cell, (x, y))
        
        # Add grid lines
        for i in range(1, grid_cols):
            line_x = i * cell_size
            draw.line([(line_x, 0), (line_x, canvas_height)], fill='gray', width=1)
        
        for i in range(1, grid_rows):
            line_y = i * cell_size
            draw.line([(0, line_y), (canvas_width, line_y)], fill='gray', width=1)
        
        # Add title
        draw.text((10, 10), f"FreeFuse Mask Banks {bank_start}-{bank_end}", fill='white', font=font)
        
        # Convert to tensor (altijd op CPU)
        img_array = np.array(canvas).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]  # Blijft op CPU
        
        # Build info string
        info_header = [
            "=" * 60,
            f"FREEFUSE MASK BANKS {bank_start}-{bank_end}",
            "=" * 60,
            f"Total Adapters: {len(mask_dict)}",
            f"Active Banks: {len(active_banks)}",
            f"Banks with data: {active_banks}",
            "-" * 40,
        ]
        info = "\n".join(info_header + bank_info_lines + ["=" * 60])
        
        return (img_tensor, info, bank_data_dict)


class FreeFuseBankInspector:
    """
    Inspect a specific mask bank in detail.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("FREEFUSE_MASKS",),
                "freefuse_data": ("FREEFUSE_DATA",),
                "bank_index": ("INT", {"default": 0, "min": 0, "max": 56, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE", "FREEFUSE_MASKS")
    RETURN_NAMES = ("mask_preview", "bank_details", "similarity_preview", "bank_mask")
    FUNCTION = "inspect_bank"
    CATEGORY = "FreeFuse/Debug"
    
    def inspect_bank(self, masks, freefuse_data, bank_index=0):
        import torch
        import torch.nn.functional as F
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        
        mask_dict = masks.get("masks", {})
        sim_maps = masks.get("similarity_maps", {})
        token_pos_maps = freefuse_data.get("token_pos_maps", {})
        concepts = freefuse_data.get("concepts", {})
        
        details = []
        details.append("=" * 50)
        details.append(f"BANK {bank_index} INSPECTION")
        details.append("=" * 50)
        
        # Find adapter at this bank index
        adapter_names = list(mask_dict.keys())
        mask_tensor = None
        sim_tensor = None
        adapter_name = None
        
        if bank_index < len(adapter_names):
            adapter_name = adapter_names[bank_index]
            mask_tensor = mask_dict[adapter_name]
            sim_tensor = sim_maps.get(adapter_name) if sim_maps else None
            
            concept_text = concepts.get(adapter_name, "")
            
            details.append(f"Adapter: {adapter_name}")
            details.append(f"Concept: {concept_text}")
            details.append(f"Mask Shape: {tuple(mask_tensor.shape)}")
            details.append(f"Mask Type: {mask_tensor.dtype}")
            
            # Move to CPU for statistics
            mask_cpu = mask_tensor.cpu()
            details.append(f"Mask Min: {mask_cpu.min().item():.4f}")
            details.append(f"Mask Max: {mask_cpu.max().item():.4f}")
            details.append(f"Mask Mean: {mask_cpu.mean().item():.4f}")
            
            # Token positions
            if adapter_name in token_pos_maps:
                positions_list = token_pos_maps[adapter_name]
                if positions_list and len(positions_list) > 0 and positions_list[0]:
                    details.append(f"Token Positions: {positions_list[0]}")
            
            if sim_tensor is not None:
                details.append(f"Similarity Shape: {tuple(sim_tensor.shape)}")
                sim_cpu = sim_tensor.cpu()
                details.append(f"Similarity Min: {sim_cpu.min().item():.6f}")
                details.append(f"Similarity Max: {sim_cpu.max().item():.6f}")
        else:
            details.append(f"No adapter found at bank {bank_index}")
            details.append(f"Total adapters: {len(adapter_names)}")
            details.append(f"Available banks: 0-{len(adapter_names)-1}")
        
        details.append("-" * 40)
        details.append("All Adapters:")
        for i, name in enumerate(adapter_names):
            marker = ">" if i == bank_index else " "
            details.append(f"{marker} Bank {i:02d}: {name}")
        
        # Create mask preview
        if mask_tensor is not None:
            # Move to CPU for processing
            mask_vis = mask_tensor.cpu().float()
            if mask_vis.dim() == 3:
                mask_vis = mask_vis[0]  # Take first channel
            
            # Resize to 512x512 for preview
            mask_resized = F.interpolate(
                mask_vis.unsqueeze(0).unsqueeze(0),
                size=(512, 512),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)
            
            mask_np = mask_resized.numpy()
            mask_np = (mask_np * 255).clip(0, 255).astype(np.uint8)
            mask_img = Image.fromarray(mask_np, mode='L').convert('RGB')
            
            # Add info text
            draw = ImageDraw.Draw(mask_img)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            draw.text((10, 10), f"Bank {bank_index}: {adapter_name}", fill='red', font=font)
            
            mask_array = np.array(mask_img).astype(np.float32) / 255.0
            mask_preview = torch.from_numpy(mask_array)[None,]
        else:
            # Empty preview
            empty = Image.new('RGB', (512, 512), color='black')
            draw = ImageDraw.Draw(empty)
            draw.text((256, 256), f"Bank {bank_index}\nNo Mask", fill='red', anchor='mm')
            mask_array = np.array(empty).astype(np.float32) / 255.0
            mask_preview = torch.from_numpy(mask_array)[None,]
        
        # Create similarity preview - FIXED VERSION
        if sim_tensor is not None:
            # Move to CPU and convert to float32
            sim_vis = sim_tensor.cpu().float()
            
            # Debug print
            print(f"[DEBUG] Similarity tensor shape: {sim_vis.shape}")
            
            # Handle different tensor shapes
            if sim_vis.dim() == 3 and sim_vis.shape[2] == 1:
                # Shape is [1, seq_len, 1] - remove last dimension
                sim_vis = sim_vis.squeeze(-1)  # Now [1, seq_len]
                
                # Try to determine spatial dimensions
                # For Flux, the sequence length is latent_h * latent_w
                # Default to 32x32 = 1024 if we can't determine
                seq_len = sim_vis.shape[1]
                
                # Try to get latent size from freefuse_data
                latent_h = 32  # Default
                latent_w = 32  # Default
                
                # Check if we have latent size info
                if "latent_h" in freefuse_data and "latent_w" in freefuse_data:
                    latent_h = freefuse_data.get("latent_h", 32)
                    latent_w = freefuse_data.get("latent_w", 32)
                
                # Calculate expected sequence length
                expected_len = latent_h * latent_w
                
                if seq_len == expected_len:
                    # Perfect match - reshape to spatial
                    sim_vis = sim_vis.reshape(latent_h, latent_w)
                    print(f"[DEBUG] Reshaped to {latent_h}x{latent_w}")
                else:
                    # Try to find a square that matches
                    side = int(seq_len ** 0.5)
                    if side * side == seq_len:
                        sim_vis = sim_vis.reshape(side, side)
                        print(f"[DEBUG] Reshaped to square {side}x{side}")
                    else:
                        # Take first N pixels that form a square
                        square_size = min(32, int((seq_len ** 0.5)))
                        sim_vis = sim_vis[0, :square_size*square_size].reshape(square_size, square_size)
                        print(f"[DEBUG] Cropped to {square_size}x{square_size}")
            
            # Normalize for better visibility
            sim_min = sim_vis.min()
            sim_max = sim_vis.max()
            
            if sim_max > sim_min:
                # Normalize to 0-1 range
                sim_vis = (sim_vis - sim_min) / (sim_max - sim_min)
                # Apply gamma correction to enhance contrast
                sim_vis = torch.pow(sim_vis, 0.7)
            else:
                # If all values are the same, create a gray image
                sim_vis = torch.ones_like(sim_vis) * 0.5
            
            print(f"[DEBUG] After normalization - min: {sim_vis.min():.4f}, max: {sim_vis.max():.4f}")
            
            # Resize to 512x512 for preview
            sim_resized = F.interpolate(
                sim_vis.unsqueeze(0).unsqueeze(0),
                size=(512, 512),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)
            
            sim_np = sim_resized.numpy()
            sim_np = (sim_np * 255).clip(0, 255).astype(np.uint8)
            sim_img = Image.fromarray(sim_np, mode='L').convert('RGB')
            
            draw = ImageDraw.Draw(sim_img)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            draw.text((10, 10), f"Similarity {bank_index}", fill='cyan', font=font)
            
            sim_array = np.array(sim_img).astype(np.float32) / 255.0
            sim_preview = torch.from_numpy(sim_array)[None,]
        else:
            empty = Image.new('RGB', (512, 512), color='black')
            draw = ImageDraw.Draw(empty)
            draw.text((256, 256), f"Bank {bank_index}\nNo Similarity", fill='gray', anchor='mm')
            sim_array = np.array(empty).astype(np.float32) / 255.0
            sim_preview = torch.from_numpy(sim_array)[None,]
        
        # Extract just this bank's mask
        if mask_tensor is not None:
            bank_mask_data = {
                "masks": {f"bank_{bank_index}": mask_tensor.cpu()},
                "similarity_maps": {f"bank_{bank_index}": sim_tensor.cpu()} if sim_tensor is not None else {},
            }
        else:
            bank_mask_data = {"masks": {}, "similarity_maps": {}}
        
        return (mask_preview, "\n".join(details), sim_preview, bank_mask_data)


# Export node mappings for debug nodes
NODE_CLASS_MAPPINGS = {
    "FreeFuseMaskBankDebug": FreeFuseMaskBankDebug,
    "FreeFuseBankInspector": FreeFuseBankInspector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeFuseMaskBankDebug": "FreeFuse Mask Bank Debug",
    "FreeFuseBankInspector": "FreeFuse Bank Inspector",
}
