"""
FreeFuse Raw Similarity Overlay

Shows all raw similarity maps overlaid in one image.
Also outputs separate grayscale images for each concept.
Uses perceptuele mapping om kleine verschillen zichtbaar te maken.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Optional, List, Tuple
import math

# Importeer de echte mask_utils functies
from ..freefuse_core.mask_utils import generate_masks, stabilized_balanced_argmax


class FreeFuseRawSimilarityOverlay:
    """
    Combine all raw similarity maps into one overlay image.
    Also outputs separate grayscale images for each concept.
    Shows WHERE each concept's attention is located in the composition.
    Uses perceptuele mapping to reveal tiny differences in attention.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "raw_similarity": ("FREEFUSE_MASKS", {
                    "tooltip": "Raw similarity maps from FreeFusePhase1Sampler"
                }),
                "freefuse_data": ("FREEFUSE_DATA", {
                    "tooltip": "Freefuse data with concept names"
                }),
            },
            "optional": {
                "latent": ("LATENT", {
                    "tooltip": "Optional latent to determine output aspect ratio"
                }),
                "preview_size": ("INT", {
                    "default": 1024, "min": 512, "max": 2048,
                    "tooltip": "Longest side of the output preview (maintains aspect ratio)"
                }),
                "sensitivity": ("FLOAT", {
                    "default": 5.0, "min": 1.0, "max": 20.0, "step": 0.5,
                    "tooltip": "How much to amplify small differences (higher = more contrast)"
                }),
                "show_background": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Include background in overlay and separate images"
                }),
                "show_legend": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show color legend on overlay"
                }),
                
                # ===== ARGMAX ALGORITHM PARAMETERS =====
                "argmax_method": (["simple", "stabilized"], {
                    "default": "stabilized",
                    "tooltip": "simple = direct argmax, stabilized = balanced with spatial constraints"
                }),
                
                # Core balancing parameters
                "max_iter": ("INT", {
                    "default": 15, "min": 1, "max": 50, "step": 1,
                    "tooltip": "Number of iterations for stabilized argmax (more = more balanced)"
                }),
                "balance_lr": ("FLOAT", {
                    "default": 0.01, "min": 0.001, "max": 0.1, "step": 0.001,
                    "tooltip": "Learning rate for bias updates (higher = faster balancing)"
                }),
                
                # Spatial coherence parameters
                "gravity_weight": ("FLOAT", {
                    "default": 0.00004, "min": 0.0, "max": 0.001, "step": 0.00001,
                    "tooltip": "How strongly pixels are pulled to concept centroid (higher = more compact)"
                }),
                "spatial_weight": ("FLOAT", {
                    "default": 0.00004, "min": 0.0, "max": 0.001, "step": 0.00001,
                    "tooltip": "How much neighboring pixels influence assignment (higher = smoother)"
                }),
                "momentum": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 0.9, "step": 0.05,
                    "tooltip": "Smoothing between iterations (higher = more stable but slower)"
                }),
                
                # Spatial geometry parameters
                "anisotropy": ("FLOAT", {
                    "default": 1.3, "min": 0.5, "max": 2.0, "step": 0.1,
                    "tooltip": "Horizontal stretch factor (important for non-square latents!)"
                }),
                "centroid_margin": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.5, "step": 0.05,
                    "tooltip": "Keep centroids away from borders (higher = more margin)"
                }),
                "border_penalty": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Penalty for assigning pixels near borders"
                }),
                
                # Background and cleanup
                "bg_scale": ("FLOAT", {
                    "default": 0.95, "min": 0.5, "max": 2.0, "step": 0.05,
                    "tooltip": "Background channel multiplier (higher = more background)"
                }),
                "use_morphological_cleaning": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Clean masks with morphological operations"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING", "FREEFUSE_MASKS")
    RETURN_NAMES = ("overlay", "concept_1", "concept_2", "concept_3", "concept_4", "argmax_winner", "info", "refined_masks")
    FUNCTION = "visualize"
    CATEGORY = "FreeFuse/Research"

    DESCRIPTION = """Combines all raw similarity maps into one overlay image.
Also outputs separate grayscale images for each concept (max 4) and an argmax winner map.

The ARGMAX WINNER shows which concept "wins" at each pixel (hard assignment).
Each pixel gets the color of the concept with highest similarity after balancing.
Perfect for seeing segmentation boundaries and concept dominance.

NEW: Outputs REFINED_MASKS (FREEFUSE_MASKS) that can be used by FreeFuseMaskApplicator.
This allows you to fine-tune similarity maps with sensitivity, top_k, argmax parameters,
then use the refined maps for generation.

Connect to FreeFuseMaskRefiner for hole-filling and cleanup before MaskApplicator.

Uses perceptuele mapping to reveal tiny differences in attention.
Shows WHERE each concept's attention is located in the composition."""
    
    def visualize(self, 
                  raw_similarity, 
                  freefuse_data, 
                  latent=None,
                  preview_size=1024,
                  sensitivity=5.0, 
                  show_background=False, 
                  show_legend=True,
                  
                  # Argmax parameters
                  argmax_method="stabilized",
                  max_iter=15,
                  balance_lr=0.01,
                  gravity_weight=0.00004,
                  spatial_weight=0.00004,
                  momentum=0.2,
                  anisotropy=1.3,
                  centroid_margin=0.0,
                  border_penalty=0.0,
                  bg_scale=0.95,
                  use_morphological_cleaning=True):
        
        sim_maps = raw_similarity.get("similarity_maps", {})
        concepts = freefuse_data.get("concepts", {})
        
        if not sim_maps:
            empty = torch.zeros(1, preview_size, preview_size, 3)
            return (empty, empty, empty, empty, empty, empty, "No similarity maps found")
        
        # Bepaal output dimensies met behoud van aspect ratio
        if latent is not None and "samples" in latent:
            # Gebruik latent voor aspect ratio
            latent_tensor = latent["samples"]
            
            # 🔥 FIX: Handle 5D latents (LTX-Video: B, C, T, H, W)
            if latent_tensor.dim() == 5:
                # LTX-Video: (B, C, T, H, W)
                latent_h = latent_tensor.shape[3]  # H dimension
                latent_w = latent_tensor.shape[4]  # W dimension
                print(f"[FreeFuse Raw Similarity Overlay] LTX-Video 5D latent detected: {latent_tensor.shape}")
                print(f"  Using spatial dimensions: H={latent_h}, W={latent_w}")
            elif latent_tensor.dim() == 4:
                # Standard: (B, C, H, W)
                latent_h = latent_tensor.shape[2]
                latent_w = latent_tensor.shape[3]
                print(f"[FreeFuse Raw Similarity Overlay] Standard 4D latent detected: {latent_tensor.shape}")
            else:
                print(f"[FreeFuse Raw Similarity Overlay] Unexpected latent shape: {latent_tensor.shape}, assuming square")
                latent_h = latent_w = preview_size

            # Schaal naar output size (behoud aspect ratio)
            if latent_h >= latent_w:
                out_h = preview_size
                out_w = int(preview_size * latent_w / latent_h)
            else:
                out_w = preview_size
                out_h = int(preview_size * latent_h / latent_w)
            
            print(f"[FreeFuse Raw Similarity Overlay] Output size: {out_w}x{out_h} (aspect ratio preserved from {latent_w}x{latent_h})")
        else:
            # Geen latent: gebruik preview_size voor beide (vierkant)
            out_h = out_w = preview_size
            print(f"[FreeFuse Raw Similarity Overlay] No latent provided, using square: {out_w}x{out_h}")
        
        # Filter background if needed
        concept_maps = []
        concept_names = []
        
        for name, sim in sim_maps.items():
            if name == "__background__" and not show_background:
                continue
            concept_maps.append(sim)
            concept_names.append(name)
        
        if not concept_maps:
            empty = torch.zeros(1, out_h, out_w, 3)
            return (empty, empty, empty, empty, empty, empty, "No concepts to display")

        # Filter out non-tensor entries (metadata, etc.)
        concept_maps = [m for m in concept_maps if isinstance(m, torch.Tensor)]
        
        if not concept_maps:
            print(f"[FreeFuse Raw Similarity Grid] No valid similarity maps found")
            empty = torch.zeros(1, out_h, out_w, 3)
            return (empty, empty, empty, empty, empty, empty, "No concepts to display")

        # Find spatial dimensions
        first_map = concept_maps[0]
        if first_map.dim() == 3:
            N = first_map.shape[1]
            h = w = int(N ** 0.5)
            if h * w != N:
                for i in range(int(N ** 0.5), 0, -1):
                    if N % i == 0:
                        h = i
                        w = N // i
                        break
        else:
            h, w = first_map.shape[-2:]
        
        # Bepaal device van eerste map (voor consistentie)
        device = concept_maps[0].device
        
        # Colors for overlay and argmax
        colors = [
            (1.0, 0.0, 0.0),  # Rood
            (0.0, 1.0, 0.0),  # Groen
            (0.0, 0.0, 1.0),  # Blauw
            (1.0, 1.0, 0.0),  # Geel
            (1.0, 0.0, 1.0),  # Magenta
            (0.0, 1.0, 1.0),  # Cyaan
            (1.0, 0.5, 0.0),  # Oranje
            (0.5, 0.0, 1.0),  # Paars
            (1.0, 0.0, 0.5),  # Roze
            (0.5, 1.0, 0.0),  # Lichtgroen
            (0.3, 0.3, 0.3),  # Donkergrijs (background)
        ]
        
        # Maak leeg canvas voor overlay op hetzelfde device
        overlay = torch.zeros(3, out_h, out_w, device=device)
        
        # Prepare separate concept images (max 4) op hetzelfde device
        concept_images = []
        for _ in range(4):
            concept_images.append(torch.zeros(3, out_h, out_w, device=device))
        
        # Prepare argmax winner image op hetzelfde device
        argmax_winner = torch.zeros(3, out_h, out_w, device=device)
        
        info_lines = []
        info_lines.append("=" * 60)
        info_lines.append("RAW SIMILARITY OVERLAY + PER CONCEPT + ARGMAX WINNER")
        info_lines.append("=" * 60)
        info_lines.append(f"Sensitivity: {sensitivity}x")
        info_lines.append(f"Output size: {out_w}x{out_h}")
        info_lines.append(f"Argmax method: {argmax_method}")
        if argmax_method == "stabilized":
            info_lines.append(f"  Iterations: {max_iter}")
            info_lines.append(f"  Balance LR: {balance_lr}")
            info_lines.append(f"  Gravity: {gravity_weight}")
            info_lines.append(f"  Spatial: {spatial_weight}")
            info_lines.append(f"  Momentum: {momentum}")
            info_lines.append(f"  Anisotropy: {anisotropy}")
            info_lines.append(f"  Border penalty: {border_penalty}")
            info_lines.append(f"  BG scale: {bg_scale}")
            info_lines.append(f"  Morph cleaning: {use_morphological_cleaning}")
        info_lines.append("-" * 40)
        
        # Vind globale min/max voor alle concepten samen
        all_mins = []
        all_maxs = []
        
        for name, sim in zip(concept_names, concept_maps):
            # Verplaats naar CPU voor min/max berekening
            if sim.dim() == 3:
                sim_2d = sim[0, :, 0].float().cpu()
            else:
                sim_2d = sim.float().cpu()
            
            all_mins.append(sim_2d.min().item())
            all_maxs.append(sim_2d.max().item())
        
        global_min = min(all_mins)
        global_max = max(all_maxs)
        global_range = global_max - global_min
        
        info_lines.append(f"Global range: {global_min:.6f} - {global_max:.6f} (Δ={global_range:.6f})")
        info_lines.append("-" * 40)
        
        # Prepare tensors voor overlay en per-concept beelden
        concept_tensors_resized = []
        
        # Verwerk elke concept voor overlay en per-concept beelden
        for idx, (name, sim) in enumerate(zip(concept_names, concept_maps)):
            color = colors[idx % len(colors)]
            
            # Haal 2D versie en verplaats naar device
            if sim.dim() == 3:
                sim_2d = sim[0, :, 0].view(h, w).float().to(device)
            else:
                sim_2d = sim.float().to(device)
            
            # 🔥 PERCEPTUELE MAPPING voor kleine verschillen
            sim_min = sim_2d.min()
            sim_max = sim_2d.max()
            sim_range = sim_max - sim_min  # 👈 DEFINIEER sim_range HIER!
            
            # Eerst lineair naar 0-1
            if sim_max > sim_min:
                sim_norm = (sim_2d - sim_min) / (sim_max - sim_min)
            else:
                sim_norm = torch.ones_like(sim_2d) * 0.5
            
            # Perceptuele mapping: versterk kleine verschillen
            sim_vis = torch.sigmoid((sim_norm - 0.5) * sensitivity)
            
            # Optioneel: voeg een kleine offset toe zodat 0 niet helemaal zwart is
            sim_vis = sim_vis * 0.8 + 0.1  # Spreid over 0.1 - 0.9
            
            # Resize naar output size (behoud aspect ratio)
            sim_resized = F.interpolate(
                sim_vis.unsqueeze(0).unsqueeze(0),
                size=(out_h, out_w),
                mode='bilinear'
            ).squeeze(0).squeeze(0)
            
            # Store voor argmax (de genormaliseerde versie)
            concept_tensors_resized.append(sim_resized)
            
            # Voeg toe aan overlay met kleur
            for c in range(3):
                overlay[c] += sim_resized * color[c]
            
            # Maak aparte grijswaarden afbeelding voor deze concept
            if idx < 4:
                for c in range(3):
                    concept_images[idx][c] = sim_resized
            
            # Info voor deze concept
            concept_text = concepts.get(name, name)
            if len(concept_text) > 40:
                concept_text = concept_text[:37] + "..."
            info_lines.append(f"{idx+1}. {name}: {concept_text}")
            info_lines.append(f"   range={sim_min:.6f}-{sim_max:.6f} (Δ={sim_range:.6f})")
        
        # === ARGMAX WINNER BEREKENING MET ECHTE ALGORITME ===
        info_lines.append("-" * 40)
        info_lines.append(f"ARGMAX WINNER: {argmax_method} method")
        
        if argmax_method == "stabilized" and len(concept_tensors_resized) > 1:
            try:
                # Gebruik de echte stabilized_balanced_argmax
                # Converteer naar (1, C, N) formaat
                C = len(concept_tensors_resized)
                N = out_h * out_w
                
                # Stack en reshape naar (1, C, N)
                stacked = torch.stack(concept_tensors_resized, dim=0).unsqueeze(0)  # (1, C, H, W)
                logits = stacked.view(1, C, N)  # (1, C, N)
                
                # Voer stabilized_balanced_argmax uit
                max_indices = stabilized_balanced_argmax(
                    logits, out_h, out_w,
                    max_iter=max_iter,
                    lr=balance_lr,
                    gravity_weight=gravity_weight,
                    spatial_weight=spatial_weight,
                    momentum=momentum,
                    centroid_margin=centroid_margin,
                    border_penalty=border_penalty,
                    anisotropy=anisotropy,
                    debug=False,
                )  # (1, N)
                
                # Reshape terug naar (H, W)
                winner_indices = max_indices[0].view(out_h, out_w)
                
                info_lines.append(f"  Stabilized with {max_iter} iterations")
                info_lines.append(f"  gravity={gravity_weight}, spatial={spatial_weight}")
                
            except Exception as e:
                # Fallback naar simple argmax bij error
                info_lines.append(f"  WARNING: Stabilized failed: {e}")
                info_lines.append(f"  Falling back to simple argmax")
                
                stacked = torch.stack(concept_tensors_resized, dim=0)
                winner_indices = torch.argmax(stacked, dim=0)
        else:
            # Simple argmax
            if len(concept_tensors_resized) > 0:
                stacked = torch.stack(concept_tensors_resized, dim=0)
                winner_indices = torch.argmax(stacked, dim=0)
                info_lines.append(f"  Simple argmax (no balancing)")
            else:
                winner_indices = torch.zeros(out_h, out_w, dtype=torch.long, device=device)
        
        # Kleur de winner indices
        if 'winner_indices' in locals():
            for idx, color in enumerate(colors[:len(concept_names)]):
                mask = (winner_indices == idx).float()
                for c in range(3):
                    argmax_winner[c] += mask * color[c]
        
        # Clamp alles
        argmax_winner = argmax_winner.clamp(0, 1)
        overlay = overlay.clamp(0, 1)
        
        # 🔥 ALLES NAAR CPU VOOR PIL VERWERKING
        overlay_cpu = overlay.cpu()
        argmax_cpu = argmax_winner.cpu()
        
        # Converteer overlay naar PIL voor legend
        overlay_np = (overlay_cpu.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        overlay_pil = Image.fromarray(overlay_np)
        
        # Converteer argmax winner naar PIL voor eventuele legend
        argmax_np = (argmax_cpu.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        argmax_pil = Image.fromarray(argmax_np)
        
        if show_legend:
            # Add legend to overlay
            draw = ImageDraw.Draw(overlay_pil)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Voeg legenda toe in hoek
            y_pos = 20
            for idx, name in enumerate(concept_names[:8]):  # Max 8 in legend
                color = colors[idx % len(colors)]
                rgb_color = tuple(int(c * 255) for c in color)
                
                concept_text = concepts.get(name, name)
                if len(concept_text) > 25:
                    concept_text = concept_text[:22] + "..."
                
                draw.rectangle([20, y_pos, 40, y_pos+20], fill=rgb_color)
                draw.text((50, y_pos), f"{concept_text}", fill='white', font=font)
                y_pos += 25
            
            # Also add legend to argmax winner image
            draw_argmax = ImageDraw.Draw(argmax_pil)
            y_pos = 20
            
            for idx, name in enumerate(concept_names[:8]):
                color = colors[idx % len(colors)]
                rgb_color = tuple(int(c * 255) for c in color)
                
                concept_text = concepts.get(name, name)
                if len(concept_text) > 25:
                    concept_text = concept_text[:22] + "..."
                
                draw_argmax.rectangle([20, y_pos, 40, y_pos+20], fill=rgb_color)
                draw_argmax.text((50, y_pos), f"{concept_text}", fill='white', font=font)
                y_pos += 25
            
            # Add title with method
            method_text = f"ARGMAX: {argmax_method}"
            if argmax_method == "stabilized":
                method_text += f" (iter={max_iter}, g={gravity_weight:.5f})"
            draw_argmax.text((20, y_pos + 10), method_text, fill='yellow', font=font)
        
        # Terug naar tensor voor overlay (weer op CPU voor ComfyUI)
        overlay_tensor = torch.from_numpy(np.array(overlay_pil).astype(np.float32) / 255.0)[None,]
        
        # Terug naar tensor voor argmax winner
        argmax_tensor = torch.from_numpy(np.array(argmax_pil).astype(np.float32) / 255.0)[None,]
        
        # Converteer aparte concept beelden naar tensors
        concept_tensors = []
        for img in concept_images:
            # Clamp en converteer (eerst naar CPU)
            img_cpu = img.cpu().clamp(0, 1)
            img_np = (img_cpu.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            
            # Voeg label toe aan aparte beelden
            if show_legend:
                pil_img = Image.fromarray(img_np)
                draw = ImageDraw.Draw(pil_img)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
                except:
                    font = ImageFont.load_default()
                draw.text((10, 10), "Grayscale", fill='red', font=font)
                img_np = np.array(pil_img)
            
            concept_tensors.append(torch.from_numpy(img_np.astype(np.float32) / 255.0)[None,])
        
        info_lines.append("=" * 60)
        info = "\n".join(info_lines)

        # Vul ontbrekende concept tensors met zeros
        while len(concept_tensors) < 4:
            concept_tensors.append(torch.zeros(1, out_h, out_w, 3))

        # === CREATE REFINED SIMILARITY MAPS OUTPUT ===
        # This allows the refined maps to be used by FreeFuseMaskApplicator
        refined_masks = {}
        refined_similarity = {}
        
        for name, sim in zip(concept_names, concept_maps):
            # Store original similarity maps (refined by sensitivity/parameters)
            refined_similarity[name] = sim
            
        # Also store the argmax masks
        for idx, name in enumerate(concept_names):
            if 'winner_indices' in locals():
                mask = (winner_indices == idx).float()
                refined_masks[name] = mask
        
        # Add background if present
        if "__background__" in sim_maps:
            refined_similarity["__background__"] = sim_maps["__background__"]
        
        refined_output = {
            "masks": refined_masks,
            "similarity_maps": refined_similarity,
            "token_pos_maps": freefuse_data.get("token_pos_maps", {}),
        }

        return (overlay_tensor,
                concept_tensors[0],
                concept_tensors[1],
                concept_tensors[2],
                concept_tensors[3],
                argmax_tensor,
                info,
                refined_output)


# Export node mappings
NODE_CLASS_MAPPINGS = {
    "FreeFuseRawSimilarityOverlay": FreeFuseRawSimilarityOverlay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeFuseRawSimilarityOverlay": "🔬 FreeFuse Raw Similarity Overlay",
}
