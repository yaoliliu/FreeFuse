"""
FreeFuse Majority Voting Utilities

Shared consensus voting algorithm used by both sampler.py and blocks_analysis.py.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


def collect_block_votes(
    block_masks: Dict[str, torch.Tensor],
    concept_names: List[str],
    target_h: int,
    target_w: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Convert per-block masks to a vote tensor where each pixel gets ONE concept ID (1-based).
    
    This is the exact algorithm from blocks_analysis.py _create_consensus_image.
    
    Args:
        block_masks: Dict mapping concept name -> binary mask tensor
        concept_names: List of all concept names (including background if applicable)
        target_h, target_w: Target spatial dimensions
        device: Device to create tensors on
        
    Returns:
        Vote tensor of shape (H, W) with dtype torch.long
        - 0 = unassigned/background
        - 1..N = concept index (1-based for bincount compatibility)
    """
    votes = torch.zeros(target_h, target_w, dtype=torch.long, device=device)
    
    for idx, concept_name in enumerate(concept_names):
        if concept_name not in block_masks:
            continue
        
        mask = block_masks[concept_name]
        
        # Ensure mask is 2D
        if mask.dim() == 3:
            mask_2d = mask[0]
        else:
            mask_2d = mask
        
        mask_2d = mask_2d.float().cpu()
        
        # Resize to target dimensions if needed
        if mask_2d.shape[0] != target_h or mask_2d.shape[1] != target_w:
            mask_2d = F.interpolate(
                mask_2d.unsqueeze(0).unsqueeze(0),
                size=(target_h, target_w),
                mode='nearest'
            ).squeeze(0).squeeze(0)
            mask_2d = mask_2d.to(device=device)
        
        # Where mask > 0.5, this concept is active - assign concept ID
        votes[mask_2d > 0.5] = idx + 1  # +1 so 0 = unassigned
    
    return votes


def majority_voting_per_pixel(
    votes_stack: torch.Tensor,
    num_concepts: int,
) -> torch.Tensor:
    """
    Perform majority voting per pixel using torch.bincount.
    
    This is the exact algorithm from blocks_analysis.py _create_consensus_image.
    
    Args:
        votes_stack: Tensor of shape (num_blocks, H, W) with concept IDs (1-based)
        num_concepts: Number of concepts (for bincount minlength)
        
    Returns:
        Tensor of shape (H, W) with dtype torch.long containing winner indices (1-based)
        - 0 = no clear winner / unassigned
        - 1..N = winning concept index
    """
    num_blocks, h, w = votes_stack.shape
    
    # Initialize result: majority winner index per pixel (1-based)
    result_class = torch.zeros(h, w, dtype=torch.long, device=votes_stack.device)
    
    # Majority voting per pixel
    for i in range(h):
        for j in range(w):
            # Get votes for this pixel from all blocks
            pixel_votes = votes_stack[:, i, j]
            valid_votes = pixel_votes[pixel_votes > 0]  # Exclude unassigned (0)
            
            if len(valid_votes) > 0:
                # Count votes for each concept using bincount
                counts = torch.bincount(valid_votes, minlength=num_concepts + 1)
                
                # Find concept with most votes (excluding index 0)
                if counts[1:].max() > 0:
                    winner_idx = counts[1:].argmax().item()
                    result_class[i, j] = winner_idx + 1  # 1-based index
    
    return result_class


def create_consensus_similarity_maps(
    all_blocks_masks: Dict[str, Dict[str, torch.Tensor]],
    concept_names: List[str],
    latent_h: int,
    latent_w: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Create similarity maps from multiple blocks using majority voting.
    
    This combines collect_block_votes and majority_voting_per_pixel into a single
    convenience function that outputs similarity maps in the format expected by
    generate_masks.
    
    Args:
        all_blocks_masks: Dict mapping block_key -> (concept_name -> mask tensor)
        concept_names: List of all concept names (including background if applicable)
        latent_h, latent_w: Latent spatial dimensions
        device: Device to create tensors on
        
    Returns:
        Dict mapping concept name -> similarity map tensor of shape (B, img_len, 1)
        Winner-take-all: winning concept gets 1.0, others get 0.0 at each pixel
    """
    # Collect votes from all blocks
    all_votes = []
    
    for block_key, block_masks in all_blocks_masks.items():
        votes = collect_block_votes(
            block_masks=block_masks,
            concept_names=concept_names,
            target_h=latent_h,
            target_w=latent_w,
            device=device,
        )
        all_votes.append(votes)
    
    if not all_votes:
        # Return empty similarity maps
        img_len = latent_h * latent_w
        return {name: torch.zeros(1, img_len, 1, device=device) for name in concept_names}
    
    # Stack all votes: (num_blocks, H, W)
    votes_tensor = torch.stack(all_votes)
    
    # Perform majority voting
    majority_indices = majority_voting_per_pixel(
        votes_stack=votes_tensor,
        num_concepts=len(concept_names),
    )
    
    # Convert winner indices to similarity maps format (B, img_len, 1)
    img_len = latent_h * latent_w
    similarity_maps = {}
    
    for idx, name in enumerate(concept_names):
        # Binary mask: 1.0 where this concept won, 0.0 elsewhere
        concept_mask = (majority_indices == idx + 1).float()
        # Reshape to (B, img_len, 1)
        sim_map = concept_mask.view(1, img_len, 1)
        similarity_maps[name] = sim_map
    
    return similarity_maps


def create_consensus_image_rgb(
    all_blocks_masks: Dict[str, Dict[str, torch.Tensor]],
    concept_names: List[str],
    preview_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Create RGB consensus image from multiple blocks using majority voting.
    
    This is the full algorithm from blocks_analysis.py _create_consensus_image,
    producing a visual RGB preview where each concept has a distinct color.
    
    Args:
        all_blocks_masks: Dict mapping block_key -> (concept_name -> mask tensor)
        concept_names: List of all concept names (including background if applicable)
        preview_size: Target preview size (square)
        device: Device to create tensors on
        
    Returns:
        RGB image tensor of shape (1, preview_size, preview_size, 3)
        - Each concept has a distinct color
        - Background/unassigned = black (0, 0, 0)
    """
    if not all_blocks_masks or not concept_names:
        return torch.zeros(1, preview_size, preview_size, 3, device=device)
    
    # Get spatial dimensions from first mask
    first_block = next(iter(all_blocks_masks.values()))
    first_mask = next(iter(first_block.values()))
    
    if first_mask.dim() == 2:
        h, w = first_mask.shape
    elif first_mask.dim() == 3:
        h, w = first_mask.shape[1], first_mask.shape[2]
    else:
        h = w = int(first_mask.numel() ** 0.5)
    
    # Colors for concepts (background = black)
    colors = [
        (255, 100, 100),  # Red
        (100, 255, 100),  # Green
        (100, 100, 255),  # Blue
        (255, 255, 100),  # Yellow
        (255, 100, 255),  # Magenta
        (100, 255, 255),  # Cyan
    ]
    
    # Collect votes from all blocks
    all_votes = []
    
    for block_key, block_masks in all_blocks_masks.items():
        votes = collect_block_votes(
            block_masks=block_masks,
            concept_names=concept_names,
            target_h=h,
            target_w=w,
            device=device,
        )
        all_votes.append(votes)
    
    if not all_votes:
        return torch.zeros(1, preview_size, preview_size, 3, device=device)
    
    # Stack all votes
    stacked = torch.stack(all_votes)  # (num_blocks, h, w)
    
    # Majority voting per pixel
    result_class = torch.zeros(h, w, dtype=torch.long, device=device)
    result_rgb = torch.zeros(3, h, w, device=device)
    
    for i in range(h):
        for j in range(w):
            votes = stacked[:, i, j]
            valid_votes = votes[votes > 0]
            
            if len(valid_votes) > 0:
                counts = torch.bincount(valid_votes, minlength=len(concept_names) + 1)
                
                if counts[1:].max() > 0:
                    winner_idx = counts[1:].argmax().item()
                    result_class[i, j] = winner_idx + 1
                    
                    # Color the pixel
                    if winner_idx < len(colors):
                        color = colors[winner_idx]
                        for c in range(3):
                            result_rgb[c, i, j] = color[c] / 255.0
    
    # Resize to preview_size
    if h != preview_size or w != preview_size:
        result_rgb = F.interpolate(
            result_rgb.unsqueeze(0),
            size=(preview_size, preview_size),
            mode='nearest'
        ).squeeze(0)
    
    result_rgb = result_rgb.permute(1, 2, 0).unsqueeze(0)
    
    return result_rgb
