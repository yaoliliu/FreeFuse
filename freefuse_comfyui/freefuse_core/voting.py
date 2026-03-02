"""
FreeFuse consensus voting helpers for multi-block similarity aggregation.
"""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


def _flatten_sim_map(sim_map: torch.Tensor, img_len: int, device: torch.device) -> Optional[torch.Tensor]:
    """Convert similarity map to shape (img_len,)."""
    if sim_map is None:
        return None

    if sim_map.dim() == 3:
        # (B, img_len, 1) or (B, 1, img_len)
        if sim_map.shape[0] > 1:
            sim_map = sim_map[:1]
        sim_map = sim_map.reshape(1, -1)
    elif sim_map.dim() == 2:
        # (B, img_len)
        if sim_map.shape[0] > 1:
            sim_map = sim_map[:1]
        sim_map = sim_map.reshape(1, -1)
    else:
        sim_map = sim_map.reshape(1, -1)

    flat = sim_map[0]
    if flat.numel() != img_len:
        return None
    return flat.to(device=device, dtype=torch.float32)


def create_consensus_similarity_maps(
    all_blocks_similarity_maps: Dict[int, Dict[str, torch.Tensor]],
    concept_names: List[str],
    latent_h: int,
    latent_w: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Majority-vote aggregation across multiple blocks.

    Returns similarity maps in FreeFuse format: (B, img_len, 1), hard one-hot.
    """
    img_len = latent_h * latent_w

    if not all_blocks_similarity_maps or not concept_names:
        return {}

    per_block_winners: List[torch.Tensor] = []
    num_concepts = len(concept_names)

    for _, block_maps in all_blocks_similarity_maps.items():
        if not block_maps:
            continue

        per_concept = []
        for name in concept_names:
            flat = _flatten_sim_map(block_maps.get(name), img_len, device)
            if flat is None:
                flat = torch.zeros(img_len, device=device, dtype=torch.float32)
            per_concept.append(flat)

        # (num_concepts, img_len)
        stacked = torch.stack(per_concept, dim=0)
        block_max, block_winner = torch.max(stacked, dim=0)
        # Mark pixels with no votes from this block.
        block_winner = torch.where(
            block_max > 0,
            block_winner.to(dtype=torch.long),
            torch.full_like(block_winner, -1, dtype=torch.long),
        )
        per_block_winners.append(block_winner)

    if not per_block_winners:
        return {}

    votes = torch.stack(per_block_winners, dim=0)  # (num_blocks, img_len)
    valid = votes >= 0
    one_hot = F.one_hot(votes.clamp(min=0), num_classes=num_concepts)
    one_hot = one_hot * valid.unsqueeze(-1)
    vote_counts = one_hot.sum(dim=0)  # (img_len, num_concepts)

    winner = vote_counts.argmax(dim=-1)
    has_vote = vote_counts.sum(dim=-1) > 0
    winner = torch.where(has_vote, winner, torch.full_like(winner, -1))

    similarity_maps: Dict[str, torch.Tensor] = {}
    for idx, name in enumerate(concept_names):
        mask = (winner == idx).to(dtype=torch.float32)
        similarity_maps[name] = mask.view(1, img_len, 1)

    return similarity_maps


__all__ = ["create_consensus_similarity_maps"]

