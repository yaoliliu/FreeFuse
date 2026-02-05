"""
FreeFuse Attention Hooks for ComfyUI

Hook into transformer attention to collect similarity maps.
Compatible with both SD/SDXL (BasicTransformerBlock) and Flux (DoubleStreamBlock).

Flux uses joint attention where img and txt are concatenated:
- txt_img_key = [encoder_key, img_key] with RoPE applied
- txt_img_query = [encoder_query, img_query] with RoPE applied

FreeFuse similarity map computation (same as reference implementation):
1. Extract concept keys from encoder_key at token positions
2. Cross-attention top-k: select top-k image tokens with highest attention to concept
3. Concept attention: compute hidden state inner product between core tokens and all tokens
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List


class FreeFuseAttentionCollector:
    """
    Collects attention patterns during Phase 1 sampling.
    
    For SDXL: Uses attn2 patch (cross-attention)
    For Flux: Uses double_block_replace to intercept attention computation
    """
    
    def __init__(self, state: Dict[str, Any], collect_blocks: list = None):
        """
        Args:
            state: Shared state dict for storing collected data
            collect_blocks: Which blocks to collect from (default: [18] for Flux)
        """
        self.state = state
        self.collect_blocks = collect_blocks or [18]
        self.collected_attention = {}
        
    def create_attn_output_patch(self, model_type: str = "flux"):
        """
        Create an attention output patch function.
        
        Returns:
            Patch function compatible with ComfyUI's attn1_output_patch
        """
        state = self.state
        collect_blocks = self.collect_blocks
        
        def attn_output_patch(out: torch.Tensor, extra_options: Dict) -> torch.Tensor:
            """
            Called after attention computation.
            
            Args:
                out: Attention output (B, seq_len, dim)
                extra_options: Contains block info, n_heads, etc.
            """
            # Check if we should collect
            phase = state.get("phase", "generate")
            if phase != "collect":
                return out
            
            current_step = state.get("current_step", 0)
            collect_step = state.get("collect_step", 4)
            
            if current_step != collect_step:
                return out
            
            # Get block info
            block_index = extra_options.get("block_index", -1)
            
            if block_index not in collect_blocks:
                return out
            
            # Store the attention output for later analysis
            if "attention_outputs" not in state:
                state["attention_outputs"] = {}
            
            state["attention_outputs"][block_index] = out.detach().clone()
            
            print(f"[FreeFuse] Collected attention at block {block_index}, "
                  f"step {current_step}, shape {out.shape}")
            
            return out
        
        return attn_output_patch
    
    def create_attn2_patch(self):
        """
        Create cross-attention patch for SDXL.
        
        This is called BEFORE attention, allowing us to modify inputs
        or just observe them.
        """
        state = self.state
        
        def attn2_patch(n: torch.Tensor, context: torch.Tensor, 
                        value: torch.Tensor, extra_options: Dict
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Cross-attention patch for SDXL.
            
            n: query (image features)
            context: key (text features)  
            value: value (text features)
            """
            # Store for later use
            phase = state.get("phase", "generate")
            if phase == "collect":
                current_step = state.get("current_step", 0)
                collect_step = state.get("collect_step", 4)
                
                if current_step == collect_step:
                    block_index = extra_options.get("block", (0, 0))
                    if "sdxl_cross_attn" not in state:
                        state["sdxl_cross_attn"] = {}
                    
                    state["sdxl_cross_attn"][str(block_index)] = {
                        "query": n.detach().clone(),
                        "key": context.detach().clone(),
                        "value": value.detach().clone(),
                    }
            
            return n, context, value
        
        return attn2_patch


class FreeFuseFluxDoubleBlockPatch:
    """
    Patch for Flux's DoubleStreamBlock.
    
    Flux uses joint attention between image and text streams,
    which requires special handling to extract the attention internals.
    """
    
    def __init__(self, state: Dict[str, Any], collect_block: int = 18):
        self.state = state
        self.collect_block = collect_block
        
    def create_double_block_patch(self):
        """
        Create a patch for Flux double blocks.
        
        The patch is called with (img, txt, extra_options) after
        the double block forward pass.
        """
        state = self.state
        collect_block = self.collect_block
        
        def double_block_patch(img: torch.Tensor, txt: torch.Tensor, 
                               extra_options: Dict
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Patch for Flux DoubleStreamBlock output.
            """
            phase = state.get("phase", "generate")
            if phase != "collect":
                return img, txt
            
            current_step = state.get("current_step", 0)
            collect_step = state.get("collect_step", 4)
            
            if current_step != collect_step:
                return img, txt
            
            block_index = extra_options.get("block_index", -1)
            
            if block_index != collect_block:
                return img, txt
            
            # Store for similarity map computation
            if "flux_outputs" not in state:
                state["flux_outputs"] = {}
            
            state["flux_outputs"]["img"] = img.detach().clone()
            state["flux_outputs"]["txt"] = txt.detach().clone()
            
            print(f"[FreeFuse] Collected Flux double block at {block_index}, "
                  f"img: {img.shape}, txt: {txt.shape}")
            
            return img, txt
        
        return double_block_patch


def compute_flux_similarity_maps(
    img_hidden_states: torch.Tensor,
    txt_hidden_states: torch.Tensor,
    token_pos_maps: Dict[str, List[List[int]]],
    top_k_ratio: float = 0.3,
    temperature: float = 1000.0,
) -> Dict[str, torch.Tensor]:
    """
    Compute similarity maps for Flux model using the FreeFuse algorithm.
    
    Algorithm (matches reference implementation):
    1. For each concept, extract concept embeddings at token positions
    2. Cross-attention top-k: compute attention scores between img and concept tokens,
       select top-k image tokens with highest scores
    3. Concept attention: compute inner product between core image tokens and all image tokens
    4. Normalize with softmax to get final sim map
    
    Args:
        img_hidden_states: Image hidden states after attention (B, img_len, dim)
        txt_hidden_states: Text hidden states after attention (B, txt_len, dim)
        token_pos_maps: Dict mapping concept name to list of token positions
        top_k_ratio: Ratio of image tokens to select as "core" (default 0.3)
        temperature: Temperature for softmax normalization (default 4000)
        
    Returns:
        Dict mapping concept name to similarity map (B, img_len, 1)
    """
    concept_sim_maps = {}
    
    if not token_pos_maps:
        return concept_sim_maps
    
    B, img_len, dim = img_hidden_states.shape
    _, txt_len, _ = txt_hidden_states.shape
    device = img_hidden_states.device
    
    # Normalize for cross-attention scoring
    img_norm = F.normalize(img_hidden_states, dim=-1)
    txt_norm = F.normalize(txt_hidden_states, dim=-1)
    
    scale = 1.0 / 1000.0  # Same scale as reference
    
    # First pass: compute all cross-attention scores (for competitive exclusion)
    all_cross_attn_scores = {}
    for lora_name, positions_list in token_pos_maps.items():
        if lora_name.startswith("__"):  # Skip special keys like __background__
            continue
            
        pos = positions_list[0] if positions_list else []
        if not pos:
            continue
        
        # Extract concept embeddings
        pos_tensor = torch.tensor(pos, device=device, dtype=torch.long)
        pos_tensor = pos_tensor.clamp(0, txt_len - 1)  # Safety clamp
        concept_embeds = txt_norm[:, pos_tensor, :]  # (B, concept_len, dim)
        
        # Cross-attention: img_hidden_states @ concept_embeds^T
        # (B, img_len, dim) @ (B, dim, concept_len) -> (B, img_len, concept_len)
        cross_attn = torch.bmm(img_norm, concept_embeds.transpose(-1, -2)) * scale
        cross_attn = F.softmax(cross_attn, dim=1)  # Softmax over img_len
        
        # Average over concept tokens to get per-image-token scores
        cross_attn_scores = cross_attn.mean(dim=-1)  # (B, img_len)
        all_cross_attn_scores[lora_name] = cross_attn_scores
    
    # Second pass: compute similarity maps with competitive exclusion
    for lora_name, positions_list in token_pos_maps.items():
        if lora_name.startswith("__"):  # Handle background separately
            continue
            
        pos = positions_list[0] if positions_list else []
        if not pos:
            continue
        
        if lora_name not in all_cross_attn_scores:
            continue
        
        # Competitive exclusion: enhance this concept's score relative to others
        cross_attn_scores = all_cross_attn_scores[lora_name] * len(all_cross_attn_scores)
        for other_name, other_scores in all_cross_attn_scores.items():
            if other_name != lora_name:
                cross_attn_scores = cross_attn_scores - other_scores
        
        # Select top-k image tokens
        k = max(1, int(img_len * top_k_ratio))
        _, top_k_indices = torch.topk(cross_attn_scores, k, dim=-1)  # (B, k)
        
        # Extract core image tokens
        top_k_indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, dim)
        core_image_tokens = torch.gather(img_hidden_states, dim=1, index=top_k_indices_expanded)  # (B, k, dim)
        
        # Concept attention: core tokens inner product with all tokens
        # (B, k, dim) @ (B, dim, img_len) -> (B, k, img_len)
        self_modal_sim = torch.bmm(core_image_tokens, img_hidden_states.transpose(-1, -2))
        
        # Average over core tokens
        self_modal_sim_avg = self_modal_sim.mean(dim=1, keepdim=True).transpose(1, 2)  # (B, img_len, 1)
        
        # Normalize with softmax
        concept_sim_map = F.softmax(self_modal_sim_avg / temperature, dim=1)
        concept_sim_maps[lora_name] = concept_sim_map
    
    # Handle background
    bg_key = None
    bg_positions = None
    
    if "__background__" in token_pos_maps:
        bg_key = "__background__"
        bg_positions = token_pos_maps["__background__"][0]
    elif "__bg__" in token_pos_maps:
        bg_key = "__bg__"
        bg_positions = token_pos_maps["__bg__"][0]
    
    if bg_positions:
        pos_tensor = torch.tensor(bg_positions, device=device, dtype=torch.long)
        pos_tensor = pos_tensor.clamp(0, txt_len - 1)
        bg_embeds = txt_norm[:, pos_tensor, :]
        
        # Cross-attention for background
        bg_cross_attn = torch.bmm(img_norm, bg_embeds.transpose(-1, -2)) * scale
        bg_cross_attn = F.softmax(bg_cross_attn, dim=1)
        bg_cross_attn_scores = bg_cross_attn.mean(dim=-1)
        
        # Select top-k
        bg_k = max(1, int(img_len * top_k_ratio))
        _, bg_top_k_indices = torch.topk(bg_cross_attn_scores, bg_k, dim=-1)
        
        # Extract and compute similarity
        bg_top_k_expanded = bg_top_k_indices.unsqueeze(-1).expand(-1, -1, dim)
        bg_core_tokens = torch.gather(img_hidden_states, dim=1, index=bg_top_k_expanded)
        
        bg_self_modal_sim = torch.bmm(bg_core_tokens, img_hidden_states.transpose(-1, -2))
        bg_self_modal_sim_avg = bg_self_modal_sim.mean(dim=1, keepdim=True).transpose(1, 2)
        bg_sim_map = F.softmax(bg_self_modal_sim_avg / temperature, dim=1)
        
        concept_sim_maps[bg_key] = bg_sim_map
    
    return concept_sim_maps


def compute_sdxl_similarity_maps(
    cross_attn_data: Dict[str, Dict[str, torch.Tensor]],
    hidden_states: torch.Tensor,
    token_pos_maps: Dict[str, List[List[int]]],
    top_k_ratio: float = 0.3,
    temperature: float = 1000.0,
) -> Dict[str, torch.Tensor]:
    """
    Compute similarity maps for SDXL model.
    
    SDXL uses separate cross-attention (image queries attend to text keys/values),
    so we adapt the FreeFuse algorithm accordingly.
    
    Args:
        cross_attn_data: Dict of block -> {query, key, value} tensors
        hidden_states: Image hidden states (B, img_len, dim)
        token_pos_maps: Dict mapping concept name to list of token positions
        top_k_ratio: Ratio of image tokens to select as "core"
        temperature: Temperature for softmax normalization
        
    Returns:
        Dict mapping concept name to similarity map (B, img_len, 1)
    """
    concept_sim_maps = {}
    
    if not cross_attn_data or not token_pos_maps:
        return concept_sim_maps
    
    # Use the last/deepest block's cross-attention
    last_block = list(cross_attn_data.keys())[-1]
    attn_data = cross_attn_data[last_block]
    
    query = attn_data["query"]  # (B, img_len, dim)
    key = attn_data["key"]      # (B, txt_len, dim)
    
    B, img_len, dim = query.shape
    _, txt_len, _ = key.shape
    device = query.device
    
    # Normalize
    query_norm = F.normalize(query, dim=-1)
    key_norm = F.normalize(key, dim=-1)
    hidden_norm = F.normalize(hidden_states, dim=-1) if hidden_states is not None else query_norm
    
    scale = 1.0 / (dim ** 0.5)
    
    # First pass: compute all cross-attention scores
    all_cross_attn_scores = {}
    for lora_name, positions_list in token_pos_maps.items():
        if lora_name.startswith("__"):
            continue
            
        pos = positions_list[0] if positions_list else []
        if not pos:
            continue
        
        pos_tensor = torch.tensor(pos, device=device, dtype=torch.long)
        pos_tensor = pos_tensor.clamp(0, txt_len - 1)
        concept_key = key_norm[:, pos_tensor, :]
        
        # Cross-attention scores
        cross_attn = torch.bmm(query_norm, concept_key.transpose(-1, -2)) * scale
        cross_attn = F.softmax(cross_attn, dim=1)
        cross_attn_scores = cross_attn.mean(dim=-1)
        all_cross_attn_scores[lora_name] = cross_attn_scores
    
    # Second pass: compute sim maps
    for lora_name, positions_list in token_pos_maps.items():
        if lora_name.startswith("__"):
            continue
            
        pos = positions_list[0] if positions_list else []
        if not pos:
            continue
        
        if lora_name not in all_cross_attn_scores:
            continue
        
        # Competitive exclusion
        cross_attn_scores = all_cross_attn_scores[lora_name] * len(all_cross_attn_scores)
        for other_name, other_scores in all_cross_attn_scores.items():
            if other_name != lora_name:
                cross_attn_scores = cross_attn_scores - other_scores
        
        # Top-k selection
        k = max(1, int(img_len * top_k_ratio))
        _, top_k_indices = torch.topk(cross_attn_scores, k, dim=-1)
        
        # Use hidden_states if available, else use query
        feat_source = hidden_states if hidden_states is not None else query
        
        top_k_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, feat_source.shape[-1])
        core_tokens = torch.gather(feat_source, dim=1, index=top_k_expanded)
        
        # Self-modal similarity
        self_sim = torch.bmm(core_tokens, feat_source.transpose(-1, -2))
        self_sim_avg = self_sim.mean(dim=1, keepdim=True).transpose(1, 2)
        
        sim_map = F.softmax(self_sim_avg / temperature, dim=1)
        concept_sim_maps[lora_name] = sim_map
    
    # Handle background
    for bg_key in ["__background__", "__bg__"]:
        if bg_key in token_pos_maps:
            bg_positions = token_pos_maps[bg_key][0]
            if bg_positions:
                pos_tensor = torch.tensor(bg_positions, device=device, dtype=torch.long)
                pos_tensor = pos_tensor.clamp(0, txt_len - 1)
                bg_key_embeds = key_norm[:, pos_tensor, :]
                
                bg_cross_attn = torch.bmm(query_norm, bg_key_embeds.transpose(-1, -2)) * scale
                bg_cross_attn = F.softmax(bg_cross_attn, dim=1)
                bg_scores = bg_cross_attn.mean(dim=-1)
                
                bg_k = max(1, int(img_len * top_k_ratio))
                _, bg_top_k = torch.topk(bg_scores, bg_k, dim=-1)
                
                feat_source = hidden_states if hidden_states is not None else query
                bg_top_k_expanded = bg_top_k.unsqueeze(-1).expand(-1, -1, feat_source.shape[-1])
                bg_core = torch.gather(feat_source, dim=1, index=bg_top_k_expanded)
                
                bg_self_sim = torch.bmm(bg_core, feat_source.transpose(-1, -2))
                bg_self_sim_avg = bg_self_sim.mean(dim=1, keepdim=True).transpose(1, 2)
                bg_sim_map = F.softmax(bg_self_sim_avg / temperature, dim=1)
                
                concept_sim_maps[bg_key] = bg_sim_map
            break
    
    return concept_sim_maps


def compute_similarity_maps_from_attention(
    attention_outputs: Dict[int, torch.Tensor],
    concepts: Dict[str, str],
    token_positions: Dict[str, list],
    image_size: Tuple[int, int] = (64, 64)
) -> Dict[str, torch.Tensor]:
    """
    Compute per-concept similarity maps from collected attention.
    
    This is a legacy/simplified interface. For full functionality,
    use compute_flux_similarity_maps or compute_sdxl_similarity_maps.
    
    Args:
        attention_outputs: Dict of block_index -> attention output tensor
        concepts: Dict of concept_name -> concept_text
        token_positions: Dict of concept_name -> list of token indices
        image_size: Target spatial size (H, W)
        
    Returns:
        Dict of concept_name -> (H, W) similarity map tensor
    """
    similarity_maps = {}
    
    if not attention_outputs:
        print("[FreeFuse] Warning: No attention outputs collected")
        return similarity_maps
    
    # Use the last collected block
    last_block = max(attention_outputs.keys())
    attn_out = attention_outputs[last_block]  # (B, seq_len, dim)
    
    B, seq_len, dim = attn_out.shape
    h, w = image_size
    
    # Determine if this looks like Flux (joint attention) or SDXL
    # Flux typically has img_len + txt_len tokens
    img_tokens = h * w
    
    if seq_len > img_tokens * 2:
        # Likely Flux with txt + img concatenated
        # Try to split - assume txt comes first
        txt_len = seq_len - img_tokens
        txt_hidden = attn_out[:, :txt_len, :]
        img_hidden = attn_out[:, txt_len:, :]
        
        # Convert token_positions to the expected format
        token_pos_maps = {}
        for name, pos_list in token_positions.items():
            if isinstance(pos_list, list) and len(pos_list) > 0:
                if isinstance(pos_list[0], list):
                    token_pos_maps[name] = pos_list
                else:
                    token_pos_maps[name] = [pos_list]
        
        sim_maps = compute_flux_similarity_maps(img_hidden, txt_hidden, token_pos_maps)
        
        # Convert to spatial format
        for name, sim_map in sim_maps.items():
            if sim_map.shape[1] == img_tokens:
                spatial = sim_map.squeeze(-1).view(B, h, w)
                # Take first batch, resize if needed
                spatial_2d = spatial[0]
                if (h, w) != image_size:
                    spatial_2d = F.interpolate(
                        spatial_2d.unsqueeze(0).unsqueeze(0),
                        size=image_size,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()
                similarity_maps[name] = spatial_2d
    else:
        # SDXL or similar - use simplified method
        for name, positions in token_positions.items():
            pos_list = positions[0] if isinstance(positions[0], list) else positions
            if not pos_list:
                continue
            
            # Simple feature similarity
            img_features = attn_out[:, :img_tokens, :]
            img_norm = F.normalize(img_features, dim=-1)
            
            # Self-similarity
            self_sim = torch.bmm(img_norm, img_norm.transpose(-1, -2))
            sim_map = self_sim[0].mean(dim=0).view(h, w)
            
            # Normalize
            sim_min, sim_max = sim_map.min(), sim_map.max()
            if sim_max > sim_min:
                sim_map = (sim_map - sim_min) / (sim_max - sim_min)
            
            similarity_maps[name] = sim_map.detach()
    
    return similarity_maps


def apply_freefuse_patches(model, state: Dict[str, Any], 
                           model_type: str = "auto") -> None:
    """
    Apply FreeFuse attention patches to a ComfyUI model.
    
    Args:
        model: ComfyUI ModelPatcher object
        state: FreeFuse state dict
        model_type: "flux", "sdxl", or "auto"
    """
    # Auto-detect model type
    if model_type == "auto":
        model_name = model.model.__class__.__name__.lower()
        if "flux" in model_name:
            model_type = "flux"
        else:
            model_type = "sdxl"
    
    print(f"[FreeFuse] Applying patches for {model_type} model")
    
    if model_type == "flux":
        # For Flux, use double_block patch
        collector = FreeFuseFluxDoubleBlockPatch(state, collect_block=18)
        patch = collector.create_double_block_patch()
        model.set_model_patch(patch, "double_block")
        
    else:
        # For SDXL/SD, use attention patches
        collector = FreeFuseAttentionCollector(state, collect_blocks=[1, 2, 4, 8])
        
        # Output patch for attn1 (self-attention)
        patch = collector.create_attn_output_patch()
        model.set_model_attn1_output_patch(patch)
        
        # Cross-attention patch
        attn2_patch = collector.create_attn2_patch()
        model.set_model_attn2_patch(attn2_patch)
    
    print(f"[FreeFuse] Patches applied successfully")


# Utility exports
__all__ = [
    "FreeFuseAttentionCollector",
    "FreeFuseFluxDoubleBlockPatch",
    "compute_similarity_maps_from_attention",
    "compute_flux_similarity_maps",
    "compute_sdxl_similarity_maps",
    "apply_freefuse_patches",
]
