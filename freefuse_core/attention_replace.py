"""
FreeFuse Attention Replace Patches for ComfyUI

Uses ComfyUI's replace patch mechanism to fully replace attention computation
and extract internal QKV states for similarity map calculation.

Key difference from output patches:
- Replace patches receive QKV AFTER projection, allowing access to internal attention states
- For Flux: Uses block replace to intercept entire DoubleStreamBlock and access q, k, v with RoPE
- For SDXL: Uses attn2_replace to intercept cross-attention with projected q, k, v

This enables the full FreeFuse algorithm:
1. Extract concept keys from encoder_key at token positions
2. Cross-attention top-k: select top-k image tokens with highest attention to concept
3. Concept attention: compute hidden state inner product between core tokens and all tokens
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List, Callable
import logging


class FreeFuseState:
    """
    Shared state for FreeFuse across sampling steps.
    
    Stores:
    - Current phase (collect / generate)
    - Token position maps for each concept
    - Collected similarity maps
    - Generated masks
    """
    
    def __init__(self):
        self.phase: str = "collect"  # "collect" or "generate"
        self.current_step: int = 0
        self.collect_step: int = 4  # Which step to collect attention at
        self.collect_block: int = 18  # Which block to collect from (for Flux)
        
        # Concept information
        self.token_pos_maps: Dict[str, List[List[int]]] = {}
        self.background_positions: Optional[List[int]] = None
        
        # Collected data
        self.similarity_maps: Dict[str, torch.Tensor] = {}
        self.collected_outputs: Dict[str, torch.Tensor] = {}
        
        # Generated masks
        self.masks: Dict[str, torch.Tensor] = {}
        
        # Settings
        self.top_k_ratio: float = 0.1
        self.temperature: float = 4000.0
        self.include_background: bool = True
        
    def reset_collection(self):
        """Reset collected data for new generation."""
        self.similarity_maps = {}
        self.collected_outputs = {}
        
    def is_collect_step(self, step: int, block_index: int = None) -> bool:
        """Check if we should collect at this step/block."""
        if self.phase != "collect":
            return False
        if step != self.collect_step:
            return False
        if block_index is not None and block_index != self.collect_block:
            return False
        return True


class FreeFuseFluxBlockReplace:
    """
    Replace patch for Flux DoubleStreamBlock.
    
    This is an AGGRESSIVE implementation that directly accesses the block's
    internal projection layers (img_attn.qkv, txt_attn.qkv) to compute QKV
    ourselves, apply RoPE, and extract similarity maps.
    
    We pass the actual block reference during initialization, allowing us to:
    1. Call block.img_attn.qkv() and block.txt_attn.qkv() to get QKV
    2. Apply block.img_attn.norm() and block.txt_attn.norm() for normalization
    3. Apply RoPE using the pe tensor
    4. Compute cross-attention scores for top-k selection
    5. Compute concept attention for final similarity maps
    
    This matches the original FreeFuse algorithm exactly.
    
    Usage:
        # Get the actual block from the model
        block = model.model.diffusion_model.double_blocks[18]
        replacer = FreeFuseFluxBlockReplace(state, block, block_index=18)
        model.set_model_patch_replace(
            replacer.create_block_replace(),
            "dit", 
            "double_block",
            18
        )
    """
    
    def __init__(self, state: FreeFuseState, block, block_index: int = 18):
        self.state = state
        self.block = block  # Actual DoubleStreamBlock reference
        self.block_index = block_index
        
    def create_block_replace(self) -> Callable:
        """Create a block replace function for Flux DoubleStreamBlock."""
        state = self.state
        block_index = self.block_index
        block = self.block  # Capture the block reference
        
        def block_replace(args: Dict, extra_args: Dict) -> Dict:
            """
            Replace function for DoubleStreamBlock.
            
            We manually compute QKV and attention to extract similarity maps,
            matching the original FreeFuse algorithm.
            """
            img = args["img"]
            txt = args["txt"]
            vec = args["vec"]
            pe = args["pe"]
            attn_mask = args.get("attn_mask")
            transformer_options = args.get("transformer_options", {})
            
            original_block = extra_args["original_block"]
            
            # Get current step from transformer_options
            # ComfyUI passes sigmas through transformer_options
            current_step = transformer_options.get("sigmas_index", state.current_step)
            
            # DEBUG: Log to understand the call pattern
            if block_index == state.collect_block:
                logging.info(f"[FreeFuse] block_replace: block={block_index}, "
                            f"current_step={current_step}, collect_step={state.collect_step}, "
                            f"phase={state.phase}, state.current_step={state.current_step}")
            
            # Check if we should collect at this block
            should_collect = state.is_collect_step(current_step, block_index)
            
            if not should_collect:
                # Just run original block normally
                return original_block(args)
            
            # === COLLECTION MODE: Manually compute QKV and extract similarity maps ===
            
            try:
                # Import the attention and rope functions from ComfyUI
                from comfy.ldm.flux.math import attention, apply_rope
                from comfy.ldm.flux.layers import apply_mod
                
                # Get modulation values (same as original block)
                if block.modulation:
                    img_mod1, img_mod2 = block.img_mod(vec)
                    txt_mod1, txt_mod2 = block.txt_mod(vec)
                else:
                    (img_mod1, img_mod2), (txt_mod1, txt_mod2) = vec
                
                # === Compute img QKV ===
                img_modulated = block.img_norm1(img)
                img_modulated = apply_mod(img_modulated, (1 + img_mod1.scale), img_mod1.shift, None)
                img_qkv = block.img_attn.qkv(img_modulated)
                img_q, img_k, img_v = img_qkv.view(
                    img_qkv.shape[0], img_qkv.shape[1], 3, block.num_heads, -1
                ).permute(2, 0, 3, 1, 4)
                img_q, img_k = block.img_attn.norm(img_q, img_k, img_v)
                
                # === Compute txt QKV ===
                txt_modulated = block.txt_norm1(txt)
                txt_modulated = apply_mod(txt_modulated, (1 + txt_mod1.scale), txt_mod1.shift, None)
                txt_qkv = block.txt_attn.qkv(txt_modulated)
                txt_q, txt_k, txt_v = txt_qkv.view(
                    txt_qkv.shape[0], txt_qkv.shape[1], 3, block.num_heads, -1
                ).permute(2, 0, 3, 1, 4)
                txt_q, txt_k = block.txt_attn.norm(txt_q, txt_k, txt_v)
                
                # === Concatenate based on flipped_img_txt flag ===
                if getattr(block, 'flipped_img_txt', False):
                    q = torch.cat((img_q, txt_q), dim=2)
                    k = torch.cat((img_k, txt_k), dim=2)
                    v = torch.cat((img_v, txt_v), dim=2)
                    txt_first = False
                else:
                    q = torch.cat((txt_q, img_q), dim=2)
                    k = torch.cat((txt_k, img_k), dim=2)
                    v = torch.cat((txt_v, img_v), dim=2)
                    txt_first = True
                
                # === Apply RoPE ===
                # q, k shape: (B, heads, seq_len, head_dim)
                # pe (freqs_cis) shape depends on ComfyUI implementation
                if pe is not None:
                    q_rope, k_rope = apply_rope(q, k, pe)
                else:
                    q_rope, k_rope = q, k
                
                # === Extract similarity maps using cross-attention + concept attention ===
                txt_len = txt.shape[1]
                img_len = img.shape[1]
                
                if txt_first:
                    txt_k_rope = k_rope[:, :, :txt_len, :]  # (B, heads, txt_len, head_dim)
                    img_q_rope = q_rope[:, :, txt_len:, :]  # (B, heads, img_len, head_dim)
                else:
                    txt_k_rope = k_rope[:, :, img_len:, :]
                    img_q_rope = q_rope[:, :, :img_len, :]
                
                # Compute similarity maps using the FreeFuse algorithm
                if state.token_pos_maps:
                    sim_maps = compute_flux_similarity_maps_with_qkv(
                        img_q_rope=img_q_rope,
                        txt_k_rope=txt_k_rope,
                        q_rope=q_rope,
                        k_rope=k_rope,
                        v=v,
                        txt_len=txt_len,
                        img_len=img_len,
                        txt_first=txt_first,
                        token_pos_maps=state.token_pos_maps,
                        background_positions=state.background_positions,
                        top_k_ratio=state.top_k_ratio,
                        temperature=state.temperature,
                        num_heads=block.num_heads,
                    )
                    state.similarity_maps.update(sim_maps)
                    
                    logging.info(f"[FreeFuse] Collected similarity maps at block {block_index}, "
                               f"step {current_step}: {list(sim_maps.keys())}")
                
            except Exception as e:
                logging.warning(f"[FreeFuse] Failed to extract QKV: {e}")
                import traceback
                traceback.print_exc()
            
            # Run original block for the actual output
            return original_block(args)
        
        return block_replace


def compute_flux_similarity_maps_with_qkv(
    img_q_rope: torch.Tensor,
    txt_k_rope: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    v: torch.Tensor,
    txt_len: int,
    img_len: int,
    txt_first: bool,
    token_pos_maps: Dict[str, List[List[int]]],
    background_positions: Optional[List[int]] = None,
    top_k_ratio: float = 0.3,
    temperature: float = 1000.0,
    num_heads: int = 24,
) -> Dict[str, torch.Tensor]:
    """
    Compute similarity maps for Flux using the full FreeFuse algorithm.
    
    This uses cross-attention (Q-K dot product with RoPE) for top-k selection
    and concept attention (hidden states inner product) for final similarity maps.
    
    Args:
        img_q_rope: Image query with RoPE, shape (B, heads, img_len, head_dim)
        txt_k_rope: Text key with RoPE, shape (B, heads, txt_len, head_dim)
        q_rope, k_rope: Full Q, K tensors with RoPE applied
        v: Full V tensor (no RoPE)
        txt_len, img_len: Sequence lengths
        txt_first: Whether text comes before image in concatenated sequence
        token_pos_maps: Dict mapping concept name to token positions
        background_positions: Optional background token positions
        top_k_ratio: Ratio of top-k image tokens to select
        temperature: Softmax temperature
        num_heads: Number of attention heads
    
    Returns:
        Dict mapping concept name to similarity map (B, img_len, 1)
    """
    concept_sim_maps = {}
    
    if not token_pos_maps:
        return concept_sim_maps
    
    device = img_q_rope.device
    B, heads, _, head_dim = img_q_rope.shape
    scale = 1.0 / 1000.0  # Same scale as FreeFuse
    
    # Compute full attention output for concept attention
    # We compute attention manually to get the output hidden states
    try:
        # q_rope, k_rope: (B, heads, seq_len, head_dim)
        # v: (B, heads, seq_len, head_dim)
        attn_weights = torch.einsum('bhid,bhjd->bhij', q_rope, k_rope) / (head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.einsum('bhij,bhjd->bhid', attn_weights, v)
        # Reshape to (B, seq_len, hidden_dim)
        attn_output = attn_output.transpose(1, 2).reshape(B, -1, heads * head_dim)
        
        # Split attention output
        if txt_first:
            txt_out = attn_output[:, :txt_len, :]
            img_out = attn_output[:, txt_len:, :]
        else:
            img_out = attn_output[:, :img_len, :]
            txt_out = attn_output[:, img_len:, :]
            
    except Exception as e:
        logging.warning(f"[FreeFuse] Failed to compute attention output: {e}")
        return concept_sim_maps
    
    # First pass: compute all cross-attention scores for competitive exclusion
    all_cross_attn_scores = {}
    
    for lora_name, positions_list in token_pos_maps.items():
        if lora_name.startswith("__"):
            continue
        
        pos = positions_list[0] if positions_list else []
        if not pos:
            continue
        
        pos_tensor = torch.tensor(pos, device=device, dtype=torch.long)
        pos_tensor = pos_tensor.clamp(0, txt_len - 1)
        
        # Extract concept keys at token positions
        # txt_k_rope: (B, heads, txt_len, head_dim)
        concept_k = txt_k_rope[:, :, pos_tensor, :]  # (B, heads, concept_len, head_dim)
        
        # Cross-attention: img_q @ concept_k^T
        # img_q_rope: (B, heads, img_len, head_dim)
        cross_attn = torch.einsum('bhid,bhjd->bhij', img_q_rope, concept_k) * scale
        cross_attn = F.softmax(cross_attn, dim=2)  # softmax over img dimension
        cross_attn_scores = cross_attn.mean(dim=1).mean(dim=-1)  # (B, img_len)
        all_cross_attn_scores[lora_name] = cross_attn_scores
    
    # Second pass: compute similarity maps with competitive exclusion
    for lora_name, positions_list in token_pos_maps.items():
        if lora_name.startswith("__"):
            continue
        
        pos = positions_list[0] if positions_list else []
        if not pos or lora_name not in all_cross_attn_scores:
            continue
        
        # Competitive exclusion
        cross_attn_scores = all_cross_attn_scores[lora_name] * len(all_cross_attn_scores)
        for other_name, other_scores in all_cross_attn_scores.items():
            if other_name != lora_name:
                cross_attn_scores = cross_attn_scores - other_scores
        
        # Top-k selection based on cross-attention scores
        k_select = max(1, int(img_len * top_k_ratio))
        _, top_k_indices = torch.topk(cross_attn_scores, k_select, dim=-1)  # (B, k)
        
        # Extract core image tokens from attention output
        # img_out: (B, img_len, hidden_dim)
        top_k_indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, img_out.shape[-1])
        core_image_tokens = torch.gather(img_out, dim=1, index=top_k_indices_expanded)  # (B, k, hidden_dim)
        
        # Concept attention: core tokens @ all image tokens
        self_modal_sim = torch.bmm(core_image_tokens, img_out.transpose(-1, -2))  # (B, k, img_len)
        self_modal_sim_avg = self_modal_sim.mean(dim=1, keepdim=True).transpose(1, 2)  # (B, img_len, 1)
        
        # Normalize with softmax
        concept_sim_map = F.softmax(self_modal_sim_avg / temperature, dim=1)
        concept_sim_maps[lora_name] = concept_sim_map
    
    # Handle background
    bg_positions = background_positions
    bg_key = None
    
    if not bg_positions:
        for key in ["__background__", "__bg__"]:
            if key in token_pos_maps:
                bg_positions = token_pos_maps[key][0]
                bg_key = key
                break
    else:
        bg_key = "__bg__"
    
    if bg_positions:
        pos_tensor = torch.tensor(bg_positions, device=device, dtype=torch.long)
        pos_tensor = pos_tensor.clamp(0, txt_len - 1)
        
        bg_concept_k = txt_k_rope[:, :, pos_tensor, :]
        bg_cross_attn = torch.einsum('bhid,bhjd->bhij', img_q_rope, bg_concept_k) * scale
        bg_cross_attn = F.softmax(bg_cross_attn, dim=2)
        bg_cross_attn_scores = bg_cross_attn.mean(dim=1).mean(dim=-1)
        
        bg_k = max(1, int(img_len * top_k_ratio))
        _, bg_top_k_indices = torch.topk(bg_cross_attn_scores, bg_k, dim=-1)
        
        bg_top_k_expanded = bg_top_k_indices.unsqueeze(-1).expand(-1, -1, img_out.shape[-1])
        bg_core_tokens = torch.gather(img_out, dim=1, index=bg_top_k_expanded)
        
        bg_self_modal_sim = torch.bmm(bg_core_tokens, img_out.transpose(-1, -2))
        bg_self_modal_sim_avg = bg_self_modal_sim.mean(dim=1, keepdim=True).transpose(1, 2)
        bg_sim_map = F.softmax(bg_self_modal_sim_avg / temperature, dim=1)
        
        concept_sim_maps[bg_key] = bg_sim_map
    
    return concept_sim_maps


class FreeFuseFluxAttentionReplace:
    """
    Alternative: Replace the attention function within Flux blocks.
    
    This hooks into the `attention` function call in layers.py to intercept
    q, k, v with RoPE already applied.
    
    For ComfyUI, we can set this via transformer_options["optimized_attention_override"]
    """
    
    def __init__(self, state: FreeFuseState, collect_block: int = 18):
        self.state = state
        self.collect_block = collect_block
        
    def create_attention_override(self) -> Callable:
        """Create an attention override function."""
        state = self.state
        collect_block = self.collect_block
        
        def attention_override(
            original_fn: Callable,
            q: torch.Tensor,
            k: torch.Tensor, 
            v: torch.Tensor,
            *args,
            transformer_options: Dict = {},
            **kwargs
        ) -> torch.Tensor:
            """
            Override optimized_attention to intercept QKV.
            
            Args:
                original_fn: The original attention function
                q, k, v: Query, Key, Value tensors with RoPE already applied
                         Shape: (B, heads, seq_len, head_dim) when skip_reshape=True
            """
            # Get block info
            block_index = transformer_options.get("block_index", -1)
            block_type = transformer_options.get("block_type", "")
            current_step = transformer_options.get("sigmas_index", state.current_step)
            
            # Check if we should collect
            should_collect = (
                state.phase == "collect" and
                current_step == state.collect_step and
                block_index == collect_block and
                block_type == "double"
            )
            
            # Run original attention
            out = original_fn(q, k, v, *args, transformer_options=transformer_options, **kwargs)
            
            if should_collect and state.token_pos_maps:
                # q, k, v have shape (B, heads, seq_len, head_dim) when skip_reshape=True
                # For Flux: seq_len = txt_len + img_len (concatenated)
                
                # Store for later processing
                state.collected_outputs["qkv"] = {
                    "q": q.detach().clone(),
                    "k": k.detach().clone(),
                    "v": v.detach().clone(),
                    "out": out.detach().clone(),
                }
                
                logging.info(f"[FreeFuse] Collected QKV at block {block_index}, "
                           f"q shape: {q.shape}")
            
            return out
        
        return attention_override


class FreeFuseSDXLAttnReplace:
    """
    Replace patch for SDXL cross-attention (attn2) with SelfConcept method.
    
    In SDXL, cross-attention has:
    - q: image features projected
    - k, v: text features projected
    
    This implementation uses:
    1. attn1_output_patch: Cache self-attention output (hidden_states)
    2. attn2_replace: Use cached hidden_states for SelfConcept similarity computation
    
    The SelfConcept method:
    1. Use cross-attention to select top-k image tokens for each concept
    2. Use hidden_states inner product to compute final similarity maps
    
    Usage:
        replacer = FreeFuseSDXLAttnReplace(state)
        replacer.apply_to_model(model)
    """
    
    def __init__(self, state: FreeFuseState, collect_blocks: List[Tuple] = None):
        self.state = state
        # Default: collect from output block 0, transformer_block 3
        # This corresponds to diffusers' up_blocks.0.attentions.0.transformer_blocks.3.attn2
        # Format: (block_name, block_num, transformer_index)
        self.collect_blocks = collect_blocks or [
            ("output", 0, 3),  # up_blocks.0.attentions.0.transformer_blocks.3.attn2
        ]
        # Cache for self-attention hidden states, keyed by block
        self._attn1_cache: Dict[str, torch.Tensor] = {}
        
    def apply_to_model(self, model):
        """Apply both attn1_output_patch and attn2_replace patches to the model."""
        # Add attn1_output_patch to cache self-attention outputs
        model.set_model_attn1_output_patch(self._create_attn1_output_patch())
        
        # Add attn2_replace patches for the collect blocks
        for block_spec in self.collect_blocks:
            if len(block_spec) == 3:
                block_name, block_num, transformer_index = block_spec
            else:
                block_name, block_num = block_spec
                transformer_index = None
            
            model.set_model_attn2_replace(
                self.create_attn_replace(block_name, block_num, transformer_index),
                block_name,
                block_num,
                transformer_index,
            )
            logging.info(f"[FreeFuse] Set attn2_replace for ({block_name}, {block_num}, {transformer_index})")
    
    def _create_attn1_output_patch(self) -> Callable:
        """Create patch to cache self-attention output."""
        def attn1_output_patch(n: torch.Tensor, extra_options: Dict) -> torch.Tensor:
            """Cache self-attention output for SelfConcept computation."""
            block = extra_options.get("block", ("unknown", 0))
            if len(block) >= 3:
                block_key = f"{block[0]}_{block[1]}_{block[2]}"
            else:
                block_key = f"{block[0]}_{block[1]}"
            self._attn1_cache[block_key] = n.clone()
            return n
        return attn1_output_patch
        
    def create_attn_replace(self, block_name: str, block_num: int, transformer_index: int = None) -> Callable:
        """Create an attn2 replace function for a specific block."""
        state = self.state
        
        # Build the block key for matching
        if transformer_index is not None:
            target_block = (block_name, block_num, transformer_index)
        else:
            target_block = (block_name, block_num)
        
        def attn2_replace(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            extra_options: Dict,
        ) -> torch.Tensor:
            """
            Replace function for attn2 (cross-attention).
            
            Uses CrossAttn + SelfConcept method from FreeFuse SDXL:
            1. CrossAttn: Use cross-attention weights to select top-k image tokens
            2. SelfConcept: Use hidden_states inner product for final similarity
            
            Args:
                q: Query from image features (B, seq_len, n_heads*dim_head)
                k: Key from text features
                v: Value from text features
                extra_options: Contains block info, n_heads, dim_head, etc.
            
            Returns:
                Attention output
            """
            n_heads = extra_options.get("n_heads", 8)
            dim_head = extra_options.get("dim_head", 64)
            current_step = extra_options.get("sigmas_index", state.current_step)
            
            # Check if this is a block we want to collect from
            # The block info in extra_options matches what we registered
            should_collect = (
                state.phase == "collect" and
                current_step == state.collect_step
            )
            
            # Compute standard attention
            scale = dim_head ** -0.5
            
            # Reshape for attention if needed
            if q.dim() == 3:
                # ComfyUI passes q,k,v as (B, seq_len, n_heads*dim_head)
                # Reshape to (B, n_heads, seq_len, dim_head)
                batch_size = q.shape[0]
                img_seq_len = q.shape[1]
                q_4d = q.view(batch_size, img_seq_len, n_heads, dim_head).transpose(1, 2)
                k_4d = k.view(batch_size, -1, n_heads, dim_head).transpose(1, 2)
                v_4d = v.view(batch_size, -1, n_heads, dim_head).transpose(1, 2)
            else:
                batch_size = q.shape[0]
                img_seq_len = q.shape[2]
                q_4d, k_4d, v_4d = q, k, v
            
            # Compute attention weights
            attn_weights = torch.matmul(q_4d, k_4d.transpose(-1, -2)) * scale
            attn_weights = F.softmax(attn_weights, dim=-1)
            
            # Compute attention output
            out_4d = torch.matmul(attn_weights, v_4d)
            
            # Reshape back to (B, seq_len, n_heads*dim_head)
            out = out_4d.transpose(1, 2).reshape(batch_size, -1, n_heads * dim_head)
            
            if should_collect and state.token_pos_maps:
                # Try to get cached hidden_states from attn1_output_patch
                block = extra_options.get("block", (block_name, block_num))
                if len(block) >= 3:
                    block_key = f"{block[0]}_{block[1]}_{block[2]}"
                else:
                    block_key = f"{block[0]}_{block[1]}"
                cached_hidden_states = self._attn1_cache.get(block_key)
                
                if cached_hidden_states is not None:
                    # Use cached self-attention output for SelfConcept
                    hidden_for_sim = cached_hidden_states
                    logging.info(f"[FreeFuse] Using cached attn1 hidden_states for {block_key}")
                else:
                    # Fallback: use cross-attention output
                    hidden_for_sim = out
                    logging.info(f"[FreeFuse] No cached attn1, using cross-attn output for {block_key}")
                
                # Get cond_or_uncond from extra_options for proper CFG batch handling
                cond_or_uncond = extra_options.get("cond_or_uncond", None)
                
                # Extract concept similarity maps using SelfConcept method
                sim_maps = self._extract_sdxl_similarity_maps_self_concept(
                    attn_weights=attn_weights,
                    hidden_states=hidden_for_sim,
                    cond_or_uncond=cond_or_uncond,
                )
                state.similarity_maps.update(sim_maps)
                
                logging.info(f"[FreeFuse] Collected SDXL attn at {block_name}/{block_num}")
            
            return out
        
        return attn2_replace
    
    def _extract_sdxl_similarity_maps_self_concept(
        self,
        attn_weights: torch.Tensor,
        hidden_states: torch.Tensor,
        cond_or_uncond: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract similarity maps using CrossAttn + SelfConcept method.
        
        This matches the FreeFuseSDXLAttnProcessor implementation:
        1. Use cross-attention weights to select top-k image tokens for each concept
        2. Use hidden_states inner product (SelfConcept) to compute final similarity
        
        IMPORTANT: With CFG (classifier-free guidance), the batch structure depends on
        cond_or_uncond list from ComfyUI's transformer_options:
        - cond_or_uncond[i] = 0 means batch[i] is cond (positive prompt with concept tokens)
        - cond_or_uncond[i] = 1 means batch[i] is uncond (negative prompt, no meaningful concepts)
        
        We MUST use only cond batches (where cond_or_uncond[i] == 0) for similarity computation.
        This supports arbitrary batch sizes, not just B=2.
        
        Args:
            attn_weights: Cross-attention weights (B, heads, img_seq_len, txt_seq_len)
            hidden_states: Attention output (B, img_seq_len, C)
            cond_or_uncond: List indicating which batch indices are cond (0) or uncond (1)
        
        Returns:
            Dict mapping concept name to similarity map (1, img_seq_len, 1)
        """
        state = self.state
        concept_sim_maps = {}
        
        B, heads, img_len, txt_len = attn_weights.shape
        
        # Extract only cond batches using cond_or_uncond info from ComfyUI
        # cond_or_uncond[i] == 0 means cond (positive prompt), == 1 means uncond (negative)
        if cond_or_uncond is not None and len(cond_or_uncond) > 0:
            # Find indices of cond batches (where value == 0)
            # Batch structure: if cond_or_uncond = [1, 0], then B items per condition
            # Total batch = B_total, items_per_cond = B_total // len(cond_or_uncond)
            items_per_cond = B // len(cond_or_uncond)
            
            cond_indices = []
            for i, c in enumerate(cond_or_uncond):
                if c == 0:  # cond batch
                    start_idx = i * items_per_cond
                    end_idx = (i + 1) * items_per_cond
                    cond_indices.extend(range(start_idx, end_idx))
            
            if cond_indices:
                cond_indices_tensor = torch.tensor(cond_indices, device=attn_weights.device, dtype=torch.long)
                attn_weights = attn_weights[cond_indices_tensor]  # (n_cond, heads, img_len, txt_len)
                hidden_states = hidden_states[cond_indices_tensor]  # (n_cond, img_len, C)
                B = len(cond_indices)
                logging.info(f"[SelfConcept] CFG detected via cond_or_uncond={cond_or_uncond}, "
                           f"using {B} cond batches at indices {cond_indices}")
            else:
                logging.warning(f"[SelfConcept] cond_or_uncond={cond_or_uncond} has no cond (0) entries, using all")
        elif B >= 2:
            # Fallback: assume [uncond, cond] structure when cond_or_uncond not available
            # Take only the second half (cond batches)
            half_B = B // 2
            attn_weights = attn_weights[half_B:]  # (half_B, heads, img_len, txt_len)
            hidden_states = hidden_states[half_B:]  # (half_B, img_len, C)
            B = attn_weights.shape[0]
            logging.info(f"[SelfConcept] CFG fallback: B>=2, using second half as cond, new B={B}")
        
        # Collect all cross-attention scores for contrastive selection
        all_cross_attn_scores = {}
        
        for lora_name, positions_list in state.token_pos_maps.items():
            if lora_name.startswith("__"):
                continue
            pos = positions_list[0] if positions_list else []
            pos = [p for p in pos if 0 <= p < txt_len]
            if not pos:
                continue
            
            pos_tensor = torch.tensor(pos, device=attn_weights.device, dtype=torch.long)
            concept_attn = attn_weights[:, :, :, pos_tensor]  # (B, heads, img_len, concept_len)
            # Mean over heads and concept tokens -> (B, img_len)
            cross_attn_scores = concept_attn.mean(dim=-1).mean(dim=1)
            all_cross_attn_scores[lora_name] = cross_attn_scores
        
        # Compute contrastive scores and final sim maps using SelfConcept
        for lora_name in all_cross_attn_scores.keys():
            # Contrastive score: enhance current concept, subtract others
            cross_attn_scores = all_cross_attn_scores[lora_name] * len(all_cross_attn_scores)
            for other_name, other_scores in all_cross_attn_scores.items():
                if other_name != lora_name:
                    cross_attn_scores = cross_attn_scores - other_scores
            
            # Select top-k image tokens based on cross-attention scores
            k = max(1, int(img_len * state.top_k_ratio))
            _, top_k_indices = torch.topk(cross_attn_scores, k, dim=-1)  # (B, k)
            
            # SelfConcept: Use hidden_states inner product
            # Extract core image tokens
            top_k_indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
            core_tokens = torch.gather(hidden_states, dim=1, index=top_k_indices_expanded)  # (B, k, C)
            
            # Compute self-modal similarity: core tokens @ all tokens
            self_modal_sim = torch.bmm(core_tokens, hidden_states.transpose(-1, -2))  # (B, k, img_len)
            
            # Average over core tokens
            concept_sim_map = self_modal_sim.mean(dim=1)  # (B, img_len)
            
            logging.info(f"[SelfConcept] {lora_name}: hidden_states shape={hidden_states.shape}, "
                        f"self_modal_sim range=[{self_modal_sim.min():.2f}, {self_modal_sim.max():.2f}], "
                        f"mean={concept_sim_map.mean():.2f}")
            
            # Apply softmax with temperature
            concept_sim_map = F.softmax(concept_sim_map / state.temperature, dim=-1)
            
            concept_sim_maps[lora_name] = concept_sim_map.unsqueeze(-1)  # (B, img_len, 1)
        
        # Handle background
        for bg_key in ["__background__", "__bg__"]:
            if bg_key in state.token_pos_maps:
                bg_pos = state.token_pos_maps[bg_key][0]
                bg_pos = [p for p in bg_pos if 0 <= p < txt_len]
                if bg_pos:
                    pos_tensor = torch.tensor(bg_pos, device=attn_weights.device, dtype=torch.long)
                    bg_attn = attn_weights[:, :, :, pos_tensor]
                    bg_scores = bg_attn.mean(dim=-1).mean(dim=1)
                    
                    # Use cross-attention based similarity for background (no SelfConcept)
                    bg_sim_map = F.softmax(bg_scores / state.temperature, dim=-1)
                    concept_sim_maps[bg_key] = bg_sim_map.unsqueeze(-1)
                break
        
        return concept_sim_maps


def compute_flux_similarity_maps_from_outputs(
    img_hidden_states: torch.Tensor,
    txt_hidden_states: torch.Tensor,
    token_pos_maps: Dict[str, List[List[int]]],
    background_positions: Optional[List[int]] = None,
    top_k_ratio: float = 0.3,
    temperature: float = 1000.0,
) -> Dict[str, torch.Tensor]:
    """
    Compute similarity maps for Flux using the FreeFuse algorithm.
    
    Uses hidden states after attention (concept attention method).
    
    Args:
        img_hidden_states: Image hidden states after attention (B, img_len, dim)
        txt_hidden_states: Text hidden states after attention (B, txt_len, dim)
        token_pos_maps: Dict mapping concept name to token positions
        background_positions: Optional background token positions
        top_k_ratio: Ratio of top-k image tokens to select
        temperature: Softmax temperature
    
    Returns:
        Dict mapping concept name to similarity map (B, img_len, 1)
    """
    concept_sim_maps = {}
    
    if not token_pos_maps:
        return concept_sim_maps
    
    B, img_len, dim = img_hidden_states.shape
    _, txt_len, _ = txt_hidden_states.shape
    device = img_hidden_states.device
    
    # Normalize for computing similarity
    img_norm = F.normalize(img_hidden_states, dim=-1)
    txt_norm = F.normalize(txt_hidden_states, dim=-1)
    
    scale = 1.0 / 1000.0
    
    # First pass: compute all cross-attention scores for competitive exclusion
    all_cross_attn_scores = {}
    
    for lora_name, positions_list in token_pos_maps.items():
        if lora_name.startswith("__"):
            continue
        
        pos = positions_list[0] if positions_list else []
        if not pos:
            continue
        
        pos_tensor = torch.tensor(pos, device=device, dtype=torch.long)
        pos_tensor = pos_tensor.clamp(0, txt_len - 1)
        concept_embeds = txt_norm[:, pos_tensor, :]  # (B, concept_len, dim)
        
        # Cross-attention: img @ concept^T
        cross_attn = torch.bmm(img_norm, concept_embeds.transpose(-1, -2)) * scale
        cross_attn = F.softmax(cross_attn, dim=1)
        cross_attn_scores = cross_attn.mean(dim=-1)  # (B, img_len)
        all_cross_attn_scores[lora_name] = cross_attn_scores
    
    # Second pass: compute similarity maps with competitive exclusion
    for lora_name, positions_list in token_pos_maps.items():
        if lora_name.startswith("__"):
            continue
        
        pos = positions_list[0] if positions_list else []
        if not pos or lora_name not in all_cross_attn_scores:
            continue
        
        # Competitive exclusion
        cross_attn_scores = all_cross_attn_scores[lora_name] * len(all_cross_attn_scores)
        for other_name, other_scores in all_cross_attn_scores.items():
            if other_name != lora_name:
                cross_attn_scores = cross_attn_scores - other_scores
        
        # Top-k selection
        k = max(1, int(img_len * top_k_ratio))
        _, top_k_indices = torch.topk(cross_attn_scores, k, dim=-1)
        
        # Extract core image tokens
        top_k_indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, dim)
        core_image_tokens = torch.gather(img_hidden_states, dim=1, index=top_k_indices_expanded)
        
        # Concept attention: core tokens @ all tokens
        self_modal_sim = torch.bmm(core_image_tokens, img_hidden_states.transpose(-1, -2))
        self_modal_sim_avg = self_modal_sim.mean(dim=1, keepdim=True).transpose(1, 2)
        
        # Normalize with softmax
        concept_sim_map = F.softmax(self_modal_sim_avg / temperature, dim=1)
        concept_sim_maps[lora_name] = concept_sim_map
    
    # Handle background
    bg_positions = None
    bg_key = None
    
    if background_positions:
        bg_positions = background_positions
        bg_key = "__bg__"
    else:
        for key in ["__background__", "__bg__"]:
            if key in token_pos_maps:
                bg_positions = token_pos_maps[key][0]
                bg_key = key
                break
    
    if bg_positions:
        pos_tensor = torch.tensor(bg_positions, device=device, dtype=torch.long)
        pos_tensor = pos_tensor.clamp(0, txt_len - 1)
        bg_embeds = txt_norm[:, pos_tensor, :]
        
        bg_cross_attn = torch.bmm(img_norm, bg_embeds.transpose(-1, -2)) * scale
        bg_cross_attn = F.softmax(bg_cross_attn, dim=1)
        bg_cross_attn_scores = bg_cross_attn.mean(dim=-1)
        
        bg_k = max(1, int(img_len * top_k_ratio))
        _, bg_top_k_indices = torch.topk(bg_cross_attn_scores, bg_k, dim=-1)
        
        bg_top_k_expanded = bg_top_k_indices.unsqueeze(-1).expand(-1, -1, dim)
        bg_core_tokens = torch.gather(img_hidden_states, dim=1, index=bg_top_k_expanded)
        
        bg_self_modal_sim = torch.bmm(bg_core_tokens, img_hidden_states.transpose(-1, -2))
        bg_self_modal_sim_avg = bg_self_modal_sim.mean(dim=1, keepdim=True).transpose(1, 2)
        bg_sim_map = F.softmax(bg_self_modal_sim_avg / temperature, dim=1)
        
        concept_sim_maps[bg_key] = bg_sim_map
    
    return concept_sim_maps


def apply_freefuse_replace_patches(
    model,
    state: FreeFuseState,
    model_type: str = "auto",
) -> None:
    """
    Apply FreeFuse replace patches to a ComfyUI model.
    
    This uses an AGGRESSIVE approach: we directly access the model's internal
    blocks and their projection layers to compute QKV ourselves.
    
    Args:
        model: ComfyUI ModelPatcher object
        state: FreeFuse state
        model_type: "flux", "sdxl", or "auto"
    """
    # Auto-detect model type
    if model_type == "auto":
        model_name = model.model.__class__.__name__.lower()
        if "flux" in model_name:
            model_type = "flux"
        else:
            model_type = "sdxl"
    
    logging.info(f"[FreeFuse] Applying AGGRESSIVE replace patches for {model_type} model")
    
    if model_type == "flux":
        # Get the actual diffusion model
        diffusion_model = model.model.diffusion_model
        
        # Get the target block
        block_index = state.collect_block
        if hasattr(diffusion_model, 'double_blocks') and len(diffusion_model.double_blocks) > block_index:
            block = diffusion_model.double_blocks[block_index]
            
            # Create replacer with actual block reference
            replacer = FreeFuseFluxBlockReplace(state, block=block, block_index=block_index)
            block_replace = replacer.create_block_replace()
            
            # Set the replace patch using ComfyUI's API
            model.set_model_patch_replace(
                block_replace,
                "dit",
                "double_block",
                block_index,
            )
            
            logging.info(f"[FreeFuse] Set AGGRESSIVE block replace for double_block {block_index}")
            logging.info(f"[FreeFuse] Block type: {type(block).__name__}")
        else:
            logging.error(f"[FreeFuse] Cannot find double_blocks[{block_index}] in diffusion model")
        
    else:
        # For SDXL, use attn1_output_patch + attn2 replace with SelfConcept method
        replacer = FreeFuseSDXLAttnReplace(state)
        replacer.apply_to_model(model)
        
        logging.info(f"[FreeFuse] Set attn1_output_patch + attn2 replace for blocks: {replacer.collect_blocks}")


# Utility exports
__all__ = [
    "FreeFuseState",
    "FreeFuseFluxBlockReplace",
    "FreeFuseFluxAttentionReplace",
    "FreeFuseSDXLAttnReplace",
    "compute_flux_similarity_maps_from_outputs",
    "compute_flux_similarity_maps_with_qkv",
    "apply_freefuse_replace_patches",
]
