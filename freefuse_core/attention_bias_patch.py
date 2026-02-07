"""
FreeFuse Attention Bias Patches for ComfyUI

Patches to inject attention bias into transformer blocks during generation.
Supports both Flux (joint attention) and SDXL (cross-attention) architectures.

For Flux:
- Uses block_replace to intercept the entire DoubleStreamBlock
- Modifies the attention computation to include bias

For SDXL:
- Uses attn2_replace to intercept cross-attention
- Applies bias to the Q-K attention scores
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Callable, List, Tuple
import logging

from .attention_bias import (
    construct_attention_bias,
    construct_attention_bias_sdxl,
    apply_attention_bias_to_weights,
    get_attention_bias_for_layer,
    AttentionBiasConfig,
)


class FreeFuseFluxBiasBlockReplace:
    """
    Block replace patch for Flux that applies attention bias during generation.
    
    This patch intercepts the attention computation in DoubleStreamBlock and adds
    the FreeFuse attention bias to guide cross-modal attention.
    
    The bias is applied to the pre-softmax attention scores:
        attn_weights = Q @ K^T / sqrt(d_k) + attention_bias
        attn_output = softmax(attn_weights) @ V
    
    IMPORTANT: Bias is constructed dynamically at runtime based on actual sequence
    lengths, as txt_seq_len can vary between tokenizers and models.
    """
    
    def __init__(
        self,
        lora_masks: Dict[str, torch.Tensor],
        token_pos_maps: Dict[str, List],
        config: AttentionBiasConfig,
        block_index: int,
        block=None,  # Actual DoubleStreamBlock reference
    ):
        """
        Args:
            lora_masks: Dict of LoRA name -> (B, img_seq_len) binary mask
            token_pos_maps: Dict of LoRA name -> [[token positions], ...]
            config: Attention bias configuration
            block_index: Index of this transformer block
            block: The actual DoubleStreamBlock reference (needed for modulation/FFN)
        """
        self.lora_masks = lora_masks
        self.token_pos_maps = token_pos_maps
        self.config = config
        self.block_index = block_index
        self.block = block  # Capture block reference
        # Cache for constructed bias (keyed by (txt_len, img_len))
        self._bias_cache = {}
        
    def _get_or_build_bias(
        self,
        txt_len: int,
        img_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Get cached bias or build new one for the given dimensions."""
        cache_key = (txt_len, img_len)
        if cache_key in self._bias_cache:
            bias = self._bias_cache[cache_key]
            if bias.device != device:
                bias = bias.to(device)
            if bias.dtype != dtype:
                bias = bias.to(dtype)
            return bias
        
        # Build new bias
        bias = construct_attention_bias(
            lora_masks=self.lora_masks,
            token_pos_maps=self.token_pos_maps,
            txt_seq_len=txt_len,
            img_seq_len=img_len,
            bias_scale=self.config.bias_scale,
            positive_bias_scale=self.config.positive_bias_scale,
            bidirectional=self.config.bidirectional,
            use_positive_bias=self.config.use_positive_bias,
            device=device,
            dtype=dtype,
        )
        
        if bias is not None:
            self._bias_cache[cache_key] = bias
            
        return bias
        
    def create_block_replace(self) -> Callable:
        """Create a block replace function that applies attention bias."""
        config = self.config
        block_index = self.block_index
        block = self.block  # Capture block reference from self
        # Capture self for dynamic bias construction
        bias_builder = self
        
        def block_replace(args: Dict, extra_args: Dict) -> Dict:
            """
            Replace function for DoubleStreamBlock with attention bias.
            
            We intercept the block, apply attention bias, and return modified output.
            """
            img = args["img"]
            txt = args["txt"]
            vec = args["vec"]
            pe = args["pe"]
            attn_mask = args.get("attn_mask")
            
            original_block = extra_args["original_block"]
            # block is captured from self.block, not from extra_args
            
            # Check if we should apply bias to this block
            block_name = f"transformer_blocks.{block_index}"
            if not config.should_apply_to_block(block_name):
                return original_block(args)
            
            # Check if block reference is valid
            if block is None:
                logging.warning(f"[FreeFuse] Block {block_index} is None, falling back to original")
                return original_block(args)
            
            # Get actual sequence lengths
            txt_len = txt.shape[1]
            img_len = img.shape[1]
            
            # Dynamically build bias for actual dimensions
            attention_bias = bias_builder._get_or_build_bias(
                txt_len=txt_len,
                img_len=img_len,
                device=img.device,
                dtype=img.dtype,
            )
            
            if attention_bias is None:
                return original_block(args)
            
            try:
                # Import ComfyUI's attention utilities
                from comfy.ldm.flux.math import attention, apply_rope
                from comfy.ldm.flux.layers import apply_mod
                
                # Get modulation values
                if hasattr(block, 'modulation') and block.modulation:
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
                if pe is not None:
                    q, k = apply_rope(q, k, pe)
                
                # === Compute attention with bias ===
                B, heads, seq_len, head_dim = q.shape
                scale = head_dim ** -0.5
                
                # Compute attention scores
                attn_weights = torch.einsum('bhid,bhjd->bhij', q, k) * scale
                
                # Apply attention bias
                bias_for_attn = attention_bias
                if bias_for_attn.device != attn_weights.device:
                    bias_for_attn = bias_for_attn.to(attn_weights.device)
                if bias_for_attn.dtype != attn_weights.dtype:
                    bias_for_attn = bias_for_attn.to(attn_weights.dtype)
                
                # Expand bias for heads dimension: (B, seq, seq) -> (B, 1, seq, seq)
                if bias_for_attn.dim() == 3:
                    bias_for_attn = bias_for_attn.unsqueeze(1)
                
                # Handle sequence order (txt_first or img_first)
                if not txt_first:
                    # Need to reorder bias from (txt, img) to (img, txt)
                    img_len = img.shape[1]
                    txt_len = txt.shape[1]
                    # Create permutation indices
                    perm = torch.cat([
                        torch.arange(txt_len, txt_len + img_len),
                        torch.arange(txt_len)
                    ]).to(bias_for_attn.device)
                    # Reorder both dimensions
                    bias_for_attn = bias_for_attn[:, :, perm][:, :, :, perm]
                
                attn_weights = attn_weights + bias_for_attn
                
                # Apply attention mask if provided
                if attn_mask is not None:
                    attn_weights = attn_weights + attn_mask
                
                # Softmax and compute output
                attn_weights = F.softmax(attn_weights, dim=-1)
                attn_output = torch.einsum('bhij,bhjd->bhid', attn_weights, v)
                
                # Reshape output
                attn_output = attn_output.transpose(1, 2).reshape(B, seq_len, heads * head_dim)
                
                # Split back to txt and img
                if txt_first:
                    txt_attn_out = attn_output[:, :txt_len, :]
                    img_attn_out = attn_output[:, txt_len:, :]
                else:
                    img_attn_out = attn_output[:, :img_len, :]
                    txt_attn_out = attn_output[:, img_len:, :]
                
                # Apply output projections
                img_attn_out = block.img_attn.proj(img_attn_out)
                txt_attn_out = block.txt_attn.proj(txt_attn_out)
                
                # Apply modulation and residual
                # Note: apply_mod(tensor, m_mult, m_add, modulation_dims)
                # For gated residual: m_mult=gate, m_add=None
                img_out = img + apply_mod(img_attn_out, img_mod1.gate, None, None)
                txt_out = txt + apply_mod(txt_attn_out, txt_mod1.gate, None, None)
                
                # FFN for img
                img_ffn = block.img_norm2(img_out)
                img_ffn = apply_mod(img_ffn, (1 + img_mod2.scale), img_mod2.shift, None)
                img_ffn = block.img_mlp(img_ffn)
                img_out = img_out + apply_mod(img_ffn, img_mod2.gate, None, None)
                
                # FFN for txt
                txt_ffn = block.txt_norm2(txt_out)
                txt_ffn = apply_mod(txt_ffn, (1 + txt_mod2.scale), txt_mod2.shift, None)
                txt_ffn = block.txt_mlp(txt_ffn)
                txt_out = txt_out + apply_mod(txt_ffn, txt_mod2.gate, None, None)
                
                return {"img": img_out, "txt": txt_out}
                
            except Exception as e:
                logging.warning(f"[FreeFuse] Failed to apply attention bias at block {block_index}: {e}")
                import traceback
                traceback.print_exc()
                # Fall back to original block
                return original_block(args)
        
        return block_replace


class FreeFuseFluxBiasSingleBlockReplace:
    """
    Block replace patch for Flux SingleStreamBlock with attention bias.
    
    SingleStreamBlock processes concatenated txt+img as a single sequence.
    The bias is applied to the full sequence attention.
    
    Key difference from DoubleStreamBlock:
    - Input x is already [txt, img] concatenated
    - We need to know txt_len to correctly apply the bias
    - txt_len is inferred from total_seq_len - img_seq_len (from masks)
    """
    
    def __init__(
        self,
        lora_masks: Dict[str, torch.Tensor],
        token_pos_maps: Dict[str, List[List[int]]],
        config: AttentionBiasConfig,
        block_index: int,
        block=None,  # Actual SingleStreamBlock reference
    ):
        """
        Args:
            lora_masks: Spatial masks for each LoRA concept {name: (H, W)}
            token_pos_maps: Token positions for each concept {name: [[positions]]}
            config: Attention bias configuration
            block_index: Index of this transformer block
            block: The actual SingleStreamBlock reference
        """
        self.lora_masks = lora_masks
        self.token_pos_maps = token_pos_maps
        self.config = config
        self.block_index = block_index
        self.block = block
        
        # Cache for constructed bias (keyed by sequence length)
        self._bias_cache = {}
        
    def create_block_replace(self) -> Callable:
        """Create a block replace function for SingleStreamBlock."""
        lora_masks = self.lora_masks
        token_pos_maps = self.token_pos_maps
        config = self.config
        block_index = self.block_index
        block = self.block
        bias_cache = self._bias_cache
        
        def block_replace(args: Dict, extra_args: Dict) -> Dict:
            """
            Replace function for SingleStreamBlock with attention bias.
            
            Note: In ComfyUI's Flux model, single blocks receive the concatenated
            txt+img sequence via the 'img' key (not 'x'), and return {'img': output}.
            """
            # In ComfyUI Flux single block replace, the input is passed as 'img'
            # even though it's actually the concatenated [txt, img] sequence
            x = args["img"]  # This is actually [txt, img] concatenated
            vec = args["vec"]
            pe = args["pe"]
            attn_mask = args.get("attn_mask")
            transformer_options = args.get("transformer_options", {})
            
            original_block = extra_args["original_block"]
            
            block_name = f"single_transformer_blocks.{block_index}"
            if not config.should_apply_to_block(block_name):
                return original_block(args)
            
            if block is None:
                logging.warning(f"[FreeFuse] Single block {block_index} is None, falling back to original")
                return original_block(args)
            
            if not lora_masks:
                return original_block(args)
            
            try:
                from comfy.ldm.flux.math import apply_rope
                from comfy.ldm.flux.layers import apply_mod
                
                B, total_seq_len, _ = x.shape
                
                # Infer img_seq_len from masks
                # Masks are flattened: (B, img_seq_len)
                first_mask = next(iter(lora_masks.values()))
                if first_mask.dim() == 2:
                    img_seq_len = first_mask.shape[1]
                else:
                    img_seq_len = first_mask.numel()
                txt_len = total_seq_len - img_seq_len
                
                # Construct attention bias dynamically
                cache_key = (total_seq_len, txt_len, img_seq_len)
                if cache_key not in bias_cache:
                    attention_bias = construct_attention_bias(
                        lora_masks=lora_masks,
                        token_pos_maps=token_pos_maps,
                        txt_seq_len=txt_len,
                        img_seq_len=img_seq_len,
                        bias_scale=config.bias_scale,
                        positive_bias_scale=config.positive_bias_scale if config.use_positive_bias else 0.0,
                        bidirectional=config.bidirectional,
                        use_positive_bias=config.use_positive_bias,
                    )
                    if attention_bias is not None:
                        bias_cache[cache_key] = attention_bias
                        if block_index == 0:
                            logging.info(f"[FreeFuse] Single block bias: shape={attention_bias.shape}")
                    else:
                        bias_cache[cache_key] = None
                
                attention_bias = bias_cache.get(cache_key)
                
                if attention_bias is None:
                    return original_block(args)
                
                # Get modulation
                if block.modulation:
                    mod, _ = block.modulation(vec)
                else:
                    mod = vec
                
                # Apply pre-norm and modulation
                x_normed = block.pre_norm(x)
                x_mod = apply_mod(x_normed, (1 + mod.scale), mod.shift, None)
                
                # Split QKV and MLP
                qkv, mlp = torch.split(
                    block.linear1(x_mod),
                    [3 * block.hidden_size, block.mlp_hidden_dim_first],
                    dim=-1
                )
                
                # Reshape QKV
                q, k, v = qkv.view(B, total_seq_len, 3, block.num_heads, -1).permute(2, 0, 3, 1, 4)
                del qkv
                
                # Normalize Q and K
                q, k = block.norm(q, k, v)
                
                # Apply RoPE if present
                if pe is not None:
                    q, k = apply_rope(q, k, pe)
                
                # Compute attention with bias
                head_dim = q.shape[-1]
                scale = head_dim ** -0.5
                
                attn_weights = torch.einsum('bhid,bhjd->bhij', q, k) * scale
                del q, k
                
                # Apply attention bias
                bias_for_attn = attention_bias
                if bias_for_attn.device != attn_weights.device:
                    bias_for_attn = bias_for_attn.to(attn_weights.device)
                if bias_for_attn.dtype != attn_weights.dtype:
                    bias_for_attn = bias_for_attn.to(attn_weights.dtype)
                
                if bias_for_attn.dim() == 3:
                    bias_for_attn = bias_for_attn.unsqueeze(1)
                
                # Handle size mismatch
                if bias_for_attn.shape[-1] != attn_weights.shape[-1]:
                    # Bias was constructed for different size, skip
                    logging.warning(f"[FreeFuse] Single block {block_index} bias size mismatch: "
                                  f"bias={bias_for_attn.shape}, attn={attn_weights.shape}")
                    return original_block(args)
                
                attn_weights = attn_weights + bias_for_attn
                
                # Apply attention mask if provided
                if attn_mask is not None:
                    attn_weights = attn_weights + attn_mask
                
                # Softmax and compute output
                attn_weights = F.softmax(attn_weights, dim=-1)
                attn_out = torch.einsum('bhij,bhjd->bhid', attn_weights, v)
                del v, attn_weights
                
                attn_out = attn_out.transpose(1, 2).reshape(B, total_seq_len, -1)
                
                # MLP activation
                if block.yak_mlp:
                    mlp = block.mlp_act(mlp[..., block.mlp_hidden_dim_first // 2:]) * mlp[..., :block.mlp_hidden_dim_first // 2]
                else:
                    mlp = block.mlp_act(mlp)
                
                # Combine attention and MLP through linear2
                output = block.linear2(torch.cat((attn_out, mlp), 2))
                
                # Apply gating and residual
                x = x + apply_mod(output, mod.gate, None, None)
                
                # Handle fp16 overflow
                if x.dtype == torch.float16:
                    x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
                
                # Return in expected format for ComfyUI Flux single block
                return {"img": x}
                
            except Exception as e:
                logging.warning(f"[FreeFuse] Failed to apply bias to single block {block_index}: {e}")
                import traceback
                traceback.print_exc()
                return original_block(args)
        
        return block_replace


class FreeFuseSDXLBiasAttnReplace:
    """
    Cross-attention replace patch for SDXL that applies attention bias.
    
    SDXL uses separate cross-attention where Q comes from image features
    and K/V come from text. The bias is applied to Q @ K^T scores.
    """
    
    def __init__(
        self,
        lora_masks: Dict[str, torch.Tensor],
        token_pos_maps: Dict[str, List[List[int]]],
        config: AttentionBiasConfig,
        latent_size: Tuple[int, int],
    ):
        """
        Args:
            lora_masks: Spatial masks for each LoRA
            token_pos_maps: Token positions for each LoRA
            config: Attention bias configuration
            latent_size: (H, W) of the latent space
        """
        self.lora_masks = lora_masks
        self.token_pos_maps = token_pos_maps
        self.config = config
        self.latent_size = latent_size
        
        # Cache computed biases for different resolutions
        self._bias_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def _infer_spatial_size(self, img_len: int) -> Tuple[int, int]:
        """Infer (img_h, img_w) from flattened img_len using latent_size aspect ratio.

        SDXL UNet preserves aspect ratio through all downsample stages.
        Each layer is (latent_h // d, latent_w // d) for d in {1, 2, 4, 8, ...}.
        We try these factors against the known latent_size first, then fall back
        to a generic aspect-ratio-preserving factorisation.
        """
        lat_h, lat_w = self.latent_size

        # Strategy 1: try power-of-2 downscale factors of latent_size
        for d in (1, 2, 4, 8, 16):
            h, w = lat_h // d, lat_w // d
            if h > 0 and w > 0 and h * w == img_len:
                return (h, w)

        # Strategy 2: aspect-ratio-preserving decomposition
        if lat_h > 0 and lat_w > 0:
            ratio = lat_w / lat_h
            h = max(int(round((img_len / max(ratio, 1e-8)) ** 0.5)), 1)
            w = img_len // h
            if h * w == img_len:
                return (h, w)

        # Strategy 3: try square
        side = int(round(img_len ** 0.5))
        if side * side == img_len:
            return (side, side)

        # Strategy 4: closest-to-square factor search (general fallback)
        import math as _math
        best_h, best_w = 1, img_len
        for h in range(1, int(_math.isqrt(img_len)) + 1):
            if img_len % h == 0:
                w = img_len // h
                if abs(w - h) < abs(best_w - best_h):
                    best_h, best_w = h, w
        return (best_h, best_w)

    def _get_bias_for_size(
        self,
        img_h: int,
        img_w: int,
        txt_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Get or compute attention bias for a specific feature map size."""
        cache_key = (img_h, img_w, txt_len)
        
        if cache_key not in self._bias_cache:
            bias = construct_attention_bias_sdxl(
                lora_masks=self.lora_masks,
                token_pos_maps=self.token_pos_maps,
                txt_seq_len=txt_len,
                img_h=img_h,
                img_w=img_w,
                bias_scale=self.config.bias_scale,
                positive_bias_scale=self.config.positive_bias_scale,
                use_positive_bias=self.config.use_positive_bias,
                device=device,
                dtype=dtype,
            )
            self._bias_cache[cache_key] = bias
        
        return self._bias_cache[cache_key]
    
    def create_attn_replace(self, block_name: str, block_num: int) -> Callable:
        """Create an attn2 replace function for a specific SDXL block."""
        config = self.config
        
        full_block_name = f"{block_name}.{block_num}"
        
        def attn2_replace(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            extra_options: Dict,
        ) -> torch.Tensor:
            """
            Replace function for SDXL cross-attention with bias.
            
            Args:
                q: Query from image (B*heads, img_len, head_dim) or (B, heads, img_len, head_dim)
                k: Key from text
                v: Value from text
                extra_options: Block info including n_heads, dim_head
            """
            if not config.should_apply_to_block(full_block_name):
                # Use standard attention without bias
                return _compute_standard_attention(q, k, v, extra_options)
            
            n_heads = extra_options.get("n_heads", 8)
            dim_head = extra_options.get("dim_head", 64)
            
            # Reshape if needed
            if q.dim() == 3:
                # ComfyUI passes q,k,v as (B, seq_len, n_heads*dim_head)
                # Reshape to (B, n_heads, seq_len, dim_head)
                batch_size = q.shape[0]
                img_len = q.shape[1]
                txt_len = k.shape[1]
                
                q = q.view(batch_size, img_len, n_heads, dim_head).transpose(1, 2)
                k = k.view(batch_size, txt_len, n_heads, dim_head).transpose(1, 2)
                v = v.view(batch_size, txt_len, n_heads, dim_head).transpose(1, 2)
            else:
                batch_size, _, img_len, _ = q.shape
                txt_len = k.shape[2]
            
            # Infer spatial size from img_len using latent_size aspect ratio
            img_h, img_w = self._infer_spatial_size(img_len)
            
            # Get bias for this resolution
            bias = self._get_bias_for_size(img_h, img_w, txt_len, q.device, q.dtype)
            
            # Compute attention
            scale = dim_head ** -0.5
            attn_weights = torch.einsum('bhid,bhjd->bhij', q, k) * scale
            
            # Apply bias
            if bias is not None:
                if bias.dim() == 3:
                    bias = bias.unsqueeze(1)  # Add heads dim
                attn_weights = attn_weights + bias
            
            attn_weights = F.softmax(attn_weights, dim=-1)
            out = torch.einsum('bhij,bhjd->bhid', attn_weights, v)
            
            # Reshape back
            out = out.transpose(1, 2).reshape(batch_size, img_len, n_heads * dim_head)
            
            return out
        
        return attn2_replace


def _compute_standard_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    extra_options: Dict,
) -> torch.Tensor:
    """Compute standard attention without bias (for blocks where bias is disabled)."""
    n_heads = extra_options.get("n_heads", 8)
    dim_head = extra_options.get("dim_head", 64)
    
    if q.dim() == 3:
        # ComfyUI passes q,k,v as (B, seq_len, n_heads*dim_head)
        # Reshape to (B, n_heads, seq_len, dim_head)
        batch_size = q.shape[0]
        seq_len_q = q.shape[1]
        seq_len_kv = k.shape[1]
        
        q = q.view(batch_size, seq_len_q, n_heads, dim_head).transpose(1, 2)
        k = k.view(batch_size, seq_len_kv, n_heads, dim_head).transpose(1, 2)
        v = v.view(batch_size, seq_len_kv, n_heads, dim_head).transpose(1, 2)
    else:
        batch_size = q.shape[0]
        seq_len_q = q.shape[2]
    
    scale = dim_head ** -0.5
    attn_weights = torch.einsum('bhid,bhjd->bhij', q, k) * scale
    attn_weights = F.softmax(attn_weights, dim=-1)
    out = torch.einsum('bhij,bhjd->bhid', attn_weights, v)
    out = out.transpose(1, 2).reshape(batch_size, seq_len_q, n_heads * dim_head)
    
    return out


def apply_attention_bias_patches(
    model_patcher,
    attention_bias: torch.Tensor,
    config: AttentionBiasConfig,
    txt_seq_len: int,
    model_type: str = "flux",
    lora_masks: Dict[str, torch.Tensor] = None,
    token_pos_maps: Dict[str, List[List[int]]] = None,
    latent_size: Tuple[int, int] = None,
):
    """
    Apply attention bias patches to a model.
    
    Args:
        model_patcher: ComfyUI ModelPatcher instance
        attention_bias: Pre-computed attention bias for Flux (ignored - use lora_masks instead)
        config: Attention bias configuration
        txt_seq_len: Text sequence length estimate (not used for Flux, actual length determined at runtime)
        model_type: "flux" or "sdxl"
        lora_masks: Spatial masks for each LoRA (flattened to img_seq_len)
        token_pos_maps: Token positions for each LoRA
        latent_size: Required for SDXL - (H, W) of latent space
    """
    if not config.enabled:
        logging.info("[FreeFuse] Attention bias disabled, skipping patches")
        return
    
    if model_type == "flux":
        _apply_flux_bias_patches(model_patcher, lora_masks, token_pos_maps, config)
    elif model_type == "sdxl":
        if lora_masks is None or token_pos_maps is None or latent_size is None:
            logging.warning("[FreeFuse] SDXL attention bias requires lora_masks, token_pos_maps, and latent_size")
            return
        _apply_sdxl_bias_patches(model_patcher, lora_masks, token_pos_maps, config, latent_size)
    else:
        logging.warning(f"[FreeFuse] Unknown model type for attention bias: {model_type}")


def _apply_flux_bias_patches(
    model_patcher,
    lora_masks: Dict[str, torch.Tensor],
    token_pos_maps: Dict[str, List[List[int]]],
    config: AttentionBiasConfig,
):
    """Apply attention bias patches for Flux model."""
    if lora_masks is None or not lora_masks:
        logging.warning("[FreeFuse] No LoRA masks provided for Flux attention bias")
        return
    
    if token_pos_maps is None or not token_pos_maps:
        logging.warning("[FreeFuse] No token position maps provided for Flux attention bias")
        return
    
    # Get model structure
    try:
        diffusion_model = model_patcher.model.diffusion_model
        num_double_blocks = len(diffusion_model.double_blocks)
        num_single_blocks = len(diffusion_model.single_blocks)
    except AttributeError:
        logging.warning("[FreeFuse] Could not access Flux model structure")
        return
    
    double_patches = 0
    single_patches = 0
    
    # Apply to double blocks
    for i in range(num_double_blocks):
        block_name = f"transformer_blocks.{i}"
        if config.should_apply_to_block(block_name):
            block = diffusion_model.double_blocks[i]
            replacer = FreeFuseFluxBiasBlockReplace(
                lora_masks=lora_masks,
                token_pos_maps=token_pos_maps,
                config=config,
                block_index=i,
                block=block,  # Pass actual block reference
            )
            model_patcher.set_model_patch_replace(
                replacer.create_block_replace(),
                "dit",
                "double_block",
                i,
            )
            double_patches += 1
    
    # Apply to single blocks
    for i in range(num_single_blocks):
        block_name = f"single_transformer_blocks.{i}"
        if config.should_apply_to_block(block_name):
            block = diffusion_model.single_blocks[i]
            replacer = FreeFuseFluxBiasSingleBlockReplace(
                lora_masks=lora_masks,
                token_pos_maps=token_pos_maps,
                config=config,
                block_index=i,
                block=block,  # Pass actual block reference
            )
            model_patcher.set_model_patch_replace(
                replacer.create_block_replace(),
                "dit",
                "single_block",
                i,
            )
            single_patches += 1
    
    logging.info(f"[FreeFuse] Applied attention bias to {double_patches} double blocks + {single_patches} single blocks")


def _apply_sdxl_bias_patches(
    model_patcher,
    lora_masks: Dict[str, torch.Tensor],
    token_pos_maps: Dict[str, List[List[int]]],
    config: AttentionBiasConfig,
    latent_size: Tuple[int, int],
):
    """Apply attention bias patches for SDXL model."""
    replacer = FreeFuseSDXLBiasAttnReplace(
        lora_masks=lora_masks,
        token_pos_maps=token_pos_maps,
        config=config,
        latent_size=latent_size,
    )
    
    patches_applied = 0
    
    # SDXL UNet structure: input_blocks, middle_block, output_blocks
    # We apply to cross-attention (attn2) in each
    
    # Define SDXL block structure (typical for SDXL)
    sdxl_blocks = [
        # Input blocks with cross-attention
        ("input", 1), ("input", 2),
        ("input", 4), ("input", 5),
        ("input", 7), ("input", 8),
        # Middle block
        ("middle", 0),
        # Output blocks with cross-attention
        ("output", 0), ("output", 1), ("output", 2),
        ("output", 3), ("output", 4), ("output", 5),
        ("output", 6), ("output", 7), ("output", 8),
    ]
    
    for block_name, block_num in sdxl_blocks:
        full_name = f"{block_name}.{block_num}"
        if config.should_apply_to_block(full_name):
            model_patcher.set_model_attn2_replace(
                replacer.create_attn_replace(block_name, block_num),
                block_name,
                block_num,
            )
            patches_applied += 1
    
    logging.info(f"[FreeFuse] Applied attention bias to {patches_applied} SDXL cross-attention blocks")
