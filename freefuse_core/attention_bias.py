"""
FreeFuse Attention Bias Module

Constructs soft attention bias matrices to encourage image tokens to attend to their
corresponding LoRA's text tokens and discourage attention to other LoRAs' text tokens.

This implements the core FreeFuse attention bias mechanism:
- Negative bias: Suppress cross-LoRA attention (image region A attending to LoRA B's tokens)
- Positive bias: Enhance same-LoRA attention (image region A attending to LoRA A's tokens)
- Bidirectional: Apply bias for both image→text and text→image directions

Mathematical formulation:
    Standard attention: softmax(QK^T / sqrt(d_k)) * V
    With bias: softmax(QK^T / sqrt(d_k) + bias) * V

Supports both Flux (joint attention) and SDXL (cross attention) architectures.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging


def construct_attention_bias(
    lora_masks: Dict[str, torch.Tensor],
    token_pos_maps: Dict[str, List[List[int]]],
    txt_seq_len: int,
    img_seq_len: int,
    bias_scale: float = 5.0,
    positive_bias_scale: float = 1.0,
    bidirectional: bool = True,
    use_positive_bias: bool = True,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Construct soft attention bias matrix to encourage image tokens to attend to their
    corresponding LoRA's text tokens and discourage attention to other LoRAs' text tokens.
    
    This is the core FreeFuse attention bias mechanism, ported from the diffusers implementation.
    
    Args:
        lora_masks: Dict mapping lora_name -> (B, img_seq_len) binary mask indicating
                    which image positions belong to this LoRA
        token_pos_maps: Dict mapping lora_name -> [[token positions in prompt], ...]
                       Token positions in the text embedding (e.g., T5 for Flux, CLIP for SDXL)
        txt_seq_len: Length of text sequence
        img_seq_len: Length of image sequence (H*W after packing for Flux, or spatial for SDXL)
        bias_scale: Strength of the NEGATIVE bias (larger = stronger suppression). Default 5.0.
        positive_bias_scale: Strength of the POSITIVE bias for same-LoRA attention.
                            Should typically be smaller than negative scale. Default 1.0.
        bidirectional: If True, also apply bias for text->image direction. Default True.
        use_positive_bias: If True, add positive bias for same-LoRA attention pairs,
                          in addition to negative bias for cross-LoRA pairs. Default True.
        device: Device for the output tensor
        dtype: Data type for the output tensor
        
    Returns:
        attention_bias: (B, txt_seq_len + img_seq_len, txt_seq_len + img_seq_len)
                       Soft bias values: positive for same-LoRA pairs (if use_positive_bias),
                       negative for cross-LoRA pairs, 0 for neutral pairs
                       
    Note:
        The bias matrix structure for Flux (joint attention):
        
        [txt_seq_len | img_seq_len] x [txt_seq_len | img_seq_len]
        
        ┌─────────────────┬─────────────────┐
        │  txt → txt      │  txt → img      │
        │  (no bias)      │  (bidirectional)│
        ├─────────────────┼─────────────────┤
        │  img → txt      │  img → img      │
        │  (main bias)    │  (no bias)      │
        └─────────────────┴─────────────────┘
    """
    if not lora_masks:
        logging.warning("[FreeFuse] No LoRA masks provided for attention bias construction")
        return None
    
    # Get batch size from first mask
    first_mask = next(iter(lora_masks.values()))
    if first_mask.dim() == 1:
        # Add batch dimension if missing
        batch_size = 1
        lora_masks = {k: v.unsqueeze(0) for k, v in lora_masks.items()}
    else:
        batch_size = first_mask.shape[0]
    
    # Infer device and dtype from masks if not provided
    if device is None:
        device = first_mask.device
    if dtype is None:
        dtype = first_mask.dtype if first_mask.dtype.is_floating_point else torch.float32
    
    total_seq_len = txt_seq_len + img_seq_len
    
    # Initialize bias as zeros (no bias = attend freely)
    attention_bias = torch.zeros(
        batch_size, total_seq_len, total_seq_len,
        device=device, dtype=dtype
    )
    
    # Build a mapping: for each text token position, which LoRA does it belong to?
    # -1 means no LoRA (shared/common tokens like articles, punctuation)
    text_token_to_lora = torch.full((txt_seq_len,), -1, device=device, dtype=torch.long)
    
    # Create name to index mapping (exclude special keys like __background__)
    lora_names = [name for name in lora_masks.keys() if not name.startswith("_")]
    lora_name_to_idx = {name: idx for idx, name in enumerate(lora_names)}
    
    # Populate text_token_to_lora mapping
    for lora_name, positions_list in token_pos_maps.items():
        if lora_name not in lora_name_to_idx:
            continue
        lora_idx = lora_name_to_idx[lora_name]
        
        # positions_list is [[positions for batch 0], [positions for batch 1], ...]
        # For simplicity, use first batch's positions (assuming same across batches)
        if len(positions_list) > 0 and len(positions_list[0]) > 0:
            for pos in positions_list[0]:
                if 0 <= pos < txt_seq_len:
                    text_token_to_lora[pos] = lora_idx
    
    # For each LoRA, get its image mask and text token positions
    for lora_name, img_mask in lora_masks.items():
        if lora_name not in lora_name_to_idx:
            continue
        lora_idx = lora_name_to_idx[lora_name]
        
        # Ensure img_mask has correct shape (B, img_seq_len)
        if img_mask.dim() == 2 and img_mask.shape[1] != img_seq_len:
            # Might be (H, W), flatten it
            img_mask = img_mask.view(batch_size, -1)
        elif img_mask.dim() == 1:
            img_mask = img_mask.unsqueeze(0).expand(batch_size, -1)
        
        # Ensure mask is on correct device
        img_mask = img_mask.to(device=device, dtype=dtype)
        
        # === NEGATIVE BIAS: Cross-LoRA suppression ===
        # For image positions in this LoRA's region,
        # add negative bias to text tokens belonging to OTHER LoRAs
        
        # Create mask for text tokens belonging to other LoRAs (not this one, not shared)
        other_lora_text_mask = (text_token_to_lora != lora_idx) & (text_token_to_lora != -1)
        other_lora_text_mask = other_lora_text_mask.float()  # (txt_seq_len,)
        
        # Image->Text bias: 
        # attention_bias[b, txt_seq_len + i, j] for image position i, text position j
        # We want: bias[b, txt_seq_len+i, j] -= scale * img_mask[b, i] * other_lora_text_mask[j]
        
        # Outer product: (B, img_seq_len, 1) * (1, 1, txt_seq_len) -> (B, img_seq_len, txt_seq_len)
        img_to_txt_bias = img_mask.unsqueeze(-1) * other_lora_text_mask.unsqueeze(0).unsqueeze(0)
        img_to_txt_bias = img_to_txt_bias * (-bias_scale)
        
        # Add to attention_bias at image->text positions
        attention_bias[:, txt_seq_len:, :txt_seq_len] += img_to_txt_bias
        
        # === POSITIVE BIAS: Same-LoRA enhancement ===
        if use_positive_bias:
            # this_lora_text_mask: (txt_seq_len,) - 1 for tokens belonging to this LoRA
            this_lora_text_mask = (text_token_to_lora == lora_idx).float()
            
            # img_to_txt_positive_bias[b, i, j] = img_mask[b, i] * this_lora_text_mask[j] * (+scale)
            img_to_txt_positive_bias = img_mask.unsqueeze(-1) * this_lora_text_mask.unsqueeze(0).unsqueeze(0)
            img_to_txt_positive_bias = img_to_txt_positive_bias * positive_bias_scale
            
            attention_bias[:, txt_seq_len:, :txt_seq_len] += img_to_txt_positive_bias
        
        # === BIDIRECTIONAL: Text->Image bias ===
        if bidirectional:
            # For text tokens belonging to this LoRA,
            # suppress attention to image positions NOT in this LoRA's mask
            
            this_lora_text_mask = (text_token_to_lora == lora_idx).float()
            
            # not_this_lora_img_mask: (B, img_seq_len) - 1 for positions NOT in this LoRA
            not_this_lora_img_mask = 1.0 - img_mask
            
            # txt_to_img_bias[b, j, i] = this_lora_text_mask[j] * not_this_lora_img_mask[b, i] * (-scale)
            # Shape: (1, txt_seq_len, 1) * (B, 1, img_seq_len) -> (B, txt_seq_len, img_seq_len)
            txt_to_img_bias = this_lora_text_mask.unsqueeze(0).unsqueeze(-1) * not_this_lora_img_mask.unsqueeze(1)
            txt_to_img_bias = txt_to_img_bias * (-bias_scale)
            
            # Add to attention_bias at text->image positions
            attention_bias[:, :txt_seq_len, txt_seq_len:] += txt_to_img_bias
            
            # Positive bias: encourage this LoRA's text tokens to attend to this LoRA's image region
            if use_positive_bias:
                txt_to_img_positive_bias = this_lora_text_mask.unsqueeze(0).unsqueeze(-1) * img_mask.unsqueeze(1)
                txt_to_img_positive_bias = txt_to_img_positive_bias * positive_bias_scale
                
                attention_bias[:, :txt_seq_len, txt_seq_len:] += txt_to_img_positive_bias
    
    logging.info(f"[FreeFuse] Constructed attention bias: shape={attention_bias.shape}, "
                f"bias_scale={bias_scale}, positive_scale={positive_bias_scale}, "
                f"bidirectional={bidirectional}, positive_bias={use_positive_bias}")
    
    return attention_bias


def construct_attention_bias_sdxl(
    lora_masks: Dict[str, torch.Tensor],
    token_pos_maps: Dict[str, List[List[int]]],
    txt_seq_len: int,
    img_h: int,
    img_w: int,
    bias_scale: float = 5.0,
    positive_bias_scale: float = 1.0,
    use_positive_bias: bool = True,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Construct attention bias for SDXL cross-attention.
    
    SDXL uses separate cross-attention (Q from image, K/V from text), so the bias
    structure is simpler: (B, img_seq_len, txt_seq_len) for each cross-attention layer.
    
    Args:
        lora_masks: Dict mapping lora_name -> (B, H, W) or (H, W) spatial masks
        token_pos_maps: Dict mapping lora_name -> [[token positions], ...]
        txt_seq_len: Length of text sequence (77 for CLIP)
        img_h, img_w: Spatial dimensions of the feature map at this layer
        bias_scale: Strength of negative bias
        positive_bias_scale: Strength of positive bias
        use_positive_bias: Whether to add positive bias
        device: Device for output tensor
        dtype: Data type for output tensor
        
    Returns:
        attention_bias: (B, img_h*img_w, txt_seq_len) bias matrix for cross-attention
    """
    if not lora_masks:
        return None
    
    img_seq_len = img_h * img_w
    
    # Get batch size and device from first mask
    first_mask = next(iter(lora_masks.values()))
    if device is None:
        device = first_mask.device
    if dtype is None:
        dtype = first_mask.dtype if first_mask.dtype.is_floating_point else torch.float32
    
    # Determine batch size
    if first_mask.dim() == 2:
        batch_size = 1
    else:
        batch_size = first_mask.shape[0]
    
    # Initialize bias
    attention_bias = torch.zeros(
        batch_size, img_seq_len, txt_seq_len,
        device=device, dtype=dtype
    )
    
    # Build text token to LoRA mapping
    text_token_to_lora = torch.full((txt_seq_len,), -1, device=device, dtype=torch.long)
    lora_names = [name for name in lora_masks.keys() if not name.startswith("_")]
    lora_name_to_idx = {name: idx for idx, name in enumerate(lora_names)}
    
    for lora_name, positions_list in token_pos_maps.items():
        if lora_name not in lora_name_to_idx:
            continue
        lora_idx = lora_name_to_idx[lora_name]
        if len(positions_list) > 0 and len(positions_list[0]) > 0:
            for pos in positions_list[0]:
                if 0 <= pos < txt_seq_len:
                    text_token_to_lora[pos] = lora_idx
    
    # Process each LoRA
    for lora_name, mask in lora_masks.items():
        if lora_name not in lora_name_to_idx:
            continue
        lora_idx = lora_name_to_idx[lora_name]
        
        # Resize mask to current feature map size
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # Add batch dim
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # Add channel dim for interpolation
        
        # Resize to (B, 1, img_h, img_w)
        mask_resized = F.interpolate(
            mask.float(),
            size=(img_h, img_w),
            mode='bilinear',
            align_corners=False
        )
        # Flatten to (B, img_seq_len)
        img_mask = mask_resized.view(batch_size, -1).to(device=device, dtype=dtype)
        
        # Negative bias: suppress attention to other LoRAs' tokens
        other_lora_text_mask = (text_token_to_lora != lora_idx) & (text_token_to_lora != -1)
        other_lora_text_mask = other_lora_text_mask.float()
        
        cross_bias = img_mask.unsqueeze(-1) * other_lora_text_mask.unsqueeze(0).unsqueeze(0)
        attention_bias += cross_bias * (-bias_scale)
        
        # Positive bias
        if use_positive_bias:
            this_lora_text_mask = (text_token_to_lora == lora_idx).float()
            pos_bias = img_mask.unsqueeze(-1) * this_lora_text_mask.unsqueeze(0).unsqueeze(0)
            attention_bias += pos_bias * positive_bias_scale
    
    return attention_bias


class AttentionBiasConfig:
    """Configuration for attention bias application."""
    
    def __init__(
        self,
        enabled: bool = True,
        bias_scale: float = 5.0,
        positive_bias_scale: float = 1.0,
        bidirectional: bool = True,
        use_positive_bias: bool = True,
        apply_to_blocks: Optional[List[str]] = None,
    ):
        """
        Args:
            enabled: Whether attention bias is enabled
            bias_scale: Strength of negative bias
            positive_bias_scale: Strength of positive bias
            bidirectional: Apply bias in both directions (Flux only)
            use_positive_bias: Whether to use positive bias
            apply_to_blocks: List of block names to apply bias to.
                           None = all blocks
                           For Flux: ["transformer_blocks.0", "transformer_blocks.18", ...]
                           For SDXL: ["input.0", "middle.0", "output.0", ...]
                           Special presets: "double_stream_only", "single_stream_only", 
                                          "last_half_double", "all"
        """
        self.enabled = enabled
        self.bias_scale = bias_scale
        self.positive_bias_scale = positive_bias_scale
        self.bidirectional = bidirectional
        self.use_positive_bias = use_positive_bias
        self.apply_to_blocks = apply_to_blocks
        
    def should_apply_to_block(self, block_name: str) -> bool:
        """Check if bias should be applied to a specific block."""
        if not self.enabled:
            return False
        
        if self.apply_to_blocks is None:
            return True
        
        # Handle presets
        if isinstance(self.apply_to_blocks, str):
            preset = self.apply_to_blocks
            if preset == "all":
                return True
            elif preset == "double_stream_only":
                return "transformer_blocks" in block_name and "single" not in block_name
            elif preset == "single_stream_only":
                return "single_transformer_blocks" in block_name
            elif preset == "last_half_double":
                if "transformer_blocks" not in block_name or "single" in block_name:
                    return False
                # Extract block number
                try:
                    block_num = int(block_name.split(".")[-1])
                    return block_num >= 10  # Last half of 19 blocks
                except ValueError:
                    return False
            else:
                return block_name == preset
        
        # List of specific blocks
        return block_name in self.apply_to_blocks
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage in model options."""
        return {
            "enabled": self.enabled,
            "bias_scale": self.bias_scale,
            "positive_bias_scale": self.positive_bias_scale,
            "bidirectional": self.bidirectional,
            "use_positive_bias": self.use_positive_bias,
            "apply_to_blocks": self.apply_to_blocks,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "AttentionBiasConfig":
        """Create from dictionary."""
        return cls(**d)


def apply_attention_bias_to_weights(
    attn_weights: torch.Tensor,
    attention_bias: torch.Tensor,
    txt_len: int = None,
    is_cross_attention: bool = False,
) -> torch.Tensor:
    """
    Apply attention bias to pre-softmax attention weights.
    
    Args:
        attn_weights: Pre-softmax attention weights
                     For Flux: (B, heads, seq_len, seq_len) where seq = txt + img
                     For SDXL cross-attn: (B, heads, img_len, txt_len)
        attention_bias: Bias matrix to add
                       For Flux: (B, seq_len, seq_len) or (B, 1, seq_len, seq_len)
                       For SDXL: (B, img_len, txt_len) or (B, 1, img_len, txt_len)
        txt_len: Text sequence length (for logging/validation)
        is_cross_attention: Whether this is cross-attention (SDXL) or joint (Flux)
        
    Returns:
        Modified attention weights with bias applied
    """
    if attention_bias is None:
        return attn_weights
    
    # Ensure bias has head dimension
    if attention_bias.dim() == 3:
        attention_bias = attention_bias.unsqueeze(1)  # (B, 1, seq, seq)
    
    # Broadcast and add
    # The bias will be broadcast across heads
    return attn_weights + attention_bias


def get_attention_bias_for_layer(
    full_bias: torch.Tensor,
    layer_img_size: Tuple[int, int],
    latent_size: Tuple[int, int],
    txt_seq_len: int,
    model_type: str = "flux",
) -> torch.Tensor:
    """
    Get attention bias resized for a specific layer's feature map size.
    
    For SDXL, different layers have different spatial resolutions, so we need
    to resize the bias accordingly.
    
    Args:
        full_bias: Full attention bias at latent resolution
        layer_img_size: (H, W) of the current layer's feature map
        latent_size: (H, W) of the latent space
        txt_seq_len: Text sequence length
        model_type: "flux" or "sdxl"
        
    Returns:
        Attention bias resized for this layer
    """
    if full_bias is None:
        return None
    
    if model_type == "flux":
        # Flux uses packed latents, all layers have same sequence length
        return full_bias
    
    # SDXL: need to resize spatial component
    latent_h, latent_w = latent_size
    layer_h, layer_w = layer_img_size
    
    if latent_h == layer_h and latent_w == layer_w:
        return full_bias
    
    B = full_bias.shape[0]
    latent_img_len = latent_h * latent_w
    layer_img_len = layer_h * layer_w
    
    # Extract image->text portion and resize
    # full_bias: (B, latent_img_len, txt_seq_len)
    bias_spatial = full_bias.view(B, latent_h, latent_w, txt_seq_len)
    bias_spatial = bias_spatial.permute(0, 3, 1, 2)  # (B, txt, H, W)
    
    bias_resized = F.interpolate(
        bias_spatial,
        size=(layer_h, layer_w),
        mode='bilinear',
        align_corners=False
    )
    
    bias_resized = bias_resized.permute(0, 2, 3, 1)  # (B, H, W, txt)
    bias_resized = bias_resized.reshape(B, layer_img_len, txt_seq_len)
    
    return bias_resized
