# LTX-Video Architecture Notes

## Key Discovery: Self-Attention Style with Projection

**The ComfyUI LTX-Video implementation uses self-attention style where both video and text are projected to 4096 dimensions before attention.**

This is different from standard cross-attention where `to_k`/`to_v` handle different input dimensions.

---

## Architecture Comparison

### LTX-Video (ComfyUI Implementation - Self-Attention Style)

```
BasicAVTransformerBlock
└── Attention module
    ├── to_q: Linear(4096, 4096)    ← Video query projection
    ├── to_k: Linear(4096, 4096)    ← Key projection (expects 4096!)
    ├── to_v: Linear(4096, 4096)    ← Value projection
    ├── q_norm: RMSNorm             ← Query normalization
    └── k_norm: RMSNorm             ← Key normalization

Text Projection (learned, created on-the-fly):
    └── text_proj_k: Linear(2048, 4096)  ← Projects text to video dim
```

**Forward pass:**
```python
# Video stream (query) - 4096 dim
q = to_q(video_hidden)      # (B, 1024, 4096) → (B, 1024, 4096)

# Text stream needs projection (2048 → 4096)
text_proj = Linear(2048, 4096)  # Created on first use
text_projected = text_proj(text_hidden)  # (B, 1024, 2048) → (B, 1024, 4096)
k = to_k(text_projected)           # (B, 1024, 4096) → (B, 1024, 4096) ✓
v = to_v(text_projected)           # (B, 1024, 4096) → (B, 1024, 4096) ✓

# Cross-attention: video queries attend to projected text keys
attention = softmax(q @ k^T / sqrt(128)) @ v
```

### Original LTX-Video (Cross-Attention Mode)

```
Attention module
└── to_q: Linear(4096, 4096)       ← Video query
    to_k: Linear(2048, 4096)       ← Text key (different input dim!)
    to_v: Linear(2048, 4096)       ← Text value (different input dim!)
```

**Note:** The original LTX-Video paper describes cross-attention with different input dimensions, but the ComfyUI implementation uses self-attention style.

---

## Key Differences

| Aspect | LTX-Video (ComfyUI) | Original LTX-Video | Qwen-Image (MMDiT) |
|--------|-------------------|-------------------|-------------------|
| **Attention type** | Self-attention style | Cross-attention | MMDiT |
| **to_k input dim** | 4096 | 2048 | 4096 |
| **to_v input dim** | 4096 | 2048 | 4096 |
| **Text handling** | Learned projection (2048→4096) | Direct (to_k handles 2048) | Separate add_*_proj |
| **Projection layers** | Created on-the-fly | Built into to_k/to_v | Separate layers |
| **Norm layers** | `q_norm`, `k_norm` | `q_norm`, `k_norm` | `norm_q`, `norm_k`, `norm_added_*` |

---

## Implementation for FreeFuse

### LTX-Video Similarity Computation (Final Implementation)

```python
# Check if MMDiT or CrossAttention
is_mmdit = (hasattr(attn, 'add_q_proj') and 
            hasattr(attn, 'add_k_proj') and 
            hasattr(attn, 'add_v_proj'))

if is_mmdit:
    # Qwen-Image style: separate projections
    q = to_q(img_attn_in)
    k = to_k(img_attn_in)
    txt_q = add_q_proj(txt_attn_in)
    txt_k = add_k_proj(txt_attn_in)
else:
    # LTX-Video style (ComfyUI): self-attention with projection
    q = to_q(img_attn_in)  # Video query: (B, img_seq, 4096)
    txt_q = to_q(text_proj(txt_attn_in))  # Project text first
    
    # Try direct cross-attention first, fall back to projection
    try:
        k = to_k(txt_attn_in)  # Try with text directly
    except RuntimeError:
        # to_k expects 4096 dim - create projection
        if not hasattr(self, 'text_proj_k'):
            self.text_proj_k = torch.nn.Linear(2048, 4096, bias=False)
        k = to_k(self.text_proj_k(txt_attn_in))
```

---

## Block Parameters (LTX-Video)

From the ComfyUI logs:
```
[LTXAttentionHook] Captured LTX: 
  v_context=torch.Size([2, 1024, 4096])  ← Video: (B, seq, dim)
  a_context=torch.Size([2, 1024, 2048])  ← Text: (B, seq, dim)

[LTXAttentionHook] Attention module type: CrossAttention
[LTXAttentionHook] Attention attrs: ['to_q', 'to_k', 'to_v', 'q_norm', 'k_norm']

[LTXAttentionHook] Self-attention mode: to_k expects image dim
[LTXAttentionHook] Projecting text from 2048 to 4096
[LTXAttentionHook] Text projected and processed successfully
```

**Note:** The block receives text at 2048 dim, but the attention `to_k` layer expects 4096 dim. Our implementation creates learned projection layers automatically.

---

## Debugging Tips

### Check Attention Type

```python
# In the hook
is_mmdit = (hasattr(attn, 'add_q_proj') and 
            hasattr(attn, 'add_k_proj'))

if is_mmdit:
    print("Using MMDiT architecture (Qwen-Image)")
else:
    print("Using CrossAttention (LTX-Video)")
    
# Try to determine if projection is needed
try:
    k_test = to_k(txt_hidden)  # Try with text dim
    print("Cross-attention mode: to_k accepts text dim")
except RuntimeError:
    print("Self-attention mode: to_k expects image dim")
    print("Need projection layer (2048 → 4096)")
```

### Log Attention Structure

```python
attn_attrs = [a for a in dir(attn) if not a.startswith('_') and 
              ('q' in a.lower() or 'k' in a.lower() or 'v' in a.lower() or 'norm' in a.lower())]
print(f"Attention attrs: {attn_attrs}")

# LTX-Video (ComfyUI): ['to_q', 'to_k', 'to_v', 'q_norm', 'k_norm']
# Qwen-Image: ['to_q', 'to_k', 'to_v', 'add_q_proj', 'add_k_proj', 'add_v_proj', 
#              'norm_q', 'norm_k', 'norm_added_q', 'norm_added_k']
```

### Check Dimensions

```python
print(f"Video hidden: {img_hidden.shape}")   # (B, seq, 4096)
print(f"Text hidden: {txt_hidden.shape}")    # (B, seq, 2048)

# After projection
if not hasattr(self, 'text_proj_k'):
    self.text_proj_k = torch.nn.Linear(2048, 4096, bias=False)
txt_projected = self.text_proj_k(txt_hidden)
k = to_k(txt_projected)

print(f"Q shape: {q.shape}")  # (B, seq, 4096)
print(f"K shape: {k.shape}")  # (B, seq, 4096)
```

---

## References

- LTX-Video GitHub: https://github.com/Lightricks/LTX-2
- LTX-Video HuggingFace: https://huggingface.co/Lightricks/LTX-Video
- LTX-Video Attention code: https://github.com/Lightricks/LTX-Video/blob/main/ltx_video/models/transformers/attention.py

---

## Changelog

- **2026-03-14**: Complete end-to-end workflow tested
  - ✅ Manual mask workflow (MaskTap + MaskReassemble + MaskApplicator)
  - ✅ Attention bias for LTX-Video (48 transformer blocks)
  - ✅ Full video generation (10 frames, tiled VAE decode)
  - ✅ Execution time: ~127 seconds
  - ✅ VRAM usage: 4-5 GB with low_vram_mode

- **2026-03-14**: Discovered ComfyUI implementation details
  - ✅ Identified self-attention style (to_k expects 4096 dim)
  - ✅ Implemented learned text projection (2048 → 4096)
  - ✅ Added automatic projection layer creation
  - ✅ Successfully extracted similarity maps
  - ✅ Fixed preview generation for different resolutions

- **2026-03-14**: Initial implementation
  - ✅ LTX model type detection
  - ✅ Gemma 3 tokenizer support
  - ✅ CrossAttention hook implementation
  - ✅ Documentation created (README, ARCHITECTURE, WORKFLOW)

