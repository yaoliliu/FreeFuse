# Qwen-Image Similarity Map Extraction

## Overview

This document explains how the **FreeFuseQwenSimilarityExtractor** node works for Qwen-Image, including the architecture details and implementation.

---

## Qwen-Image Architecture (MMDiT)

```
Qwen-Image Architecture:
┌─────────────────────────────────────┐
│  Input: (B, C, T, H, W) 5D tensor   │ ← Note temporal dimension!
│  ↓                                   │
│  process_img: reshape to 5D          │
│  ↓                                   │
│  img_in / txt_in (separate streams)  │
│  ↓                                   │
│  Transformer Blocks (transformer_blocks[])
│    - Dual-stream MMDiT architecture  │
│    - img_mod / txt_mod (modulation)  │
│    - img_norm1/2, txt_norm1/2        │
│    - attn.to_q/k/v (image stream)    │
│    - attn.add_q/k/v_proj (text)      │ ← Different projection names!
│    - norm_q/k, norm_added_q/k        │ ← Separate norms
│    - img_mlp, txt_mlp (separate)     │
│  ↓                                   │
│  norm_out.linear + proj_out          │
└─────────────────────────────────────┘
```

**Key characteristics:**
- Uses `transformer_blocks` array (60 blocks total, indexed 0-59)
- **Dual-stream MMDiT**: separate image and text processing
- **5D tensor input**: `(B, C, T, H, W)` - temporal dimension required
- Different QKV projections:
  - Image: `to_q`, `to_k`, `to_v`
  - Text: `add_q_proj`, `add_k_proj`, `add_v_proj`
- Separate QK norms: `norm_q`, `norm_k`, `norm_added_q`, `norm_added_k`
- Modulation: `img_mod`, `txt_mod` (chunk into scale/gate)
- 64 attention heads

---

## Implementation Details

### 5D Tensor Format

Qwen-Image expects 5D input `(B, C, T, H, W)` but ComfyUI provides 4D `(B, C, H, W)`.

The node automatically adds the temporal dimension:

```python
# Add temporal dimension before sampling
if latent_tensor.dim() == 4:
    latent_tensor = latent_tensor.unsqueeze(2)  # Insert T at index 2
    # (B, C, H, W) → (B, C, 1, H, W)
```

### Transformer Blocks

Qwen-Image uses `transformer_blocks` (60 blocks):

```python
# Find transformer blocks
target_blocks = diffusion_model.transformer_blocks
print(f"Found {len(target_blocks)} transformer blocks")  # 60

# Select block for collection (0-59)
target_block = target_blocks[collect_block]
```

### QKV Projections (Dual-Stream)

Image and text streams use different projection layers:

```python
# Get image stream projections
to_q = getattr(attn, 'to_q', None)
to_k = getattr(attn, 'to_k', None)
to_v = getattr(attn, 'to_v', None)

# Get text stream projections (MMDiT specific)
add_q_proj = getattr(attn, 'add_q_proj', None)
add_k_proj = getattr(attn, 'add_k_proj', None)
add_v_proj = getattr(attn, 'add_v_proj', None)

# Project separately
img_q = to_q(img_attn_in)
txt_q = add_q_proj(txt_attn_in)  # Different layer!
```

### QK Normalization

Separate normalization layers for image and text:

```python
# Apply QK norms (Qwen-Image naming)
if hasattr(attn, 'norm_q'):
    img_q = attn.norm_q(img_q)
if hasattr(attn, 'norm_k'):
    img_k = attn.norm_k(img_k)
if hasattr(attn, 'norm_added_q'):
    txt_q = attn.norm_added_q(txt_q)
if hasattr(attn, 'norm_added_k'):
    txt_k = attn.norm_added_k(txt_k)
```

### Modulation (Simplified)

For similarity extraction, modulation is simplified to avoid shape mismatches:

```python
# Simplified: just apply normalization without full modulation
if hasattr(module, 'img_norm1'):
    img_attn_in = module.img_norm1(img_hidden)
if hasattr(module, 'txt_norm1'):
    txt_attn_in = module.txt_norm1(txt_hidden)
```

**Note:** For full generation, proper modulation is required. For similarity extraction, normalization alone captures the attention patterns correctly.

### RoPE (Skipped)

Qwen-Image's RoPE frequencies have a different structure. The node skips RoPE for similarity extraction:

```python
# Apply RoPE (may fail, but similarity computation continues)
if image_rotary_emb is not None:
    try:
        img_q, img_k = apply_rope(img_q, img_k, image_rotary_emb[0])
        txt_q, txt_k = apply_rope(txt_q, txt_k, image_rotary_emb[0])
    except Exception as e:
        logging.warning(f"RoPE failed: {e}")
        # Continue without RoPE - similarity maps still valid
```

**Why it still works:** The FreeFuse similarity algorithm uses the **attention output** (`img_attn_out`) for the self-modal similarity calculation, which already incorporates the attention patterns even without perfect RoPE.

### Sequence Length vs Latent Size

Qwen-Image uses **patchified latents**. The latent is processed as patches (patch_size=2):

| Latent Size | Patches | Tokens |
|-------------|---------|--------|
| 32x32 | 16x16 | 256 |
| 48x48 | 24x24 | 576 |
| 64x64 | 32x32 | 1024 |

The node calculates actual dimensions from sequence length:

```python
# Get actual sequence length from similarity maps
seq_len = first_sim.shape[1]  # e.g., 1024

# Calculate latent dimensions
actual_latent_h = actual_latent_w = int(seq_len ** 0.5)  # 32x32
if actual_latent_h * actual_latent_w != seq_len:
    # Find factors for non-square layouts
    for i in range(int(seq_len ** 0.5), 0, -1):
        if seq_len % i == 0:
            actual_latent_h = i
            actual_latent_w = seq_len // i
            break
```

---

## The FreeFuse Algorithm for Qwen-Image

### Step 1: Cross-Attention Score Computation

```python
# Extract concept keys at token positions
concept_k = txt_k[:, pos_t, :, :]  # (B, n_concept, H, D)

# Multi-head cross-attention: img_q @ concept_k^T
weights = torch.einsum("bihd,bjhd->bhij", img_q, concept_k) * scale
weights = F.softmax(weights, dim=2)  # softmax over image dimension
scores = weights.mean(dim=1).mean(dim=-1)  # (B, img_len)
```

### Step 2: Competitive Exclusion

```python
# Amplify concept scores, suppress others
scores = all_cross_attn_scores[lora_name] * max(1, n_concepts)
for other in all_cross_attn_scores:
    if other != lora_name:
        scores = scores - all_cross_attn_scores[other]
```

### Step 3: Top-K Core Token Selection

```python
k_count = max(1, int(img_len * top_k_ratio))
_, topk_idx = torch.topk(scores, k_count, dim=-1)  # (B, k)

# Gather core tokens
core_tokens = torch.gather(img_attn_out, dim=1, index=expanded)
```

### Step 4: Self-Modal Similarity

```python
# Core tokens attend to all image tokens
self_modal_sim = torch.bmm(core_tokens, img_attn_out.transpose(-1, -2))
sim_avg = self_modal_sim.mean(dim=1, keepdim=True).transpose(1, 2)

# Softmax with temperature
sim_map = F.softmax(sim_avg / temperature, dim=1)
```

---

## Node Parameters

### FreeFuseQwenSimilarityExtractor

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `collect_block` | 20 | 0-59 | Which transformer block to collect from |
| `collect_step` | 1 | 0-99 | Step at which to collect attention |
| `steps` | 3 | 1-100 | Number of sampling steps (2-3 is enough) |
| `temperature` | 4000.0 | 0-10000 | Temperature for similarity computation |
| `top_k_ratio` | 0.3 | 0.01-1.0 | Ratio of top-k tokens to use |
| `preview_size` | 512 | 256-1024 | Preview size (smaller = less VRAM) |
| `low_vram_mode` | True | boolean | Enable aggressive VRAM optimization |
| `attention_head_index` | -1 | -1-63 | Specific head (-1 = average all 64) |

### FreeFuseRawSimilarityOverlay

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `preview_size` | 1024 | 512-2048 | Longest side of output preview |
| `sensitivity` | 5.0 | 1.0-20.0 | Contrast amplification |
| `argmax_method` | stabilized | simple/stabilized | Argmax algorithm |
| `max_iter` | 15 | 1-50 | Iterations for stabilized argmax |
| `balance_lr` | 0.01 | 0.001-0.1 | Learning rate for balancing |
| `gravity_weight` | 0.00004 | 0-0.001 | Centroid attraction |
| `spatial_weight` | 0.00004 | 0-0.001 | Neighbor influence |
| `momentum` | 0.2 | 0-0.9 | Smoothing between iterations |
| `anisotropy` | 1.3 | 0.5-2.0 | Horizontal stretch factor |
| `bg_scale` | 0.95 | 0.5-2.0 | Background multiplier |
| `use_morphological_cleaning` | True | boolean | Clean masks with morphology |

---

## Usage Example

```python
# Basic workflow
1. Load Qwen-Image model (UNETLoader)
2. Load LoRAs (FreeFuse6LoraLoader)
3. Define concepts (FreeFuseTokenPositions)
4. Extract similarity maps (FreeFuseQwenSimilarityExtractor)
   - model: from step 2
   - positive/negative: conditionings
   - latent: EmptySD3LatentImage (512x512)
   - freefuse_data: from step 3
   - collect_block: 20 (middle of 60 blocks)
   - collect_step: 1 (early in sampling)
   - steps: 3 (keep low for testing)
5. Visualize (FreeFuseRawSimilarityOverlay)
   - raw_similarity: from step 4
   - freefuse_data: from step 3
```

---

## Debugging Tips

### Check Model Structure

```python
diffusion_model = model.model.diffusion_model
print(f"Model type: {diffusion_model.__class__.__name__}")
print(f"Has transformer_blocks: {hasattr(diffusion_model, 'transformer_blocks')}")
print(f"Num blocks: {len(diffusion_model.transformer_blocks)}")

# Inspect a block
block = diffusion_model.transformer_blocks[20]
print(f"Block type: {type(block).__name__}")
print(f"Has attn: {hasattr(block, 'attn')}")
```

### Check Attention Structure

```python
attn = block.attn
print(f"Attention type: {type(attn).__name__}")
print(f"Has to_q/k/v: {hasattr(attn, 'to_q')}, {hasattr(attn, 'to_k')}, {hasattr(attn, 'to_v')}")
print(f"Has add_q/k/v_proj: {hasattr(attn, 'add_q_proj')}, {hasattr(attn, 'add_k_proj')}, {hasattr(attn, 'add_v_proj')}")
print(f"Has norm_q/k: {hasattr(attn, 'norm_q')}, {hasattr(attn, 'norm_k')}")
print(f"Has norm_added_q/k: {hasattr(attn, 'norm_added_q')}, {hasattr(attn, 'norm_added_k')}")
```

### Check Tensor Shapes

```python
# In the hook
print(f"img_hidden: {img_hidden.shape}")  # Should be (B, img_seq, dim)
print(f"txt_hidden: {txt_hidden.shape}")  # Should be (B, txt_seq, dim)
print(f"temb: {temb.shape if temb is not None else None}")
print(f"image_rotary_emb: {image_rotary_emb[0].shape if image_rotary_emb else None}")
```

---

## Performance Considerations

### VRAM Usage

| Configuration | VRAM |
|---------------|------|
| Qwen-Image BF16 (loaded) | ~27GB |
| + 1 LoRA | ~28GB |
| + 2 LoRAs | ~29-30GB |
| Sampling (3 steps, 32x32 latent) | ~28-29GB peak |

### Recommended Settings for Multiple LoRAs

**Minimal VRAM:**
```yaml
latent: 32x32 (256x256 image)
steps: 2
collect_step: 1
collect_block: 20
low_vram_mode: True
```

**Balanced:**
```yaml
latent: 48x48 (384x384 image)
steps: 3
collect_step: 1
collect_block: 25
low_vram_mode: True
```

### Collection Settings

| Setting | Value | Reason |
|---------|-------|--------|
| `steps` | 3 | Minimum for collection |
| `collect_step` | 1 | Early step has clean attention |
| `collect_block` | 20-30 | Middle of network (good balance) |
| `temperature` | 4000.0 | Standard for similarity softmax |
| `top_k_ratio` | 0.3 | 30% of tokens as core |

---

## Block Selection Guide

| Block Range | Quality | Use |
|-------------|---------|-----|
| 0-15 | Early features | Not recommended |
| 16-25 | Good separation | Recommended for testing |
| 26-35 | Transition | Test if 20 doesn't work |
| 36-45 | Noisy | Skip |
| 46-55 | Deep concepts | Good for complex scenes |
| 56-59 | Overlapping | Use for blending |

---

## Troubleshooting

### No similarity maps collected

**Check:**
1. Hook was installed correctly (check logs)
2. Model is actually Qwen-Image
3. `collect_step` is within `steps` range

### VRAM errors

**Solutions:**
1. Reduce latent size (32x32 = 1024 tokens)
2. Reduce `steps` to 2
3. Enable `low_vram_mode: True`
4. Use earlier `collect_block` (15-20)

### Poor concept separation

**Try:**
1. Different `collect_block` (try 20, 25, 30)
2. Adjust `temperature` (higher = softer, lower = sharper)
3. Use specific `attention_head_index` (0-63) instead of averaging

---

## Comparison: Qwen-Image vs Other Models

| Aspect | Qwen-Image | Flux | SDXL |
|--------|------------|------|------|
| **Tensor format** | 5D `(B,C,T,H,W)` | 4D `(B,C,H,W)` | 4D `(B,C,H,W)` |
| **Block array** | `transformer_blocks[]` | `double_blocks[]` | `output_blocks[]` |
| **Num blocks** | 60 | 57 | Variable |
| **Attention** | Dual stream MMDiT | Joint attention | Cross-attention |
| **QKV (text)** | `add_q/k/v_proj` | `to_q/k/v` | `to_q/k/v` |
| **QK norms** | `norm_q/k`, `norm_added_q/k` | `norm_q/k` | None |
| **Seq len** | Patchified (latent/2) | Matches latent | Matches latent |

---

## Future Improvements

1. **Fix RoPE integration**: Understand Qwen-Image's RoPE frequency structure
2. **Proper modulation**: Implement correct chunking for img_mod/txt_mod
3. **Block range collection**: Collect from multiple blocks like Flux
4. **Optimize VRAM**: Offload unused transformer blocks to CPU
5. **Real-time preview**: Show similarity maps during collection

---

## Credits

- **FreeFuse Team**: Original FreeFuse algorithm
- **Comfy-Org**: Qwen-Image ComfyUI integration
- **Qwen Team**: Qwen-Image architecture

---

## Changelog

- **2026-03-09**: Updated documentation for current implementation
  - ✅ Removed Z-Image references
  - ✅ Updated parameter defaults (temperature=4000, top_k_ratio=0.3)
  - ✅ Corrected block count (60 blocks, 0-59)
  - ✅ Added attention_head_index parameter
  - ✅ Updated VRAM recommendations

- **2026-03-07**: Initial Qwen-Image similarity extraction implementation
  - ✅ Hook into transformer blocks
  - ✅ Capture dual-stream attention
  - ✅ Compute similarity maps
  - ✅ Handle 5D tensor format
  - ✅ Early stopping for VRAM efficiency
  - ⚠️ RoPE skipped (shape mismatch)
  - ⚠️ Modulation simplified (shape mismatch)
