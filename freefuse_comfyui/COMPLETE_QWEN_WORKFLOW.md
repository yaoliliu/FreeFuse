# Complete FreeFuse Qwen-Image Workflow

## Optimal Node Chain for Qwen-Image with Multiple LoRAs

```
[LoRA Loaders] → [Token Positions] → [QwenSimilarityExtractor]
                                                ↓
                                   [RawSimilarityOverlay]
                                   - sensitivity: 5.0
                                   - max_iter: 15
                                   - balance_lr: 0.01
                                   - gravity: 0.00004
                                   - spatial: 0.00004
                                   - momentum: 0.2
                                                ↓ (refined_masks)
                                   [MaskRefiner] ← Optional
                                   - fill_holes: True
                                   - morph_operation: close
                                   - remove_small: True
                                                ↓ (refined_masks)
                                   [MaskApplicator]
                                   - mask_source: similarity_maps
                                   - enable_attention_bias: True
                                   - bias_scale: 5.0
                                                ↓
                                   [KSampler Phase 2]
                                                ↓
                                   [SaveImage]
```

---

## Node Settings

### 1. FreeFuseQwenSimilarityExtractor

```yaml
collect_block: 20-30        # Best for Qwen-Image (middle blocks)
collect_step: 1             # Collect at first step
steps: 2-3                  # Minimum for extraction
temperature: 4000.0         # Default temperature
top_k_ratio: 0.3            # 30% of tokens
low_vram_mode: True         # Essential for multiple LoRAs
attention_head_index: -1    # -1 = average all 64 heads
```

### 2. FreeFuseRawSimilarityOverlay

```yaml
sensitivity: 5.0            # Moderate contrast
preview_size: 1024          # Output preview size
argmax_method: stabilized   # Balanced with spatial constraints
max_iter: 15                # Iterations for stabilized argmax
balance_lr: 0.01            # Learning rate
gravity_weight: 0.00004     # Centroid attraction
spatial_weight: 0.00004     # Neighbor influence
momentum: 0.2               # Smoothing
anisotropy: 1.3             # Horizontal stretch
bg_scale: 0.95              # Background multiplier
use_morphological_cleaning: True
```

**Output:** `refined_masks` (FREEFUSE_MASKS)

### 3. FreeFuseMaskRefiner ← Optional

```yaml
fill_holes: True
max_hole_size: 50
morph_operation: close
kernel_size: 3
iterations: 1
remove_small_regions: True
min_region_size: 100
smooth_boundaries: False
apply_threshold: False
```

**Purpose:** Clean up masks before applying to LoRAs

### 4. FreeFuseMaskApplicator

```yaml
mask_source: "similarity_maps"    # Use soft similarity weights
enable_token_masking: True
enable_attention_bias: True
bias_scale: 5.0
positive_bias_scale: 1.0
bidirectional: True
```

### 5. KSampler (Phase 2)

```yaml
steps: 20-28
cfg: 3.5
sampler_name: euler / dpmpp_2m
scheduler: simple / beta
```

---

## Why This Workflow Works

| Node | Purpose | Key Benefit |
|------|---------|-------------|
| **QwenSimilarityExtractor** | Extract at block 20-30 | Best concept separation |
| **RawSimilarityOverlay** | Fine-tune with your params | Your optimized settings |
| **MaskRefiner** | Fill holes, cleanup | Clean masks for generation |
| **MaskApplicator** | Apply as soft weights | Smooth blending |
| **KSampler** | Generate final image | High quality output |

---

## Parameter Guide

### Block Selection

| Block Range | Quality | Use |
|-------------|---------|-----|
| 0-15 | Early features | Not recommended |
| **16-25** | **Good separation** | **Recommended** |
| 26-35 | Transition | Test if 20 doesn't work |
| 36-45 | Noisy | Skip |
| 46-55 | Deep concepts | Complex scenes |
| 56-59 | Overlapping | Use for blending |

### MaskRefiner Settings

| Problem | Solution |
|---------|----------|
| Holes in masks | `fill_holes: True`, `max_hole_size: 50-100` |
| Rough edges | `morph_operation: close`, `kernel_size: 3` |
| Noise speckles | `remove_small_regions: True`, `min_region_size: 100` |
| Hard boundaries | `smooth_boundaries: True`, `blur_sigma: 1.0` |

### MaskApplicator: mask_source

| Option | Effect | When to Use |
|--------|--------|-------------|
| `argmax_masks` | Hard masks, one concept per pixel | Clean separation needed |
| `similarity_maps` | Soft weights, blending allowed | Natural transitions |

---

## Troubleshooting

### Problem: Masks have holes

**Solution 1:** Increase `max_hole_size` in MaskRefiner
```yaml
max_hole_size: 100  # Increase from 50
```

**Solution 2:** Use morphological closing
```yaml
morph_operation: close
kernel_size: 5      # Increase from 3
iterations: 2       # Increase from 1
```

### Problem: Concepts overlap/bleed

**Solution 1:** Lower temperature in QwenSimilarityExtractor
```yaml
temperature: 3000  # Decrease from 4000
```

**Solution 2:** Use argmax_masks instead
```yaml
MaskApplicator.mask_source: "argmax_masks"
```

**Solution 3:** Increase gravity/spatial weights
```yaml
gravity_weight: 0.00006
spatial_weight: 0.00006
```

### Problem: Poor separation at block 20

**Solution:** Try nearby blocks
```yaml
collect_block: 25  # Or 30, 15
```

### Problem: Slow generation

**Solution:** Reduce RawSimilarityOverlay preview size
```yaml
preview_size: 512  # Decrease from 1024
```

### Problem: VRAM errors / OOM

**Solution 1:** Reduce latent size
```yaml
# Use 32x32 latent (256x256 image)
```

**Solution 2:** Reduce steps
```yaml
steps: 2  # Minimum for collection
```

**Solution 3:** Enable low VRAM mode
```yaml
low_vram_mode: True  # Should be default
```

---

## Files in This Package

| File | Purpose |
|------|---------|
| `nodes/qwen_similarity_extractor.py` | Qwen-Image similarity extraction |
| `nodes/raw_similarity_grid.py` | Visualization and mask generation |
| `nodes/mask_refiner.py` | Mask cleanup utilities |
| `nodes/mask_applicator.py` | Apply masks to LoRAs |
| `nodes/mask_exporter.py` | Export masks to disk |
| `freefuse_core/mask_utils.py` | Mask generation utilities |
| `freefuse_core/json_serialization.py` | JSON serialization fixes |

---

## Quick Start

1. Load your workflow
2. Add **FreeFuseQwenSimilarityExtractor** after token positions
3. Set your parameters (collect_block=20, steps=2, etc.)
4. Add **FreeFuseRawSimilarityOverlay** for visualization
5. Optionally add **FreeFuseMaskRefiner** for cleanup
6. Add **FreeFuseMaskApplicator** to apply masks
7. Connect to **KSampler** for Phase 2 generation
8. Generate!

---

## VRAM Recommendations

| LoRAs | Latent Size | Steps | collect_block |
|-------|-------------|-------|---------------|
| 1-2 | 64x64 (512x512) | 3 | 25 |
| 2-3 | 48x48 (384x384) | 3 | 20 |
| 3+ | 32x32 (256x256) | 2 | 20 |

**Always enable `low_vram_mode: True` for 2+ LoRAs!**

---

## See Also

- `QWEN_IMAGE_OPTIMAL_SETTINGS.md` - Detailed parameter guide
- `QWEN_IMAGE_SIMILARITY_EXTRACTION.md` - Technical implementation details
- `MULTI_LORA_VRAM_OPTIMIZATION.md` - VRAM optimization strategies
- `TENSOR_JSON_ERROR_FIX.md` - Fix for JSON serialization errors
