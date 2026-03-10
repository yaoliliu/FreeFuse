# FreeFuse Qwen-Image Optimal Settings

## Tested Configuration
**Model:** Qwen-Image
**LoRAs:** 2+ (elle, fox, etc.)
**Best Blocks:** 20-30 (middle range)

---

## FreeFuseQwenSimilarityExtractor

```yaml
collect_block: 20-30          # Middle blocks for best separation
collect_step: 1               # Collect at first step
steps: 2-3                    # Minimum for extraction
temperature: 4000.0           # Default temperature
top_k_ratio: 0.3              # 30% of tokens as core
low_vram_mode: True           # Recommended for multiple LoRAs
attention_head_index: -1      # -1 = average all 64 heads
```

---

## FreeFuseRawSimilarityOverlay

```yaml
# Preview settings
preview_size: 1024            # Output preview size
sensitivity: 5.0              # Contrast amplification

# Argmax algorithm
argmax_method: stabilized     # stabilized = balanced with spatial constraints
max_iter: 15                  # Iterations for stabilized argmax
balance_lr: 0.01              # Learning rate for balancing

# Spatial coherence
gravity_weight: 0.00004       # Centroid attraction
spatial_weight: 0.00004       # Neighbor influence
momentum: 0.2                 # Smoothing between iterations

# Spatial geometry
anisotropy: 1.3               # Horizontal stretch factor
centroid_margin: 0.0          # Keep centroids away from borders
border_penalty: 0.0           # Penalty for border assignment

# Background and cleanup
bg_scale: 0.95                # Background multiplier
use_morphological_cleaning: True  # Clean masks with morphology
```

---

## Why These Values Work

| Parameter | Value | Reason |
|-----------|-------|--------|
| `collect_block` | 20-30 | Middle blocks have best concept separation |
| `collect_step` | 1 | Clean attention patterns early in sampling |
| `temperature` | 4000.0 | Balanced sharpness for Qwen-Image |
| `top_k_ratio` | 0.3 | Good token coverage (30%) |
| `max_iter` | 15 | Fast convergence for stabilized argmax |
| `balance_lr` | 0.01 | Stable convergence rate |
| `gravity_weight` | 0.00004 | Mild centroid attraction |
| `spatial_weight` | 0.00004 | Mild spatial smoothing |
| `momentum` | 0.2 | Smooth optimization without overshooting |
| `sensitivity` | 5.0 | Moderate contrast enhancement |
| `anisotropy` | 1.3 | Slight horizontal preference |

---

## Block Selection Guide

| Block Range | Quality | Use Case |
|-------------|---------|----------|
| 0-15 | Early features | Not recommended |
| **16-25** | **Good separation** | **Recommended for testing** |
| 26-35 | Transition | Test if 20 doesn't work |
| 36-45 | Noisy | Skip |
| 46-55 | Deep concepts | Good for complex scenes |
| 56-59 | Overlapping | Use for blending |

**Note:** Block 56 was tested in early versions but blocks 20-30 generally provide better concept separation for Qwen-Image.

---

## Attention Head Selection

| Head Index | Effect | When to Use |
|------------|--------|-------------|
| -1 (default) | Average all 64 heads | General purpose |
| 0-15 | Early attention | Fine details |
| 16-31 | Mid attention | Concept boundaries |
| 32-47 | Deep attention | Semantic features |
| 48-63 | Very deep | Abstract concepts |

**Tip:** Use a specific head (e.g., `attention_head_index: 24`) for cleaner, more focused hotspots.

---

## Latent Size Recommendations

| Latent Size | Tokens | VRAM | Quality | Use |
|-------------|--------|------|---------|-----|
| 32x32 (256x256) | 256 | Low | Good | Testing, multiple LoRAs |
| 48x48 (384x384) | 576 | Medium | Better | Balanced |
| 64x64 (512x512) | 1024 | High | Best | Final generation |

**For 2+ LoRAs:** Use 32x32 or 48x48 to avoid OOM.

---

## Troubleshooting

### Problem: Poor concept separation

**Solution 1:** Try different blocks
```yaml
collect_block: 25  # Try 20, 25, 30
```

**Solution 2:** Adjust temperature
```yaml
temperature: 3000  # Lower = sharper boundaries
temperature: 5000  # Higher = softer blending
```

**Solution 3:** Use specific attention head
```yaml
attention_head_index: 24  # Try different heads (0-63)
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
low_vram_mode: True  # Default
```

**Solution 4:** Use earlier block
```yaml
collect_block: 15  # Earlier blocks use less VRAM
```

### Problem: Masks have holes

**Solution 1:** Enable morphological cleaning
```yaml
use_morphological_cleaning: True
```

**Solution 2:** Increase top_k_ratio
```yaml
top_k_ratio: 0.35  # More tokens = better coverage
```

### Problem: Concepts overlap/bleed

**Solution 1:** Lower temperature
```yaml
temperature: 3000  # Sharper boundaries
```

**Solution 2:** Use stabilized argmax
```yaml
argmax_method: stabilized
max_iter: 20  # More iterations for better balancing
```

**Solution 3:** Increase gravity/spatial weights
```yaml
gravity_weight: 0.00006
spatial_weight: 0.00006
```

### Problem: Slow convergence

**Solution 1:** Increase learning rate
```yaml
balance_lr: 0.02  # Faster but may overshoot
```

**Solution 2:** Increase iterations
```yaml
max_iter: 25  # More iterations
```

---

## Notes

- Qwen-Image needs **minimal sampling** (2-3 steps is enough for extraction)
- **Middle blocks (20-30)** provide best concept separation
- **Stabilized argmax** works better than simple argmax for balanced masks
- **Lower temperature than expected** (Qwen-Image works well with 3000-5000)
- **Specific attention heads** can give cleaner hotspots than averaging all 64
- **Low VRAM mode** is essential for 2+ LoRAs

---

## Quick Reference

### Minimal VRAM Setup (2+ LoRAs)
```yaml
# FreeFuseQwenSimilarityExtractor
latent: 32x32 (256x256 image)
steps: 2
collect_step: 1
collect_block: 20
temperature: 4000.0
top_k_ratio: 0.3
low_vram_mode: True

# FreeFuseRawSimilarityOverlay
argmax_method: stabilized
max_iter: 15
use_morphological_cleaning: True
```

### Best Quality Setup (1-2 LoRAs)
```yaml
# FreeFuseQwenSimilarityExtractor
latent: 64x64 (512x512 image)
steps: 3
collect_step: 1
collect_block: 25
temperature: 4000.0
top_k_ratio: 0.3
attention_head_index: 24

# FreeFuseRawSimilarityOverlay
argmax_method: stabilized
max_iter: 20
balance_lr: 0.01
gravity_weight: 0.00004
spatial_weight: 0.00004
use_morphological_cleaning: True
```
