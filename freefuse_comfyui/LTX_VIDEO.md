# LTX-Video FreeFuse Guide

## Overview

FreeFuse nodes for LTX-Video multi-concept LoRA composition using attention-based masks.

**Status:** ✅ Fully implemented and tested (March 2026)

![LTX-Video 2.3 Workflow](https://codeberg.org/skynet/FreeFuse/raw/branch/main/images/ltx-ltx23-workflow.png)

---

## Quick Start

### Automatic Workflow
```
6-LoRA Loader → Background Loader → Token Positions → LTXSimilarityExtractor
                                                      ↓
                                         RawSimilarityOverlay → MaskApplicator → KSampler
```

### Manual Mask Workflow
```
Similarity Extractor → RawSimilarityOverlay → MaskTap → MaskReassemble → MaskApplicator → KSampler
```

---

## LTX-Video Architecture (ComfyUI)

| Component | Value |
|-----------|-------|
| **Model Class** | AVTransformer3DModel |
| **Transformer Blocks** | 48 |
| **Attention Heads** | 32 |
| **Head Dim** | 128 |
| **Video Dim** | 4096 |
| **Text Dim** | 2048 (projected to 4096) |
| **QK Norm** | RMSNorm |
| **Latent Format** | 5D: (B, C, T, H, W) |
| **Text Encoder** | Gemma 3 (12B) |

**Sequence Length:** T × (W/32) × (H/32) tokens

**Example:** 1024×1024, 10 frames = 10 × 32 × 32 = 10,240 tokens

---

## Nodes

### FreeFuseLTXSimilarityExtractor

Extract similarity maps from a specific transformer block.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `collect_block` | 10 | Block to collect from (0-47) |
| `collect_step` | 1 | Step at which to collect |
| `steps` | 3 | Sampling steps (2-3 enough) |
| `temperature` | 500.0 | Similarity temperature |
| `top_k_ratio` | 0.3 | Top-k token ratio |
| `attention_head_index` | -1 | -1 = average all 32 heads |
| `low_vram_mode` | True | VRAM optimization |

### FreeFuseLTXBlockGridExtractor

Scan multiple blocks and display as grid for visual analysis.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `block_start` | 10 | First block to scan |
| `block_end` | 20 | Last block to scan |
| `cell_size` | 128 | Block cell size in grid |
| `argmax_method` | stabilized | simple/stabilized |

### FreeFuseRawSimilarityOverlay

Visualize and refine similarity maps.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sensitivity` | 5.0 | Contrast amplification |
| `max_iter` | 15 | Stabilized argmax iterations |
| `gravity_weight` | 0.00004 | Centroid attraction |
| `spatial_weight` | 0.00004 | Neighbor influence |
| `anisotropy` | 1.3 | Horizontal stretch (video) |

---

## Block Selection Guide

| Block Range | Quality | Use |
|-------------|---------|-----|
| 0-9 | Early features | Debug only |
| **10-16** | **Best separation** | **Recommended** |
| 17-30 | Transition | Test if needed |
| 31-47 | Deep concepts | Complex scenes |

**Tip:** Use `LTXBlockGridExtractor` to scan blocks 10-20 and find the best one visually.

### Block Heat Maps

Visual analysis of attention patterns across all 48 transformer blocks:

![Blocks 0-20](https://codeberg.org/skynet/FreeFuse/raw/branch/main/images/ltx-heat-maps-b0-b20.png)
*Heat maps for blocks 0-20*

![Blocks 21-35](https://codeberg.org/skynet/FreeFuse/raw/branch/main/images/ltx-heat-maps-b21-b35.png)
*Heat maps for blocks 21-35*

![Blocks 35-47](https://codeberg.org/skynet/FreeFuse/raw/branch/main/images/ltx-heat-maps-b35-b47.png)
*Heat maps for blocks 35-47*

---

## Model Files

```
Diffusion (dev):      ltx-2-3-22b-dev-model.safetensors (BF16)
                      ltx-2.3-22b-dev_transformer_only_fp8_scaled.safetensors (FP8)
                      ltx-2-3-22b-dev-XX_X.gguf (GGUF)

Diffusion (distilled): ltx-2-3-22b-distilled-model.safetensors (BF16)
                       ltx-2.3-22b-distilled_transformer_only_fp8_scaled.safetensors (FP8)
                       ltx-2-3-22b-distilled-XX_X.gguf (GGUF)

Text Encoders:        gemma_3_12B_it.safetensors
                      ltx-2-3-22b-text_encoder.safetensors

VAE:                  ltx-2-3-22b-VAE.safetensors
                      ltx-2-3-22b-audio_vae.safetensors

LoRAs:                ltx-2.3-22b-distilled-lora-384.safetensors

Upscale:              ltx-2.3-spatial-upscaler-x2-1.0.safetensors
                      ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors
```

---

## Performance

### VRAM Usage

| Configuration | VRAM |
|---------------|------|
| LTX-Video BF16 | ~45GB |
| LTX-Video FP8 | ~25-30GB |
| + 1 LoRA | +1-2GB |

### Recommended Settings

**Minimal VRAM (FP8):**
```yaml
latent: 32x32, 8 frames
steps: 2, collect_step: 1
low_vram_mode: True
```

**Balanced:**
```yaml
latent: 48x48, 8 frames
steps: 3, collect_step: 1
low_vram_mode: True
```

**Quality:**
```yaml
latent: 64x64, 16 frames
steps: 3, collect_step: 1
low_vram_mode: False
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No similarity maps | Check hook installed, collect_step < steps |
| VRAM errors | Use FP8, reduce latent size, enable low_vram_mode |
| Poor separation | Try different block (10, 12, 14), adjust temperature |
| Token positions empty | Verify concept text in prompt, check Gemma 3 tokenizer |
| Masks have holes | Use `similarity_maps` (soft), increase max_iter |
| Concepts overlap | Lower temperature (300), use argmax_masks |

---

## Technical Notes

### Mask Application

Masks applied to video-side projections only:
- ✅ `attn1.to_q/k/v/out` - Video self-attention
- ✅ `attn2.to_q/out` - Video side of cross-attention
- ❌ `attn2.to_k/v` - Text side (skip)

### Spatio-Temporal Masks

For video with T frames:
1. Resize spatial mask to target resolution (H, W)
2. Repeat for all T frames
3. Final mask: (T × H × W,)

### Attention Bias

- Default: `enable_attention_bias: False` (spatial masks sufficient)
- If needed: `bias_scale: 1.0-3.0` (avoid 10+)

---

## References

- **LTX-Video Paper**: https://arxiv.org/abs/2501.00103
- **LTX-2 Paper**: https://arxiv.org/abs/2601.03233
- **HuggingFace**: https://huggingface.co/Lightricks/LTX-Video
