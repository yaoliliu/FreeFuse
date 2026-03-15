# Qwen-Image FreeFuse Guide

## Overview

FreeFuse nodes for Qwen-Image multi-concept LoRA composition using attention-based masks.

**Status:** ✅ Fully implemented and tested

![Qwen-Image Workflow](images/qwen-image-z-image-workflows.png)

---

## Quick Start

### Automatic Workflow
```
6-LoRA Loader → Token Positions → QwenSimilarityExtractor
                                         ↓
                            RawSimilarityOverlay → MaskApplicator → KSampler
```

### Manual Mask Workflow
```
Similarity Extractor → RawSimilarityOverlay → MaskTap → MaskReassemble → MaskApplicator → KSampler
```

---

## Qwen-Image Architecture

| Component | Value |
|-----------|-------|
| **Model Class** | QwenImageModel |
| **Architecture** | Dual-stream MMDiT |
| **Transformer Blocks** | 60 |
| **Attention Heads** | 64 |
| **Head Dim** | 128 |
| **QK Norm** | RMSNorm (separate for img/txt) |
| **Latent Format** | 5D: (B, C, T, H, W) |
| **Text Encoder** | Qwen 2.5 VL (7B) |
| **Tokenization** | Patchified (patch_size=2) |

**Sequence Length:** (H/2) × (W/2) tokens

**Example:** 64×64 latent = 32×32 = 1,024 tokens

---

## Nodes

### FreeFuseQwenSimilarityExtractor

Extract similarity maps from a specific transformer block.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `collect_block` | 20 | Block to collect from (0-59) |
| `collect_step` | 1 | Step at which to collect |
| `steps` | 3 | Sampling steps (2-3 enough) |
| `temperature` | 4000.0 | Similarity temperature |
| `top_k_ratio` | 0.3 | Top-k token ratio |
| `attention_head_index` | -1 | -1 = average all 64 heads |
| `low_vram_mode` | True | VRAM optimization |

### FreeFuseRawSimilarityOverlay

Visualize and refine similarity maps.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sensitivity` | 5.0 | Contrast amplification |
| `max_iter` | 15 | Stabilized argmax iterations |
| `balance_lr` | 0.01 | Learning rate |
| `gravity_weight` | 0.00004 | Centroid attraction |
| `spatial_weight` | 0.00004 | Neighbor influence |
| `momentum` | 0.2 | Smoothing |
| `anisotropy` | 1.3 | Horizontal stretch |

### FreeFuseMaskRefiner (Optional)

Clean up masks before applying.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fill_holes` | True | Fill holes in masks |
| `max_hole_size` | 50 | Max hole size to fill |
| `morph_operation` | close | Morphological operation |
| `remove_small_regions` | True | Remove small noise |

---

**Tip:** Use blocks 20-30 for best concept separation.

---

## Model Files

```
Diffusion:          qwen_image_2512_bf16.safetensors
                    https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main/split_files/diffusion_models

LoRA (Lightning):   Qwen-Image-Lightning-4steps-V2.0.safetensors (strength: 1.0)
                    https://huggingface.co/lightx2v/Qwen-Image-Lightning

Text Encoder:       qwen_2.5_vl_7b_fp8_scaled.safetensors (type: qwen_image)
                    https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/blob/main/split_files/text_encoders/

VAE:                qwen_image_vae.safetensors
                    https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/blob/main/split_files/vae/

Model Sampling:     AuraFlow shift=3.10
KSampler:           steps=4, cfg=1.0, sampler_name=res_2s, scheduler=beta
```

---

## Performance

### VRAM Usage

| Configuration | VRAM |
|---------------|------|
| Qwen-Image BF16 | ~27GB |
| + 1 LoRA | ~28GB |
| + 2 LoRAs | ~29-30GB |

### Recommended Settings

**Minimal VRAM (2+ LoRAs):**
```yaml
latent: 32x32 (256x256 image)
steps: 2, collect_step: 1
collect_block: 20
low_vram_mode: True
```

**Balanced:**
```yaml
latent: 48x48 (384x384 image)
steps: 3, collect_step: 1
collect_block: 25
low_vram_mode: True
```

**Quality (1-2 LoRAs):**
```yaml
latent: 64x64 (512x512 image)
steps: 3, collect_step: 1
collect_block: 25
attention_head_index: 24
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No similarity maps | Check hook installed, collect_step < steps |
| VRAM errors | Reduce latent size, enable low_vram_mode, use fewer steps |
| Poor separation | Try different block (20, 25, 30), adjust temperature |
| Token positions empty | Verify concept text in prompt |
| Masks have holes | Enable morphological cleaning, increase top_k_ratio |
| Concepts overlap | Lower temperature (3000), use argmax_masks |

---

## Technical Notes

### Dual-Stream Attention

Qwen-Image uses separate image and text streams:

| Stream | Projections | Norm |
|--------|-------------|------|
| **Image** | `to_q/k/v` | `norm_q/k` |
| **Text** | `add_q/k/v_proj` | `norm_added_q/k` |

### 5D Tensor Format

Qwen-Image expects 5D input `(B, C, T, H, W)`. The node auto-adds temporal dimension:

```python
# (B, C, H, W) → (B, C, 1, H, W)
latent_tensor = latent_tensor.unsqueeze(2)
```

### Patchified Latents

| Latent Size | Patches | Tokens |
|-------------|---------|--------|
| 32x32 | 16×16 | 256 |
| 48x48 | 24×24 | 576 |
| 64x64 | 32×32 | 1,024 |

### Attention Bias

- Default: `enable_attention_bias: True`
- Recommended: `bias_scale: 5.0`, `positive_bias_scale: 1.0`

---

## References

- **Qwen-Image**: https://huggingface.co/Qwen
- **ComfyUI Integration**: https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI
