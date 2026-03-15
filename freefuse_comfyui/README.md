# FreeFuse

**FreeFuse** is a ComfyUI custom nodes package for multi-concept LoRA composition using attention-based mask generation.

![Version](https://img.shields.io/badge/version-2.1-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![ComfyUI](https://img.shields.io/badge/ComfyUI-v1.0+-lightgrey)

## Workflows

![Qwen-Image & Z-Image Workflows](https://codeberg.org/skynet/FreeFuse/raw/branch/main/images/qwen-image-z-image-workflows.png)

*Qwen-Image, Z-Image, and LTX-Video workflows with FreeFuse multi-concept LoRA composition*

![LTX-Video 2.3 Workflow](https://codeberg.org/skynet/FreeFuse/raw/branch/main/images/ltx-ltx23-workflow.png)

*Complete LTX-Video 2.3 workflow with FreeFuse multi-concept LoRA composition*

## LTX-Video 2.3 Results - Text to Video with 2 LoRAs

FreeFuse successfully composites multiple LoRA concepts in LTX-Video 2.3 using attention-based mask generation.

### Example 1

**Video:** [ltx23-t2v-2lora-0.mp4](https://codeberg.org/skynet/FreeFuse/raw/branch/main/video/ltx23-t2v-2lora-0.mp4)

| Video Preview | Last Frame |
|---------------|------------|
| <video src="https://codeberg.org/skynet/FreeFuse/raw/branch/main/video/ltx23-t2v-2lora-0.mp4" width="320"></video> | <img src="https://codeberg.org/skynet/FreeFuse/raw/branch/main/images/ltx23-t2v-lora-last-frame-0.png" width="320"> |

### Example 2

**Video:** [ltx23-t2v-2lora-1.mp4](https://codeberg.org/skynet/FreeFuse/raw/branch/main/video/ltx23-t2v-2lora-1.mp4)

| Video Preview | Last Frame |
|---------------|------------|
| <video src="https://codeberg.org/skynet/FreeFuse/raw/branch/main/video/ltx23-t2v-2lora-1.mp4" width="320"></video> | <img src="https://codeberg.org/skynet/FreeFuse/raw/branch/main/images/ltx23-t2v-lora-last-frame-1.png" width="320"> |

*Last frame previews from LTX-Video 2.3 text-to-video generation using 2 LoRAs with FreeFuse multi-concept composition.*

## What FreeFuse Does

![FreeFuse Comparison](https://codeberg.org/skynet/FreeFuse/raw/branch/main/assets/compare_all.png?raw=true)

*FreeFuse enables precise multi-concept LoRA composition by generating attention-based masks that separate different characters/objects.*

## Features

- **Multi-LoRA Composition** - Stack up to 6 LoRAs per character
- **Attention-Based Masks** - Auto-generate masks from attention patterns
- **Manual Mask Support** - Use your own masks or SAM-generated masks
- **Multi-Model Support** - Qwen-Image, Z-Image, LTX-Video 2.3
- **VRAM Optimized** - Aggressive memory management for multi-LoRA

## Our Contributions

- **Qwen-Image Support** - Full integration for Qwen-Image MMDiT architecture (60 transformer blocks)
- **Stacked LoRA Loader** - Load up to 6 LoRAs per character adapter
- **SAM Integration** - Segment Anything Model support for AI-generated masks
- **Updated TokenPositions** - Improved token position computation
- **VRAM Optimization** - Aggressive memory management for multi-LoRA workflows

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://codeberg.org/skynet/FreeFuse.git FreeFuse
```

Or use **ComfyUI Manager** → Search "FreeFuse" → Install

## Key Nodes

| Node | Purpose |
|------|---------|
| `FreeFuse 6-LoRA Stacked Loader` | Load up to 6 LoRAs per adapter |
| `FreeFuseTokenPositions` | Define character locations |
| `FreeFuseQwenSimilarityExtractor` | Extract attention masks (Qwen) |
| `FreeFuseLTXSimilarityExtractor` | Extract attention masks (LTX-Video) |
| `FreeFuseLTXBlockGridExtractor` | Scan multiple blocks visually (LTX-Video) |
| `FreeFuse Raw Similarity Overlay` | Generate & refine masks |
| `FreeFuseMaskApplicator` | Apply masks to LoRAs |
| `FreeFuse Sampler` | Collect attention masks (Z-Image) |

## Quick Tips

### Qwen-Image
- **Blocks**: Use 20-30 for best separation
- **Latent size**: 32x32 for testing, 64x64 for final
- **Steps**: 2-3 steps enough for mask extraction

### LTX-Video
- **Blocks**: Use 10-16 for best concept separation (48 total blocks in ComfyUI)
- **Sequence Length**: T × (W/32) × (H/32) tokens
- **Attention Bias**: Start with 1.0-3.0, avoid high values (10+)
- **Mask Interpolation**: Nearest-neighbor preserves hard edges
- **attn2 Handling**: Masks applied to to_q/to_out only, NOT to_k/to_v

### General
- **VRAM**: Enable `low_vram_mode` for 2+ LoRAs
- **enable_attention_bias**: False by default (use spatial masks only)
- **bias_scale**: 2.0 if enabling attention bias

## TODO / Future Work

- [ ] **LTX-Video Similarity Maps Investigation** - Need to investigate LTX-Video similarity maps further to enable fully automated FreeFuse process. Current workflow requires manual block selection and parameter tuning.

## Credits

This project is based on the original **FreeFuse** implementation by **Yaoli Liu** [@yaoliliu](https://github.com/yaoliliu), a Master's student in Computer Science at [Zhejiang University](https://www.zju.edu.cn/), specializing in Generative Models.

**Contributor Recognition:** Michel "Skynet" has been added as a contributor to the FreeFuse dev channel for his Qwen-Image & LTX-Video integration work.

---

**Michel "Skynet"** - Author
**Qwen** - State-of-the-Art AI Coder | 80GB Active Memory | Polishing 16 CPU Cores | [Qwen3-Coder-Next-Q8](https://huggingface.co/Qwen) | *"He is my buddy"*

**FreeFuse** - Precise multi-concept generation through attention-based mask fusion.
