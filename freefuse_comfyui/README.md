# FreeFuse

**FreeFuse** is a ComfyUI custom nodes package for multi-concept LoRA composition using attention-based mask generation.

![Version](https://img.shields.io/badge/version-2.1-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![ComfyUI](https://img.shields.io/badge/ComfyUI-v1.0+-lightgrey)

## Qwen-Image & Z-Image Workflow

![Qwen-Image & Z-Image Workflows](qwen-image-workflow.png)

*Qwen-Image, Z-Image, and LTX-Video workflows with FreeFuse multi-concept LoRA composition*

### LTX-Video Workflow

![LTX-Video 2.3 Workflow](ltx2-3-workflow.png)

*Complete LTX-Video 2.3 workflow with FreeFuse multi-concept LoRA composition*

## What FreeFuse Does

![FreeFuse Comparison](https://github.com/yaoliliu/FreeFuse/blob/master/assets/compare_all.png?raw=true)

*FreeFuse enables precise multi-concept LoRA composition by generating attention-based masks that separate different characters/objects in the image.*

## Example Workflows

| Workflow | Description |
|----------|-------------|
| [**Qwen-Image SAM Masks**](dev_workflows/FreeFuse-qwen-image-sam-mask.json) | Qwen-Image with SAM (Segment Anything) AI masks |
| [**Qwen-Image Manual Masks**](dev_workflows/FreeFuse-qwen-image-manual-mask.json) | Qwen-Image with manual mask input |
| [**Z-Image SAM Masks**](dev_workflows/FreeFuse-zimage-sam-mask.json) | Z-Image with SAM (Segment Anything) AI masks |
| [**Z-Image Manual Masks**](dev_workflows/FreeFuse-zimage-manual-mask.json) | Z-Image with manual mask input |
| [**Z-Image Standard**](dev_workflows/FreeFuse-zimage-standard.json) | Z-Image with attention-based masks |
| [**LTX-Video Manual Masks**](dev_workflows/FreeFuse-ltx-manual-masks.json) | LTX-Video 2.3 with manual mask input |

## Features

- **Multi-LoRA Composition** - Stack up to 6 LoRAs per character
- **Attention-Based Masks** - Auto-generate masks from attention patterns
- **Manual Mask Support** - Use your own masks or SAM-generated masks
- **Qwen-Image & Z-Image** - Full support for both architectures
- **LTX-Video 2.3** - Complete video generation with FreeFuse integration
- **VRAM Optimized** - Aggressive memory management for multi-LoRA

## Our Contributions

This fork extends the original FreeFuse implementation with:

- **Qwen-Image Support** - Full integration for Qwen-Image MMDiT architecture (60 transformer blocks)
- **Stacked LoRA Loader** - Load up to 6 LoRAs per character adapter (original had none)
- **SAM Integration** - Segment Anything Model support for AI-generated masks
- **Updated TokenPositions** - Improved token position computation for better concept mapping
- **VRAM Optimization** - Aggressive memory management for multi-LoRA workflows
- **Enhanced Workflow** - Streamlined node chain for both Qwen-Image and Z-Image

## The Qwen-Image Challenge

Getting FreeFuse to work with Qwen-Image was no small feat. The model's architecture was a black box - no documentation, no reference implementation, just a 40GB maze of transformer blocks waiting to be decoded.

**What we discovered through reverse engineering:**

- **5D Tensor Mystery** - Qwen-Image expects 5D tensors `(B, C, T, H, W)` instead of the standard 4D. ComfyUI provides 4D. We had to figure out where and how to inject the temporal dimension.

- **60 Transformer Blocks** - Unlike Flux (57 blocks) or SDXL (~30 blocks), Qwen-Image has 60 transformer blocks in a dual-stream MMDiT architecture. Finding the right blocks for attention extraction required extensive testing across the entire range.

- **Dual-Stream Attention** - Qwen-Image processes image and text in separate streams with different QKV projections (`to_q/k/v` for image, `add_q/k/v_proj` for text). We had to map both streams and understand how they interact.

- **QK Normalization Layers** - Separate normalization for image and text queries/keys (`norm_q/k` vs `norm_added_q/k`). Missing this caused silent failures in attention computation.

- **RoPE Frequency Shapes** - Qwen-Image's rotary position embeddings have a completely different structure than Flux. We discovered the similarity maps still work without perfect RoPE by using the attention output directly.

- **Patchified Latents** - The 64x64 latent is processed as 32x32 patches (patch_size=2), giving 1024 tokens instead of 4096. This caused massive confusion until we traced the sequence length back to the patchification.

- **VRAM Nightmares** - Qwen-Image BF16 alone uses ~27GB. With multiple LoRAs, we hit OOM constantly. Solution: aggressive early stopping, moving maps to CPU immediately, and strategic `torch.cuda.empty_cache()` calls at critical points.

**The breakthrough:** After days of debugging, hook installation/removal cycles, and tensor shape detective work, we cracked the basic architecture. But the truth? Qwen-Image is still full of mysteries. The model is surprisingly unpredictable - attention patterns shift between blocks, and what works once might not work again. We've opened the door, but there's still unexplored territory ahead: the 5D tensor space, transformer block interactions, attention head behaviors, and who knows what else. This is ongoing research, not a finished solution.

This wasn't just plugin development - this was digital archaeology, reverse engineering a closed architecture brick by brick. The journey continues.

## The LTX-Video Challenge

LTX-Video brought a whole new set of challenges. Video generation adds a temporal dimension - literally. We went from 4D to 5D tensors, 48 transformer blocks, and a completely different attention architecture.

**What we discovered through reverse engineering:**

- **5D Video Tensors** - LTX-Video expects `(B, C, T, H, W)` for video latents. The temporal dimension multiplies your sequence length: 10 frames × 24×16 spatial = 3,840 tokens per sample. Memory usage explodes fast.

- **Self-Attention Style Surprise** - The ComfyUI LTX implementation uses self-attention style where `to_k` expects 4096 dim input, but text comes in at 2048 dim. We had to create learned projection layers (2048 → 4096) on-the-fly. This wasn't documented anywhere.

- **48 Transformer Blocks** - LTX-Video has 48 blocks (vs 60 in Qwen-Image, 57 in Flux). Finding the sweet spot for concept separation required scanning the entire range. Blocks 16-30 turned out to be the goldilocks zone.

- **32 Attention Heads** - Half as many as Qwen-Image (64 heads). We implemented head averaging across all 32 heads, but discovered some heads show cleaner separation for specific concepts.

- **Gemma 3 Text Encoder** - 12B parameter text encoder with chat template. Token position finding had to be adapted from Qwen 2.5 VL to handle Gemma 3's different tokenization and chat wrapping.

- **RMS Norm for QK** - LTX-Video uses RMS normalization instead of standard layer norm. Missing this caused silent failures in attention computation.

- **VRAM Beast** - LTX-Video 22B BF16 alone uses ~45GB loaded. FP8 brings it down to ~25-30GB, but add multiple LoRAs and video sampling, and you're looking at OOM city. Solution: aggressive `low_vram_mode`, CPU offload, and strategic memory cleanup.

- **Temporal Consistency** - Video means temporal flickering. Masks need to be consistent across frames. We extract similarity at a single step and reuse across all frames, but true temporal smoothing is still on the roadmap.

**The breakthrough:** After extensive debugging of the attention hook system, we discovered the self-attention style with projection requirement. Once we implemented the learned text projection layers, similarity maps started flowing. The manual mask workflow (MaskTap + MaskReassemble + MaskApplicator) completed the pipeline, enabling full video generation with attention bias support.

This was architectural detective work at its finest - tracing tensor shapes through 48 blocks, decoding attention patterns, and building a complete video generation pipeline from scratch.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone <repository-url> FreeFuse
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

## Documentation

| File | Description |
|------|-------------|
| [COMPLETE_QWEN_WORKFLOW.md](COMPLETE_QWEN_WORKFLOW.md) | Full Qwen-Image guide |
| [QWEN_IMAGE_OPTIMAL_SETTINGS.md](QWEN_IMAGE_OPTIMAL_SETTINGS.md) | Best parameters |
| [MULTI_LORA_VRAM_OPTIMIZATION.md](MULTI_LORA_VRAM_OPTIMIZATION.md) | VRAM tips |
| [QWEN_IMAGE_SIMILARITY_EXTRACTION.md](QWEN_IMAGE_SIMILARITY_EXTRACTION.md) | Technical details |
| [LTX_VIDEO_README.md](LTX_VIDEO_README.md) | LTX-Video similarity extraction |
| [LTX_VIDEO_WORKFLOW.md](LTX_VIDEO_WORKFLOW.md) | LTX-Video workflow guide |
| [LTX_VIDEO_ARCHITECTURE.md](LTX_VIDEO_ARCHITECTURE.md) | LTX-Video architecture details |

## Quick Tips

- **Qwen-Image blocks**: Use 20-30 for best separation
- **LTX-Video blocks**: Use 16-30 for best concept separation
- **VRAM**: Enable `low_vram_mode` for 2+ LoRAs
- **Steps**: 2-3 steps enough for mask extraction
- **Latent size**: 32x32 for testing, 64x64 for final

## Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions

## Credits

This project is based on the original **FreeFuse** implementation by **Yaoli Liu** [@yaoliliu](https://github.com/yaoliliu), a Master's student in Computer Science at [Zhejiang University](https://www.zju.edu.cn/), specializing in Generative Models.

**Contributor Recognition:** Michel "Skynet" has been added as a contributor to the FreeFuse dev channel for his Qwen-Image & LTX-Video integration work.

---

**Michel "Skynet"** - Author  
**Qwen** - State-of-the-Art AI Coder | 80GB Active Memory | Polishing 16 CPU Cores | [Qwen3-Coder-Next-Q8](https://huggingface.co/Qwen) | *"He is my buddy"*

**FreeFuse** - Precise multi-concept generation through attention-based mask fusion.
