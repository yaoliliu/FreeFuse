# Multi-LoRA VRAM Optimization for Qwen-Image

## Problem

Qwen-Image BF16 uses ~27GB VRAM. With multiple LoRAs, you'll hit OOM (Out Of Memory).

## Solution: Aggressive VRAM Management

The `FreeFuseQwenSimilarityExtractor` includes built-in VRAM optimizations:

### Automatic Optimizations
- **Early stopping**: Stops after step 1-2 (no need to complete full sampling)
- **Move maps to CPU**: Similarity maps moved to CPU immediately after collection
- **Aggressive cleanup**: `torch.cuda.empty_cache()` + `torch.cuda.synchronize()` at key points
- **VRAM monitoring**: Logs free VRAM at each stage

### Manual Optimizations (User Control)

#### 1. Use Smaller Latents

| Latent Size | Tokens | VRAM Usage | Quality |
|-------------|--------|------------|---------|
| 32x32 | 256 | **Lowest** | Good for testing |
| 48x48 | 576 | Medium | Good balance |
| 64x64 | 1024 | High | Best quality |

**Recommendation:** Use **32x32** for testing with multiple LoRAs.

#### 2. Minimize Sampling Steps

```yaml
steps: 2-3              # Minimum for collection
collect_step: 1         # Collect at first step
```

You don't need high quality samples - just enough to capture attention patterns.

#### 3. Enable Low VRAM Mode

```yaml
low_vram_mode: True     # Default - enables all optimizations
```

This enables:
- Extra VRAM cleanup before sampling
- Moving similarity maps to CPU immediately
- Synchronization to ensure cleanup completes

#### 4. Use Middle Blocks

```yaml
collect_block: 20-30    # Middle of 60 blocks
```

Middle blocks have good concept separation without needing deep processing.

#### 5. Use Specific Attention Head

```yaml
attention_head_index: 24  # Single head instead of averaging all 64
```

Using a single head reduces memory bandwidth during similarity computation.

---

## Recommended Settings for 2+ LoRAs

### Minimal VRAM Configuration (Test First)

```yaml
# FreeFuseQwenSimilarityExtractor
latent: 32x32           # 256x256 image → 256 tokens
steps: 2
collect_step: 1
collect_block: 20
temperature: 4000.0
top_k_ratio: 0.3
low_vram_mode: True
attention_head_index: -1

# FreeFuseRawSimilarityOverlay
preview_size: 512       # Smaller preview = less VRAM
```

**Expected VRAM:** ~28-29GB peak

### Balanced Configuration (If VRAM Allows)

```yaml
# FreeFuseQwenSimilarityExtractor
latent: 48x48           # 384x384 image → 576 tokens
steps: 3
collect_step: 1
collect_block: 25
temperature: 4000.0
top_k_ratio: 0.3
low_vram_mode: True

# FreeFuseRawSimilarityOverlay
preview_size: 1024
```

**Expected VRAM:** ~29-30GB peak

### Best Quality Configuration (1-2 LoRAs)

```yaml
# FreeFuseQwenSimilarityExtractor
latent: 64x64           # 512x512 image → 1024 tokens
steps: 3
collect_step: 1
collect_block: 25
temperature: 4000.0
top_k_ratio: 0.3
low_vram_mode: True
attention_head_index: 24

# FreeFuseRawSimilarityOverlay
preview_size: 1024
argmax_method: stabilized
max_iter: 20
```

**Expected VRAM:** ~30-31GB peak

---

## Workflow Example (2 LoRAs)

```
1. Load Qwen-Image model
   └─> UNETLoader: qwen_image_*.safetensors

2. Load 2 LoRAs
   └─> FreeFuse6LoraLoader
       - lora_name_1: character1.safetensors (strength: 1.0)
       - lora_name_2: character2.safetensors (strength: 1.0)
       - concept_text_1: "character one, description..."
       - concept_text_2: "character two, description..."

3. Define token positions
   └─> FreeFuseTokenPositions
       - user_text: "image contains character1 and character2"

4. Extract similarity maps ⚠️ USE SMALL LATENT
   └─> FreeFuseQwenSimilarityExtractor
       - latent: 32x32 (256x256 image)
       - steps: 2
       - collect_step: 1
       - collect_block: 20
       - low_vram_mode: True

5. Visualize
   └─> FreeFuseRawSimilarityOverlay
       - preview_size: 512
```

---

## VRAM Monitoring

The node logs VRAM at each stage:

```
[FreeFuse Qwen Extract] VRAM: 21540672 MB free / 32000000 MB total
[FreeFuse Qwen Extract] VRAM before sampling: 21000000 MB available
[FreeFuse Qwen Extract] Early stop triggered
[FreeFuse Qwen Extract] VRAM after early stop: 22000000 MB available
[FreeFuse Qwen Extract] Hook removed
[FreeFuse Qwen Extract] VRAM after hook removal: 23000000 MB available
[FreeFuse Qwen Extract] Moving similarity maps to CPU...
[FreeFuse Qwen Extract] VRAM after moving to CPU: 24000000 MB available
```

**If you see < 1GB free at any point, reduce latent size!**

---

## Troubleshooting OOM

### "Allocation would exceed allowed memory"

**Before sampling:**
- Reduce latent size (64x64 → 32x32)
- Close other applications using GPU
- Reduce `--max-memory` if set

**During sampling:**
- Reduce `steps` to 2
- Set `collect_step` to 0 or 1
- Use earlier `collect_block` (15-20)

**After collection:**
- Ensure `low_vram_mode: True`
- Check maps are moved to CPU (should see log message)

### "CUDA out of memory"

Same as above, but more urgent. Try:
1. **Restart ComfyUI** (clears fragmented VRAM)
2. **Use 32x32 latent** (256 tokens)
3. **Test with 1 LoRA first**, then add second
4. **Reduce preview_size** to 512

---

## Why This Works

The key insight: **We only need 1-2 sampling steps** to capture attention patterns.

Traditional sampling needs many steps to:
1. Gradually denoise the image
2. Refine details
3. Converge to final result

Similarity extraction only needs:
1. **One forward pass** through the target block
2. Capture QKV states at that block
3. Compute similarity from captured states

Everything after step 1-2 is wasted VRAM for our purpose!

---

## VRAM Comparison: Model Configurations

| Model | Base VRAM | +1 LoRA | +2 LoRAs | +3 LoRAs |
|-------|-----------|---------|----------|----------|
| Qwen-Image BF16 | ~27GB | ~28GB | ~29-30GB | ~31-32GB |
| Qwen-Image FP8 | ~20GB | ~21GB | ~22-23GB | ~24-25GB |

**Note:** FP8 models use significantly less VRAM but may have reduced quality.

---

## Why Qwen-Image Uses More VRAM

Qwen-Image uses more VRAM than other models because:
- **Larger model** (40.9GB vs ~20GB for Flux)
- **Dual-stream architecture** (img + txt processing)
- **60 transformer blocks** vs ~30-40 for others
- **5D tensor format** (temporal dimension)
- **64 attention heads** vs 16-24 for others

---

## Tips for Large Workflows

### 1. Load LoRAs Efficiently

Use `FreeFuse6LoraLoader` instead of multiple single LoRA loaders - it's optimized for VRAM.

### 2. Process in Stages

```
Stage 1: Extract similarity maps (small latent)
  ↓
Stage 2: Generate masks (CPU processing)
  ↓
Stage 3: Apply masks and generate (full latent)
```

### 3. Clear VRAM Between Stages

Add a simple node that runs:
```python
torch.cuda.empty_cache()
torch.cuda.synchronize()
```

### 4. Use Preview Nodes Wisely

Large previews consume VRAM. Use smaller preview sizes during testing:
```yaml
preview_size: 512   # Testing
preview_size: 1024  # Final
```

---

## Future Optimizations

Potential improvements (not yet implemented):

1. **Direct forward hook**: Skip sampling entirely, just run one forward pass
2. **Block caching**: Cache block outputs to avoid recomputation
3. **Gradient checkpointing**: Trade compute for VRAM
4. **CPU offloading**: Offload unused transformer blocks to CPU
5. **FP8 support**: Native FP8 extraction for lower VRAM

---

## See Also

- `QWEN_IMAGE_OPTIMAL_SETTINGS.md` - Parameter recommendations
- `QWEN_IMAGE_SIMILARITY_EXTRACTION.md` - Technical details
- `COMPLETE_QWEN_WORKFLOW.md` - Full workflow guide
