# FreeFuse Qwen-Image Architecture Diagram

## Complete Data Flow & Hook Injection Points

### Model Configuration (from qwen-model.txt)

```
Diffusion Model:  qwen_image_2512_bf16.safetensors (default weight_dtype)
LoRA:             Qwen-Image-Lightning-4steps-V2.0.safetensors (strength: 1.0)
CLIP:             qwen_2.5_vl_7b_fp8_scaled.safetensors (type: qwen_image)
VAE:              qwen_image_vae.safetensors
Model Sampling:   AuraFlow shift=3.10
KSampler:         steps=4, cfg=1.0, sampler=res_2s, scheduler=beta
```

**Download Links:**
- Diffusion Model: https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main/split_files/diffusion_models
- Lightning LoRA: https://huggingface.co/lightx2v/Qwen-Image-Lightning/blob/main/Qwen-Image-Lightning-4steps-V2.0.safetensors
- CLIP: https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/blob/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors
- VAE: https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/blob/main/split_files/vae/qwen_image_vae.safetensors

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           FREEFUSE QWEN-IMAGE WORKFLOW                                   │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 0: MODEL PREPARATION                                                              │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                            │
│  │ UNETLoader   │────▶│ LoraLoader   │────▶│ FreeFuse     │                            │
│  │              │     │ ModelOnly    │     │ 6-LoRA       │                            │
│  │ Qwen-Image   │     │ Lightning    │     │ Stacked      │                            │
│  │ BF16/FP8     │     │ (optional)   │     │ Loader       │                            │
│  └──────────────┘     └──────────────┘     └──────────────┘                            │
│         │                    │                      │                                   │
│         │                    │                      │                                   │
│         ▼                    ▼                      ▼                                   │
│  ┌────────────────────────────────────────────────────────────────────────────┐        │
│  │                         MODEL CLONE (model.clone())                        │        │
│  │                                                                            │        │
│  │  At this point we have:                                                    │        │
│  │  • Base Qwen-Image diffusion_model                                         │        │
│  │  • 60 transformer_blocks (MMDiT architecture)                              │        │
│  │  • LoRA adapters loaded in bypass mode                                     │        │
│  │  • FREEFUSE_DATA with concept definitions                                  │        │
│  └────────────────────────────────────────────────────────────────────────────┘        │
│                                    │                                                    │
│                                    ▼                                                    │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: HOOK INSTALLATION                                                              │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────┐         │
│  │              QwenAttentionHook Installation (on model clone)               │         │
│  │                                                                            │         │
│  │  Target: diffusion_model.transformer_blocks[collect_block]                │         │
│  │          (typically block 20-30 out of 60)                                │         │
│  │                                                                            │         │
│  │  ┌──────────────────────────────────────────────────────────────────────┐ │         │
│  │  │  Forward Pre-Hook (captures inputs)                                  │ │         │
│  │  │  ─────────────────────────────────────────────────────────────────   │ │         │
│  │  │  • hidden_states: (B, img_seq, dim) - already flattened from 5D     │ │         │
│  │  │  • encoder_hidden_states: (B, txt_seq, dim)                         │ │         │
│  │  │  • temb: timestep embedding                                          │ │         │
│  │  │  • image_rotary_emb: RoPE frequencies                                │ │         │
│  │  │                                                                      │ │         │
│  │  │  Action: Cache all inputs for similarity computation                │ │         │
│  │  └──────────────────────────────────────────────────────────────────────┘ │         │
│  │                                    │                                       │         │
│  │                                    ▼                                       │         │
│  │  ┌──────────────────────────────────────────────────────────────────────┐ │         │
│  │  │  Forward Hook (computes similarity maps)                             │ │         │
│  │  │  ─────────────────────────────────────────────────────────────────   │ │         │
│  │  │  1. Extract QKV from block's attention module:                      │ │         │
│  │  │     • img_q = to_q(img_attn_in)         [Image stream]              │ │         │
│  │  │     • img_k = to_k(img_attn_in)         [Image stream]              │ │         │
│  │  │     • txt_q = add_q_proj(txt_attn_in)   [Text stream - MMDiT]       │ │         │
│  │  │     • txt_k = add_k_proj(txt_attn_in)   [Text stream - MMDiT]       │ │         │
│  │  │                                                                      │ │         │
│  │  │  2. Apply QK norms:                                                 │ │         │
│  │  │     • norm_q, norm_k           (image)                              │ │         │
│  │  │     • norm_added_q, norm_added_k (text)                             │ │         │
│  │  │                                                                      │ │         │
│  │  │  3. Compute cross-attention scores:                                 │ │         │
│  │  │     • concept_k = txt_k[:, pos_t, :, :]  (at concept positions)     │ │         │
│  │  │     • weights = einsum(img_q, concept_k) * scale                    │ │         │
│  │  │     • scores = softmax(weights).mean()                              │ │         │
│  │  │                                                                      │ │         │
│  │  │  4. Competitive exclusion:                                          │ │         │
│  │  │     • Amplify concept scores                                        │ │         │
│  │  │     • Suppress other concepts                                       │ │         │
│  │  │                                                                      │ │         │
│  │  │  5. Top-K core token selection:                                     │ │         │
│  │  │     • k_count = img_len * top_k_ratio (typically 30%)               │ │         │
│  │  │     • _, topk_idx = torch.topk(scores, k_count)                     │ │         │
│  │  │                                                                      │ │         │
│  │  │  6. Self-modal similarity:                                          │ │         │
│  │  │     • core_tokens = gather(img_attn_out, topk_idx)                  │ │         │
│  │  │     • self_modal_sim = bmm(core_tokens, img_attn_out.transpose())   │ │         │
│  │  │     • sim_map = softmax(sim_avg / temperature)                      │ │         │
│  │  │                                                                      │ │         │
│  │  │  Output: similarity_maps[lora_name] = sim_map                       │ │         │
│  │  └──────────────────────────────────────────────────────────────────────┘ │         │
│  └────────────────────────────────────────────────────────────────────────────┘         │
│                                    │                                                    │
│                                    ▼                                                    │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: SAMPLING & COLLECTION                                                          │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────┐         │
│  │                    ComfyUI Sampling Loop (k_sample)                        │         │
│  │                                                                            │         │
│  │  Step 0 ──────────────────────────────────────────────────────┐           │         │
│  │     │                                                         │           │         │
│  │     ▼                                                         │           │         │
│  │  ┌─────────────────────────────────────────────────────────┐  │           │         │
│  │  │ Noise + Latent Preparation                               │  │           │         │
│  │  │ • latent: (B, C, H, W) → (B, C, 1, H, W) [5D reshape]   │  │           │         │
│  │  │ • noise: prepared for sampling                           │  │           │         │
│  │  └─────────────────────────────────────────────────────────┘  │           │         │
│  │     │                                                         │           │         │
│  │     ▼                                                         │           │         │
│  │  Step 1 ───────────────────────────────────────────────────►  │           │         │
│  │     │              [COLLECT_STEP - Hook Fires Here!]          │           │         │
│  │     │                                                         │           │         │
│  │     ▼                                                         │           │         │
│  │  ┌─────────────────────────────────────────────────────────┐  │           │         │
│  │  │ Transformer Block Forward Pass                           │  │           │         │
│  │  │                                                          │  │           │         │
│  │  │  Pre-Hook: Cache inputs                                  │  │           │         │
│  │  │       ↓                                                  │  │           │         │
│  │  │  Block Processing:                                       │  │           │         │
│  │  │  • img_norm1(img_hidden)                                │  │           │         │
│  │  │  • txt_norm1(txt_hidden)                                │  │           │         │
│  │  │  • Attention (QKV extraction happens here) ◄───────┐    │  │           │         │
│  │  │  • img_mlp, txt_mlp                               │    │  │           │         │
│  │  │  • norm_out.linear + proj_out                     │    │  │           │         │
│  │  │                                                    │    │  │           │         │
│  │  │  Post-Hook: Compute similarity maps ───────────────┼────┘  │           │         │
│  │  │                                                    │       │           │         │
│  │  └─────────────────────────────────────────────────────────┘  │           │         │
│  │     │                                                         │           │         │
│  │     ▼                                                         │           │         │
│  │  Step 2 ───────────────────────────────────────────────────►  │           │         │
│  │     │              [Early Stop Triggered!]                    │           │         │
│  │     │                                                         │           │         │
│  │     ▼                                                         │           │         │
│  │  ┌─────────────────────────────────────────────────────────┐  │           │         │
│  │  │ Early Stop Exception                                     │  │           │         │
│  │  │ • similarity_maps collected ✓                           │  │           │         │
│  │  │ • step > collect_step ✓                                 │  │           │         │
│  │  │ • raise EarlyStopException("Collection done")           │  │           │         │
│  │  └─────────────────────────────────────────────────────────┘  │           │         │
│  │                                                                 │           │         │
│  └─────────────────────────────────────────────────────────────────┘           │         │
│                                    │                                            │         │
│                                    ▼                                            │         │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: POST-PROCESSING & VRAM CLEANUP                                                 │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────┐         │
│  │                    After Sampling Completion                               │         │
│  │                                                                            │         │
│  │  1. Hook Removal:                                                         │         │
│  │     • hook.remove()                                                        │         │
│  │     • Clears pre_hook_handle and hook_handle                              │         │
│  │                                                                            │         │
│  │  2. VRAM Cleanup (Aggressive):                                            │         │
│  │     • torch.cuda.empty_cache()                                            │         │
│  │     • torch.cuda.synchronize()                                            │         │
│  │     • Log: "VRAM after hook removal: XXXX MB available"                   │         │
│  │                                                                            │         │
│  │  3. Move Maps to CPU (Low VRAM Mode):                                     │         │
│  │     • for name in collected_sim_maps:                                     │         │
│  │         collected_sim_maps[name] = collected_sim_maps[name].cpu()         │         │
│  │     • torch.cuda.empty_cache()                                            │         │
│  │     • Log: "VRAM after moving to CPU: XXXX MB available"                  │         │
│  │                                                                            │         │
│  └────────────────────────────────────────────────────────────────────────────┘         │
│                                    │                                                    │
│                                    ▼                                                    │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: SIMILARITY MAP VISUALIZATION                                                   │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────┐         │
│  │              FreeFuseRawSimilarityOverlay Processing                       │         │
│  │                                                                            │         │
│  │  Input: raw_similarity (FREEFUSE_MASKS)                                   │         │
│  │         {                                                                  │         │
│  │           "masks": {},                                                     │         │
│  │           "similarity_maps": {                                             │         │
│  │             "character1": tensor(B, 1024),  # 32x32 patches               │         │
│  │             "character2": tensor(B, 1024)   # 32x32 patches               │         │
│  │           }                                                                │         │
│  │         }                                                                  │         │
│  │                                                                            │         │
│  │  Processing Steps:                                                        │         │
│  │  ─────────────────                                                        │         │
│  │                                                                            │         │
│  │  1. Reshape similarity maps:                                              │         │
│  │     • seq_len = 1024 (from patchified 64x64 → 32x32)                      │         │
│  │     • actual_latent_h = actual_latent_w = 32                              │         │
│  │     • sim_2d = sim[0, :, 0].view(32, 32)                                  │         │
│  │                                                                            │         │
│  │  2. Perceptual Mapping (enhance small differences):                       │         │
│  │     • sim_norm = (sim - min) / (max - min)     [0-1 range]               │         │
│  │     • sim_vis = sigmoid((sim_norm - 0.5) * sensitivity)                  │         │
│  │     • sensitivity=5.0 amplifies contrast                                  │         │
│  │                                                                            │         │
│  │  3. Resize to output size:                                                │         │
│  │     • out_h, out_w = preview_size (e.g., 1024x1024)                       │         │
│  │     • F.interpolate(sim_vis, size=(out_h, out_w), mode='bilinear')        │         │
│  │                                                                            │         │
│  │  4. Stabilized Balanced Argmax (optional):                                │         │
│  │     • Input: stacked concept tensors (1, C, N)                            │         │
│  │     • Iterative optimization (max_iter=15):                               │         │
│  │       - Update bias with gradient descent (lr=0.01)                       │         │
│  │       - Apply gravity_weight (centroid attraction)                        │         │
│  │       - Apply spatial_weight (neighbor voting)                            │         │
│  │       - Smooth with momentum (0.2)                                        │         │
│  │     • Output: winner_indices (H, W)                                       │         │
│  │                                                                            │         │
│  │  5. Color overlay creation:                                               │         │
│  │     • character1 → Red channel                                            │         │
│  │     • character2 → Green channel                                          │         │
│  │     • Blend additively                                                    │         │
│  │                                                                            │         │
│  │  6. Argmax winner map:                                                    │         │
│  │     • For each concept index:                                             │         │
│  │         mask = (winner_indices == idx).float()                           │         │
│  │         overlay[c] += mask * color[c]                                     │         │
│  │                                                                            │         │
│  │  Outputs:                                                                 │         │
│  │  ─────────                                                                │         │
│  │  • overlay: (1, out_h, out_w, 3) - Colored composite                      │         │
│  │  • concept_1..4: Individual grayscale maps                               │         │
│  │  • argmax_winner: Hard assignment map                                    │         │
│  │  • refined_masks: FREEFUSE_MASKS for MaskApplicator                      │         │
│  │                                                                            │         │
│  └────────────────────────────────────────────────────────────────────────────┘         │
│                                    │                                                    │
│                                    ▼                                                    │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 5: MASK APPLICATION & GENERATION                                                  │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────┐         │
│  │              FreeFuseMaskApplicator (Phase 2)                              │         │
│  │                                                                            │         │
│  │  Input:                                                                    │         │
│  │  • model: Model with LoRAs in bypass mode                                 │         │
│  │  • masks: FREEFUSE_MASKS from RawSimilarityOverlay                        │         │
│  │  • freefuse_data: Adapter definitions                                     │         │
│  │                                                                            │         │
│  │  Processing:                                                              │         │
│  │  ────────────                                                             │         │
│  │                                                                            │         │
│  │  1. Install MultiAdapterBypassForwardHook:                                │         │
│  │     • For each adapter in freefuse_data["adapters"]:                      │         │
│  │         - Find matching bypass manager                                    │         │
│  │         - Install hook on adapter's forward pass                          │         │
│  │                                                                            │         │
│  │  2. Mask Application (per adapter):                                       │         │
│  │     • h(x) = adapter_output * mask[adapter_name]                          │         │
│  │     • Soft weights from similarity_maps OR                               │         │
│  │     • Hard weights from argmax_masks                                      │         │
│  │                                                                            │         │
│  │  3. Attention Bias (optional):                                            │         │
│  │     • Construct bias matrix from masks                                    │         │
│  │     • Apply to text-image cross-attention                                 │         │
│  │     • Suppress cross-LoRA attention (bias_scale=5.0)                      │         │
│  │     • Enhance same-LoRA attention (positive_bias_scale=1.0)               │         │
│  │                                                                            │         │
│  │  Output: model (patched with masked LoRA application)                     │         │
│  │                                                                            │         │
│  └────────────────────────────────────────────────────────────────────────────┘         │
│                                    │                                                    │
│                                    ▼                                                    │
│  ┌────────────────────────────────────────────────────────────────────────────┐         │
│  │                    KSampler (Phase 2 - Full Generation)                    │         │
│  │                                                                            │         │
│  │  Workflow Settings (FreeFuse-qwen-image-manual-mask.json):                │         │
│  │  • steps: 28                                                               │         │
│  │  • cfg: 3.5                                                                │         │
│  │  • sampler_name: euler                                                     │         │
│  │  • scheduler: simple                                                       │         │
│  │  • ModelSamplingAuraFlow: shift=3.0                                       │         │
│  │                                                                            │         │
│  │  Recommended Lightning Settings (from qwen-model.txt):                    │         │
│  │  • steps: 4  (with Lightning LoRA)                                         │         │
│  │  • cfg: 1.0                                                                │         │
│  │  • sampler_name: res_2s                                                    │         │
│  │  • scheduler: beta                                                         │         │
│  │  • ModelSamplingAuraFlow: shift=3.10                                      │         │
│  │                                                                            │         │
│  │  During sampling:                                                         │         │
│  │  • LoRA weights applied with spatial masks                                │         │
│  │  • character1 appears only in masked regions                              │         │
│  │  • character2 appears only in masked regions                              │         │
│  │  • Clean separation achieved                                              │         │
│  │                                                                            │         │
│  └────────────────────────────────────────────────────────────────────────────┘         │
│                                    │                                                    │
│                                    ▼                                                    │
│  ┌────────────────────────────────────────────────────────────────────────────┐         │
│  │                    VAEDecode + SaveImage                                   │         │
│  │                                                                            │         │
│  │  • Decode latent to image space                                           │         │
│  │  • Save final result                                                      │         │
│  │                                                                            │         │
│  └────────────────────────────────────────────────────────────────────────────┘         │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘

```

## Key Technical Details

### Qwen-Image Architecture Specifics

| Component | Specification | FreeFuse Handling |
|-----------|--------------|-------------------|
| **Tensor Format** | 5D `(B, C, T, H, W)` | Auto-reshape from 4D |
| **Transformer Blocks** | 60 blocks (MMDiT) | Hook on block 20-30 |
| **Attention Heads** | 64 heads | Average all or select specific |
| **QKV Projections** | Dual-stream | `to_q/k/v` (img) + `add_q/k/v_proj` (txt) |
| **QK Norms** | 4 separate layers | `norm_q/k` + `norm_added_q/k` |
| **Sequence Length** | 1024 tokens (32x32 patches) | Auto-detect from sim maps |
| **Patch Size** | 2 (64x64 → 32x32) | Handled in visualization |

### Hook Injection Points

```
Qwen-Image Transformer Block:
┌─────────────────────────────────────────┐
│  Forward Pre-Hook                       │
│  ├─ Capture: hidden_states             │
│  ├─ Capture: encoder_hidden_states     │
│  ├─ Capture: temb                      │
│  └─ Capture: image_rotary_emb          │
├─────────────────────────────────────────┤
│  Forward Pass                           │
│  ├─ img_norm1(img_hidden)              │
│  ├─ txt_norm1(txt_hidden)              │
│  ├─ Attention (QKV extraction) ◄───────┼── Hook reads QKV here
│  ├─ img_mlp, txt_mlp                   │
│  └─ norm_out.linear + proj_out         │
├─────────────────────────────────────────┤
│  Forward Hook                           │
│  ├─ Extract QKV from attention         │
│  ├─ Compute cross-attention scores     │
│  ├─ Competitive exclusion              │
│  ├─ Top-K selection                    │
│  └─ Self-modal similarity → sim_map   │
└─────────────────────────────────────────┘
```

### VRAM Management Timeline

```
Time ─────────────────────────────────────────────────────────────►

│
├─ Start: Log VRAM (free/total)
│
├─ Before Sampling:
│  • torch.cuda.empty_cache()
│  • torch.cuda.synchronize()
│  • Log VRAM before sampling
│
├─ During Sampling (step 1):
│  • Hook fires
│  • Similarity maps computed
│  • Maps stored in collected_sim_maps
│
├─ Early Stop (step 2):
│  • raise EarlyStopException
│  • torch.cuda.empty_cache()
│  • Log VRAM after early stop
│
├─ Hook Removal:
│  • hook.remove()
│  • torch.cuda.empty_cache()
│  • Log VRAM after hook removal
│
├─ Move to CPU (low_vram_mode):
│  • for name in collected_sim_maps:
│      collected_sim_maps[name] = .cpu()
│  • torch.cuda.empty_cache()
│  • Log VRAM after moving to CPU
│
└─ End: Maps on CPU, VRAM freed
```

### LoRA Activity Monitoring

```
Bypass LoRA Loader creates:
┌─────────────────────────────────────────────────────────────────┐
│  Bypass Manager (per adapter)                                   │
│                                                                 │
│  • adapter_name: "character1"                                  │
│  • lora_name: "character_lora.safetensors"                     │
│  • strength_model: 1.0                                         │
│  • strength_clip: 1.0                                          │
│  • bypass: False (weights applied, not merged)                │
│                                                                 │
│  Hooks installed on:                                           │
│  • transformer_blocks forward pass                            │
│  • adapter forward (h(x) computation)                         │
│                                                                 │
│  During MaskApplicator Phase 2:                               │
│  • Hook intercepts adapter output                              │
│  • Applies spatial mask: h(x) * mask[character1]              │
│  • Returns masked contribution                                 │
│                                                                 │
│  Activity can be monitored via:                               │
│  • FreeFuseMaskDebug node                                     │
│  • FreeFuseBankInspector node                                 │
│  • Logging in bypass_lora_loader.py                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Comparison Points

### Before FreeFuse (Standard LoRA Blending)

```
LoRA1 + LoRA2 → Blended Weights → Single Output
                                    │
                                    ▼
                          Characters merge/bleed together
                          No spatial control
```

### After FreeFuse (Masked Application)

```
LoRA1 ──┐
        ├→ Mask1 → Character1 (left side only)
        │
LoRA2 ──┤
        ├→ Mask2 → Character2 (right side only)
        │
        └→ Clean separation achieved
```

---

**Diagram created for:** FreeFuse GitHub Review  
**Author:** Michel "Skynet"  
**AI Assistant:** Qwen3-Coder-Next-Q8 (80GB, 16 CPU cores)  
**Date:** March 9, 2026
