# ComfyUI-FreeFuse

FreeFuse for ComfyUI: multi-concept LoRA composition with spatial awareness.

## Workflows (Complete only)

- [workflows/flux_freefuse_complete.json](workflows/flux_freefuse_complete.json)
- [workflows/sdxl_freefuse_complete.json](workflows/sdxl_freefuse_complete.json)
- [workflows/zimage_freefuse_complete.json](workflows/zimage_freefuse_complete.json)

## Installation

```bash
git clone <this-repo>
ln -s /path/to/FreeFuse/comfyui ComfyUI/custom_nodes
```

## Example LoRAs and Prompt (from test_parameters.py)

**LoRA download links**

- Daiyu: https://huggingface.co/lsmpp/freefuse_community_loras/resolve/main/daiyu_lin.safetensors?download=true
- Harry: https://huggingface.co/lsmpp/freefuse_community_loras/resolve/main/harry_potter.safetensors?download=true
- Jinx (Z-Image-Turbo): https://huggingface.co/lsmpp/freefuse_example_loras/resolve/main/Jinx_Arcane_zit.safetensors?download=true
- Skeletor (Z-Image-Turbo): https://huggingface.co/lsmpp/freefuse_example_loras/resolve/main/skeletor_zit.safetensors?download=true

> The workflows expect these filenames by default:
> - Flux: harry_potter_flux.safetensors, daiyu_lin_flux.safetensors
> - SDXL: harry_potter_xl.safetensors, daiyu_lin_xl.safetensors
> - Z-Image-Turbo: Jinx_Arcane_zit.safetensors, skeletor_zit.safetensors
> If you use the downloads above, rename the files or update the workflow nodes.

**Prompt**

Realistic photography, harry potter, an European photorealistic style teenage wizard boy with messy black hair, round wire-frame glasses, and bright green eyes, wearing a white shirt, burgundy and gold striped tie, and dark robes hugging daiyu_lin, a young East Asian photorealistic style woman in traditional Chinese hanfu dress, elaborate black updo hairstyle adorned with delicate white floral hairpins and ornaments, dangling red tassel earrings, soft pink and red color palette, gentle smile with knowing expression, autumn leaves blurred in the background, high quality, detailed

**Negative Prompt (SDXL only)**

low quality, blurry, deformed, ugly, bad anatomy

**Concept Map**

- harry: harry potter, an European photorealistic style teenage wizard boy with messy black hair, round wire-frame glasses, and bright green eyes, wearing a white shirt, burgundy and gold striped tie, and dark robes
- daiyu: daiyu_lin, a young East Asian photorealistic style woman in traditional Chinese hanfu dress, elaborate black updo hairstyle adorned with delicate white floral hairpins and ornaments, dangling red tassel earrings, soft pink and red color palette, gentle smile with knowing expression
- background_text: autumn leaves blurred in the background

## Hyperparameters

### Phase 1 (FreeFuse Phase1 Sampler)

- `steps`: Total steps for Phase 2 (keep consistent for the same noise schedule)
- `collect_step`: Which step to collect attention and early-stop
- `collect_block`: Transformer block to extract attention (Flux: 0-56; SDXL: usually ≤20)
- `temperature`: Softmax temperature for similarity; 0 = auto (Flux=4000, SDXL=300)
- `top_k_ratio`: Ratio of top-k tokens used for similarity
- `disable_lora_phase1`: Disable LoRA in Phase 1 (recommended for cleaner attention)
- `bg_scale`: Background similarity scale (higher = more background)
- `use_morphological_cleaning`: Apply morphological cleanup
- `balance_iterations`: Iterations for balanced argmax (higher = more stable, slower)

### Phase 2 (FreeFuse Mask Applicator)

- `enable_token_masking`: Token-level masking (zero out other concept tokens)
- `enable_attention_bias`: Enable attention bias
- `bias_scale`: Negative bias strength (suppresses wrong concepts)
- `positive_bias_scale`: Positive bias strength (enhances correct concepts)
- `bidirectional`: Flux-only bidirectional bias (text↔image)
- `use_positive_bias`: Enable positive bias
- `bias_blocks`: Which blocks to apply bias (recommended all or double_stream_only)

### Sampling (KSampler / FluxGuidance)

- Flux uses FluxGuidance for CFG; set KSampler CFG to 1.0
- SDXL uses KSampler CFG directly (recommended 7.0)

## Preview Image

The workflows include a preview image:
freefuse_flux_square_1024_output.png. It shows up in the Preview when the workflow loads.

## License

Apache 2.0
