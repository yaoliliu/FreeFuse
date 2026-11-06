# ComfyUI-FreeFuse

FreeFuse for ComfyUI: multi-concept LoRA composition with spatial awareness.

## Workflows (Complete only)

- [workflows/flux_freefuse_complete.json](workflows/flux_freefuse_complete.json)
- [workflows/flux2_klein_4b_freefuse_complete.json](workflows/flux2_klein_4b_freefuse_complete.json)
- [workflows/flux2_klein_9b_freefuse_complete.json](workflows/flux2_klein_9b_freefuse_complete.json)
- [workflows/sdxl_freefuse_complete.json](workflows/sdxl_freefuse_complete.json)
- [workflows/zimage_freefuse_complete.json](workflows/zimage_freefuse_complete.json)
- [workflows/krea2_freefuse_with_editor.json](workflows/krea2_freefuse_with_editor.json)
- [workflows/moss_image_freefuse_complete.json](workflows/moss_image_freefuse_complete.json)
- [workflows/anima_freefuse_complete.json](workflows/anima_freefuse_complete.json)
- [workflows/minimax_h3_freefuse_complete.json](workflows/minimax_h3_freefuse_complete.json)
- [workflows/krea2_freefuse_depth_autoprompt_api.json](workflows/krea2_freefuse_depth_autoprompt_api.json) — community workflow: Krea2 AutoPrompt chain + depth-control branch ([docs](../docs/krea2_depth_autoprompt.md))

## Installation

```bash
cd /path/to/ComfyUI/custom_nodes
git clone --depth 1 --branch comfyui https://github.com/yaoliliu/FreeFuse.git freefuse_comfyui
```

## MiniMax H3 video

The existing FreeFuse nodes automatically recognize native H3 joint video/audio
latents. Start with the complete H3 workflow above; it saves with/without MP4s
and previews temporal regions using one shared prompt, seed and step count.
Use the full BF16 H3 model, native Qwen3-VL 32B text encoder (`minimax`), H3 video
and audio VAEs, and H3-trained character LoRAs.

The example uses 768 × 768, 124 frames, CFG 1, 20 `res_multistep` / `simple`
steps, collection step 4/block 20, temperature 100000, top-k 0.1, and
`h3_temporal_window=3`. Masks are `[time,height,width]`; the preview batch shows
latent time positions. H3 currently supports a single T2VA clip and one collection
block. Keyframe/reference guides and other CFG values are rejected in Phase 1.
Text/audio LoRA contributions remain global, and attention bias is video-to-text.

See the [H3 setup, weight links and validated scope](https://github.com/yaoliliu/FreeFuse/blob/master/docs/minimax_h3.md).

## Example LoRAs and Prompt (from test_parameters.py)

**LoRA download links**

- Daiyu: https://huggingface.co/lsmpp/freefuse_community_loras/resolve/main/daiyu_lin.safetensors?download=true
- Harry: https://huggingface.co/lsmpp/freefuse_community_loras/resolve/main/harry_potter.safetensors?download=true
- Jinx (Z-Image-Turbo): https://huggingface.co/lsmpp/freefuse_example_loras/resolve/main/Jinx_Arcane_zit.safetensors?download=true
- Skeletor (Z-Image-Turbo): https://huggingface.co/lsmpp/freefuse_example_loras/resolve/main/skeletor_zit.safetensors?download=true
- Archer Queen (Krea2): https://huggingface.co/lsmpp/freefuse_example_loras/resolve/main/archer_queen_krea2.safetensors?download=true
- Kim Possible (Krea2): https://huggingface.co/lsmpp/freefuse_example_loras/resolve/main/kim_possible_krea2.safetensors?download=true
- Satoru Gojo (Krea2): https://huggingface.co/lsmpp/freefuse_example_loras/resolve/main/satoru_gojo_krea2.safetensors?download=true

> The workflows expect these filenames by default:
> - Flux: harry_potter_flux.safetensors, daiyu_lin_flux.safetensors
> - Flux2.Klein 4B: flux-2-klein-4b.safetensors + qwen_3_4b.safetensors + flux2-vae.safetensors
> - Flux2.Klein 9B: flux-2-klein-9b-fp8.safetensors + qwen_3_8b_fp8mixed.safetensors + flux2-vae.safetensors
> - SDXL: harry_potter_xl.safetensors, daiyu_lin_xl.safetensors
> - Z-Image-Turbo: Jinx_Arcane_zit.safetensors, skeletor_zit.safetensors
> - Krea2 Turbo: krea2_turbo_fp8_scaled.safetensors + qwen3vl_4b_fp8_scaled.safetensors + qwen_image_vae.safetensors, or the larger bf16 pair krea2_turbo_bf16.safetensors + qwen3vl_4b_bf16.safetensors for Apple MPS testing
> - Krea2 example LoRAs: archer_queen_krea2.safetensors, kim_possible_krea2.safetensors, satoru_gojo_krea2.safetensors
> If you use the downloads above, rename the files or update the workflow nodes.

**Prompt**

Realistic photography, harry potter, an European photorealistic style teenage wizard boy with messy black hair, round wire-frame glasses, and bright green eyes, wearing a white shirt, burgundy and gold striped tie, and dark robes hugging daiyu_lin, a young East Asian photorealistic style woman in traditional Chinese hanfu dress, elaborate black updo hairstyle adorned with delicate white floral hairpins and ornaments, dangling red tassel earrings, soft pink and red color palette, gentle smile with knowing expression, autumn leaves blurred in the background, high quality, detailed

**Negative Prompt (SDXL only)**

low quality, blurry, deformed, ugly, bad anatomy

**Concept Map**

- harry: harry potter, an European photorealistic style teenage wizard boy with messy black hair, round wire-frame glasses, and bright green eyes, wearing a white shirt, burgundy and gold striped tie, and dark robes
- daiyu: daiyu_lin, a young East Asian photorealistic style woman in traditional Chinese hanfu dress, elaborate black updo hairstyle adorned with delicate white floral hairpins and ornaments, dangling red tassel earrings, soft pink and red color palette, gentle smile with knowing expression
- background_text: autumn leaves blurred in the background

## Important Prompt Rule

- Every **subject** `concept_text` (adapter trigger phrase) must appear verbatim in the **main prompt**.
- If any subject concept is missing from the main prompt, `FreeFuseTokenPositions` / `FreeFuseConceptMapSimple` now raises an error in ComfyUI.
- `background_text` is optional for runtime safety: if provided but not found in the main prompt, FreeFuse only prints a warning and continues.

Example:
- `concept_text = "harry potter"` means your main prompt must contain `"harry potter"`.

## Hyperparameters

### Phase 1 (FreeFuse Phase1 Sampler)

- `steps`: Total steps for Phase 2 (keep consistent for the same noise schedule)
- `collect_step`: Which step to collect attention and early-stop
- `collect_block`: Transformer block/layer to extract attention (Flux: `transformer_blocks.<idx>`, Flux2: `single_transformer_blocks.<idx>`, Z-Image: `layers.<idx>`, Krea2: `blocks.<idx>`, SDXL ignored)
- `collect_block_end`: Optional inclusive end index for range-mode collection (Flux/Flux2/Z-Image/Krea2). Set `collect_block_end > collect_block` to enable majority-vote aggregation across blocks.
- `temperature`: Softmax temperature for similarity; 0 = auto (Flux/Flux2=4000, SDXL=300)
- `top_k_ratio`: Ratio of top-k tokens used for similarity
- `disable_lora_phase1`: Disable LoRA in Phase 1 (recommended for cleaner attention)
- `bg_scale`: Background similarity scale (higher = more background)
- `use_morphological_cleaning`: Apply morphological cleanup
- `balance_iterations`: Iterations for balanced argmax (higher = more stable, slower)

### Phase 2 (FreeFuse Mask Applicator)

- `enable_token_masking`: Suppress each character LoRA on other characters' positive-prompt text tokens. Shared/overlapping concept positions, background text and negative CFG branches are preserved. Applies to bypass LoRAs on Krea2, Flux/Flux2, Z-Image, MOSS Image, Anima and SDXL; H3 keeps text/audio contributions global.
- `enable_attention_bias`: Enable attention bias
- `bias_scale`: Negative bias strength (suppresses wrong concepts)
- `positive_bias_scale`: Positive bias strength (enhances correct concepts)
- `bidirectional`: Flux/Flux2 bidirectional bias (text↔image)
- `use_positive_bias`: Enable positive bias
- `bias_blocks`: Which blocks to apply bias (recommended all or double_stream_only)

### Sampling (KSampler / FluxGuidance)

- Flux uses FluxGuidance for CFG; set KSampler CFG to 1.0
- Flux2.Klein uses CLIPTextEncode + CLIPLoader(type=`flux2`); keep KSampler CFG at 1.0 as a safe default
- Krea2 uses CLIPTextEncode + CLIPLoader(type=`krea2`); the included comparison runner keeps Phase 1 and Phase 2 at 12 steps by default
- SDXL uses KSampler CFG directly (recommended 7.0)

## Krea2 Comparison Tests

Krea2 fp8 weights should be tested on a CUDA host. On macOS, ComfyUI auto device selection uses Apple MPS, which cannot run the fp8 Krea2 tensors. Comfy-Org also publishes bf16 Krea2 files, which avoid that fp8 dtype failure on MPS, but they are much larger: `krea2_turbo_bf16.safetensors` is about 26.3 GB and `qwen3vl_4b_bf16.safetensors` is about 8.88 GB before runtime memory.

Prepare ComfyUI, the venv, and the default fp8 Krea2 model files under `/tmp` for CUDA testing:

```bash
WORK_DIR=/tmp/freefuse-krea2 \
LORA_SOURCE_DIR=/tmp/freefuse-krea2/input_loras \
bash freefuse_comfyui/scripts/setup_krea2_cuda_test.sh
```

The setup script expects the two character LoRAs in `LORA_SOURCE_DIR` unless `KIM_LORA_FILE` and `VIOLET_LORA_FILE` are set explicitly.

For an explicit Mac/MPS bf16 smoke route, use bf16 weights and disable the setup-time CUDA requirement:

```bash
WORK_DIR=/tmp/freefuse-krea2 \
KREA2_PRECISION=bf16 \
KREA2_DEVICE=mps \
REQUIRE_CUDA=0 \
LORA_SOURCE_DIR=/tmp/freefuse-krea2/input_loras \
bash freefuse_comfyui/scripts/setup_krea2_cuda_test.sh
```

Then run a smaller local comparison first. If this fails due to memory pressure, treat it as an MPS capacity limit rather than a FreeFuse algorithm failure:

```bash
/tmp/freefuse-krea2/.venv/bin/python freefuse_comfyui/scripts/run_krea2_compare.py \
  --comfyui-dir /tmp/freefuse-krea2/ComfyUI \
  --plugin-dir freefuse_comfyui \
  --device mps \
  --unet-name krea2_turbo_bf16.safetensors \
  --clip-name qwen3vl_4b_bf16.safetensors \
  --kim-lora-file "/tmp/freefuse-krea2/input_loras/archer_queen_krea2.safetensors" \
  --violet-lora-file "/tmp/freefuse-krea2/input_loras/satoru_gojo_krea2.safetensors" \
  --output-dir /tmp/freefuse-krea2/results/krea2_mps_bf16 \
  --mode both --width 512 --height 512 --steps 12 --phase1-steps 12 --collect-step 3
```

To stage the local character LoRAs before copying them to a CUDA host:

```bash
WORK_DIR=/tmp/freefuse-krea2 \
bash freefuse_comfyui/scripts/pack_krea2_loras.sh
```

Run the 8-GPU seed matrix after setup:

```bash
WORK_DIR=/tmp/freefuse-krea2 \
OUTPUT_DIR=/tmp/freefuse-krea2/results/krea2_matrix \
bash freefuse_comfyui/scripts/run_krea2_cuda_matrix.sh
```

Or run the full remote flow from the LoRA tar in one command:

```bash
WORK_DIR=/tmp/freefuse-krea2 \
bash freefuse_comfyui/scripts/run_krea2_remote_full_test.sh
```

After choosing the best `krea2_baseline_vs_freefuse.png`, copy it back and update the top-level README:

```bash
python3 freefuse_comfyui/scripts/update_krea2_readme_result.py \
  --contact-sheet /path/to/krea2_baseline_vs_freefuse.png
```

If you copied back the full remote result bundle, import it and then select the best seed explicitly:

```bash
RESULT_BUNDLE=/tmp/freefuse-krea2/krea2_cuda_results.tar.gz \
UPDATE_README=0 \
bash freefuse_comfyui/scripts/import_krea2_cuda_results.sh
```

If no company CUDA host is reachable, prepare a Hugging Face Jobs run without launching it:

```bash
python3 freefuse_comfyui/scripts/submit_krea2_hf_job.py --dry-run
```

## Preview Image

The workflows include a preview image:
freefuse_flux_square_1024_output.png. It shows up in the Preview when the workflow loads.

## License

Apache 2.0


## MOSS Image (experimental)

Native MOSS Image Base/SFT models are supported through the existing FreeFuse
nodes, including split/fused QKV. Use CLIPLoader type `moss_image`,
EmptyFlux2LatentImage, the FLUX.2 VAE and MOSS-specific LoRAs.
Open `workflows/moss_image_freefuse_complete.json`.
Collection block 20 / step 5 are provisional starting settings.
See [the full guide](https://github.com/yaoliliu/FreeFuse/blob/master/docs/moss_image.md) for exporting an official Diffusers
bundle, downloads, comparisons, parameter sweeps and SSH forwarding.

### MOSS Image example

The MOSS workflow uses a 1024 × 1024 Tinkerbell / Albedo close portrait, seed 42, 30 steps and CFG 3. Both LoRAs are active in the baseline and FreeFuse images. See the [setup and exact example settings](https://github.com/yaoliliu/FreeFuse/blob/master/docs/moss_image.md).

![MOSS Image comparison](https://raw.githubusercontent.com/yaoliliu/FreeFuse/master/assets/moss_image_compare.png)

## Anima

Native Anima Base v1.0 works with the existing FreeFuse nodes. Open
[anima_freefuse_complete.json](workflows/anima_freefuse_complete.json).
Use Qwen-3 0.6B with CLIPLoader type stable_diffusion (auto-detected as Anima),
qwen_image_vae.safetensors and EmptyCosmosLatentVideo with length 1, batch size 1.
Use Anima-specific LoRAs; collection temperature 0 selects the Anima default 1.0.
See the [setup, example and limits](https://github.com/yaoliliu/FreeFuse/blob/master/docs/anima.md).

![Anima comparison](https://raw.githubusercontent.com/yaoliliu/FreeFuse/master/assets/anima_compare.png)
