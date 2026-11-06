# FreeFuse: Multi-Subject LoRA Fusion via Adaptive Token-Level Routing at Test Time

<p align="center">
  <img src="assets/teaser.png" alt="Teaser" width="100%">
</p>

<p align="center">
  <em>FreeFuse enables seamless multi-subject composition by fusing multiple LoRAs without additional training or user-defined masks.</em>
</p>

[![arXiv](https://img.shields.io/badge/arXiv-2510.23515-b31b1b.svg)](https://arxiv.org/abs/2510.23515)

## 📌 TODO

- [x] Release basic inference code
- [x] Release inference code for controlnet, ipadapter and redux
- [x] Supports ComfyUI
- [x] Supports Z-Image-turbo in diffusers
- [x] Supports Z-Image-turbo in ComfyUI
- [x] Supports FLUX.2-klein-4B/9B  in diffusers
- [x] Supports FLUX.2-klein-4B/9B in ComfyUI
- [x] Supports Krea2 in ComfyUI
- [x] Experimental MOSS Image Base/SFT support in Diffusers and ComfyUI
- [x] [Anima Base v1.0 support in ComfyUI](docs/anima.md)
- [x] Support ControlNet+FreeFuse in ComfyUI
- [x] [Native MiniMax H3 text-to-audio-video support in ComfyUI](docs/minimax_h3.md)

## 📖 Abstract

This paper proposes **FreeFuse**, a training-free framework for multi-subject text-to-image generation through automatic fusion of multiple subject LoRAs. In contrast to prior studies that focus on retraining LoRA to alleviate feature conflicts, our analysis reveals that simply spatially confining the subject LoRA's output to its target region and preventing other LoRAs from directly intruding into this area is sufficient for effective mitigation. Accordingly, we implement **Adaptive Token-Level Routing** during the inference phase. We introduce **FreeFuseAttn**, a mechanism that exploits the flow matching model's intrinsic semantic alignment to dynamically match subject-specific tokens to their corresponding spatial regions at early denoising timesteps, thereby bypassing the need for external segmentors. FreeFuse distinguishes itself through high practicality: it necessitates no additional training, model modifications, or user-defined masks spatial conditions. Users need only provide subject activation words to achieve seamless integration into standard workflows. Extensive experiments validate that FreeFuse outperforms existing approaches in both identity preservation and compositional fidelity.

## 🎨 Results

### Results on Flux.dev

<p align="center">
  <img src="assets/flux_res.png" alt="Flux Results" width="100%">
</p>

<p align="center">
  <em>Qualitative results on Flux.dev. FreeFuse achieves superior identity preservation and compositional fidelity.</em>
</p>

### Results on SDXL

<p align="center">
  <img src="assets/il_res.png" alt="IL Results" width="100%">
</p>

<p align="center">
  <em>Additional comparisons demonstrating FreeFuse's effectiveness on SDXL.</em>
</p>

## 🏗️ Architecture

<p align="center">
  <img src="assets/arch.png" alt="Architecture" width="100%">
</p>

<p align="center">
  <em>Overview of FreeFuse architecture. FreeFuseAttn dynamically routes subject-specific LoRAs to their corresponding spatial regions via adaptive token-level routing, enabling training-free multi-subject LoRA fusion.</em>
</p>

## 🚀 Quick Start

### Installation with ComfyUI

```shell
# use comfy-cli
pip install comfy-cli
comfy node install freefuse

# or manual installation
cd <your ComfyUI path>/custom_nodes
git clone --depth 1 --branch comfyui https://github.com/yaoliliu/FreeFuse.git freefuse_comfyui
```

You can find example workflows in the `freefuse_comfyui/workflows` folder,
including the [MiniMax H3 video comparison workflow](freefuse_comfyui/workflows/minimax_h3_freefuse_complete.json).

### Installation with diffusers

You just need clone this repo and use [uv](https://docs.astral.sh/uv/) to install dependencies.

``` shell
apt-get update && apt-get install -y git-lfs && git lfs install
git clone https://github.com/yaoliliu/FreeFuse.git
git lfs pull
# or using huggingface if lfs budget is used up
hf download lsmpp/freefuse_example_loras --local-dir=./loras

uv venv
source .venv/bin/activate
uv sync
```

### Running on Flux

without background identification

``` shell
python main_freefuse_flux.py
```

with background identification and style LoRA

``` shell
python main_freefuse_flux_bg_except.py
```

with redux
``` shell
python redux_freefuse_flux_bg_except.py
```

with ipadapter
``` shell
python ipadapter_freefuse_flux_bg_except.py
```

with controlnet
``` shell
python controlnet_freefuse_flux_bg_except.py
```

*If you running out of your vram, you can set the quantization variable in the script to 'True' to enable quantization, which will reduce the vram usage.*

### Running on SDXL

``` shell
python main_freefuse_sdxl.py
```

### Running on Z-Image-turbo

``` shell
python main_freefuse_z_image.py
```

## Acknowledgement

Thanks for [Tekemo](https://civarchive.com/users/Tekemo) and [jrewingwannabe947](https://civitai.com/user/jrewingwannabe947)'s ZIT LoRAs, which are used in our Z-Image-turbo examples.

Thanks to [gpl1967](https://github.com/gpl1967) and [merermero](https://github.com/merermero) for the Krea2 support request and discussion in [issue #26](https://github.com/yaoliliu/FreeFuse/issues/26).

Thanks to [Rudolf (GPL1967)](https://github.com/GPL1967) and Agent Cow (monomatic, Hamburg) for contributing the [Krea2 AutoPrompt and depth-control workflow](docs/krea2_depth_autoprompt.md) in [PR #32](https://github.com/yaoliliu/FreeFuse/pull/32).

Thanks to [mertkasar](https://github.com/mertkasar) for identifying the bypass LoRA token-masking issue and sharing a proposed fix in [issue #34](https://github.com/yaoliliu/FreeFuse/issues/34).

## More model comparisons

Single-LoRA character references are shown at the left of each row. See [reference generation settings](docs/character_references.md).

<p align="center">
  <img src="assets/compare_all_with_references.png" alt="Single-LoRA references and FreeFuse comparisons" width="100%">
</p>

<p align="center">
  <em>FreeFuse solves feature conflicts between multiple subject LoRAs without additional training in flux1.dev, sdxl, z-image and more.</em>
</p>

### Results on MiniMax H3 video (ComfyUI)

Single-LoRA references, generated separately with single-character prompts (seed 42):

| Starfire · Teen Titans LoRA only | Albedo · Albedo LoRA only |
|:---:|:---:|
| ![Starfire single-LoRA reference](assets/minimax_h3_starfire_reference.png) | ![Albedo single-LoRA reference](assets/minimax_h3_albedo_reference.png) |

https://github.com/user-attachments/assets/401615de-1685-4e76-8975-03529ce28443

Global LoRAs on the left; automatic FreeFuse on the right, with the same prompt,
seed and sampling settings. This preview is silent; the native workflow generates
768 × 768 videos with 124 frames and audio. See the [H3 workflow and model setup](docs/minimax_h3.md).
