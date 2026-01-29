# FreeFuse: Multi-Subject LoRA Fusion via Adaptive Token-Level Routing at Test Time

<p align="center">
  <img src="assets/teaser.png" alt="Teaser" width="100%" style="background-color: white; padding: 10px;">
</p>

<p align="center">
  <em>FreeFuse enables seamless multi-subject composition by fusing multiple LoRAs without additional training or user-defined masks.</em>
</p>

## 📌 TODO

- [ ] Release inference code

## 📖 Abstract

This paper proposes **FreeFuse**, a training-free framework for multi-subject text-to-image generation through automatic fusion of multiple subject LoRAs. In contrast to prior studies that focus on retraining LoRA to alleviate feature conflicts, our analysis reveals that simply spatially confining the subject LoRA's output to its target region and preventing other LoRAs from directly intruding into this area is sufficient for effective mitigation. Accordingly, we implement **Adaptive Token-Level Routing** during the inference phase. We introduce **FreeFuseAttn**, a mechanism that exploits the flow matching model's intrinsic semantic alignment to dynamically match subject-specific tokens to their corresponding spatial regions at early denoising timesteps, thereby bypassing the need for external segmentors. FreeFuse distinguishes itself through high practicality: it necessitates no additional training, model modifications, or user-defined masks spatial conditions. Users need only provide subject activation words to achieve seamless integration into standard workflows. Extensive experiments validate that FreeFuse outperforms existing approaches in both identity preservation and compositional fidelity.

## 🎨 Results

### Comparison on Flux

<p align="center">
  <img src="assets/flux_res.png" alt="Flux Results" width="100%" style="background-color: white; padding: 10px;">
</p>

<p align="center">
  <em>Qualitative comparison with existing methods on Flux. FreeFuse achieves superior identity preservation and compositional fidelity.</em>
</p>

### Comparison on Other Methods

<p align="center">
  <img src="assets/il_res.png" alt="IL Results" width="100%" style="background-color: white; padding: 10px;">
</p>

<p align="center">
  <em>Additional comparisons demonstrating FreeFuse's effectiveness on SDXL.</em>
</p>

## 🏗️ Architecture

<p align="center">
  <img src="assets/arch.png" alt="Architecture" width="100%" style="background-color: white; padding: 10px;">
</p>

<p align="center">
  <em>Overview of FreeFuse architecture. FreeFuseAttn dynamically routes subject-specific LoRAs to their corresponding spatial regions via adaptive token-level routing, enabling training-free multi-subject LoRA fusion.</em>
</p>

## 🚀 Quick Start

*Code coming soon...*
