# 1.3.2

- Fix `enable_token_masking` not reaching bypass LoRA hooks. Route other subjects' text positions for Krea2, Flux/Flux2, Z-Image, MOSS Image, Anima and SDXL using each model's native text layout.
- Preserve shared and background text, negative CFG branches, per-prompt positions and option resets across model clones and reinjection. H3 retains its video-only routing and global text/audio behavior.
- Correct native SDXL text-layer and Z-Image refiner classification; handle Krea2 text-fusion layers whose token axis is folded into the batch dimension.
- Thanks to @mertkasar for reporting and investigating the missing token-mask plumbing in [issue #34](https://github.com/yaoliliu/FreeFuse/issues/34).

# 1.3.1

- Fix Krea2 `mlp.gate/up/down` LoRA masks being stretched across text and image tokens after being misclassified as Anima layers. Krea2 MLPs now retain the same text/image alignment as their attention layers.
- Preserve image-only routing for Anima/Cosmos `mlp.layer1/layer2` and add CPU regressions for sequence alignment and two-adapter spatial routing.
- Thanks to @osuvense for the reproduction and diagnosis in [issue #33](https://github.com/yaoliliu/FreeFuse/issues/33).

# 1.3.0

- Add native MiniMax H3 text-to-audio-video composition through the existing FreeFuse nodes, with temporal mask volumes, packed video-only LoRA routing and memory-efficient video-to-text attention bias.
- Add complete UI/API comparison workflows with shared prompt, seed and schedule, native video/audio decoding and MP4 outputs. Initial scope is T2VA, one video, CFG 1 and one collection block.
- Preserve temporal mask previews and fail clearly for incompatible video grids or unsupported reference/guide conditioning.
- Keep LoRA disable state across rebuilt hooks, and compose standalone H3 attention bias with existing spatial routing.
- Preserve bypass adapters when native fused input-activation helpers are enabled; restore the module's previous flag on ejection.
- Support Unicode concept spans with ComfyUI's slow Qwen tokenizer for MOSS Image.

# 1.2.0

- Add native Anima Base v1.0 composition with T5-aligned collection, float32 similarity maps, spatial LoRA routing and image-to-text attention bias.
- Add an Anima workflow, reproducible 1024 × 1024 comparisons and isolated character references.

- Keep developer API clients out of the Registry package; runtime debug messages use standard logging.
- Remove the legacy adapter precision override and implicit debug tensor dumps; FP8/GGUF adapters retain floating precision.

# 1.1.0

- Add experimental native MOSS Image Base/SFT support, including split and fused QKV checkpoints.
- Route subject LoRAs with spatial masks and apply concept attention bias only to conditional CFG branches.
- Include a 1024 × 1024 MOSS workflow and reproducible comparison settings.
- Correct Registry dependency metadata so requirement-file comments are not treated as package names.
