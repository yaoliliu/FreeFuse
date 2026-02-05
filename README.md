# ComfyUI-FreeFuse

Multi-concept LoRA composition with spatial awareness.

## Design

**Simplified v2**: Maximum reuse of ComfyUI internals.

- Uses `load_bypass_lora_for_models()` - LoRA not merged into base model
- Phase 1 node collects masks
- Phase 2 uses native `KSampler`

## Nodes

| Node | Purpose |
|------|---------|
| **FreeFuse LoRA Loader** | Load LoRA in bypass mode (chain multiple) |
| **FreeFuse Concept Map** | Map adapter names to concept text |
| **FreeFuse Phase 1** | Collect attention & generate masks |
| **FreeFuse Mask Preview** | Visualize generated masks |

## Workflow

```
Load Checkpoint
      ↓
FreeFuse LoRA Loader (character1)
      ↓
FreeFuse LoRA Loader (character2)
      ↓
FreeFuse Concept Map ← (define concepts)
      ↓
FreeFuse Phase 1 (5 steps) ← Empty Latent
      ↓
KSampler (28 steps) ← Same seed!
      ↓
VAE Decode → Save Image
```

## Installation

```bash
git clone <this-repo> 
ln -s /path/to/FreeFuse/comfyui ComfyUI/custom_nodes
```

## Usage

1. Load your base model (Flux/SDXL)
2. Chain `FreeFuse LoRA Loader` nodes for each character
3. Create `FreeFuse Concept Map` - match adapter names to your prompt text
4. Connect to `FreeFuse Phase 1` - runs 5 steps to collect masks
5. Connect output model to standard `KSampler` - use **same seed**
6. Decode and save

## License

Apache 2.0
