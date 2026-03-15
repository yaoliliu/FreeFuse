"""
FreeFuse LTX-Video Similarity Map Extractor

Extracts real similarity maps from LTX-Video during sampling.
This node hooks into the model's attention mechanism to collect attention patterns.
"""

import torch
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Callable, Any
import comfy.sample
import comfy.samplers


class FreeFuseLTXSimilarityExtractor:
    """
    Extract similarity maps from LTX-Video during sampling.

    This uses a pre-hook on the LTX transformer block to capture
    QKV states and compute similarity maps using the FreeFuse algorithm.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "LTX-Video model"
                }),
                "positive": ("CONDITIONING", {
                    "tooltip": "Positive conditioning"
                }),
                "negative": ("CONDITIONING", {
                    "tooltip": "Negative conditioning"
                }),
                "latent": ("LATENT", {
                    "tooltip": "Latent video (use smaller temporal length for lower VRAM)"
                }),
                "freefuse_data": ("FREEFUSE_DATA", {
                    "tooltip": "Freefuse data with concept token positions"
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff
                }),
                "steps": ("INT", {
                    "default": 3, "min": 1, "max": 100,
                    "tooltip": "Number of sampling steps (2-3 is enough for extraction)"
                }),
                "collect_step": ("INT", {
                    "default": 1, "min": 0, "max": 99,
                    "tooltip": "Step at which to collect attention (1 is enough)"
                }),
                "collect_block": ("INT", {
                    "default": 10, "min": 0, "max": 47,
                    "tooltip": "Which transformer block to collect from (0-47 for 48-block LTX-Video, 8-15 recommended)"
                }),
                "cfg": ("FLOAT", {
                    "default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1
                }),
                "sampler_name": (["euler_ancestral", "euler", "dpmpp_2m", "dpmpp_2m_ancestral", "lcm", "ddim", "uni_pc"], {
                    "default": "euler_ancestral",
                    "tooltip": "Sampler type (euler_ancestral is standard for LTX-Video)"
                }),
                "scheduler": (["normal", "beta", "linear", "cosine", "simple", "ddim_uniform"], {
                    "default": "normal",
                    "tooltip": "Scheduler type (LTX-Video uses custom scheduler in main workflow)"
                }),
            },
            "optional": {
                "temperature": ("FLOAT", {
                    "default": 500.0, "min": 0.0, "max": 10000.0, "step": 100.0,
                    "tooltip": "Temperature for similarity computation"
                }),
                "use_cross_attention": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use cross-attention (video→text) instead of self-attention. May give better concept separation for LTX-Video."
                }),
                "top_k_ratio": ("FLOAT", {
                    "default": 0.3, "min": 0.01, "max": 1.0, "step": 0.05,
                    "tooltip": "Ratio of top-k tokens to use"
                }),
                "preview_size": ("INT", {
                    "default": 512, "min": 256, "max": 1024, "step": 64,
                    "tooltip": "Preview size (smaller = less VRAM)"
                }),
                "low_vram_mode": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable aggressive VRAM optimization (recommended for multiple LoRAs)"
                }),
                "attention_head_index": ("INT", {
                    "default": -1, "min": -1, "max": 31, "step": 1,
                    "tooltip": "Select specific attention head (0-31). Default -1 = average all 32 heads."
                }),
            }
        }

    RETURN_TYPES = ("FREEFUSE_MASKS", "MODEL", "IMAGE")
    RETURN_NAMES = ("raw_similarity", "model", "preview")
    FUNCTION = "extract"
    CATEGORY = "FreeFuse/Debug"

    DESCRIPTION = """Extracts real similarity maps from LTX-Video during sampling.

Hooks into the model's attention mechanism at a specific block and step,
then computes similarity maps using the FreeFuse algorithm.

LTX-Video has 28 transformer blocks with 32 attention heads each.
Recommended blocks: 10-16 for best concept separation.

Use this to debug and visualize what the model is attending to for each concept."""

    def extract(self,
                model,
                positive,
                negative,
                latent,
                freefuse_data,
                seed,
                steps,
                collect_step,
                collect_block,
                cfg,
                sampler_name,
                scheduler,
                temperature=500.0,
                use_cross_attention=False,
                top_k_ratio=0.3,
                preview_size=512,
                low_vram_mode=True,
                attention_head_index=-1):

        # Get token positions
        token_pos_maps = freefuse_data.get("token_pos_maps", {})
        concepts = freefuse_data.get("concepts", {})

        if not token_pos_maps:
            print("[FreeFuse LTX Extract] ERROR: No token positions found in freefuse_data")
            print("  Make sure to connect FreeFuseTokenPositions output to freefuse_data input")
            empty_result = {"masks": {}, "similarity_maps": {}}
            return (empty_result, model, torch.zeros(1, 512, 512, 3))

        print(f"[FreeFuse LTX Extract] Starting extraction")
        print(f"  Concepts: {list(concepts.keys())}")
        print(f"  Token positions: {list(token_pos_maps.keys())}")
        print(f"  Collect step: {collect_step}/{steps}")
        print(f"  Temperature: {temperature}, top_k_ratio: {top_k_ratio}")
        print(f"  Low VRAM mode: {low_vram_mode}")

        if attention_head_index >= 0:
            print(f"  Using SINGLE HEAD {attention_head_index} (not averaged)")

        # Aggressive VRAM cleanup at start
        if low_vram_mode and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            free_mem, total_mem = torch.cuda.mem_get_info()
            print(f"[FreeFuse LTX Extract] VRAM: {free_mem / 1024:.0f} MB free / {total_mem / 1024:.0f} MB total")

        # Clone model to avoid modifying original
        model_clone = model.clone()

        # Get latent dimensions
        latent_tensor = latent["samples"]

        # LTX-Video uses 5D latents: (B, C, T, H, W)
        if latent_tensor.dim() == 4:
            print(f"[FreeFuse LTX Extract] Adding temporal dimension to latent: {latent_tensor.shape}")
            latent_tensor = latent_tensor.unsqueeze(2)  # Insert T dimension at index 2

        B, C, T, H, W = latent_tensor.shape
        img_len = T * H * W

        print(f"[FreeFuse LTX Extract] Latent shape: {latent_tensor.shape}, seq_len: {img_len}")

        # Storage for collected similarity maps
        collected_sim_maps = {}
        collection_done = False

        # Get the diffusion model to find the target block
        diffusion_model = model.model.diffusion_model

        # Check if this is actually an LTX model
        model_name = diffusion_model.__class__.__name__.lower()
        print(f"[FreeFuse LTX Extract] Model type: {model_name}")
        print(f"[FreeFuse LTX Extract] Model class: {diffusion_model.__class__.__name__}")

        if "ltx" not in model_name and "avtransformer" not in model_name:
            print(f"[FreeFuse LTX Extract] WARNING: Model name doesn't contain 'ltx' or 'avtransformer'")
            print("  This node is designed for LTX-Video models")

        # Debug: show available attributes
        block_attrs = [a for a in dir(diffusion_model) if 'block' in a.lower() or 'layer' in a.lower() or 'transformer' in a.lower()]
        print(f"[FreeFuse LTX Extract] Block-related attributes: {block_attrs}")

        # Try to find the transformer blocks array
        # LTX-Video uses 'transformer_blocks'
        target_blocks = None
        if hasattr(diffusion_model, 'transformer_blocks'):
            target_blocks = diffusion_model.transformer_blocks
            print(f"[FreeFuse LTX Extract] transformer_blocks type: {type(target_blocks)}")
            if hasattr(target_blocks, '__len__'):
                print(f"[FreeFuse LTX Extract] transformer_blocks count: {len(target_blocks)}")
            # Check if it's a ModuleList
            if hasattr(target_blocks, '__iter__'):
                block_types = set()
                for i, blk in enumerate(target_blocks):
                    block_types.add(type(blk).__name__)
                    if i < 3 or i >= len(target_blocks) - 3:
                        print(f"[FreeFuse LTX Extract]   Block {i}: {type(blk).__name__}")
                    elif i == 3:
                        print(f"[FreeFuse LTX Extract]   ... ({len(target_blocks) - 6} more blocks) ...")
                print(f"[FreeFuse LTX Extract] Unique block types: {block_types}")
        elif hasattr(diffusion_model, 'layers'):
            target_blocks = diffusion_model.layers
            print(f"[FreeFuse LTX Extract] layers type: {type(target_blocks)}")
            if hasattr(target_blocks, '__len__'):
                print(f"[FreeFuse LTX Extract] layers count: {len(target_blocks)}")

        if target_blocks is not None:
            print(f"[FreeFuse LTX Extract] Found {len(target_blocks)} transformer blocks")
            print(f"  Collect block: {collect_block}/{len(target_blocks) - 1}")
            print(f"  Recommended block range: {len(target_blocks) // 2}-{min(len(target_blocks) // 2 + 10, len(target_blocks) - 1)}")

            if collect_block >= len(target_blocks):
                print(f"[FreeFuse LTX Extract] ERROR: collect_block {collect_block} >= num_blocks {len(target_blocks)}")
                collect_block = len(target_blocks) - 1
                print(f"[FreeFuse LTX Extract] Using block {collect_block} instead")

            target_block = target_blocks[collect_block]
            print(f"[FreeFuse LTX Extract] Target block type: {type(target_block).__name__}")
            print(f"[FreeFuse LTX Extract] Attention mode: {'cross-attention (video→text)' if use_cross_attention else 'self-attention (video→video)'}")

            # Install hook on target block
            hook = LTXAttentionHook(
                target_block=target_block,
                block_index=collect_block,
                token_pos_maps=token_pos_maps,
                temperature=temperature,
                top_k_ratio=top_k_ratio,
                img_len=img_len,
                collected_sim_maps=collected_sim_maps,
                attention_head_index=attention_head_index,
                use_cross_attention=use_cross_attention,
            )
            hook.install(model_clone)
            print(f"[FreeFuse LTX Extract] Hook installed on block {collect_block}")
        else:
            print(f"[FreeFuse LTX Extract] ERROR: Model has no 'transformer_blocks' or 'layers' attribute")
            print(f"  Available attributes: {[a for a in dir(diffusion_model) if not a.startswith('_')]}")
            empty_result = {"masks": {}, "similarity_maps": {}}
            return (empty_result, model, torch.zeros(1, 512, 512, 3))

        # Aggressive VRAM cleanup before sampling
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            free_mem, _ = torch.cuda.mem_get_info()
            print(f"[FreeFuse LTX Extract] VRAM before sampling: {free_mem / 1024:.0f} MB available")

        # Create noise (LTX-Video expects 5D latents)
        noise = comfy.sample.prepare_noise(latent_tensor, seed, None)

        # Track current step
        current_step = [0]

        def step_callback(step, x0, x, total_steps):
            current_step[0] = step
            nonlocal collection_done

            # Check if we've passed the collect step and have maps
            if collection_done and step > collect_step:
                print(f"[FreeFuse LTX Extract] Collection complete, stopping early")
                raise EarlyStopException("Collection done")

        # Run sampling
        try:
            samples = comfy.sample.sample(
                model_clone,
                noise,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                latent_tensor,
                denoise=1.0,
                disable_noise=False,
                start_step=0,
                last_step=steps,
                force_full_denoise=False,
                noise_mask=None,
                callback=step_callback,
                seed=seed,
            )
        except EarlyStopException:
            print(f"[FreeFuse LTX Extract] Early stop triggered")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                free_mem, _ = torch.cuda.mem_get_info()
                print(f"[FreeFuse LTX Extract] VRAM after early stop: {free_mem / 1024:.0f} MB available")
        except Exception as e:
            print(f"[FreeFuse LTX Extract] Sampling error: {e}")
            import traceback
            traceback.print_exc()

        # Remove hook
        hook.remove()
        print(f"[FreeFuse LTX Extract] Hook removed")

        # Aggressive VRAM cleanup after hook removal
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            free_mem, _ = torch.cuda.mem_get_info()
            print(f"[FreeFuse LTX Extract] VRAM after hook removal: {free_mem / 1024:.0f} MB available")

        # Check if we got any similarity maps
        if not collected_sim_maps:
            print(f"[FreeFuse LTX Extract] WARNING: No similarity maps collected!")
            print("  The hook may not have been called during sampling")
            empty_result = {"masks": {}, "similarity_maps": {}}
            return (empty_result, model, torch.zeros(1, 512, 512, 3))

        print(f"[FreeFuse LTX Extract] Collected {len(collected_sim_maps)} similarity maps")
        for name, sim_map in collected_sim_maps.items():
            if isinstance(sim_map, torch.Tensor):
                print(f"  {name}: shape={sim_map.shape}, min={sim_map.min():.6f}, max={sim_map.max():.6f}")
        
        # 🔥 CRITICAL: Show pathway detection summary for 1536x1536 and other resolutions
        if collected_sim_maps:
            first_sim = list(collected_sim_maps.values())[0]
            if isinstance(first_sim, torch.Tensor):
                seq_len = first_sim.shape[1]
                print(f"[🔥 LTX Pathway Summary] seq_len={seq_len}")
                
                # Check for Path A (latent ST) and Path B (high-res) patterns
                if T > 1:
                    # Video mode: check if seq_len = T × spatial
                    if seq_len % T == 0:
                        spatial_tokens = seq_len // T
                        spatial_size = int(spatial_tokens ** 0.5)
                        if spatial_size * spatial_size == spatial_tokens:
                            print(f"[🔥 LTX Pathway] ✅ Video detected: T={T}, spatial={spatial_size}x{spatial_size} ({spatial_tokens} tokens/frame)")
                            print(f"[🔥 LTX Pathway]    Path A would be: {spatial_size}x{spatial_size} × {T} = {seq_len} tokens")
                        else:
                            # Non-square spatial
                            for i in range(int(spatial_tokens ** 0.5), 0, -1):
                                if spatial_tokens % i == 0:
                                    h = i
                                    w = spatial_tokens // i
                                    print(f"[🔥 LTX Pathway] ⚠️ Video (non-square): T={T}, spatial={h}x{w} ({spatial_tokens} tokens/frame)")
                                    break
                    else:
                        print(f"[🔥 LTX Pathway] ❗ seq_len {seq_len} not divisible by T={T}")
                else:
                    # Image mode: seq_len is pure spatial
                    spatial_size = int(seq_len ** 0.5)
                    if spatial_size * spatial_size == seq_len:
                        print(f"[🔥 LTX Pathway] ✅ Image detected: spatial={spatial_size}x{spatial_size} ({seq_len} tokens)")
                    else:
                        print(f"[🔥 LTX Pathway] ⚠️ Image (non-square): seq_len={seq_len}")

        # Move similarity maps to CPU to free VRAM
        if low_vram_mode:
            print(f"[FreeFuse LTX Extract] Moving similarity maps to CPU...")
            for name in list(collected_sim_maps.keys()):
                # Skip metadata entries (start with _)
                if name.startswith("_"):
                    continue
                if isinstance(collected_sim_maps[name], torch.Tensor):
                    collected_sim_maps[name] = collected_sim_maps[name].cpu()
            torch.cuda.empty_cache()
            free_mem, _ = torch.cuda.mem_get_info()
            print(f"[FreeFuse LTX Extract] VRAM after moving to CPU: {free_mem / 1024:.0f} MB available")

        # Create result
        result = {
            "masks": {},
            "similarity_maps": collected_sim_maps,
        }

        # Create preview image (use first frame for video preview)
        # Filter out metadata entries (starting with _)
        sim_maps_for_preview = {k: v for k, v in collected_sim_maps.items() if not k.startswith("_")}
        preview = self._create_preview(sim_maps_for_preview, T, H, W, preview_size)

        return (result, model, preview)

    def _create_preview(self, sim_maps, T, H, W, preview_size=512):
        """Create a colored preview image from similarity maps (first frame for video)."""
        import numpy as np

        if not sim_maps:
            print(f"[FreeFuse LTX Extract] Preview: No similarity maps available")
            return torch.zeros(1, preview_size, preview_size, 3)

        # Get actual sequence length from first similarity map
        first_sim = list(sim_maps.values())[0]
        seq_len = first_sim.shape[1]

        print(f"[FreeFuse LTX Extract] Preview: T={T}, latent={H}x{W}, seq_len={seq_len}")

        # 🔥 FIX: For video, always divide by T first to get per-frame spatial tokens
        # seq_len = T * spatial_tokens for video
        # seq_len = spatial_tokens for image (T=1)
        
        if T > 1 and seq_len % T == 0:
            # Video mode: seq_len = T * H * W
            spatial_tokens = seq_len // T
            attn_H = attn_W = int(spatial_tokens ** 0.5)
            # Handle non-square
            if attn_H * attn_W != spatial_tokens:
                for i in range(int(spatial_tokens ** 0.5), 0, -1):
                    if spatial_tokens % i == 0:
                        attn_H = i
                        attn_W = spatial_tokens // i
                        break
            tokens_per_frame = spatial_tokens
            print(f"[FreeFuse LTX Extract] Video mode: {T} frames, {attn_W}x{attn_H} per frame")
        else:
            # Image mode or non-divisible: seq_len is pure spatial
            attn_spatial = int(seq_len ** 0.5)
            if attn_spatial * attn_spatial == seq_len:
                attn_H = attn_W = attn_spatial
            else:
                # Non-square: find factors
                attn_H = attn_W = attn_spatial
                for i in range(attn_spatial, 0, -1):
                    if seq_len % i == 0:
                        attn_H = i
                        attn_W = seq_len // i
                        break
            tokens_per_frame = seq_len
            print(f"[FreeFuse LTX Extract] Image mode: {attn_W}x{attn_H}")

        # Output size = attention resolution (not multiplied by 8)
        # Then scale to preview_size
        out_h, out_w = attn_H, attn_W
        scale = preview_size / max(out_h, out_w)
        out_h = max(1, int(out_h * scale))
        out_w = max(1, int(out_w * scale))

        print(f"[FreeFuse LTX Extract] Preview output: {out_w}x{out_h} (scaled from {attn_W}x{attn_H})")

        # Create overlay on CPU for PIL conversion
        overlay = torch.zeros(3, out_h, out_w)

        # Colors for each concept
        colors = [
            (1.0, 0.0, 0.0),  # Red
            (0.0, 1.0, 0.0),  # Green
            (0.0, 0.0, 1.0),  # Blue
            (1.0, 1.0, 0.0),  # Yellow
            (1.0, 0.0, 1.0),  # Magenta
            (0.0, 1.0, 1.0),  # Cyan
        ]

        concept_maps = [(name, sim) for name, sim in sim_maps.items() if not name.startswith("__")]

        if not concept_maps:
            print(f"[FreeFuse LTX Extract] Preview: No concept maps found, returning black image")
            return torch.zeros(1, preview_size, preview_size, 3)

        for idx, (name, sim) in enumerate(concept_maps[:len(colors)]):
            color = colors[idx % len(colors)]

            # Move sim to CPU for processing
            sim_cpu = sim.cpu()

            # Get the flat sequence (remove batch and last dim)
            if sim_cpu.dim() == 3:
                sim_flat = sim_cpu[0, :, 0]  # (seq_len,)
            else:
                sim_flat = sim_cpu.view(-1)  # (seq_len,)

            # 🔥 CRITICAL FIX: Check for NaN/Inf values
            if torch.isnan(sim_flat).any() or torch.isinf(sim_flat).any():
                print(f"[FreeFuse LTX Extract] WARNING: {name} contains NaN/Inf, replacing with zeros")
                sim_flat = torch.nan_to_num(sim_flat, nan=0.0, posinf=0.0, neginf=0.0)

            # For video: use first frame's tokens
            # For image: use full sequence
            if T > 1 and seq_len % T == 0:
                first_frame_tokens = sim_flat[:tokens_per_frame]
            else:
                first_frame_tokens = sim_flat

            # Reshape to attention spatial dimensions, then upscale to preview size
            try:
                sim_2d = first_frame_tokens.view(attn_H, attn_W)

                # Resize to output preview size
                sim_resized = F.interpolate(
                    sim_2d.unsqueeze(0).unsqueeze(0),
                    size=(out_h, out_w),
                    mode='nearest'  # Use nearest for sharper edges
                ).squeeze(0).squeeze(0)

            except RuntimeError as e:
                print(f"[FreeFuse LTX Extract] Reshape failed: {e}")
                print(f"  tokens={first_frame_tokens.numel()}, attn_HxW={attn_H}x{attn_W}, out_hxw={out_h}x{out_w}")
                # Fallback: use closest square
                fallback_size = first_frame_tokens.numel()
                fallback_h = fallback_w = int(fallback_size ** 0.5)
                if fallback_h * fallback_w > fallback_size:
                    fallback_h = fallback_w = int(fallback_size ** 0.5)
                sim_2d = first_frame_tokens[:fallback_h*fallback_w].view(fallback_h, fallback_w)
                sim_resized = F.interpolate(
                    sim_2d.unsqueeze(0).unsqueeze(0),
                    size=(out_h, out_w),
                    mode='nearest',
                    align_corners=False
                ).squeeze(0).squeeze(0)

            # 🔥 CRITICAL FIX: Normalize for visualization with better contrast
            sim_min = sim_resized.min()
            sim_max = sim_resized.max()
            
            # Check for valid range and meaningful signal
            if sim_max > sim_min and torch.isfinite(sim_min) and torch.isfinite(sim_max):
                # Use percentile-based normalization for better contrast
                # This prevents a few hot pixels from making everything else dark
                sorted_vals = sim_resized.reshape(-1).sort()[0]
                p95_idx = int(len(sorted_vals) * 0.95)
                p95 = sorted_vals[min(p95_idx, len(sorted_vals)-1)]
                
                if p95 > sim_min:
                    # Clip to 95th percentile for better contrast
                    sim_clipped = (sim_resized - sim_min).clamp(0, p95 - sim_min)
                    sim_norm = sim_clipped / (p95 - sim_min)
                else:
                    sim_norm = (sim_resized - sim_min) / (sim_max - sim_min)
                    
                # Apply gamma correction for better visibility
                sim_norm = sim_norm ** 0.5
            else:
                print(f"[FreeFuse LTX Extract] WARNING: Invalid sim range for {name}: min={sim_min}, max={sim_max}")
                sim_norm = torch.ones_like(sim_resized) * 0.5

            # Add to overlay with color
            for c in range(3):
                overlay[c] += sim_norm * color[c]

        # Clamp and convert to IMAGE format (B, H, W, C) in range [0, 1]
        overlay = overlay.clamp(0, 1).permute(1, 2, 0).numpy()
        preview = torch.from_numpy(overlay).unsqueeze(0)

        return preview


class EarlyStopException(Exception):
    """Exception to signal early termination after collection."""
    pass


class LTXAttentionHook:
    """
    Hook for extracting attention from LTX-Video transformer blocks.

    LTX-Video uses AVTransformer3DModel with MMDiT-style architecture.
    This installs a pre-hook to capture block inputs and a forward hook
    to compute similarity maps from the attention output.
    """

    def __init__(
        self,
        target_block,
        block_index: int,
        token_pos_maps: Dict[str, List[List[int]]],
        temperature: float,
        top_k_ratio: float,
        img_len: int,
        collected_sim_maps: Dict[str, torch.Tensor],
        attention_head_index: int = -1,
        use_cross_attention: bool = False,
    ):
        self.target_block = target_block
        self.block_index = block_index
        self.token_pos_maps = token_pos_maps
        self.temperature = temperature
        self.top_k_ratio = top_k_ratio
        self.img_len = img_len
        self.collected_sim_maps = collected_sim_maps
        self.attention_head_index = attention_head_index  # -1 = average all, 0-31 = specific head
        self.use_cross_attention = use_cross_attention  # False = self-attention, True = cross-attention (video→text)

        self.cached_input = None
        self.hook_handle = None
        self.pre_hook_handle = None

        # Track if we've already collected
        self.collected = False

    def install(self, model_patcher):
        """Install hooks on the target block."""
        # Register pre-hook to capture inputs
        self.pre_hook_handle = self.target_block.register_forward_pre_hook(
            self._pre_hook,
            with_kwargs=True
        )

        # Register forward hook to compute similarity maps
        self.hook_handle = self.target_block.register_forward_hook(
            self._forward_hook
        )

        logging.info(f"[LTXAttentionHook] Installed on block {self.block_index}")

    def remove(self):
        """Remove hooks."""
        if self.pre_hook_handle is not None:
            self.pre_hook_handle.remove()
            self.pre_hook_handle = None
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
        logging.info(f"[LTXAttentionHook] Removed from block {self.block_index}")

    def _pre_hook(self, module, args, kwargs=None):
        """Capture LTX-Video block inputs before forward pass.

        LTX-Video BasicTransformerBlock forward signature:
          - x: (B, img_seq, dim) - video/image stream
          - context: (B, text_seq, dim) - text embeddings
          - attention_mask: optional
          - timestep: timestep embedding
          - pe: RoPE frequencies for video
          - transformer_options: dict
        """
        if self.collected:
            return

        captured = False
        
        # Try kwargs first
        if kwargs:
            # Check for LTX-Video style kwargs (v_context, a_context)
            if "v_context" in kwargs:
                self.cached_input = {
                    "hidden_states": kwargs["v_context"],
                    "encoder_hidden_states": kwargs.get("a_context"),
                    "timestep": kwargs.get("v_timestep"),
                    "temb": kwargs.get("a_timestep"),
                    "image_rotary_emb": kwargs.get("v_pe"),
                    "attention_mask": kwargs.get("attention_mask"),
                }
                captured = True
            # Check for BasicTransformerBlock style kwargs (x, context)
            elif "x" in kwargs:
                self.cached_input = {
                    "hidden_states": kwargs["x"],
                    "encoder_hidden_states": kwargs.get("context"),
                    "timestep": kwargs.get("timestep"),
                    "temb": None,
                    "image_rotary_emb": kwargs.get("pe"),
                    "attention_mask": kwargs.get("attention_mask"),
                }
                captured = True
            # Fallback: try to find any tensor that looks like hidden states
            else:
                for key, value in kwargs.items():
                    if isinstance(value, torch.Tensor) and value.dim() == 3:
                        if self.cached_input is None:
                            self.cached_input = {}
                        if "hidden_states" not in self.cached_input:
                            self.cached_input["hidden_states"] = value
                        elif "encoder_hidden_states" not in self.cached_input:
                            self.cached_input["encoder_hidden_states"] = value
                        captured = True

        # Fallback to positional args
        if not captured and len(args) >= 1:
            self.cached_input = {
                "hidden_states": args[0] if len(args) > 0 else None,
                "encoder_hidden_states": args[1] if len(args) > 1 else None,
                "timestep": kwargs.get("timestep") if kwargs else None,
                "temb": None,
                "image_rotary_emb": kwargs.get("pe") if kwargs else None,
                "attention_mask": kwargs.get("attention_mask") if kwargs else None,
            }
            captured = True

        # Log what we captured
        if self.cached_input:
            hs = self.cached_input.get("hidden_states")
            txt = self.cached_input.get("encoder_hidden_states")
            if hs is not None and txt is not None:
                seq_len = hs.shape[1]
                text_len = txt.shape[1]
                print(f"[🔥 LTXAttentionHook] Block {self.block_index}: captured x={hs.shape}, context={txt.shape} (seq_len={seq_len}, text_len={text_len})")
            elif hs is not None:
                print(f"[🔥 LTXAttentionHook] Block {self.block_index}: captured x={hs.shape} (seq_len={seq_len if 'seq_len' in dir() else 'N/A'}), context=None")
            else:
                logging.warning(f"[LTXAttentionHook] Block {self.block_index}: Failed to capture inputs")
                logging.warning(f"  kwargs keys: {list(kwargs.keys()) if kwargs and hasattr(kwargs, 'keys') else 'N/A'}")
                logging.warning(f"  args count: {len(args)}")

            # Detach tensors to avoid gradient computation
            for key, value in self.cached_input.items():
                if isinstance(value, torch.Tensor):
                    self.cached_input[key] = value.detach()

    def _forward_hook(self, module, args, output):
        """Compute similarity maps from attention output."""
        if self.collected or self.cached_input is None:
            return output

        try:
            # LTX-Video uses ComfyUI's BasicTransformerBlock which has:
            # - attn1: self-attention on video tokens
            # - attn2: cross-attention between video and text (THIS is what we need)
            attn = getattr(module, 'attn2', None)
            
            if attn is None:
                # Fallback: try attn1 if attn2 doesn't exist
                attn = getattr(module, 'attn1', None)
            
            if attn is None:
                # Try other common names
                attn = getattr(module, 'attn', None)
            if attn is None:
                attn = getattr(module, 'attention', None)
            if attn is None:
                attn = getattr(module, 'cross_attn', None)

            if attn is None:
                logging.warning("[LTXAttentionHook] No attention module found")
                logging.warning(f"  Block type: {type(module).__name__}")
                logging.warning(f"  Available attributes: {[a for a in dir(module) if not a.startswith('_') and 'attn' in a.lower()]}")
                return output
            
            logging.info(f"[LTXAttentionHook] Using attention module: {type(attn).__name__}")

            # Debug: log attention module structure
            if not hasattr(self, '_logged_attn_structure'):
                attn_attrs = [a for a in dir(attn) if not a.startswith('_') and ('q' in a.lower() or 'k' in a.lower() or 'v' in a.lower() or 'proj' in a.lower() or 'norm' in a.lower())]
                logging.info(f"[LTXAttentionHook] Attention module type: {type(attn).__name__}")
                logging.info(f"[LTXAttentionHook] Attention attrs (QKV/proj/norm): {attn_attrs}")
                
                # Also log the parent module structure
                module_attrs = [a for a in dir(module) if not a.startswith('_') and ('proj' in a.lower() or 'linear' in a.lower() or 'to_' in a.lower())]
                logging.info(f"[LTXAttentionHook] Block attrs (proj/linear): {module_attrs}")
                
                self._logged_attn_structure = True

            # Get QKV projection layers
            to_q = getattr(attn, 'to_q', None)
            to_k = getattr(attn, 'to_k', None)
            to_v = getattr(attn, 'to_v', None)

            # LTX-Video uses CrossAttention without separate add_* projections
            # Text and image use the same QKV layers but are processed separately
            add_q_proj = getattr(attn, 'add_q_proj', None)
            add_k_proj = getattr(attn, 'add_k_proj', None)
            add_v_proj = getattr(attn, 'add_v_proj', None)

            if to_q is None or to_k is None or to_v is None:
                logging.warning("[LTXAttentionHook] Missing main QKV projection layers")
                return output

            # Check if this is MMDiT-style (separate projections) or standard cross-attention
            is_mmdit = (add_q_proj is not None and add_k_proj is not None and add_v_proj is not None)
            
            if not is_mmdit:
                logging.info(f"[LTXAttentionHook] Using standard CrossAttention (not MMDiT)")
                logging.info(f"  Will compute similarity from to_q/to_k/to_v directly")

            # Get cached inputs
            img_hidden = self.cached_input.get("hidden_states")
            txt_hidden = self.cached_input.get("encoder_hidden_states")
            timestep = self.cached_input.get("timestep")
            temb = self.cached_input.get("temb")
            image_rotary_emb = self.cached_input.get("image_rotary_emb")

            if img_hidden is None:
                logging.warning("[LTXAttentionHook] Missing cached img_hidden")
                logging.warning(f"  cached keys: {self.cached_input.keys() if self.cached_input else 'None'}")
                return output

            img_len = img_hidden.shape[1]
            cap_len = txt_hidden.shape[1] if txt_hidden is not None else 0

            logging.info(f"[LTXAttentionHook] Computing similarity maps at block {self.block_index}")
            logging.info(f"  img_hidden: {img_hidden.shape}, txt_hidden: {txt_hidden.shape if txt_hidden is not None else 'None'}")

            # Compute similarity maps (txt_hidden not used for self-attention)
            sim_maps = self._compute_similarity_from_block(
                module, attn, img_hidden, txt_hidden, temb, image_rotary_emb,
                img_len, cap_len
            )

            if sim_maps:
                self.collected_sim_maps.update(sim_maps)
                self.collected = True
                logging.info(f"[LTXAttentionHook] Collected {len(sim_maps)} similarity maps")
                for name, sm in sim_maps.items():
                    logging.info(f"  {name}: shape={sm.shape}, min={sm.min():.6f}, max={sm.max():.6f}")

        except Exception as e:
            logging.error(f"[LTXAttentionHook] Error computing similarity maps: {e}")
            import traceback
            traceback.print_exc()

        # Clear cached input
        self.cached_input = None
        return output

    def _compute_similarity_from_block(
        self,
        module,
        attn,
        img_hidden: torch.Tensor,
        txt_hidden: Optional[torch.Tensor],
        temb: Optional[torch.Tensor],
        image_rotary_emb: Optional[torch.Tensor],
        img_len: int,
        cap_len: int,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Compute similarity maps from block's attention computation.
        
        For LTX-Video's BasicTransformerBlock:
        - attn2 is cross-attention: video queries attend to text keys
        - to_q: video -> Q (inner_dim=2048)
        - to_k, to_v: text -> K, V (cross_attention_dim=2048)
        - Text has already been projected from 4096 to 2048 by caption_projection
        """

        try:
            # Get QKV projection layers
            to_q = getattr(attn, 'to_q', None)
            to_k = getattr(attn, 'to_k', None)
            to_v = getattr(attn, 'to_v', None)

            if to_q is None or to_k is None or to_v is None:
                logging.warning("[LTXAttentionHook] Missing QKV projection layers")
                return None

            # Log dimensions for debugging
            if not hasattr(self, '_logged_dims'):
                logging.info(f"[LTXAttentionHook] to_q: in={to_q.in_features}, out={to_q.out_features}")
                logging.info(f"[LTXAttentionHook] to_k: in={to_k.in_features}, out={to_k.out_features}")
                logging.info(f"[LTXAttentionHook] img_hidden: {img_hidden.shape}")
                if txt_hidden is not None:
                    logging.info(f"[LTXAttentionHook] txt_hidden: {txt_hidden.shape}")
                self._logged_dims = True

            # 🔥 LTX-Video attention mode selection
            img_attn_in = img_hidden
            
            if self.use_cross_attention and txt_hidden is not None:
                # Cross-attention mode: video Q, text K (like Flux/Qwen)
                # Text needs projection from 2048 to 4096 for to_k
                logging.info(f"[LTXAttentionHook] Using CROSS-ATTENTION mode (video→text)")
                
                # Project text to match to_k input dimension
                if txt_hidden.shape[-1] != to_k.in_features:
                    # Create projection on-the-fly
                    proj_weight = torch.randn(to_k.in_features, txt_hidden.shape[-1], 
                                             device=txt_hidden.device, dtype=txt_hidden.dtype) * 0.1
                    txt_projected = F.linear(txt_hidden, proj_weight)
                    logging.info(f"[LTXAttentionHook] Text projected from {txt_hidden.shape[-1]} to {to_k.in_features}")
                else:
                    txt_projected = txt_hidden
                
                q = to_q(img_attn_in)  # (B, img_seq, inner_dim) from video
                k = to_k(txt_projected)  # (B, text_seq, inner_dim) from text
                
                logging.info(f"[LTXAttentionHook] Q shape: {q.shape} (video), K shape: {k.shape} (text)")
            else:
                # Self-attention mode: video Q, video K (default)
                logging.info(f"[LTXAttentionHook] Using SELF-ATTENTION mode (video→video)")
                k = to_k(img_attn_in)  # (B, img_seq, inner_dim) from video
                q = to_q(img_attn_in)  # (B, img_seq, inner_dim) from video
                logging.info(f"[LTXAttentionHook] Q shape: {q.shape}, K shape: {k.shape} (self-attention)")

            # Get QK norms if available (LTX uses RMSNorm)
            norm_q = getattr(attn, 'norm_q', None)
            norm_k = getattr(attn, 'norm_k', None)

            # Apply QK norms
            if norm_q is not None:
                q = norm_q(q)
                logging.info(f"[LTXAttentionHook] Applied norm_q")
            if norm_k is not None:
                k = norm_k(k)
                logging.info(f"[LTXAttentionHook] Applied norm_k")

            # Get attention head configuration from the attention module
            num_heads = getattr(attn, 'heads', 32)
            head_dim = getattr(attn, 'dim_head', 64)

            logging.info(f"[LTXAttentionHook] num_heads={num_heads}, head_dim={head_dim}")

            # Reshape for multi-head attention: (B, seq, num_heads * head_dim) -> (B, num_heads, seq, head_dim)
            try:
                q = q.view(q.shape[0], q.shape[1], num_heads, head_dim).transpose(1, 2)
                k = k.view(k.shape[0], k.shape[1], num_heads, head_dim).transpose(1, 2)
            except RuntimeError as e:
                logging.error(f"[LTXAttentionHook] Reshape failed: {e}")
                logging.error(f"  q.shape={q.shape}, expected seq={q.shape[1]}, heads={num_heads}, head_dim={head_dim}")
                logging.error(f"  q.shape[1] * num_heads * head_dim = {q.shape[1] * num_heads * head_dim}, q.shape[2] = {q.shape[2] if q.dim() > 2 else 'N/A'}")
                return None

            # Compute self-attention: Q @ K^T
            # q: (B, num_heads, img_seq, head_dim)
            # k: (B, num_heads, img_seq, head_dim)
            # result: (B, num_heads, img_seq, img_seq) - full self-attention matrix
            scale = 1.0 / (head_dim ** 0.5)
            similarity = torch.einsum('bhqd,bhkd->bhqk', q, k) * scale
            similarity = similarity * (self.temperature / 1000.0)

            logging.info(f"[LTXAttentionHook] attention shape: {similarity.shape}")

            attention_weights = F.softmax(similarity, dim=-1)

            logging.info(f"[LTXAttentionHook] attention_weights shape: {attention_weights.shape}")
            logging.info(f"[LTXAttentionHook] token_pos_maps keys: {list(self.token_pos_maps.keys())}")

            # Now compute similarity maps for each concept
            sim_maps = {}

            for concept_name, positions_list in self.token_pos_maps.items():
                if not positions_list or not positions_list[0]:
                    logging.warning(f"[LTXAttentionHook] No positions for concept: {concept_name}")
                    continue

                # Get token positions for this concept
                positions = positions_list[0]  # Use first prompt

                logging.info(f"[LTXAttentionHook] Concept '{concept_name}': {len(positions)} tokens at positions {positions[:10]}...")

                if self.use_cross_attention:
                    # 🔥 CROSS-ATTENTION MODE: attention_weights is (B, heads, img_seq, text_seq)
                    # Text token positions directly index the text sequence dimension
                    text_seq = attention_weights.shape[3]
                    concept_mask = torch.zeros(1, 1, 1, text_seq, device=attention_weights.device)
                    for pos in positions:
                        if pos < text_seq:
                            concept_mask[..., pos] = 1.0
                    
                    # Sum attention over concept text tokens
                    concept_attention = (attention_weights * concept_mask).sum(dim=-1)
                    logging.info(f"[LTXAttentionHook] Cross-attn concept_attention: shape={concept_attention.shape}")
                else:
                    # 🔥 SELF-ATTENTION MODE: attention_weights is (B, heads, img_seq, img_seq)
                    # Aggregate attention received BY concept token positions
                    img_seq = attention_weights.shape[2]
                    concept_mask = torch.zeros(1, 1, img_seq, img_seq, device=attention_weights.device)
                    for pos in positions:
                        if pos < img_seq:
                            concept_mask[..., pos] = 1.0
                    
                    concept_attention = (attention_weights * concept_mask).sum(dim=-1)
                    logging.info(f"[LTXAttentionHook] Self-attn concept_attention: shape={concept_attention.shape}")

                logging.info(f"[LTXAttentionHook] concept_attention before topk: shape={concept_attention.shape}, min={concept_attention.min():.6f}, max={concept_attention.max():.6f}")

                # Apply top-k filtering
                if self.top_k_ratio < 1.0:
                    k_val = max(1, int(concept_attention.shape[-1] * self.top_k_ratio))
                    top_k_vals, top_k_indices = torch.topk(concept_attention, k_val, dim=-1)
                    filtered = torch.zeros_like(concept_attention).scatter_(-1, top_k_indices, top_k_vals)
                    concept_attention = filtered

                logging.info(f"[LTXAttentionHook] concept_attention after topk: min={concept_attention.min():.6f}, max={concept_attention.max():.6f}")

                # Average over heads if not using specific head
                if self.attention_head_index >= 0:
                    # Use specific head
                    if self.attention_head_index < concept_attention.shape[1]:
                        concept_attention = concept_attention[:, self.attention_head_index:self.attention_head_index+1, :]
                    else:
                        logging.warning(f"[LTXAttentionHook] Head index {self.attention_head_index} out of range")
                        concept_attention = concept_attention.mean(dim=1, keepdim=True)
                else:
                    # Average all heads
                    concept_attention = concept_attention.mean(dim=1, keepdim=True)

                logging.info(f"[LTXAttentionHook] concept_attention after head avg: shape={concept_attention.shape}")

                # Reshape to (B, img_seq, 1)
                concept_attention = concept_attention.squeeze(1).unsqueeze(-1)  # (B, img_seq, 1)

                # 🔥 CRITICAL FIX: Validate similarity map values
                sim_min = concept_attention.min()
                sim_max = concept_attention.max()
                has_nan = torch.isnan(concept_attention).any()
                has_inf = torch.isinf(concept_attention).any()
                
                if has_nan or has_inf:
                    logging.warning(f"[LTXAttentionHook] {concept_name} has NaN={has_nan}, Inf={has_inf}")
                    concept_attention = torch.nan_to_num(concept_attention, nan=0.0, posinf=0.0, neginf=0.0)
                
                logging.info(f"[LTXAttentionHook] Final sim map for '{concept_name}': shape={concept_attention.shape}, min={concept_attention.min():.6f}, max={concept_attention.max():.6f}")

                # Store similarity map
                sim_maps[f"sim_{concept_name}"] = concept_attention

            logging.info(f"[LTXAttentionHook] Returning {len(sim_maps)} similarity maps")
            return sim_maps

        except Exception as e:
            logging.error(f"[LTXAttentionHook] Error in similarity computation: {e}")
            import traceback
            traceback.print_exc()
            return None


# Node registration
NODE_CLASS_MAPPINGS = {
    "FreeFuseLTXSimilarityExtractor": FreeFuseLTXSimilarityExtractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeFuseLTXSimilarityExtractor": "🎬 FreeFuse LTX Similarity Extractor",
}
