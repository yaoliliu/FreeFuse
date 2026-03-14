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
                    "default": 24, "min": 0, "max": 47,
                    "tooltip": "Which transformer block to collect from (0-47 for LTX-Video)"
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
                    "default": 4000.0, "min": 0.0, "max": 10000.0, "step": 100.0,
                    "tooltip": "Temperature for similarity computation"
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

LTX-Video has 48 transformer blocks with 32 attention heads each.
Recommended blocks: 20-30 for best concept separation.

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
                temperature=4000.0,
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
        print(f"  Collect block: {collect_block}/47")
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

        if "ltx" not in model_name and "avtransformer" not in model_name:
            print(f"[FreeFuse LTX Extract] WARNING: Model name doesn't contain 'ltx' or 'avtransformer'")
            print("  This node is designed for LTX-Video models")

        # Try to find the transformer blocks array
        # LTX-Video uses 'transformer_blocks'
        target_blocks = None
        if hasattr(diffusion_model, 'transformer_blocks'):
            target_blocks = diffusion_model.transformer_blocks
        elif hasattr(diffusion_model, 'layers'):
            target_blocks = diffusion_model.layers

        if target_blocks is not None:
            print(f"[FreeFuse LTX Extract] Found {len(target_blocks)} transformer blocks")

            if collect_block >= len(target_blocks):
                print(f"[FreeFuse LTX Extract] ERROR: collect_block {collect_block} >= num_blocks {len(target_blocks)}")
                collect_block = len(target_blocks) - 1
                print(f"[FreeFuse LTX Extract] Using block {collect_block} instead")

            target_block = target_blocks[collect_block]
            print(f"[FreeFuse LTX Extract] Target block type: {type(target_block).__name__}")

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

        # Move similarity maps to CPU to free VRAM
        if low_vram_mode:
            print(f"[FreeFuse LTX Extract] Moving similarity maps to CPU...")
            for name in collected_sim_maps:
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
        preview = self._create_preview(collected_sim_maps, T, H, W, preview_size)

        return (result, model, preview)

    def _create_preview(self, sim_maps, T, H, W, preview_size=512):
        """Create a colored preview image from similarity maps (first frame for video)."""
        import numpy as np

        # Get actual sequence length from first similarity map
        first_sim = list(sim_maps.values())[0]
        seq_len = first_sim.shape[1]

        print(f"[FreeFuse LTX Extract] Preview: T={T}, latent={H}x{W}, seq_len={seq_len}")

        # The similarity maps might be at a different resolution than the latent
        # Calculate spatial dimensions from sequence length
        spatial_len = seq_len // T if T > 0 and seq_len > T else seq_len
        
        # Try to find factors for spatial dimensions
        actual_H = actual_W = int(spatial_len ** 0.5)
        if actual_H * actual_W != spatial_len:
            # Try to find factors
            for i in range(int(spatial_len ** 0.5), 0, -1):
                if spatial_len % i == 0:
                    actual_H = i
                    actual_W = spatial_len // i
                    break
        
        # If dimensions don't work, use the seq_len directly as 2D
        if actual_H * actual_W != spatial_len:
            actual_H = int(spatial_len ** 0.5)
            actual_W = spatial_len // actual_H
            print(f"[FreeFuse LTX Extract] Using calculated spatial: {actual_W}x{actual_H}")

        print(f"[FreeFuse LTX Extract] Preview spatial: {actual_W}x{actual_H}")

        # Scale to preview size (longest side)
        out_h, out_w = actual_H * 8, actual_W * 8
        scale = preview_size / max(out_h, out_w)
        out_h = int(out_h * scale)
        out_w = int(out_w * scale)

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

        for idx, (name, sim) in enumerate(concept_maps[:len(colors)]):
            color = colors[idx % len(colors)]

            # Move sim to CPU for processing
            sim_cpu = sim.cpu()

            # Get the flat sequence (remove batch and last dim)
            if sim_cpu.dim() == 3:
                sim_flat = sim_cpu[0, :, 0]  # (seq_len,)
            else:
                sim_flat = sim_cpu.view(-1)  # (seq_len,)
            
            # Take first frame's worth of tokens
            tokens_per_frame = seq_len // T if T > 0 else seq_len
            first_frame_tokens = sim_flat[:tokens_per_frame]
            
            # Reshape to spatial dimensions
            try:
                sim_2d = first_frame_tokens.view(actual_H, actual_W)
            except RuntimeError:
                # Fallback: reshape to whatever fits
                sim_2d = first_frame_tokens[:actual_H*actual_W].view(actual_H, actual_W)

            # Resize to output size
            sim_resized = F.interpolate(
                sim_2d.unsqueeze(0).unsqueeze(0),
                size=(out_h, out_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)

            # Normalize for visualization
            sim_min = sim_resized.min()
            sim_max = sim_resized.max()
            if sim_max > sim_min:
                sim_norm = (sim_resized - sim_min) / (sim_max - sim_min)
            else:
                sim_norm = torch.ones_like(sim_resized) * 0.5

            # Add to overlay with color
            for c in range(3):
                overlay[c] += sim_norm * color[c]

        # Clamp and convert
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
    ):
        self.target_block = target_block
        self.block_index = block_index
        self.token_pos_maps = token_pos_maps
        self.temperature = temperature
        self.top_k_ratio = top_k_ratio
        self.img_len = img_len
        self.collected_sim_maps = collected_sim_maps
        self.attention_head_index = attention_head_index  # -1 = average all, 0-31 = specific head

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

        LTX-Video BasicAVTransformerBlock forward signature (kwargs):
          - v_context: (B, img_seq, dim) - video/image stream
          - a_context: (B, seq, dim) - audio/text stream  
          - v_timestep: timestep for video
          - a_timestep: timestep for audio
          - v_pe: RoPE frequencies for video (tuple)
          - a_pe: RoPE frequencies for audio (tuple)
          - attention_mask: optional
        """
        if self.collected:
            return

        # Capture LTX-Video specific parameter names
        if kwargs:
            # Use direct indexing instead of .get() - kwargs might be special dict
            self.cached_input = {
                "hidden_states": kwargs["v_context"] if "v_context" in kwargs else None,  # Video/image stream
                "encoder_hidden_states": kwargs["a_context"] if "a_context" in kwargs else None,  # Audio/text stream
                "timestep": kwargs.get("v_timestep"),
                "temb": kwargs.get("a_timestep"),
                "image_rotary_emb": kwargs.get("v_pe"),  # RoPE for video
                "attention_mask": kwargs.get("attention_mask"),
            }
            
            # Log what we captured
            hs = self.cached_input.get("hidden_states")
            txt = self.cached_input.get("encoder_hidden_states")
            if hs is not None and txt is not None:
                logging.info(f"[LTXAttentionHook] Captured LTX: v_context={hs.shape}, a_context={txt.shape}")
            else:
                logging.warning(f"[LTXAttentionHook] Failed to capture: v_context={hs is not None}, a_context={txt is not None}")
                logging.warning(f"[LTXAttentionHook] kwargs type: {type(kwargs)}, keys: {list(kwargs.keys()) if hasattr(kwargs, 'keys') else 'N/A'}")
        elif len(args) >= 1:
            # Fallback for positional args
            self.cached_input = {
                "hidden_states": args[0] if len(args) > 0 else None,
                "encoder_hidden_states": args[1] if len(args) > 1 else kwargs.get("a_context") if kwargs else None,
                "timestep": kwargs.get("v_timestep") if kwargs else None,
                "temb": kwargs.get("a_timestep") if kwargs else None,
                "image_rotary_emb": kwargs.get("v_pe") if kwargs else None,
                "attention_mask": kwargs.get("attention_mask") if kwargs else None,
            }
            if self.cached_input["hidden_states"] is not None:
                logging.info(f"[LTXAttentionHook] Captured from args: v_context={self.cached_input['hidden_states'].shape}")

        if self.cached_input:
            # Detach tensors to avoid gradient computation
            for key, value in self.cached_input.items():
                if isinstance(value, torch.Tensor):
                    self.cached_input[key] = value.detach()

    def _forward_hook(self, module, args, output):
        """Compute similarity maps from attention output."""
        if self.collected or self.cached_input is None:
            return output

        try:
            # Get the attention module - try multiple attribute names
            attn = getattr(module, 'attn', None)
            if attn is None:
                attn = getattr(module, 'attention', None)
            if attn is None:
                # LTX-Video might use different naming
                attn = getattr(module, 'attn1', None)
            if attn is None:
                attn = getattr(module, 'cross_attn', None)
            
            if attn is None:
                logging.warning("[LTXAttentionHook] No attention module found")
                logging.warning(f"  Block type: {type(module).__name__}")
                logging.warning(f"  Available attributes: {[a for a in dir(module) if not a.startswith('_') and 'attn' in a.lower()]}")
                return output

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

            if img_hidden is None or txt_hidden is None:
                logging.warning("[LTXAttentionHook] Missing cached inputs")
                logging.warning(f"  cached keys: {self.cached_input.keys() if self.cached_input else 'None'}")
                return output

            img_len = img_hidden.shape[1]
            cap_len = txt_hidden.shape[1]

            logging.info(f"[LTXAttentionHook] Computing similarity maps at block {self.block_index}")
            logging.info(f"  img_hidden: {img_hidden.shape}, txt_hidden: {txt_hidden.shape}")

            # Compute similarity maps
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
        txt_hidden: torch.Tensor,
        temb: Optional[torch.Tensor],
        image_rotary_emb: Optional[torch.Tensor],
        img_len: int,
        cap_len: int,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Compute similarity maps from block's attention computation."""

        try:
            # LTX-Video uses CrossAttention (not MMDiT)
            # - to_q, to_k, to_v: shared projections for both image and text
            # - q_norm, k_norm: QK normalization

            # Get QKV projection layers
            to_q = getattr(attn, 'to_q', None)
            to_k = getattr(attn, 'to_k', None)
            to_v = getattr(attn, 'to_v', None)

            if to_q is None or to_k is None or to_v is None:
                logging.warning("[LTXAttentionHook] Missing QKV projection layers")
                return None

            # Check if MMDiT-style (separate projections) or standard cross-attention
            add_q_proj = getattr(attn, 'add_q_proj', None)
            add_k_proj = getattr(attn, 'add_k_proj', None)
            add_v_proj = getattr(attn, 'add_v_proj', None)
            is_mmdit = (add_q_proj is not None and add_k_proj is not None and add_v_proj is not None)

            # Apply normalization if available
            img_attn_in = img_hidden
            txt_attn_in = txt_hidden

            if hasattr(module, 'norm1'):
                img_attn_in = module.norm1(img_hidden)
            if hasattr(module, 'norm1_context'):
                txt_attn_in = module.norm1_context(txt_hidden)

            if is_mmdit:
                # MMDiT-style: separate projections for image and text
                q = to_q(img_attn_in)
                k = to_k(img_attn_in)
                txt_q = add_q_proj(txt_attn_in)
                txt_k = add_k_proj(txt_attn_in)
            else:
                # LTX-Video CrossAttention: both to_q and to_k expect 4096 dim
                # Need to project text from 2048 to 4096
                q = to_q(img_attn_in)  # Video query: (B, img_seq, 4096)
                
                # Project text to image dimension for Q
                if not hasattr(self, 'text_proj_q'):
                    self.text_proj_q = torch.nn.Linear(2048, 4096, bias=False).to(txt_attn_in.device, dtype=txt_attn_in.dtype)
                txt_projected_q = self.text_proj_q(txt_attn_in)
                txt_q = to_q(txt_projected_q)
                
                # Check if to_k expects text dim or image dim
                k = None
                try:
                    # Try with text directly (cross-attention mode)
                    k = to_k(txt_attn_in)
                    txt_k = to_k(img_attn_in)
                    logging.info(f"[LTXAttentionHook] Cross-attention mode: to_k accepts text dim")
                except RuntimeError as e:
                    if "mat1 and mat2 shapes" in str(e):
                        # to_k expects same dim as to_q (self-attention with concatenated inputs)
                        logging.info(f"[LTXAttentionHook] Self-attention mode: to_k expects image dim")
                        logging.info(f"[LTXAttentionHook] Projecting text from 2048 to 4096")
                        
                        # Create a simple projection on the fly
                        # text_proj: 2048 -> 4096
                        if not hasattr(self, 'text_proj_k'):
                            self.text_proj_k = torch.nn.Linear(2048, 4096, bias=False).to(txt_attn_in.device, dtype=txt_attn_in.dtype)
                        
                        txt_projected = self.text_proj_k(txt_attn_in)
                        k = to_k(txt_projected)
                        txt_k = to_k(img_attn_in)
                        logging.info(f"[LTXAttentionHook] Text projected and processed successfully")
                    else:
                        raise
                
                if k is None:
                    logging.warning(f"[LTXAttentionHook] Could not compute K projection")
                    return None

            # Get QK norms if available
            norm_q = getattr(attn, 'norm_q', None)
            norm_k = getattr(attn, 'norm_k', None)
            norm_added_q = getattr(attn, 'norm_added_q', None)
            norm_added_k = getattr(attn, 'norm_added_k', None)

            # Apply QK norms
            if norm_q is not None:
                q = norm_q(q)
            if norm_k is not None:
                k = norm_k(k)
            if norm_added_q is not None:
                txt_q = norm_added_q(txt_q)
            if norm_added_k is not None:
                txt_k = norm_added_k(txt_k)

            # Reshape for multi-head attention
            # LTX-Video: 32 heads, head dim 128
            num_heads = 32
            head_dim = 128

            # Reshape q, k
            q = q.view(q.shape[0], q.shape[1], num_heads, head_dim).transpose(1, 2)
            k = k.view(k.shape[0], k.shape[1], num_heads, head_dim).transpose(1, 2)
            txt_q = txt_q.view(txt_q.shape[0], txt_q.shape[1], num_heads, head_dim).transpose(1, 2)
            txt_k = txt_k.view(txt_k.shape[0], txt_k.shape[1], num_heads, head_dim).transpose(1, 2)

            # Compute similarity: img_q @ txt_k^T (image queries attending to text keys)
            scale = 1.0 / (head_dim ** 0.5)
            similarity = torch.einsum('bhqd,bhkd->bhqk', q, txt_k) * scale
            similarity = similarity * (self.temperature / 1000.0)

            attention_weights = F.softmax(similarity, dim=-1)

            # Now compute similarity maps for each concept
            sim_maps = {}

            for concept_name, positions_list in self.token_pos_maps.items():
                if not positions_list or not positions_list[0]:
                    continue

                # Get token positions for this concept
                positions = positions_list[0]  # Use first prompt

                # Create a mask for concept tokens
                concept_mask = torch.zeros_like(attention_weights[..., -cap_len:])
                for pos in positions:
                    if pos < cap_len:
                        concept_mask[..., pos] = 1.0

                # Sum attention weights over concept tokens
                concept_attention = (attention_weights[..., -cap_len:] * concept_mask).sum(dim=-1)

                # Apply top-k filtering
                if self.top_k_ratio < 1.0:
                    k_val = max(1, int(concept_attention.shape[-1] * self.top_k_ratio))
                    top_k_vals, top_k_indices = torch.topk(concept_attention, k_val, dim=-1)
                    filtered = torch.zeros_like(concept_attention).scatter_(-1, top_k_indices, top_k_vals)
                    concept_attention = filtered

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

                # Reshape to (B, img_seq, 1)
                concept_attention = concept_attention.squeeze(1).unsqueeze(-1)  # (B, img_seq, 1)

                # Store similarity map
                sim_maps[f"sim_{concept_name}"] = concept_attention

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
