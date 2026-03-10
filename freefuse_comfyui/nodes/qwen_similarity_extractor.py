"""
FreeFuse Qwen-Image Similarity Map Extractor

Extracts real similarity maps from Qwen-Image during sampling.
This node hooks into the model's attention mechanism to collect attention patterns.
"""

import torch
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Callable, Any
import comfy.sample
import comfy.samplers


class FreeFuseQwenSimilarityExtractor:
    """
    Extract similarity maps from Qwen-Image during sampling.
    
    This uses a pre-hook on the Qwen-Image transformer block to capture
    QKV states and compute similarity maps using the FreeFuse algorithm.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Qwen-Image model"
                }),
                "positive": ("CONDITIONING", {
                    "tooltip": "Positive conditioning"
                }),
                "negative": ("CONDITIONING", {
                    "tooltip": "Negative conditioning"
                }),
                "latent": ("LATENT", {
                    "tooltip": "Latent image (use 32x32=1024 tokens or 48x48=2304 tokens for lower VRAM)"
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
                    "default": 20, "min": 0, "max": 59,
                    "tooltip": "Which transformer block to collect from (0-59 for Qwen-Image)"
                }),
                "cfg": ("FLOAT", {
                    "default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
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
                    "default": -1, "min": -1, "max": 63, "step": 1,
                    "tooltip": "Select specific attention head (0-63). Default -1 = average all 64 heads. Use specific head for cleaner hotspots!"
                }),
            }
        }

    RETURN_TYPES = ("FREEFUSE_MASKS", "MODEL", "IMAGE")
    RETURN_NAMES = ("raw_similarity", "model", "preview")
    FUNCTION = "extract"
    CATEGORY = "FreeFuse/Debug"

    DESCRIPTION = """Extracts real similarity maps from Qwen-Image during sampling.

Hooks into the model's attention mechanism at a specific block and step,
then computes similarity maps using the FreeFuse algorithm.

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
            print("[FreeFuse Qwen Extract] ERROR: No token positions found in freefuse_data")
            print("  Make sure to connect FreeFuseTokenPositions output to freefuse_data input")
            empty_result = {"masks": {}, "similarity_maps": {}}
            return (empty_result, model, torch.zeros(1, 512, 512, 3))

        print(f"[FreeFuse Qwen Extract] Starting extraction")
        print(f"  Concepts: {list(concepts.keys())}")
        print(f"  Token positions: {list(token_pos_maps.keys())}")
        print(f"  Collect step: {collect_step}/{steps}")
        print(f"  Collect block: {collect_block}")
        print(f"  Temperature: {temperature}, top_k_ratio: {top_k_ratio}")
        print(f"  Low VRAM mode: {low_vram_mode}")
        
        # Aggressive VRAM cleanup at start
        if low_vram_mode and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            free_mem, total_mem = torch.cuda.mem_get_info()
            print(f"[FreeFuse Qwen Extract] VRAM: {free_mem / 1024:.0f} MB free / {total_mem / 1024:.0f} MB total")

        # Clone model to avoid modifying original
        model_clone = model.clone()
        
        # Get latent dimensions
        latent_tensor = latent["samples"]
        latent_h, latent_w = latent_tensor.shape[2], latent_tensor.shape[3]
        img_len = latent_h * latent_w
        
        # Storage for collected similarity maps
        collected_sim_maps = {}
        collection_done = False
        
        # Get the diffusion model to find the target block
        diffusion_model = model.model.diffusion_model
        
        # Check if this is actually a Qwen-Image model
        model_name = diffusion_model.__class__.__name__.lower()
        print(f"[FreeFuse Qwen Extract] Model type: {model_name}")
        
        if "qwen" not in model_name and "qwenimage" not in model_name:
            print(f"[FreeFuse Qwen Extract] WARNING: Model name doesn't contain 'qwen'")
            print("  This node is designed for Qwen-Image models")
        
        # Try to find the transformer blocks array
        # Qwen-Image uses 'transformer_blocks', not 'layers'
        target_blocks = None
        if hasattr(diffusion_model, 'transformer_blocks'):
            target_blocks = diffusion_model.transformer_blocks
        elif hasattr(diffusion_model, 'layers'):
            target_blocks = diffusion_model.layers
        
        if target_blocks is not None:
            print(f"[FreeFuse Qwen Extract] Found {len(target_blocks)} transformer blocks")
            
            if collect_block >= len(target_blocks):
                print(f"[FreeFuse Qwen Extract] ERROR: collect_block {collect_block} >= num_blocks {len(target_blocks)}")
                collect_block = len(target_blocks) - 1
                print(f"[FreeFuse Qwen Extract] Using block {collect_block} instead")
            
            target_block = target_blocks[collect_block]
            print(f"[FreeFuse Qwen Extract] Target block type: {type(target_block).__name__}")
            
            # Install hook on target block
            hook = QwenAttentionHook(
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
            print(f"[FreeFuse Qwen Extract] Hook installed on block {collect_block}")
            if attention_head_index >= 0:
                print(f"[FreeFuse Qwen Extract] Using SINGLE HEAD {attention_head_index} (not averaged)")
            else:
                print(f"[FreeFuse Qwen Extract] Averaging all 64 heads")
        else:
            print(f"[FreeFuse Qwen Extract] ERROR: Model has no 'transformer_blocks' or 'layers' attribute")
            print(f"  Available attributes: {[a for a in dir(diffusion_model) if not a.startswith('_')]}")
            empty_result = {"masks": {}, "similarity_maps": {}}
            return (empty_result, model, torch.zeros(1, 512, 512, 3))
        
        # Create noise
        # Qwen-Image expects 5D latents: (B, C, T, H, W)
        # Add temporal dimension if missing
        if latent_tensor.dim() == 4:
            print(f"[FreeFuse Qwen Extract] Adding temporal dimension to latent: {latent_tensor.shape} -> {(latent_tensor.unsqueeze(2)).shape}")
            latent_tensor = latent_tensor.unsqueeze(2)  # Insert T dimension at index 2
        
        # Aggressive VRAM cleanup before sampling
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            free_mem, _ = torch.cuda.mem_get_info()
            print(f"[FreeFuse Qwen Extract] VRAM before sampling: {free_mem / 1024:.0f} MB available")
        
        noise = comfy.sample.prepare_noise(latent_tensor, seed, None)
        
        # Track current step
        current_step = [0]
        
        def step_callback(step, x0, x, total_steps):
            current_step[0] = step
            nonlocal collection_done
            
            # Check if we've passed the collect step and have maps
            if collection_done and step > collect_step:
                print(f"[FreeFuse Qwen Extract] Collection complete, stopping early")
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
                force_full_denoise=False,  # Don't fully denoise since we exit early
                noise_mask=None,
                callback=step_callback,
                seed=seed,
            )
        except EarlyStopException:
            print(f"[FreeFuse Qwen Extract] Early stop triggered")
            # Aggressive VRAM cleanup after early stop
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                free_mem, _ = torch.cuda.mem_get_info()
                print(f"[FreeFuse Qwen Extract] VRAM after early stop: {free_mem / 1024:.0f} MB available")
        except Exception as e:
            print(f"[FreeFuse Qwen Extract] Sampling error: {e}")
            import traceback
            traceback.print_exc()
        
        # Remove hook
        hook.remove()
        print(f"[FreeFuse Qwen Extract] Hook removed")
        
        # Aggressive VRAM cleanup after hook removal
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            free_mem, _ = torch.cuda.mem_get_info()
            print(f"[FreeFuse Qwen Extract] VRAM after hook removal: {free_mem / 1024:.0f} MB available")

        # Check if we got any similarity maps
        if not collected_sim_maps:
            print(f"[FreeFuse Qwen Extract] WARNING: No similarity maps collected!")
            print("  The hook may not have been called during sampling")
            empty_result = {"masks": {}, "similarity_maps": {}}
            return (empty_result, model, torch.zeros(1, 512, 512, 3))

        print(f"[FreeFuse Qwen Extract] Collected {len(collected_sim_maps)} similarity maps")
        for name, sim_map in collected_sim_maps.items():
            if isinstance(sim_map, torch.Tensor):
                print(f"  {name}: shape={sim_map.shape}, min={sim_map.min():.6f}, max={sim_map.max():.6f}")

        # Move similarity maps to CPU to free VRAM
        if low_vram_mode:
            print(f"[FreeFuse Qwen Extract] Moving similarity maps to CPU...")
            for name in collected_sim_maps:
                collected_sim_maps[name] = collected_sim_maps[name].cpu()
            torch.cuda.empty_cache()
            free_mem, _ = torch.cuda.mem_get_info()
            print(f"[FreeFuse Qwen Extract] VRAM after moving to CPU: {free_mem / 1024:.0f} MB available")

        # Create result
        result = {
            "masks": {},
            "similarity_maps": collected_sim_maps,
        }

        # Create preview image
        preview = self._create_preview(collected_sim_maps, latent_h, latent_w, preview_size)

        return (result, model, preview)
    
    def _create_preview(self, sim_maps, latent_h, latent_w, preview_size=512):
        """Create a colored preview image from similarity maps."""
        import numpy as np
        
        # Get actual sequence length from first similarity map
        first_sim = list(sim_maps.values())[0]
        seq_len = first_sim.shape[1]
        
        # Calculate actual latent dimensions from sequence length
        # For Qwen-Image: seq_len = latent_h * latent_w
        actual_latent_h = actual_latent_w = int(seq_len ** 0.5)
        if actual_latent_h * actual_latent_w != seq_len:
            # Try to find factors
            for i in range(int(seq_len ** 0.5), 0, -1):
                if seq_len % i == 0:
                    actual_latent_h = i
                    actual_latent_w = seq_len // i
                    break
        
        print(f"[FreeFuse Qwen Extract] Preview: seq_len={seq_len}, latent={actual_latent_w}x{actual_latent_h}")
        
        # Scale to preview size (longest side)
        out_h, out_w = actual_latent_h * 8, actual_latent_w * 8
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
            
            # Reshape sim map to 2D using actual latent dimensions
            if sim_cpu.dim() == 3:
                sim_2d = sim_cpu[0, :, 0].view(actual_latent_h, actual_latent_w)
            else:
                sim_2d = sim_cpu.view(actual_latent_h, actual_latent_w)
            
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
                # Simple linear normalization
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


class QwenAttentionHook:
    """
    Hook for extracting attention from Qwen-Image transformer blocks.
    
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
        self.attention_head_index = attention_head_index  # -1 = average all, 0-63 = specific head

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
        
        logging.info(f"[QwenAttentionHook] Installed on block {self.block_index}")
    
    def remove(self):
        """Remove hooks."""
        if self.pre_hook_handle is not None:
            self.pre_hook_handle.remove()
            self.pre_hook_handle = None
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
        logging.info(f"[QwenAttentionHook] Removed from block {self.block_index}")
    
    def _pre_hook(self, module, args, kwargs=None):
        """Capture Qwen-Image block inputs before forward pass.
        
        Qwen-Image Transformer Block forward signature (kwargs):
          - hidden_states: (B, img_seq, dim) - already flattened from 5D
          - encoder_hidden_states: (B, txt_seq, dim)
          - encoder_hidden_states_mask: optional
          - temb: timestep embedding
          - image_rotary_emb: RoPE frequencies
          - timestep_zero_index: optional
        """
        if self.collected:
            return
        
        # Capture kwargs which should contain the inputs
        if kwargs:
            self.cached_input = {
                "hidden_states": kwargs.get("hidden_states"),
                "encoder_hidden_states": kwargs.get("encoder_hidden_states"),
                "temb": kwargs.get("temb"),
                "image_rotary_emb": kwargs.get("image_rotary_emb"),
                "timestep_zero_index": kwargs.get("timestep_zero_index"),
            }
        elif len(args) >= 2:
            # Fallback for positional args
            self.cached_input = {
                "hidden_states": args[0] if len(args) > 0 else None,
                "encoder_hidden_states": args[1] if len(args) > 1 else None,
                "temb": args[2] if len(args) > 2 else kwargs.get("temb"),
                "image_rotary_emb": args[3] if len(args) > 3 else kwargs.get("image_rotary_emb"),
                "timestep_zero_index": kwargs.get("timestep_zero_index"),
            }
        
        if self.cached_input:
            # Detach tensors to avoid gradient computation
            for key, value in self.cached_input.items():
                if isinstance(value, torch.Tensor):
                    self.cached_input[key] = value.detach()
        
        # Debug logging on first capture
        if self.cached_input and not hasattr(self, '_logged_info'):
            hs = self.cached_input.get("hidden_states")
            txt = self.cached_input.get("encoder_hidden_states")
            if hs is not None and txt is not None:
                logging.info(f"[QwenAttentionHook] Captured: img={hs.shape}, txt={txt.shape}")
                self._logged_info = True
    
    def _forward_hook(self, module, args, output):
        """Compute similarity maps from attention output."""
        if self.collected or self.cached_input is None:
            return output
        
        try:
            # Get the attention module
            attn = getattr(module, 'attn', None)
            if attn is None:
                logging.warning("[QwenAttentionHook] No attention module found")
                return output
            
            # Get cached inputs
            img_hidden = self.cached_input.get("hidden_states")
            txt_hidden = self.cached_input.get("encoder_hidden_states")
            temb = self.cached_input.get("temb")
            image_rotary_emb = self.cached_input.get("image_rotary_emb")
            
            if img_hidden is None or txt_hidden is None:
                logging.warning("[QwenAttentionHook] Missing cached inputs")
                logging.warning(f"  cached keys: {self.cached_input.keys() if self.cached_input else 'None'}")
                return output
            
            img_len = img_hidden.shape[1]
            cap_len = txt_hidden.shape[1]
            
            logging.info(f"[QwenAttentionHook] Computing similarity maps at block {self.block_index}")
            logging.info(f"  img_hidden: {img_hidden.shape}, txt_hidden: {txt_hidden.shape}")
            logging.info(f"  temb: {temb.shape if temb is not None else None}")
            logging.info(f"  image_rotary_emb: {image_rotary_emb[0].shape if image_rotary_emb is not None and len(image_rotary_emb) > 0 else None}")
            
            # Try to extract QKV and compute similarity maps
            sim_maps = self._compute_similarity_from_block(
                module, attn, img_hidden, txt_hidden, temb, image_rotary_emb,
                img_len, cap_len
            )
            
            if sim_maps:
                self.collected_sim_maps.update(sim_maps)
                self.collected = True
                logging.info(f"[QwenAttentionHook] Collected {len(sim_maps)} similarity maps")
                for name, sm in sim_maps.items():
                    logging.info(f"  {name}: shape={sm.shape}, min={sm.min():.6f}, max={sm.max():.6f}")
            
        except Exception as e:
            logging.error(f"[QwenAttentionHook] Error computing similarity maps: {e}")
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
            from comfy.ldm.flux.math import apply_rope
            
            # Qwen-Image MMDiT structure:
            # - to_q, to_k, to_v: main projections (image stream)
            # - add_q_proj, add_k_proj, add_v_proj: additional projections (text/fusion stream)
            # - norm_q, norm_k, norm_added_q, norm_added_k: QK norms
            
            # Get QKV projection layers
            to_q = getattr(attn, 'to_q', None)
            to_k = getattr(attn, 'to_k', None)
            to_v = getattr(attn, 'to_v', None)
            
            # For MMDiT: use add_* projections for text stream
            add_q_proj = getattr(attn, 'add_q_proj', None)
            add_k_proj = getattr(attn, 'add_k_proj', None)
            add_v_proj = getattr(attn, 'add_v_proj', None)
            
            if to_q is None or to_k is None or to_v is None:
                logging.warning("[QwenAttentionHook] Missing main QKV projection layers")
                return None
            
            if add_q_proj is None or add_k_proj is None or add_v_proj is None:
                logging.warning("[QwenAttentionHook] Missing add_* QKV projection layers (MMDiT)")
                return None
            
            # Apply modulation if available
            img_attn_in = img_hidden
            txt_attn_in = txt_hidden
            
            # Image stream modulation - simplified to avoid shape mismatch
            if hasattr(module, 'img_norm1'):
                img_attn_in = module.img_norm1(img_hidden)
            
            # Text stream modulation
            if hasattr(module, 'txt_norm1'):
                txt_attn_in = module.txt_norm1(txt_hidden)
            
            # Project QKV for image stream
            img_q = to_q(img_attn_in)
            img_k = to_k(img_attn_in)
            img_v = to_v(img_attn_in)

            # Project QKV for text stream (using add_* projections)
            txt_q = add_q_proj(txt_attn_in)
            txt_k = add_k_proj(txt_attn_in)
            txt_v = add_v_proj(txt_attn_in)

            # Note: QK-norm is applied internally by the model during attention
            # We skip explicit norm application here to avoid shape mismatches

            # Reshape to multi-head
            # CRITICAL: Detect heads from RoPE tensor, NOT from attn.num_heads
            # Qwen-Image uses 64 heads, not 24 (Flux default)
            n_heads = 64  # Default for Qwen-Image

            # Try to detect actual head count from RoPE tensor
            if image_rotary_emb is not None and isinstance(image_rotary_emb, (list, tuple)) and len(image_rotary_emb) > 0:
                rope_tensor = image_rotary_emb[0]
                if rope_tensor is not None:
                    # Qwen-Image uses 6D RoPE: [B, 1, seq_len, num_heads, 2, 2]
                    if rope_tensor.dim() == 6:
                        detected_heads = rope_tensor.shape[3]  # Index 3 for 6D RoPE
                        if 16 <= detected_heads <= 128:
                            n_heads = detected_heads
                            logging.info(f"[QwenAttentionHook] Detected {n_heads} heads from 6D RoPE")
                    elif rope_tensor.dim() == 5:
                        # Other models: [B, seq_len, num_heads, 2, 2]
                        detected_heads = rope_tensor.shape[2]
                        if 16 <= detected_heads <= 128:
                            n_heads = detected_heads
                            logging.info(f"[QwenAttentionHook] Detected {n_heads} heads from 5D RoPE")
                    elif rope_tensor.dim() == 4:
                        # Standard 4D RoPE: [B, seq_len, num_heads, dim]
                        detected_heads = rope_tensor.shape[2]
                        if 16 <= detected_heads <= 128:
                            n_heads = detected_heads
                            logging.info(f"[QwenAttentionHook] Detected {n_heads} heads from 4D RoPE")

            logging.info(f"[QwenAttentionHook] Using n_heads={n_heads}, head_dim={img_q.shape[-1] // n_heads}")

            head_dim = img_q.shape[-1] // n_heads

            img_q = img_q.view(img_q.shape[0], img_q.shape[1], n_heads, head_dim)
            img_k = img_k.view(img_k.shape[0], img_k.shape[1], n_heads, head_dim)
            img_v = img_v.view(img_v.shape[0], img_v.shape[1], n_heads, head_dim)

            txt_q = txt_q.view(txt_q.shape[0], txt_q.shape[1], n_heads, head_dim)
            txt_k = txt_k.view(txt_k.shape[0], txt_k.shape[1], n_heads, head_dim)
            txt_v = txt_v.view(txt_v.shape[0], txt_v.shape[1], n_heads, head_dim)
            
            # Apply RoPE
            if image_rotary_emb is not None:
                try:
                    img_q, img_k = apply_rope(img_q, img_k, image_rotary_emb[0])
                    txt_q, txt_k = apply_rope(txt_q, txt_k, image_rotary_emb[0])
                except Exception as e:
                    logging.warning(f"[QwenAttentionHook] RoPE failed: {e}")
            
            # Compute attention output for self-modal similarity
            # Concatenate image and text streams for joint attention
            q = torch.cat([img_q, txt_q], dim=1)
            k = torch.cat([img_k, txt_k], dim=1)
            v = torch.cat([img_v, txt_v], dim=1)
            
            q_4d = q.transpose(1, 2)
            k_4d = k.transpose(1, 2)
            v_4d = v.transpose(1, 2)
            
            attn_out_4d = F.scaled_dot_product_attention(q_4d, k_4d, v_4d, dropout_p=0.0)
            attn_out = attn_out_4d.transpose(1, 2).reshape(
                img_hidden.shape[0], -1, n_heads * head_dim
            )
            img_attn_out = attn_out[:, :img_len, :]
            
            # Compute similarity maps using FreeFuse algorithm
            sim_maps = compute_qwen_similarity_maps(
                img_q=img_q[:, :img_len, :, :],
                txt_k=txt_k,
                img_attn_out=img_attn_out,
                cap_len=cap_len,
                img_len=img_len,
                token_pos_maps=self.token_pos_maps,
                top_k_ratio=self.top_k_ratio,
                temperature=self.temperature,
                n_heads=n_heads,
                attention_head_index=self.attention_head_index,
            )

            return sim_maps
            
        except Exception as e:
            logging.error(f"[QwenAttentionHook] _compute_similarity_from_block error: {e}")
            import traceback
            traceback.print_exc()
            return None


def compute_qwen_similarity_maps(
    img_q: torch.Tensor,
    txt_k: torch.Tensor,
    img_attn_out: torch.Tensor,
    cap_len: int,
    img_len: int,
    token_pos_maps: Dict[str, List[List[int]]],
    top_k_ratio: float = 0.3,
    temperature: float = 4000.0,
    n_heads: int = 64,
    attention_head_index: int = -1,  # -1 = average all, 0-63 = specific head
) -> Dict[str, torch.Tensor]:
    """
    Compute similarity maps for Qwen-Image using FreeFuse algorithm.
    
    Args:
        attention_head_index: If >= 0, use only this specific head (0-63)
                             If -1, average across all heads (default)
    """
    concept_sim_maps = {}
    if not token_pos_maps:
        return concept_sim_maps

    device = img_q.device
    B = img_q.shape[0]
    scale = 1.0 / 1000.0
    
    # Check if using single head
    use_single_head = (attention_head_index >= 0)
    if use_single_head:
        logging.info(f"[compute_qwen_similarity_maps] Using SINGLE HEAD {attention_head_index}")
        # Extract only the specified head
        img_q = img_q[:, :, attention_head_index:attention_head_index+1, :]
        txt_k = txt_k[:, :, attention_head_index:attention_head_index+1, :]
        # Reshape img_attn_out to extract single head
        head_dim = img_attn_out.shape[-1] // n_heads
        img_attn_out = img_attn_out.view(B, img_len, n_heads, head_dim)[:, :, attention_head_index, :]
        n_heads = 1

    # First pass: cross-attn scores per concept
    all_cross_attn_scores = {}

    for lora_name, positions_list in token_pos_maps.items():
        if lora_name.startswith("__"):
            continue
        pos = positions_list[0] if positions_list else []
        if not pos:
            continue

        pos_t = torch.tensor(pos, device=device, dtype=torch.long)
        pos_t = pos_t.clamp(0, cap_len - 1)

        concept_k = txt_k[:, pos_t, :, :]

        # Multi-head cross-attention
        weights = torch.einsum("bihd,bjhd->bhij", img_q, concept_k) * scale
        weights = F.softmax(weights, dim=2)
        
        if use_single_head:
            scores = weights[:, 0, :, :].mean(dim=-1)  # Single head
        else:
            scores = weights.mean(dim=1).mean(dim=-1)  # Average all heads
        
        all_cross_attn_scores[lora_name] = scores

    # Second pass: contrastive top-k + concept attention
    n_concepts = len(all_cross_attn_scores)

    for lora_name in list(all_cross_attn_scores.keys()):
        scores = all_cross_attn_scores[lora_name] * max(1, n_concepts)
        for other in all_cross_attn_scores:
            if other != lora_name:
                scores = scores - all_cross_attn_scores[other]

        k_count = max(1, int(img_len * top_k_ratio))
        _, topk_idx = torch.topk(scores, k_count, dim=-1)

        expanded = topk_idx.unsqueeze(-1).expand(-1, -1, img_attn_out.shape[-1])
        core_tokens = torch.gather(img_attn_out, dim=1, index=expanded)

        self_modal_sim = torch.bmm(core_tokens, img_attn_out.transpose(-1, -2))
        sim_avg = self_modal_sim.mean(dim=1, keepdim=True).transpose(1, 2)

        sim_map = F.softmax(sim_avg / temperature, dim=1)
        concept_sim_maps[lora_name] = sim_map

    # Handle background
    for bg_key in ["__background__", "__bg__"]:
        if bg_key in token_pos_maps:
            bg_pos = token_pos_maps[bg_key][0] if token_pos_maps[bg_key] else []
            if bg_pos:
                bg_pos_t = torch.tensor(bg_pos, device=device, dtype=torch.long)
                bg_pos_t = bg_pos_t.clamp(0, cap_len - 1)

                bg_k = txt_k[:, bg_pos_t, :, :]
                bg_weights = torch.einsum("bihd,bjhd->bhij", img_q, bg_k) * scale
                bg_weights = F.softmax(bg_weights, dim=2)
                
                if use_single_head:
                    bg_scores = bg_weights[:, 0, :, :].mean(dim=-1)
                else:
                    bg_scores = bg_weights.mean(dim=1).mean(dim=-1)

                k_count = max(1, int(img_len * top_k_ratio))
                _, bg_topk = torch.topk(bg_scores, k_count, dim=-1)

                bg_exp = bg_topk.unsqueeze(-1).expand(-1, -1, img_attn_out.shape[-1])
                bg_core = torch.gather(img_attn_out, dim=1, index=bg_exp)

                bg_sim = torch.bmm(bg_core, img_attn_out.transpose(-1, -2))
                bg_sim_avg = bg_sim.mean(dim=1, keepdim=True).transpose(1, 2)
                bg_sim_map = F.softmax(bg_sim_avg / temperature, dim=1)

                concept_sim_maps[bg_key] = bg_sim_map
                break

    return concept_sim_maps


# Export node mappings
NODE_CLASS_MAPPINGS = {
    "FreeFuseQwenSimilarityExtractor": FreeFuseQwenSimilarityExtractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeFuseQwenSimilarityExtractor": "🔬 FreeFuse Qwen Similarity Extractor",
}
