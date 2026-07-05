"""
FreeFuse Krea 2 Support (comfy/ldm/krea2/model.py :: SingleStreamDiT)

Krea 2 is a single-stream DiT:
    combined = torch.cat((context, img), dim=1)          # [txt, img] layout, same as Z-Image
    for block in self.blocks:
        combined = block(combined, tvec, freqs, None, transformer_options=...)

    SingleStreamBlock.forward(x, vec, freqs, mask=None, transformer_options={})
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        attn_in = (1 + prescale) * self.prenorm(x) + preshift
        attn_out = self.attn(attn_in, freqs, mask, transformer_options=...)
        x = x + pregate * attn_out
        ...

    Attention.forward(x, freqs=None, mask=None, transformer_options={})
        q, k, v, gate = wq(x), wk(x), wv(x), gate(x)
        q, k = qknorm(q, k)
        q, k = apply_rope(q, k, freqs)
        # GQA expansion (repeat_interleave)
        out = optimized_attention_masked(q, k, v, heads, mask=mask, skip_reshape=True, ...)
        return wo(out * sigmoid(gate))

KEY DIFFERENCE FROM FLUX / Z-IMAGE:
Krea2's Attention/Block already thread a `mask` argument through to the real attention
call, but the top-level forward loop always passes `mask=None`. So for Phase 2 (bias
injection) we do NOT need to reimplement RoPE/GQA/attention ourselves - we just hijack
that unused `mask` argument via a forward_pre_hook on `block.attn`, right before the
*original*, unmodified Attention.forward runs.

For Phase 1 (mask collection) we still need to manually recompute q/k, because
Attention.forward doesn't expose them - only the final projected output.

IMPORTANT - lifecycle: Krea2 has no ComfyUI replace-patch system, so these hooks are
plain torch module hooks attached directly to the underlying nn.Module (shared across
ModelPatcher clones). ALWAYS call .remove() on the returned replacer object after
sampling finishes (in a try/finally), or the hooks will leak into later generations.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange

from .attention_replace import (
    FreeFuseState,
    compute_z_image_similarity_maps_with_qkv,  # architecture-agnostic, reused as-is
)
from .tensor_debug import format_tensor_stats


# ──────────────────────────────────────────────────────────────────────────
# Phase 1: similarity-map collection
# ──────────────────────────────────────────────────────────────────────────

class FreeFuseKrea2BlockReplace:
    """
    QKV-based similarity collector for Krea 2 (SingleStreamDiT).

    Uses plain PyTorch hooks (no ComfyUI replace-patch system needed):
      1) forward_hook on `diffusion_model.txtfusion` captures cap_len (text length)
         for the current forward pass - txtfusion always runs before the block loop.
      2) forward_pre_hook on each target `diffusion_model.blocks[i]` captures the
         block's inputs (x, vec, freqs) and, on the target step, manually recomputes
         QKV/RoPE/GQA/attention-output to build similarity maps (mirrors
         comfy.ldm.krea2.model.Attention.forward exactly, read-only - never modifies
         the actual forward pass).
    """

    def __init__(
        self,
        state: FreeFuseState,
        diffusion_model,
        block_entries: List[Tuple[int, Any]],
    ):
        self.state = state
        self.diffusion_model = diffusion_model
        self._blocks: Dict[int, Any] = {int(idx): blk for idx, blk in block_entries}
        self._target_blocks = set(self._blocks.keys())
        self._cap_len: Optional[int] = None
        self._hook_handles: List[Any] = []

    def install(self) -> None:
        txtfusion = getattr(self.diffusion_model, "txtfusion", None)
        if txtfusion is None:
            logging.error("[FreeFuse Krea2] Cannot find `txtfusion` on diffusion model")
            return

        self._hook_handles.append(
            txtfusion.register_forward_hook(self._txtfusion_hook, with_kwargs=True)
        )
        for block_index, block in self._blocks.items():
            self._hook_handles.append(
                block.register_forward_pre_hook(
                    self._make_pre_hook(block_index), with_kwargs=True
                )
            )
        logging.info(
            f"[FreeFuse Krea2] Installed QKV-based collectors on blocks "
            f"{sorted(self._target_blocks)}"
        )

    def remove(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def _txtfusion_hook(self, module, args, kwargs, output):
        # TextFusionTransformer.forward returns (B, cap_len, txtdim)
        if torch.is_tensor(output):
            self._cap_len = int(output.shape[1])

    def _make_pre_hook(self, block_index: int) -> Callable:
        def _pre_hook(module, args, kwargs):
            state = self.state
            if state.phase != "collect":
                return None
            # SingleStreamBlock.forward(self, x, vec, freqs, mask=None, transformer_options={})
            if len(args) < 3:
                return None
            x, vec, freqs = args[0], args[1], args[2]
            transformer_options = (kwargs or {}).get("transformer_options", {})
            current_step = transformer_options.get("sigmas_index", state.current_step)

            if not state.is_collect_step(current_step, block_index):
                return None

            try:
                self._extract_and_compute(module, x, vec, freqs, block_index)
            except Exception as e:
                logging.error(
                    f"[FreeFuse Krea2] QKV extraction failed at block {block_index}: {e}"
                )
                import traceback
                traceback.print_exc()
            return None  # never modify the Phase-1 forward pass

        return _pre_hook

    def _extract_and_compute(self, block, x, vec, freqs, block_index: int) -> None:
        from comfy.ldm.flux.math import apply_rope

        state = self.state
        cap_len = self._cap_len
        if cap_len is None:
            logging.warning("[FreeFuse Krea2] cap_len not captured yet; skipping collection")
            return

        seqlen_total = x.shape[1]
        img_len = seqlen_total - cap_len
        if img_len <= 0:
            logging.warning(f"[FreeFuse Krea2] Non-positive img_len ({img_len}); skipping")
            return

        # ---- Mirror SingleStreamBlock.forward's pre-attention modulation ----
        prescale, preshift, pregate, postscale, postshift, postgate = block.mod(vec)
        attn_in = (1 + prescale) * block.prenorm(x) + preshift

        # ---- Mirror Attention.forward up to (not including) optimized_attention_masked ----
        attn = block.attn
        q = attn.wq(attn_in)
        k = attn.wk(attn_in)
        v = attn.wv(attn_in)

        q = rearrange(q, "B L (H D) -> B H L D", H=attn.heads)
        k = rearrange(k, "B L (H D) -> B H L D", H=attn.kvheads)
        v = rearrange(v, "B L (H D) -> B H L D", H=attn.kvheads)
        q, k = attn.qknorm(q, k)
        if freqs is not None:
            q, k = apply_rope(q, k, freqs)

        if attn.kvheads != attn.heads:
            rep = attn.heads // attn.kvheads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # (B, H, L, D) -> (B, L, H, D) to match compute_z_image_similarity_maps_with_qkv
        q_bld = q.transpose(1, 2)
        k_bld = k.transpose(1, 2)

        attn_out_4d = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)  # (B, H, L, D)
        attn_out = attn_out_4d.transpose(1, 2).reshape(x.shape[0], seqlen_total, -1)

        # Krea2 layout is [txt(cap_len), img(img_len)] - same convention as Z-Image.
        img_q = q_bld[:, cap_len:, :, :]
        txt_k = k_bld[:, :cap_len, :, :]
        img_attn_out = attn_out[:, cap_len:, :]

        if state.token_pos_maps:
            sim_maps = compute_z_image_similarity_maps_with_qkv(
                img_q=img_q,
                txt_k=txt_k,
                img_attn_out=img_attn_out,
                cap_len=cap_len,
                img_len=img_len,
                token_pos_maps=state.token_pos_maps,
                top_k_ratio=state.top_k_ratio,
                temperature=state.temperature,
                n_heads=attn.heads,
            )
            state.similarity_maps.update(sim_maps)
            if "block_similarity_maps" in state.collected_outputs:
                state.collected_outputs["block_similarity_maps"][block_index] = dict(sim_maps)
            if hasattr(state, "collected_blocks"):
                state.collected_blocks.add(block_index)

            logging.info(
                f"[FreeFuse Krea2] Collected {len(sim_maps)} similarity maps at block {block_index}"
            )
            for name, sm in sim_maps.items():
                logging.info(f"   {name}: {format_tensor_stats(sm, include_shape=True)}")


# ──────────────────────────────────────────────────────────────────────────
# Phase 2: attention-bias injection
# ──────────────────────────────────────────────────────────────────────────

class FreeFuseKrea2BiasBlockReplace:
    """
    Injects attention bias into Krea 2 by hijacking the `mask` argument that
    Attention.forward already threads through to optimized_attention_masked.

    We do NOT reimplement QKV/RoPE/GQA here - we overwrite the `mask` kwarg/arg
    right before the *original*, unmodified Attention.forward runs, via a
    forward_pre_hook on `block.attn`.
    """

    def __init__(
        self,
        lora_masks: Dict[str, torch.Tensor],
        token_pos_maps: Dict[str, List[List[int]]],
        config,
        diffusion_model,
        block_indices: List[int],
    ):
        self.lora_masks = lora_masks
        self.token_pos_maps = token_pos_maps
        self.config = config
        self.diffusion_model = diffusion_model
        self.block_indices = {int(i) for i in block_indices}
        self._cap_len: Optional[int] = None
        self._bias_cache: Dict[Tuple[int, int], Optional[torch.Tensor]] = {}
        self._hook_handles: List[Any] = []
        self._construct_attention_bias = None

    def install(self) -> None:
        from .attention_bias import construct_attention_bias
        self._construct_attention_bias = construct_attention_bias

        txtfusion = getattr(self.diffusion_model, "txtfusion", None)
        if txtfusion is None:
            logging.error("[FreeFuse Krea2] Cannot find `txtfusion` for bias injection")
            return
        self._hook_handles.append(
            txtfusion.register_forward_hook(self._txtfusion_hook, with_kwargs=True)
        )

        blocks = getattr(self.diffusion_model, "blocks", None)
        if blocks is None:
            logging.error("[FreeFuse Krea2] Cannot find `blocks` for bias injection")
            return

        for idx in self.block_indices:
            if idx < 0 or idx >= len(blocks):
                continue
            attn_module = blocks[idx].attn
            self._hook_handles.append(
                attn_module.register_forward_pre_hook(
                    self._make_bias_prehook(idx), with_kwargs=True
                )
            )
        logging.info(
            f"[FreeFuse Krea2] Installed bias injection on blocks {sorted(self.block_indices)}"
        )

    def remove(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def _txtfusion_hook(self, module, args, kwargs, output):
        if torch.is_tensor(output):
            self._cap_len = int(output.shape[1])

    def _get_or_build_bias(self, img_len, cap_len, device, dtype):
        key = (img_len, cap_len)
        if key in self._bias_cache:
            cached = self._bias_cache[key]
            return cached.to(device=device, dtype=dtype) if cached is not None else None

        bias = self._construct_attention_bias(
            lora_masks=self.lora_masks,
            token_pos_maps=self.token_pos_maps,
            txt_seq_len=cap_len,
            img_seq_len=img_len,
            bias_scale=self.config.bias_scale,
            positive_bias_scale=self.config.positive_bias_scale,
            bidirectional=self.config.bidirectional,
            use_positive_bias=self.config.use_positive_bias,
            device=device,
            dtype=dtype,
        )
        self._bias_cache[key] = bias
        return bias

    def _make_bias_prehook(self, block_index: int) -> Callable:
        def _prehook(module, args, kwargs):
            # Attention.forward(self, x, freqs=None, mask=None, transformer_options={})
            if not args:
                return None
            x = args[0]
            cap_len = self._cap_len
            if cap_len is None:
                return None
            seqlen = x.shape[1]
            img_len = seqlen - cap_len
            if img_len <= 0:
                return None

            bias = self._get_or_build_bias(img_len, cap_len, x.device, x.dtype)
            if bias is None:
                return None
            if bias.dim() == 3:
                bias = bias.unsqueeze(1)  # -> (B, 1, seq, seq), broadcasts over heads

            new_args = list(args)
            new_kwargs = dict(kwargs) if kwargs else {}
            if len(new_args) >= 3:
                new_args[2] = bias
            else:
                new_kwargs["mask"] = bias
            return tuple(new_args), new_kwargs

        return _prehook


# ──────────────────────────────────────────────────────────────────────────
# Entry points (mirror apply_freefuse_replace_patches / apply_attention_bias_patches)
# ──────────────────────────────────────────────────────────────────────────

def apply_krea2_replace_patches(
    model,
    state: FreeFuseState,
    krea2_collect_blocks: Optional[List[int]] = None,
) -> Optional[FreeFuseKrea2BlockReplace]:
    """Install Phase-1 collection hooks on a Krea2 model. Returns the replacer
    (caller MUST call .remove() when sampling finishes)."""
    diffusion_model = model.model.diffusion_model
    blocks = getattr(diffusion_model, "blocks", None)
    if blocks is None:
        logging.error("[FreeFuse Krea2] Cannot find Krea2 `blocks` in diffusion model")
        return None

    if krea2_collect_blocks is None:
        krea2_collect_blocks = [state.collect_block]

    max_idx = len(blocks) - 1
    valid_blocks = [b for b in krea2_collect_blocks if 0 <= b <= max_idx]
    if not valid_blocks:
        fallback = min(max(state.collect_block, 0), max_idx)
        logging.warning(
            f"[FreeFuse Krea2] No valid blocks in requested range; falling back to {fallback}."
        )
        valid_blocks = [fallback]
        state.collect_block = fallback
        state.collect_block_end = fallback

    block_entries = [(idx, blocks[idx]) for idx in valid_blocks]
    replacer = FreeFuseKrea2BlockReplace(state, diffusion_model, block_entries)
    replacer.install()
    return replacer


def apply_krea2_bias_patches(
    model,
    lora_masks: Dict[str, torch.Tensor],
    token_pos_maps: Dict[str, List[List[int]]],
    config,
    block_indices: Optional[List[int]] = None,
):
    """
    Install Phase-2 bias injection for Krea 2, scoped safely to `model.model_options`.

    Krea2 has no per-ModelPatcher-clone replace-patch system (unlike Flux/Z-Image),
    so raw hooks installed once here and left running would remain attached to the
    *shared* underlying nn.Module for as long as the process lives - silently
    corrupting any later, unrelated Krea2 generation that reuses the same
    checkpoint (even without FreeFuse).

    Instead we wrap `model_options["model_function_wrapper"]` - a ComfyUI extension
    point called once per individual model forward call (each sampling step). We
    install the bias hooks immediately before calling the real forward function
    and remove them immediately after, inside the same call. That ties their
    lifetime exactly to model_options on THIS model clone; a fresh `model.clone()`
    used elsewhere never sees them.

    Composes with any existing model_function_wrapper instead of overwriting it,
    so this stays compatible with other custom nodes (e.g. ControlNet helpers).
    """
    diffusion_model = model.model.diffusion_model
    blocks = getattr(diffusion_model, "blocks", None)
    if blocks is None:
        logging.error("[FreeFuse Krea2] Cannot find `blocks` for bias injection")
        return None

    if block_indices is None:
        # Default: last half of blocks (mirrors Z-Image's "last_half" bias_blocks option).
        n = len(blocks)
        block_indices = list(range(n // 2, n))

    previous_wrapper = model.model_options.get("model_function_wrapper")

    def krea2_bias_wrapper(apply_model_fn, args):
        replacer = FreeFuseKrea2BiasBlockReplace(
            lora_masks, token_pos_maps, config, diffusion_model, block_indices
        )
        replacer.install()
        try:
            if previous_wrapper is not None:
                return previous_wrapper(apply_model_fn, args)
            return apply_model_fn(args["input"], args["timestep"], **args["c"])
        finally:
            replacer.remove()

    model.model_options["model_function_wrapper"] = krea2_bias_wrapper
    logging.info(
        f"[FreeFuse Krea2] Registered per-call bias wrapper for blocks "
        f"{sorted(set(int(i) for i in block_indices))} (scoped to this model clone only)"
    )
    return krea2_bias_wrapper


__all__ = [
    "FreeFuseKrea2BlockReplace",
    "FreeFuseKrea2BiasBlockReplace",
    "apply_krea2_replace_patches",
    "apply_krea2_bias_patches",
]
