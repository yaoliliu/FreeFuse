"""
FreeFuse Core Utilities

Attention hooks, mask generation, token utilities, and LoRA mask hooks
for ComfyUI integration.
"""

from .attention import (
    FreeFuseAttentionCollector,
    FreeFuseFluxDoubleBlockPatch,
    apply_freefuse_patches,
    compute_similarity_maps_from_attention,
    compute_flux_similarity_maps,
    compute_sdxl_similarity_maps,
)

# New replace patch based attention hooks (recommended)
from .attention_replace import (
    FreeFuseState,
    FreeFuseFluxBlockReplace,
    FreeFuseFluxAttentionReplace,
    FreeFuseSDXLAttnReplace,
    apply_freefuse_replace_patches,
    compute_flux_similarity_maps_from_outputs,
    compute_flux_similarity_maps_with_qkv,
)

from .mask_utils import (
    generate_masks,
    balanced_argmax,
    resize_mask,
)

from .token_utils import (
    find_concept_positions,
    find_concept_positions_t5,
    find_concept_positions_clip,
    find_background_positions,
    find_eos_position_t5,
    detect_model_type,
    get_tokenizer_for_model,
    compute_token_position_maps,
)

from .lora_mask_hook import (
    FreeFuseMaskedBypassHook,
    FreeFuseMaskManager,
    FreeFuseAdapterConfig,
    create_freefuse_injections,
)

# New bypass-based mask application (Plan A implementation)
from .freefuse_bypass import (
    FreeFuseBypassForwardHook,
    FreeFuseBypassInjectionManager,
    create_freefuse_bypass_from_model,
)

# Fixed bypass LoRA loader for Flux fused QKV weights
from .bypass_lora_loader import (
    OffsetBypassForwardHook,
    MultiAdapterBypassForwardHook,
    OffsetBypassInjectionManager,
    load_bypass_lora_for_models_fixed,
)

# Attention bias mechanism
from .attention_bias import (
    construct_attention_bias,
    construct_attention_bias_sdxl,
    apply_attention_bias_to_weights,
    get_attention_bias_for_layer,
    AttentionBiasConfig,
)

# Attention bias patches for injection
from .attention_bias_patch import (
    FreeFuseFluxBiasBlockReplace,
    FreeFuseFluxBiasSingleBlockReplace,
    FreeFuseSDXLBiasAttnReplace,
    apply_attention_bias_patches,
)

__all__ = [
    # Attention
    "FreeFuseAttentionCollector",
    "FreeFuseFluxDoubleBlockPatch",
    "apply_freefuse_patches",
    "compute_similarity_maps_from_attention",
    "compute_flux_similarity_maps",
    "compute_sdxl_similarity_maps",
    # New replace patch based hooks (recommended)
    "FreeFuseState",
    "FreeFuseFluxBlockReplace",
    "FreeFuseFluxAttentionReplace",
    "FreeFuseSDXLAttnReplace",
    "apply_freefuse_replace_patches",
    "compute_flux_similarity_maps_from_outputs",
    "compute_flux_similarity_maps_with_qkv",
    # Masks
    "generate_masks",
    "balanced_argmax",
    "resize_mask",
    # Token utilities
    "find_concept_positions",
    "find_concept_positions_t5",
    "find_concept_positions_clip",
    "find_background_positions",
    "find_eos_position_t5",
    "detect_model_type",
    "get_tokenizer_for_model",
    "compute_token_position_maps",
    # LoRA mask hooks
    "FreeFuseMaskedBypassHook",
    "FreeFuseMaskManager",
    "FreeFuseAdapterConfig",
    "create_freefuse_injections",
    # New bypass-based mask application (Plan A)
    "FreeFuseBypassForwardHook",
    "FreeFuseBypassInjectionManager",
    "create_freefuse_bypass_from_model",
    # Fixed bypass LoRA loader for Flux
    "OffsetBypassForwardHook",
    "MultiAdapterBypassForwardHook",
    "OffsetBypassInjectionManager",
    "load_bypass_lora_for_models_fixed",
    # Attention bias
    "construct_attention_bias",
    "construct_attention_bias_sdxl",
    "apply_attention_bias_to_weights",
    "get_attention_bias_for_layer",
    "AttentionBiasConfig",
    "FreeFuseFluxBiasBlockReplace",
    "FreeFuseFluxBiasSingleBlockReplace",
    "FreeFuseSDXLBiasAttnReplace",
    "apply_attention_bias_patches",
]

