"""
FreeFuse Core Utilities

Attention hooks, mask generation, token utilities, and LoRA mask hooks
for ComfyUI integration - Z-Image only version.
"""

# New replace patch based attention hooks (recommended)
from .attention_replace import (
    FreeFuseState,
    FreeFuseZImageBlockReplace,
    apply_freefuse_replace_patches,
    compute_z_image_similarity_maps_with_qkv,
)

from .mask_utils import (
    generate_masks,
    balanced_argmax,
    resize_mask,
    make_freefuse_masks_json_serializable,
)

from .json_serialization import (
    FreeFuseJSONEncoder,
    make_freefuse_data_json_serializable,
    make_freefuse_masks_json_serializable as make_masks_json_serializable,
    safe_json_dumps,
)

from .token_utils import (
    find_concept_positions,
    find_background_positions,
    compute_token_position_maps,
    LUMINA2_SYSTEM_PROMPT,
)

# Fixed bypass LoRA loader for Z-Image fused QKV weights
from .bypass_lora_loader import (
    OffsetBypassForwardHook,
    MultiAdapterBypassForwardHook,
    OffsetBypassInjectionManager,
    load_bypass_lora_for_models_fixed,
)

# Attention bias mechanism
from .attention_bias import (
    construct_attention_bias,
    apply_attention_bias_to_weights,
    get_attention_bias_for_layer,
    AttentionBiasConfig,
)

# Attention bias patches for injection
from .attention_bias_patch import (
    FreeFuseZImageBiasBlockReplace,
    apply_attention_bias_patches,
)

__all__ = [
    # New replace patch based hooks (recommended)
    "FreeFuseState",
    "FreeFuseZImageBlockReplace",
    "apply_freefuse_replace_patches",
    "compute_z_image_similarity_maps_with_qkv",
    # Masks
    "generate_masks",
    "balanced_argmax",
    "resize_mask",
    # Token utilities
    "find_concept_positions",
    "find_background_positions",
    "compute_token_position_maps",
    "LUMINA2_SYSTEM_PROMPT",
    # Fixed bypass LoRA loader for Z-Image
    "OffsetBypassForwardHook",
    "MultiAdapterBypassForwardHook",
    "OffsetBypassInjectionManager",
    "load_bypass_lora_for_models_fixed",
    # Attention bias
    "construct_attention_bias",
    "apply_attention_bias_to_weights",
    "get_attention_bias_for_layer",
    "AttentionBiasConfig",
    "FreeFuseZImageBiasBlockReplace",
    "apply_attention_bias_patches",
]
