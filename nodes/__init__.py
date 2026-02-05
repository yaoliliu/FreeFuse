"""
FreeFuse ComfyUI Nodes Package (v2 - with LoRA Mask Application)

Provides nodes for multi-concept LoRA composition using the FreeFuse algorithm.

Key Nodes:
- FreeFuseLoRALoader: Load LoRA in bypass mode (keeps weights separate)
- FreeFuseConceptMap: Define concepts and their trigger words
- FreeFuseTokenPositions: Compute token positions for concepts
- FreeFusePhase1Sampler: Collect attention and generate spatial masks
- FreeFuseMaskApplicator: Apply masks to LoRA outputs
"""

from .lora_loader import (
    FreeFuseLoRALoader,
    FreeFuseLoRALoaderSimple,
    NODE_CLASS_MAPPINGS as LORA_NODE_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as LORA_DISPLAY_MAPPINGS,
)
from .concept_map import (
    FreeFuseConceptMap,
    FreeFuseTokenPositions,
    FreeFuseConceptMapSimple,
)
from .sampler import FreeFusePhase1Sampler
from .preview import FreeFuseMaskPreview
from .mask_applicator import (
    FreeFuseMaskApplicator,
    FreeFuseMaskDebug,
    NODE_CLASS_MAPPINGS as MASK_NODE_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as MASK_DISPLAY_MAPPINGS,
)
from .attention_bias import (
    FreeFuseAttentionBias,
    FreeFuseAttentionBiasVisualize,
    NODE_CLASS_MAPPINGS as ATTN_BIAS_NODE_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as ATTN_BIAS_DISPLAY_MAPPINGS,
)

# Combine all node mappings
NODE_CLASS_MAPPINGS = {
    "FreeFuseLoRALoader": FreeFuseLoRALoader,
    "FreeFuseLoRALoaderSimple": FreeFuseLoRALoaderSimple,
    "FreeFuseConceptMap": FreeFuseConceptMap,
    "FreeFuseTokenPositions": FreeFuseTokenPositions,
    "FreeFuseConceptMapSimple": FreeFuseConceptMapSimple,
    "FreeFusePhase1Sampler": FreeFusePhase1Sampler,
    "FreeFuseMaskPreview": FreeFuseMaskPreview,
    "FreeFuseMaskApplicator": FreeFuseMaskApplicator,
    "FreeFuseMaskDebug": FreeFuseMaskDebug,
    # Attention bias nodes
    "FreeFuseAttentionBias": FreeFuseAttentionBias,
    "FreeFuseAttentionBiasVisualize": FreeFuseAttentionBiasVisualize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeFuseLoRALoader": "FreeFuse LoRA Loader (Bypass)",
    "FreeFuseLoRALoaderSimple": "FreeFuse LoRA Loader (Simple)",
    "FreeFuseConceptMap": "FreeFuse Concept Map",
    "FreeFuseTokenPositions": "FreeFuse Token Positions",
    "FreeFuseConceptMapSimple": "FreeFuse Concept Map (Simple)",
    "FreeFusePhase1Sampler": "FreeFuse Phase 1 Sampler",
    "FreeFuseMaskPreview": "FreeFuse Mask Preview",
    "FreeFuseMaskApplicator": "FreeFuse Mask Applicator",
    "FreeFuseMaskDebug": "FreeFuse Mask Debug",
    # Attention bias nodes
    "FreeFuseAttentionBias": "FreeFuse Attention Bias",
    "FreeFuseAttentionBiasVisualize": "FreeFuse Attention Bias Visualize",
}

__all__ = [
    # LoRA loaders
    "FreeFuseLoRALoader",
    "FreeFuseLoRALoaderSimple",
    # Concept mapping
    "FreeFuseConceptMap",
    "FreeFuseTokenPositions",
    "FreeFuseConceptMapSimple",
    # Sampling
    "FreeFusePhase1Sampler",
    # Mask application
    "FreeFuseMaskApplicator",
    "FreeFuseMaskDebug",
    # Attention bias
    "FreeFuseAttentionBias",
    "FreeFuseAttentionBiasVisualize",
    # Preview
    "FreeFuseMaskPreview",
    # Mappings
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
