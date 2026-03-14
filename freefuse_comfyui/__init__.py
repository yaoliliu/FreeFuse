"""
ComfyUI-FreeFuse (v2.1 - with LoRA Mask Application)
"""

from .nodes import (
    # Original LoRA loaders (keep these)
    FreeFuseLoRALoader,
    FreeFuseLoRALoaderSimple,

    # Numbered LoRA loaders (only #6 kept)
    FreeFuse6LoraLoader,

    # Background loader
    FreeFuseBackgroundLoader,

    # Concept mapping
    FreeFuseConceptMap,
    FreeFuseTokenPositions,
    FreeFuseConceptMapSimple,

    # Sampling
    FreeFusePhase1Sampler,

    # Mask application
    FreeFuseMaskApplicator,
    FreeFuseMaskDebug,

    # Mask debug nodes
    FreeFuseMaskBankDebug,
    FreeFuseBankInspector,

    # Research nodes
    FreeFuseBlocksAnalysis,
    FreeFuseBaseAnalysis,
    FreeFuseRawSimilarityOverlay,

    # Test/Debug nodes
    FreeFuseTestSimilarityMaps,
    FreeFuseQwenSimilarityExtractor,
    FreeFuseQwenBlockGridExtractor,

    # LTX-Video nodes
    FreeFuseLTXSimilarityExtractor,
    FreeFuseLTXBlockGridExtractor,

    # Preview
    FreeFuseMaskPreview,

    # Utility nodes
    FreeFuseMaskTap,
    FreeFuseMaskReassemble,

    # Mappings
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

# Re-export for ComfyUI
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
