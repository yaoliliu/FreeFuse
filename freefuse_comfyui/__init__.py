"""
ComfyUI-FreeFuse (v2.1 - with LoRA Mask Application)

Multi-concept LoRA composition with spatial awareness.

Design: Reuse ComfyUI internals as much as possible.
- Uses load_bypass_lora_for_models() for non-merged LoRA
- Phase 1 sampler for mask collection
- FreeFuseMaskApplicator to apply masks to LoRA outputs
- Phase 2 uses native KSampler

Workflow:
1. Load model
2. FreeFuse6LoraLoader for stacked LoRAs (up to 6 per adapter)
3. FreeFuseConceptMap to define trigger words
4. FreeFuseTokenPositions to compute token positions
5. FreeFusePhase1Sampler to collect attention and generate masks
6. FreeFuseMaskApplicator to apply masks
7. Standard KSampler for Phase 2 generation
"""

from .nodes import (
    # LoRA loaders
    FreeFuseLoRALoader,
    FreeFuseLoRALoaderSimple,
    FreeFuse6LoraLoader,
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
    # Attention bias nodes
    FreeFuseAttentionBias,
    FreeFuseAttentionBiasVisualize,
    # Test/Debug nodes
    FreeFuseTestSimilarityMaps,
    FreeFuseQwenSimilarityExtractor,
    FreeFuseQwenBlockGridExtractor,
    # Preview
    FreeFuseMaskPreview,
    # Utility nodes
    FreeFuseMaskTap,
    FreeFuseMaskReassemble,
    # Mappings
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
__version__ = "0.2.1"
