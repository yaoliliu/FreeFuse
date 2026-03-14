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

# Tell ComfyUI where to find JavaScript files
import os
import sys

# The directory containing this file (nodes/)
WEB_DIRECTORY = "./"

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
from .mask_debug import (
    FreeFuseMaskBankDebug,
    FreeFuseBankInspector,
    NODE_CLASS_MAPPINGS as DEBUG_NODE_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as DEBUG_DISPLAY_MAPPINGS,
)
from .blocks_analysis import (
    FreeFuseBlocksAnalysis,
    NODE_CLASS_MAPPINGS as BLOCKS_NODE_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as BLOCKS_DISPLAY_MAPPINGS,
)
from .base_analysis import (
    FreeFuseBaseAnalysis,
    NODE_CLASS_MAPPINGS as BASE_NODE_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as BASE_DISPLAY_MAPPINGS,
)
from .raw_similarity_grid import (
    FreeFuseRawSimilarityOverlay,
    NODE_CLASS_MAPPINGS as RAW_SIM_NODE_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as RAW_SIM_DISPLAY_MAPPINGS,
)
from .attention_bias import (
    FreeFuseAttentionBias,
    FreeFuseAttentionBiasVisualize,
    NODE_CLASS_MAPPINGS as ATTN_BIAS_NODE_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as ATTN_BIAS_DISPLAY_MAPPINGS,
)
from .mask_tap import (
    FreeFuseMaskTap,
    FreeFuseMaskReassemble,
)

# Import the numbered LoRA loaders (only #6 kept)
from .lora_6_loader import (
    FreeFuse6LoraLoader,
    NODE_CLASS_MAPPINGS as LORA6_NODE_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as LORA6_DISPLAY_MAPPINGS,
)

# Import the background loader
from .background_loader import (
    FreeFuseBackgroundLoader,
    NODE_CLASS_MAPPINGS as BG_NODE_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as BG_DISPLAY_MAPPINGS,
)

# Import test/debug nodes
from .test_similarity_maps import (
    FreeFuseTestSimilarityMaps,
    NODE_CLASS_MAPPINGS as TEST_NODE_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as TEST_DISPLAY_MAPPINGS,
)
from .qwen_similarity_extractor import (
    FreeFuseQwenSimilarityExtractor,
    NODE_CLASS_MAPPINGS as QWEN_EXT_NODE_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as QWEN_EXT_DISPLAY_MAPPINGS,
)
from .qwen_block_grid_extractor import (
    FreeFuseQwenBlockGridExtractor,
    NODE_CLASS_MAPPINGS as QWEN_GRID_NODE_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as QWEN_GRID_DISPLAY_MAPPINGS,
)
from .mask_exporter import (
    FreeFuseMaskExporter,
    NODE_CLASS_MAPPINGS as MASK_EXPORT_NODE_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as MASK_EXPORT_DISPLAY_MAPPINGS,
)
from .mask_refiner import (
    FreeFuseMaskRefiner,
    NODE_CLASS_MAPPINGS as MASK_REFINER_NODE_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as MASK_REFINER_DISPLAY_MAPPINGS,
)
from .ltx_similarity_extractor import (
    FreeFuseLTXSimilarityExtractor,
    NODE_CLASS_MAPPINGS as LTX_SIM_NODE_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as LTX_SIM_DISPLAY_MAPPINGS,
)
from .ltx_block_grid_extractor import (
    FreeFuseLTXBlockGridExtractor,
    NODE_CLASS_MAPPINGS as LTX_GRID_NODE_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as LTX_GRID_DISPLAY_MAPPINGS,
)

# Combine all node mappings
NODE_CLASS_MAPPINGS = {
    # Original LoRA loaders (keep these)
    "FreeFuseLoRALoader": FreeFuseLoRALoader,
    "FreeFuseLoRALoaderSimple": FreeFuseLoRALoaderSimple,

    # Numbered LoRA loaders (only #6 kept)
    "FreeFuse6LoraLoader": FreeFuse6LoraLoader,

    # Background loader
    "FreeFuseBackgroundLoader": FreeFuseBackgroundLoader,
    
    # Concept mapping
    "FreeFuseConceptMap": FreeFuseConceptMap,
    "FreeFuseTokenPositions": FreeFuseTokenPositions,
    "FreeFuseConceptMapSimple": FreeFuseConceptMapSimple,
    
    # Sampling
    "FreeFusePhase1Sampler": FreeFusePhase1Sampler,
    
    # Preview
    "FreeFuseMaskPreview": FreeFuseMaskPreview,
    
    # Mask application
    "FreeFuseMaskApplicator": FreeFuseMaskApplicator,
    "FreeFuseMaskDebug": FreeFuseMaskDebug,
    
    # Mask debug nodes
    "FreeFuseMaskBankDebug": FreeFuseMaskBankDebug,
    "FreeFuseBankInspector": FreeFuseBankInspector,
    
    # Research nodes
    "FreeFuseBlocksAnalysis": FreeFuseBlocksAnalysis,
    "FreeFuseBaseAnalysis": FreeFuseBaseAnalysis,
    "FreeFuseRawSimilarityOverlay": FreeFuseRawSimilarityOverlay,
    
    # Attention bias nodes
    "FreeFuseAttentionBias": FreeFuseAttentionBias,
    "FreeFuseAttentionBiasVisualize": FreeFuseAttentionBiasVisualize,

    # Test/Debug nodes
    "FreeFuseTestSimilarityMaps": FreeFuseTestSimilarityMaps,
    "FreeFuseQwenSimilarityExtractor": FreeFuseQwenSimilarityExtractor,
    "FreeFuseQwenBlockGridExtractor": FreeFuseQwenBlockGridExtractor,

    # LTX-Video nodes
    "FreeFuseLTXSimilarityExtractor": FreeFuseLTXSimilarityExtractor,
    "FreeFuseLTXBlockGridExtractor": FreeFuseLTXBlockGridExtractor,

    # Mask exporter
    "FreeFuseMaskExporter": FreeFuseMaskExporter,

    # Mask refiner
    "FreeFuseMaskRefiner": FreeFuseMaskRefiner,

    # Utility nodes
    "FreeFuseMaskTap": FreeFuseMaskTap,
    "FreeFuseMaskReassemble": FreeFuseMaskReassemble,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Original LoRA loaders
    "FreeFuseLoRALoader": "FreeFuse LoRA Loader (Bypass)",
    "FreeFuseLoRALoaderSimple": "FreeFuse LoRA Loader (Simple)",

    # Numbered LoRA loaders (only #6 kept)
    "FreeFuse6LoraLoader": "FreeFuse 6-LoRA Stacked Loader",

    # Background loader
    "FreeFuseBackgroundLoader": "FreeFuse Background Loader",
    
    # Concept mapping
    "FreeFuseConceptMap": "FreeFuse Concept Map",
    "FreeFuseTokenPositions": "FreeFuse Token Positions",
    "FreeFuseConceptMapSimple": "FreeFuse Concept Map (Simple)",
    
    # Sampling
    "FreeFusePhase1Sampler": "FreeFuse Phase 1 Sampler",
    
    # Preview
    "FreeFuseMaskPreview": "FreeFuse Mask Preview",
    
    # Mask application
    "FreeFuseMaskApplicator": "FreeFuse Mask Applicator",
    "FreeFuseMaskDebug": "FreeFuse Mask Debug",
    
    # Mask debug nodes
    "FreeFuseMaskBankDebug": "FreeFuse Mask Bank Debug",
    "FreeFuseBankInspector": "FreeFuse Bank Inspector",
    
    # Research nodes
    "FreeFuseBlocksAnalysis": "🔬 FreeFuse Blocks Analysis (Research)",
    "FreeFuseBaseAnalysis": "🔬 FreeFuse Base Analysis (Research)",
    "FreeFuseRawSimilarityOverlay": "🔬 FreeFuse Raw Similarity Overlay",
    
    # Attention bias nodes
    "FreeFuseAttentionBias": "FreeFuse Attention Bias",
    "FreeFuseAttentionBiasVisualize": "FreeFuse Attention Bias Visualize",

    # Test/Debug nodes
    "FreeFuseTestSimilarityMaps": "🧪 FreeFuse Test Similarity Maps",
    "FreeFuseQwenSimilarityExtractor": "🔬 FreeFuse Qwen Similarity Extractor",
    "FreeFuseQwenBlockGridExtractor": "🔍 Qwen Block Grid Extractor",

    # LTX-Video nodes
    "FreeFuseLTXSimilarityExtractor": "🎬 FreeFuse LTX Similarity Extractor",
    "FreeFuseLTXBlockGridExtractor": "🎬 FreeFuse LTX Block Grid Extractor",

    # Mask exporter
    "FreeFuseMaskExporter": "💾 FreeFuse Mask Exporter",
    "FreeFuseMaskRefiner": "🔧 FreeFuse Mask Refiner",

    # Utility nodes
    "FreeFuseMaskTap": "FreeFuse Mask Tap",
    "FreeFuseMaskReassemble": "FreeFuse Mask Reassemble",
}

# Make sure WEB_DIRECTORY is included in __all__
__all__ = [
    # Original LoRA loaders
    "FreeFuseLoRALoader",
    "FreeFuseLoRALoaderSimple",

    # Numbered LoRA loaders (only #6 kept)
    "FreeFuse6LoraLoader",

    # Background loader
    "FreeFuseBackgroundLoader",
    
    # Concept mapping
    "FreeFuseConceptMap",
    "FreeFuseTokenPositions",
    "FreeFuseConceptMapSimple",
    
    # Sampling
    "FreeFusePhase1Sampler",
    
    # Mask application
    "FreeFuseMaskApplicator",
    "FreeFuseMaskDebug",
    
    # Mask debug nodes
    "FreeFuseMaskBankDebug",
    "FreeFuseBankInspector",
    
    # Research nodes
    "FreeFuseBlocksAnalysis",
    "FreeFuseBaseAnalysis",
    "FreeFuseRawSimilarityOverlay",
    
    # Attention bias
    "FreeFuseAttentionBias",
    "FreeFuseAttentionBiasVisualize",

    # Test/Debug nodes
    "FreeFuseTestSimilarityMaps",
    "FreeFuseQwenSimilarityExtractor",
    "FreeFuseQwenBlockGridExtractor",

    # LTX-Video nodes
    "FreeFuseLTXSimilarityExtractor",
    "FreeFuseLTXBlockGridExtractor",

    # Preview
    "FreeFuseMaskPreview",

    # Utility nodes
    "FreeFuseMaskTap",
    "FreeFuseMaskReassemble",

    # Mappings
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
