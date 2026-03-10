"""
FreeFuse JSON Serialization Utilities

Provides functions to make FreeFuse data structures JSON-serializable
for ComfyUI workflow metadata storage.
"""

import torch
import numpy as np
from typing import Any, Dict, List, Union


class FreeFuseJSONEncoder:
    """
    Custom JSON encoder for FreeFuse data types.
    
    Converts PyTorch tensors and numpy arrays to lists for JSON serialization.
    """
    
    @staticmethod
    def convert_to_serializable(obj: Any) -> Any:
        """
        Recursively convert an object to JSON-serializable format.
        
        Args:
            obj: Any object that might contain tensors
            
        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, torch.Tensor):
            # Convert tensor to list (CPU, numpy, then list)
            return obj.detach().cpu().float().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            # Convert numpy array to list
            return obj.tolist()
        elif isinstance(obj, dict):
            # Recursively process dictionary
            return {key: FreeFuseJSONEncoder.convert_to_serializable(value) 
                    for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            # Recursively process list/tuple
            return [FreeFuseJSONEncoder.convert_to_serializable(item) 
                    for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            # Convert numpy integers to Python int
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            # Convert numpy floats to Python float
            return float(obj)
        else:
            # Return as-is (should be JSON-serializable)
            return obj


def make_freefuse_data_json_serializable(freefuse_data: Dict) -> Dict:
    """
    Convert FREEFUSE_DATA to JSON-serializable format.
    
    This is safe to call on any FreeFuse data structure - it will only
    convert tensor/numpy types and leave other data unchanged.
    
    Args:
        freefuse_data: FreeFuse data dictionary
        
    Returns:
        JSON-serializable dictionary
    """
    return FreeFuseJSONEncoder.convert_to_serializable(freefuse_data)


def make_freefuse_masks_json_serializable(masks_data: Dict) -> Dict:
    """
    Convert FREEFUSE_MASKS to JSON-serializable format.
    
    Converts mask tensors to lists so they can be stored in ComfyUI
    workflow metadata.
    
    Args:
        masks_data: Dictionary containing 'masks' and/or 'similarity_maps'
        
    Returns:
        Dictionary with tensors converted to lists
    """
    return FreeFuseJSONEncoder.convert_to_serializable(masks_data)


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """
    Safely dump FreeFuse data to JSON string.
    
    This automatically converts tensors to serializable format before
    passing to json.dumps.
    
    Args:
        obj: Object to serialize
        **kwargs: Additional arguments for json.dumps
        
    Returns:
        JSON string
    """
    import json
    serializable = FreeFuseJSONEncoder.convert_to_serializable(obj)
    return json.dumps(serializable, **kwargs)
