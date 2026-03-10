"""
FreeFuse Background Loader

Sets up background handling for FreeFuse multi-concept composition.
Configures background text and enables/disables background processing.
"""

import logging
import folder_paths
import comfy.utils
import comfy.sd
import comfy.lora
import comfy.lora_convert
import torch
import re

# Use our fixed bypass loader
from ..freefuse_core.bypass_lora_loader import load_bypass_lora_for_models_fixed
from ..freefuse_core.token_utils import detect_model_type


class FreeFuseBackgroundLoader:
    """
    Configure background settings for FreeFuse multi-concept composition.
    
    This node enables or disables background handling and sets the background
    description text that will appear in the prompt.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "freefuse_data": ("FREEFUSE_DATA",),
                "enable_background": ("BOOLEAN", {
                    "default": True,
                    "label": "Enable Background",
                    "tooltip": "Enable or disable background handling"
                }),
                "background_text": ("STRING", {
                    "default": "", 
                    "multiline": True,
                    "placeholder": "Enter background description here...",
                    "tooltip": "Text description of the background (e.g., 'a magical castle', 'forest at sunset')"
                }),
            },
            "optional": {}  # No optional parameters
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "FREEFUSE_DATA")
    RETURN_NAMES = ("model", "clip", "freefuse_data")
    FUNCTION = "configure_background"
    CATEGORY = "FreeFuse"
    
    DESCRIPTION = """Configure background settings for FreeFuse.
    
Set background description text and enable/disable background processing.
The background text will be included in the prompt and can be used for
spatial masking of background regions.
"""
    
    def _clean_text(self, text: str) -> str:
        """Remove ALL newlines and collapse extra spaces."""
        if not text or not isinstance(text, str):
            return ""
        # Step 1: Replace ALL newlines and carriage returns with NOTHING (not spaces)
        text = text.replace('\n', '').replace('\r', '')
        # Step 2: Replace tabs with spaces
        text = text.replace('\t', ' ')
        # Step 3: Collapse multiple spaces and strip
        return re.sub(r'\s+', ' ', text).strip()
    
    def configure_background(self, model, clip, freefuse_data, enable_background, background_text):
        
        # Clean the background text - right when it enters!
        cleaned_background = self._clean_text(background_text)
        
        # Show what was cleaned (optional)
        if background_text != cleaned_background:
            print(f"[FreeFuseBackgroundLoader] Cleaned background text:")
            print(f"  Original: '{background_text}'")
            print(f"  Cleaned:  '{cleaned_background}'")
        
        # Build/extend freefuse_data
        if freefuse_data is None:
            freefuse_data = {
                "concepts": {},
                "settings": {
                    "enable_background": enable_background,
                    "background_text": cleaned_background  # Store cleaned version
                },
                "adapters": []
            }
        else:
            # Make a copy to avoid modifying the original
            freefuse_data = dict(freefuse_data)
            
            # Ensure settings section exists
            if "settings" not in freefuse_data:
                freefuse_data["settings"] = {}
            else:
                freefuse_data["settings"] = dict(freefuse_data.get("settings", {}))
        
        # Update background settings with CLEANED text
        freefuse_data["settings"]["enable_background"] = enable_background
        freefuse_data["settings"]["background_text"] = cleaned_background
        
        # Return in the same order as RETURN_TYPES
        return (model, clip, freefuse_data)


# Export node mappings
NODE_CLASS_MAPPINGS = {
    "FreeFuseBackgroundLoader": FreeFuseBackgroundLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeFuseBackgroundLoader": "FreeFuse Background Loader",
}
