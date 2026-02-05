"""
FreeFuse LoRA Loader - uses fixed bypass mode for Flux support

This loader uses FreeFuse's fixed load_bypass_lora_for_models_fixed() to keep
LoRA weights separate (not merged), enabling FreeFuse spatial masking.

The fixed version correctly handles Flux's fused QKV weights (tuple keys)
which ComfyUI's original implementation doesn't support properly.

Key features:
- Loads LoRA in bypass mode (not merged into base weights)
- Correctly handles Flux fused QKV weights with offset
- Tracks adapter information for mask application
- Supports chaining multiple LoRAs
"""

import logging
import folder_paths
import comfy.utils
import comfy.sd
import comfy.lora
import comfy.lora_convert

# Use our fixed bypass loader instead of ComfyUI's buggy version
from ..freefuse_core.bypass_lora_loader import load_bypass_lora_for_models_fixed


class FreeFuseLoRALoader:
    """
    Load LoRA in bypass mode for FreeFuse.
    
    Uses ComfyUI's load_bypass_lora_for_models() to keep LoRA weights 
    separate from base model, enabling spatial masking in FreeFuse.
    
    The adapter_name should match the concept name used in FreeFuseConceptMap.
    """
    
    def __init__(self):
        self.loaded_lora = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "adapter_name": ("STRING", {
                    "default": "character1",
                    "tooltip": "Unique name for this adapter (must match concept map)"
                }),
                "strength_model": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                    "tooltip": "LoRA strength for UNet/transformer"
                }),
                "strength_clip": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                    "tooltip": "LoRA strength for text encoder"
                }),
            },
            "optional": {
                "freefuse_data": ("FREEFUSE_DATA",),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "FREEFUSE_DATA")
    RETURN_NAMES = ("model", "clip", "freefuse_data")
    FUNCTION = "load_lora"
    CATEGORY = "FreeFuse"
    
    DESCRIPTION = """Load LoRA in bypass mode for FreeFuse multi-concept composition.
    
Chain multiple loaders together. The adapter_name MUST match
the concept name in FreeFuseConceptMap for mask application to work.

Example workflow:
1. Load base model
2. FreeFuseLoRALoader (adapter_name="harry") 
3. FreeFuseLoRALoader (adapter_name="hermione")
4. FreeFuseConceptMap with matching concept names
5. FreeFusePhase1Sampler -> FreeFuseMaskApplicator -> KSampler"""
    
    def load_lora(self, model, clip, lora_name, adapter_name, 
                  strength_model, strength_clip, freefuse_data=None):
        
        if strength_model == 0 and strength_clip == 0:
            return (model, clip, freefuse_data or {"adapters": []})
        
        # Load LoRA file
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = None
        
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None
        
        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)
        
        # Use FreeFuse's fixed bypass mode - correctly handles Flux fused QKV weights
        # Original ComfyUI's load_bypass_lora_for_models() doesn't support tuple keys
        model_lora, clip_lora = load_bypass_lora_for_models_fixed(
            model, clip, lora, strength_model, strength_clip, adapter_name=adapter_name
        )
        
        # Build/extend freefuse_data
        if freefuse_data is None:
            freefuse_data = {"adapters": [], "adapter_keys": {}}
        else:
            freefuse_data = dict(freefuse_data)
            freefuse_data["adapters"] = list(freefuse_data.get("adapters", []))
            freefuse_data["adapter_keys"] = dict(freefuse_data.get("adapter_keys", {}))
        
        # Record adapter info
        adapter_info = {
            "name": adapter_name,
            "lora_name": lora_name,
            "lora_path": lora_path,
            "strength_model": strength_model,
            "strength_clip": strength_clip,
        }
        freefuse_data["adapters"].append(adapter_info)
        
        # Track which keys belong to this adapter (for mask application)
        key_map = {}
        if model is not None:
            key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
        
        lora_converted = comfy.lora_convert.convert_lora(lora)
        loaded = comfy.lora.load_lora(lora_converted, key_map)
        
        # Store the adapter keys for this adapter
        adapter_key_list = list(loaded.keys())
        freefuse_data["adapter_keys"][adapter_name] = adapter_key_list
        
        # Also store reverse mapping: key -> adapter name (for mask applicator)
        if "adapter_key_to_name" not in freefuse_data:
            freefuse_data["adapter_key_to_name"] = {}
        for k in adapter_key_list:
            freefuse_data["adapter_key_to_name"][k] = adapter_name
        
        # Store in model options for use by mask applicator
        if "transformer_options" not in model_lora.model_options:
            model_lora.model_options["transformer_options"] = {}
        model_lora.model_options["transformer_options"]["freefuse_data"] = freefuse_data
        
        logging.info(f"[FreeFuse] Loaded LoRA '{adapter_name}' with {len(loaded)} keys")
        
        return (model_lora, clip_lora, freefuse_data)


class FreeFuseLoRALoaderSimple:
    """
    Simplified LoRA loader for FreeFuse - uses standard patching.
    
    Use this if bypass mode causes issues. Masks are applied via
    attention hooks instead of LoRA output masking.
    """
    
    def __init__(self):
        self.loaded_lora = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "adapter_name": ("STRING", {
                    "default": "character1",
                }),
                "strength_model": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01
                }),
                "strength_clip": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01
                }),
            },
            "optional": {
                "freefuse_data": ("FREEFUSE_DATA",),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "FREEFUSE_DATA")
    RETURN_NAMES = ("model", "clip", "freefuse_data")
    FUNCTION = "load_lora"
    CATEGORY = "FreeFuse"
    
    DESCRIPTION = """Simplified LoRA loader using standard patching.
    
Use this as fallback if bypass mode has issues. FreeFuse masks
will be applied via attention modulation instead."""
    
    def load_lora(self, model, clip, lora_name, adapter_name, 
                  strength_model, strength_clip, freefuse_data=None):
        
        if strength_model == 0 and strength_clip == 0:
            return (model, clip, freefuse_data or {"adapters": []})
        
        # Load LoRA file
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = None
        
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None
        
        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)
        
        # Use standard LoRA loading (merged into weights)
        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model, clip, lora, strength_model, strength_clip
        )
        
        # Build/extend freefuse_data
        if freefuse_data is None:
            freefuse_data = {"adapters": [], "mode": "merged"}
        else:
            freefuse_data = dict(freefuse_data)
            freefuse_data["adapters"] = list(freefuse_data.get("adapters", []))
            freefuse_data["mode"] = "merged"
        
        freefuse_data["adapters"].append({
            "name": adapter_name,
            "lora_name": lora_name,
            "strength_model": strength_model,
            "strength_clip": strength_clip,
        })
        
        logging.info(f"[FreeFuse] Loaded LoRA '{adapter_name}' (merged mode)")
        
        return (model_lora, clip_lora, freefuse_data)


# Export node mappings
NODE_CLASS_MAPPINGS = {
    "FreeFuseLoRALoader": FreeFuseLoRALoader,
    "FreeFuseLoRALoaderSimple": FreeFuseLoRALoaderSimple,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeFuseLoRALoader": "FreeFuse LoRA Loader (Bypass)",
    "FreeFuseLoRALoaderSimple": "FreeFuse LoRA Loader (Simple)",
}
