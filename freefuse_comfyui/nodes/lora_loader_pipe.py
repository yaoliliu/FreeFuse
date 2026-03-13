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
- Location text input for spatial positioning
"""

# Freefuse data structure matching FreeFuseTokenPositions expected format:
# {
#     "concepts": {
#         "harry": "on the left\na young wizard with glasses",  # Location line + concept
#         "hermione": "in the middle\na clever witch"
#     },
#     "settings": {
#         "enable_background": true,
#         "background_text": ""
#     },
#     "adapters": [
#         {
#             "name": "harry",
#             "lora_name": "harry_lora.safetensors",
#             "lora_path": "/path/to/harry_lora.safetensors",
#             "strength_model": 1.0,
#             "strength_clip": 1.0,
#             "bypass": false,
#             "location_text": "on the left",
#             "internal_concept": "on the left\na young wizard with glasses",
#             "dino_prompt": "on the left"
#         }
#     ]
# }

import logging
import folder_paths
import comfy.utils
import comfy.sd
import comfy.lora
import comfy.lora_convert
import torch  # ← ADD THIS LINE


# Use our fixed bypass loader instead of ComfyUI's buggy version
from ..freefuse_core.bypass_lora_loader import load_bypass_lora_for_models_fixed
from ..freefuse_core.token_utils import detect_model_type

class FreeFuseLoraPipe:
    """
    Load Lora Freefuse with location text support
    """
    
    def __init__(self):
        self.loaded_lora = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                # Bypass toggle button above lora_name
                "bypass": ("BOOLEAN", {
                    "default": False,
                    "label": "bypass",
                    "tooltip": "Toggle to bypass LoRA loading (pass through model/clip unchanged)"
                }),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "adapter_name": ("STRING", {
                    "default": "character1",
                    "tooltip": "Unique name for this adapter (must match concept map)"
                }),
                "location_text": ("STRING", {
                    "default": "",
                    "tooltip": "Location description (e.g., 'on the left', 'in the middle', 'on the right')"
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
                # Adding concept_text as an optional input
                "concept_text": ("STRING", {
                    "default": "", 
                    "multiline": True,
                    "placeholder": "Enter your concept description here...",
                    "tooltip": "Text description of the concept this LoRA represents"
                }),
            }
        }
    
    # Updated RETURN_TYPES to include STRING output for DINO
    RETURN_TYPES = ("MODEL", "CLIP", "FREEFUSE_DATA", "STRING")
    RETURN_NAMES = ("model", "clip", "freefuse_data", "dino_prompt")
    FUNCTION = "load_lora"
    CATEGORY = "FreeFuse"
    
    DESCRIPTION = """Load Lora Freefuse with location text support
    
Location text is inserted at the beginning of the concept text with a newline.
The location text is also output as dino_prompt for GroundingDINO.

Examples:
  location_text="on the left", concept_text="a woman with brown hair"
  → Internal concept: "on the left\na woman with brown hair"
  → DINO prompt: "on the left"
"""
    
    def _clean_text(self, text: str) -> str:
        """Remove ALL newlines and collapse extra spaces."""
        if not text or not isinstance(text, str):
            return ""
        import re
        # Step 1: Replace ALL newlines and carriage returns with NOTHING (not spaces)
        text = text.replace('\n', '').replace('\r', '')
        # Step 2: Replace tabs with spaces
        text = text.replace('\t', ' ')
        # Step 3: Collapse multiple spaces and strip
        return re.sub(r'\s+', ' ', text).strip()
    
    def load_lora(self, model, clip, bypass, lora_name, adapter_name, location_text,
                  strength_model, strength_clip, freefuse_data=None, concept_text=""):
        
        # === CLEAN ALL USER INPUTS ===
        # Clean adapter name (though it shouldn't have spaces/newlines)
        cleaned_adapter_name = self._clean_text(adapter_name)
        
        # Clean location text - remove any accidental newlines/spaces
        cleaned_location = self._clean_text(location_text)
        
        # Clean concept text - remove extra whitespace but KEEP INTENTIONAL NEWLINES?
        # For concept text, we need to be careful - it might contain intentional newlines
        # Let's clean whitespace but preserve structure
        # Clean concept text - REMOVE ALL NEWLINES COMPLETELY
        if concept_text:
            # Remove ALL newlines, not just replace with spaces
            cleaned_concept = self._clean_text(concept_text)
        else:
            cleaned_concept = ""
        
        # Debug output showing cleaning
        if location_text != cleaned_location or concept_text != cleaned_concept:
            print(f"\n[FreeFuseLoraPipe] Cleaned inputs for '{cleaned_adapter_name}':")
            if location_text != cleaned_location:
                print(f"  Location: '{location_text}' → '{cleaned_location}'")
            if concept_text != cleaned_concept:
                print(f"  Concept: '{concept_text[:50]}...' → '{cleaned_concept[:50]}...'")
        
        # === Generate internal concept and external DINO prompt ===
        
        # Step 1: Build DINO prompt - use cleaned location
        dino_prompt = cleaned_location
        
        # Step 2: Build internal concept - location + newline + concept
        # We want to preserve the intentional newline between location and concept
        if cleaned_location and cleaned_concept:
            internal_concept = f"{cleaned_location}\n{cleaned_concept}"
            print(f"   Building internal concept: location + newline + concept")
        elif cleaned_location:
            internal_concept = cleaned_location
            print(f"   Using location text only")
        elif cleaned_concept:
            internal_concept = cleaned_concept
            print(f"   Using concept text only")
        else:
            internal_concept = cleaned_adapter_name
            print(f"   Using adapter name as fallback")
        
        # Debug output
        print(f"\n=== FreeFuse LoRA Loader ===")
        print(f"Adapter: {cleaned_adapter_name}")
        print(f"Location text: '{cleaned_location}'")
        print(f"Concept text: '{cleaned_concept}'")
        print(f"Internal concept:\n{internal_concept}")
        print(f"DINO prompt: '{dino_prompt}'")
        
        # Build/extend freefuse_data - Store ALL information in the expected format
        if freefuse_data is None:
            # Initialize with the complete structure
            freefuse_data = {
                "concepts": {},
                "settings": {
                    "enable_background": True,
                    "background_text": ""
                },
                "adapters": []
            }
        else:
            # Make a copy to avoid modifying the original
            freefuse_data = dict(freefuse_data)
            
            # Ensure all required sections exist
            if "concepts" not in freefuse_data:
                freefuse_data["concepts"] = {}
            else:
                freefuse_data["concepts"] = dict(freefuse_data.get("concepts", {}))
                
            if "settings" not in freefuse_data:
                freefuse_data["settings"] = {
                    "enable_background": True,
                    "background_text": ""
                }
            else:
                freefuse_data["settings"] = dict(freefuse_data.get("settings", {}))
                
            if "adapters" not in freefuse_data:
                freefuse_data["adapters"] = []
            else:
                freefuse_data["adapters"] = list(freefuse_data.get("adapters", []))
        
        # Store concept text in the concepts dictionary (using internal_concept)
        # if internal_concept:
        #     freefuse_data["concepts"][cleaned_adapter_name] = internal_concept
        
        
        # Store concept text in the concepts dictionary - COMBINE if exists
        if internal_concept:
            if cleaned_adapter_name in freefuse_data["concepts"]:
                # Already exists - combine them
                existing = freefuse_data["concepts"][cleaned_adapter_name]
                
                # Check if both have the location+newline structure
                if '\n' in existing and '\n' in internal_concept:
                    # Split into location and description
                    existing_loc, existing_desc = existing.split('\n', 1)
                    new_loc, new_desc = internal_concept.split('\n', 1)
                    
                    # Keep first location, combine descriptions
                    combined = f"{existing_loc}\n{existing_desc} {new_desc}"
                    freefuse_data["concepts"][cleaned_adapter_name] = combined
                    print(f"   Combined concept for '{cleaned_adapter_name}' (preserved location)")
                else:
                    # Simple combine
                    freefuse_data["concepts"][cleaned_adapter_name] = f"{existing} {internal_concept}"
                    print(f"   Combined concept for '{cleaned_adapter_name}'")
            else:
                # First time - just store
                freefuse_data["concepts"][cleaned_adapter_name] = internal_concept
                print(f"   Added concept for '{cleaned_adapter_name}'")
                
        
        # Create the adapter entry with ALL available information
        adapter_entry = {
            "name": cleaned_adapter_name,
            "lora_name": lora_name if not bypass else None,
            "strength_model": strength_model,
            "strength_clip": strength_clip,
            "bypass": bypass,
            "location_text": cleaned_location,
            "internal_concept": internal_concept,
            "dino_prompt": dino_prompt,
        }
        
        # Add lora_path if not bypassed
        if not bypass:
            adapter_entry["lora_path"] = folder_paths.get_full_path_or_raise("loras", lora_name)
        
        # Check if this adapter already exists in the adapters list
        adapter_exists = False
        for i, existing in enumerate(freefuse_data["adapters"]):
            if existing.get("name") == cleaned_adapter_name:
                # Update existing entry while preserving any additional fields
                updated_entry = dict(existing)
                updated_entry.update(adapter_entry)
                freefuse_data["adapters"][i] = updated_entry
                adapter_exists = True
                logging.info(f"[FreeFuse] Updated existing adapter '{cleaned_adapter_name}' with location '{cleaned_location}'")
                break
        
        # If adapter doesn't exist, append new entry
        if not adapter_exists:
            freefuse_data["adapters"].append(adapter_entry)
            logging.info(f"[FreeFuse] Added new adapter '{cleaned_adapter_name}' with location '{cleaned_location}'")
        
        # If bypass is enabled, skip LoRA loading entirely
        if bypass:
            logging.info(f"[FreeFuse] Bypass enabled - skipping LoRA loading for '{cleaned_adapter_name}'")
            
            # Store in model options for use by mask applicator
            if "transformer_options" not in model.model_options:
                model.model_options["transformer_options"] = {}
            model.model_options["transformer_options"]["freefuse_data"] = freefuse_data
            
            # Return the DINO prompt even when bypassed
            return (model, clip, freefuse_data, dino_prompt)
        
        if strength_model == 0 and strength_clip == 0:
            return (model, clip, freefuse_data, dino_prompt)
        
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
        model_lora, clip_lora = load_bypass_lora_for_models_fixed(
            model, clip, lora, strength_model, strength_clip, adapter_name=cleaned_adapter_name
        )

        # Store in model options for use by mask applicator
        if "transformer_options" not in model_lora.model_options:
            model_lora.model_options["transformer_options"] = {}
        model_lora.model_options["transformer_options"]["freefuse_data"] = freefuse_data

        logging.info(f"[FreeFuse] Loaded LoRA '{cleaned_adapter_name}' with location '{cleaned_location}'")
        
        return (model_lora, clip_lora, freefuse_data, dino_prompt)

class FreeFuseBackground:
    """
    Load Background Freefuse - Original working version
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
    
    DESCRIPTION = """Load Background Freefuse - Original working version"""
    
    def _clean_text(self, text: str) -> str:
        """Remove ALL newlines and collapse extra spaces."""
        if not text or not isinstance(text, str):
            return ""
        import re
        # Step 1: Replace ALL newlines and carriage returns with NOTHING (not spaces)
        text = text.replace('\n', '').replace('\r', '')
        # Step 2: Replace tabs with spaces
        text = text.replace('\t', ' ')
        # Step 3: Collapse multiple spaces and strip
        return re.sub(r'\s+', ' ', text).strip()
    
    def configure_background(self, model, clip, freefuse_data, enable_background, background_text):
        
        # 🔥 CLEAN THE BACKGROUND TEXT HERE - right when it enters!
        cleaned_background = self._clean_text(background_text)
        
        # Optional: Show what was cleaned
        if background_text != cleaned_background:
            print(f"[FreeFuseBackground] Cleaned background text:")
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
        freefuse_data["settings"]["background_text"] = cleaned_background  # Store cleaned version
        
        # Return in the same order as RETURN_TYPES
        return (model, clip, freefuse_data)

class FreeFuseMergedLoraPipe:
    """
    TEST NODE: Load up to 2 LoRAs in bypass mode for the same adapter
    """
    
    def __init__(self):
        self.loaded_loras = {}  # Cache for loaded LoRAs
    
    @classmethod
    def INPUT_TYPES(cls):
        lora_list = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "adapter_name": ("STRING", {
                    "default": "character1",
                    "tooltip": "Unique name for this adapter (must match concept map)"
                }),
                "location_text": ("STRING", {
                    "default": "",
                    "tooltip": "Location description (e.g., 'on the left', 'in the middle', 'on the right')"
                }),
                # First LoRA
                "lora_name_1": (lora_list,),
                "strength_model_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "strength_clip_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "concept_text_1": ("STRING", {"multiline": True, "default": ""}),
                # Second LoRA (optional)
                "lora_name_2": (["None"] + lora_list, {"default": "None"}),
                "strength_model_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "strength_clip_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "concept_text_2": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "freefuse_data": ("FREEFUSE_DATA",),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "FREEFUSE_DATA", "STRING")
    RETURN_NAMES = ("model", "clip", "freefuse_data", "dino_prompt")
    FUNCTION = "load_loras"
    CATEGORY = "FreeFuse"
    
    DESCRIPTION = """TEST: Load up to 2 LoRAs for the same adapter.
    
Both LoRAs will be applied to the same masked region.
"""
    
    def _clean_text(self, text: str) -> str:
        """Remove ALL newlines and collapse extra spaces."""
        if not text or not isinstance(text, str):
            return ""
        import re
        text = text.replace('\n', '').replace('\r', '')
        text = text.replace('\t', ' ')
        return re.sub(r'\s+', ' ', text).strip()
    
    def _load_single_lora(self, model, clip, lora_name, adapter_name, location_text,
                          strength_model, strength_clip, concept_text, freefuse_data):
        """Load a single LoRA in bypass mode and update freefuse_data."""
        
        # Clean inputs
        cleaned_adapter = self._clean_text(adapter_name)
        cleaned_location = self._clean_text(location_text)
        cleaned_concept = self._clean_text(concept_text) if concept_text else ""
        
        # Build internal concept
        if cleaned_location and cleaned_concept:
            internal_concept = f"{cleaned_location}\n{cleaned_concept}"
        elif cleaned_location:
            internal_concept = cleaned_location
        elif cleaned_concept:
            internal_concept = cleaned_concept
        else:
            internal_concept = cleaned_adapter
        
        dino_prompt = cleaned_location
        
        print(f"   Loading LoRA: {lora_name} for '{cleaned_adapter}'")
        
        # Build/extend freefuse_data
        if freefuse_data is None:
            freefuse_data = {
                "concepts": {},
                "settings": {"enable_background": True, "background_text": ""},
                "adapters": []
            }
        else:
            freefuse_data = dict(freefuse_data)
            freefuse_data["concepts"] = dict(freefuse_data.get("concepts", {}))
            freefuse_data["settings"] = dict(freefuse_data.get("settings", {}))
            freefuse_data["adapters"] = list(freefuse_data.get("adapters", []))
        
        # Store concept text - COMBINE if exists
        if internal_concept:
            if cleaned_adapter in freefuse_data["concepts"]:
                existing = freefuse_data["concepts"][cleaned_adapter]
                freefuse_data["concepts"][cleaned_adapter] = f"{existing} {internal_concept}"
                print(f"   Combined concept for '{cleaned_adapter}'")
            else:
                freefuse_data["concepts"][cleaned_adapter] = internal_concept
                print(f"   Added concept for '{cleaned_adapter}'")
        
        # Create adapter entry
        adapter_entry = {
            "name": cleaned_adapter,
            "lora_name": lora_name,
            "strength_model": strength_model,
            "strength_clip": strength_clip,
            "bypass": False,
            "location_text": cleaned_location,
            "internal_concept": internal_concept,
            "dino_prompt": dino_prompt,
        }
        
        # Update adapters list (allow multiple entries with same name)
        freefuse_data["adapters"].append(adapter_entry)
        
        # Load LoRA file
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        
        # Load from cache or file
        if lora_path in self.loaded_loras:
            lora = self.loaded_loras[lora_path]
        else:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_loras[lora_path] = lora
        
        # Load in bypass mode
        model, clip = load_bypass_lora_for_models_fixed(
            model, clip, lora,
            strength_model=strength_model,
            strength_clip=strength_clip,
            adapter_name=cleaned_adapter
        )
        
        # 🔥 ADD DEBUG CODE HERE 🔥
        print(f"[DEBUG] After loading {lora_name}:")
        if hasattr(model, 'model_options') and 'transformer_options' in model.model_options:
            managers = model.model_options['transformer_options'].get('freefuse_bypass_managers', [])
            print(f"  Bypass managers count: {len(managers)}")
            for i, m in enumerate(managers):
                print(f"    Manager {i}: adapter_name={m.get('adapter_name', 'unknown')}")
        else:
            print(f"  No transformer_options found")
        
        return model, clip, freefuse_data, dino_prompt
    
    def load_loras(self, model, clip, adapter_name, location_text,
                   lora_name_1, strength_model_1, strength_clip_1, concept_text_1,
                   lora_name_2="None", strength_model_2=1.0, strength_clip_2=1.0, concept_text_2="",
                   freefuse_data=None):
        
        print(f"\n=== FreeFuse Dual LoRA Loader ===")
        print(f"Adapter: {adapter_name}")
        
        # Load first LoRA
        model, clip, freefuse_data, _ = self._load_single_lora(
            model, clip, lora_name_1, adapter_name, location_text,
            strength_model_1, strength_clip_1, concept_text_1, freefuse_data
        )
        
        # Load second LoRA if provided and not "None"
        if lora_name_2 and lora_name_2 != "None":
            model, clip, freefuse_data, _ = self._load_single_lora(
                model, clip, lora_name_2, adapter_name, location_text,
                strength_model_2, strength_clip_2, concept_text_2, freefuse_data
            )
            
            # After both LoRAs are loaded
        print(f"\n[DEBUG] FINAL STATE:")
        print(f"freefuse_data adapters: {len(freefuse_data['adapters'])}")
        for i, a in enumerate(freefuse_data['adapters']):
            print(f"  Adapter {i}: name={a['name']}, lora={a['lora_name']}")
        
        if 'transformer_options' in model.model_options:
            managers = model.model_options['transformer_options'].get('freefuse_bypass_managers', [])
            print(f"Bypass managers: {len(managers)}")
            for i, m in enumerate(managers):
                print(f"  Manager {i}: adapter_name={m.get('adapter_name', 'unknown')}")
                
        # Store freefuse_data in model options
        if "transformer_options" not in model.model_options:
            model.model_options["transformer_options"] = {}
        model.model_options["transformer_options"]["freefuse_data"] = freefuse_data
        
        return (model, clip, freefuse_data, location_text)

# Export node mappings
NODE_CLASS_MAPPINGS = {
    "FreeFuseLoraPipe": FreeFuseLoraPipe,
    "FreeFuseBackground": FreeFuseBackground,
    "FreeFuseMergedLoraPipe" : FreeFuseMergedLoraPipe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeFuseLoraPipe": "FreeFuse LoRA Loader",
    "FreeFuseBackground": "FreeFuse Background",
    "FreeFuseMergedLoraPipe" : "FreeFuse Merged Lora Loader",
}
