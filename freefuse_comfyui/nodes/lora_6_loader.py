"""
FreeFuse 6-LoRA Stacked Loader

Loads up to 6 LoRAs stacked for the same adapter.
All LoRAs share the same mask region.
"""

import logging
import folder_paths
import comfy.utils
import comfy.sd
import comfy.lora
import comfy.lora_convert
import torch

# Use our fixed bypass loader
from ..freefuse_core.bypass_lora_loader import load_bypass_lora_for_models_fixed
from ..freefuse_core.token_utils import detect_model_type


class FreeFuse6LoraLoader:
    """
    Load up to 6 LoRAs stacked for the same adapter.
    All LoRAs share the same mask region.
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
            },
            "optional": {
                # First LoRA (optional)
                "lora_name_1": (["None"] + lora_list, {"default": "None"}),
                "strength_model_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "strength_clip_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "concept_text_1": ("STRING", {"multiline": True, "default": ""}),
                # Second LoRA (optional)
                "lora_name_2": (["None"] + lora_list, {"default": "None"}),
                "strength_model_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "strength_clip_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "concept_text_2": ("STRING", {"multiline": True, "default": ""}),
                # Third LoRA (optional)
                "lora_name_3": (["None"] + lora_list, {"default": "None"}),
                "strength_model_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "strength_clip_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "concept_text_3": ("STRING", {"multiline": True, "default": ""}),
                # Fourth LoRA (optional)
                "lora_name_4": (["None"] + lora_list, {"default": "None"}),
                "strength_model_4": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "strength_clip_4": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "concept_text_4": ("STRING", {"multiline": True, "default": ""}),
                # Fifth LoRA (optional)
                "lora_name_5": (["None"] + lora_list, {"default": "None"}),
                "strength_model_5": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "strength_clip_5": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "concept_text_5": ("STRING", {"multiline": True, "default": ""}),
                # Sixth LoRA (optional)
                "lora_name_6": (["None"] + lora_list, {"default": "None"}),
                "strength_model_6": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "strength_clip_6": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "concept_text_6": ("STRING", {"multiline": True, "default": ""}),
                "freefuse_data": ("FREEFUSE_DATA",),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "FREEFUSE_DATA", "STRING")
    RETURN_NAMES = ("model", "clip", "freefuse_data", "dino_prompt")
    FUNCTION = "load_loras"
    CATEGORY = "FreeFuse"

    DESCRIPTION = """Load up to 6 LoRAs stacked for the same adapter.

All LoRAs will be applied to the same masked region.
Concept texts are combined automatically.

Note: All LoRA slots are optional. You can use any number from 0-6 LoRAs.
If no LoRAs are selected, the node will pass through the model/clip unchanged.
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

        return model, clip, freefuse_data, dino_prompt

    def load_loras(self, model, clip, adapter_name, location_text,
                   lora_name_1="None", strength_model_1=1.0, strength_clip_1=1.0, concept_text_1="",
                   lora_name_2="None", strength_model_2=1.0, strength_clip_2=1.0, concept_text_2="",
                   lora_name_3="None", strength_model_3=1.0, strength_clip_3=1.0, concept_text_3="",
                   lora_name_4="None", strength_model_4=1.0, strength_clip_4=1.0, concept_text_4="",
                   lora_name_5="None", strength_model_5=1.0, strength_clip_5=1.0, concept_text_5="",
                   lora_name_6="None", strength_model_6=1.0, strength_clip_6=1.0, concept_text_6="",
                   freefuse_data=None):

        print(f"\n=== FreeFuse 6-LoRA Stacked Loader ===")
        print(f"Adapter: {adapter_name}")

        # Load first LoRA if provided
        if lora_name_1 and lora_name_1 != "None":
            model, clip, freefuse_data, _ = self._load_single_lora(
                model, clip, lora_name_1, adapter_name, location_text,
                strength_model_1, strength_clip_1, concept_text_1, freefuse_data
            )

        # Load second LoRA if provided
        if lora_name_2 and lora_name_2 != "None":
            model, clip, freefuse_data, _ = self._load_single_lora(
                model, clip, lora_name_2, adapter_name, location_text,
                strength_model_2, strength_clip_2, concept_text_2, freefuse_data
            )

        # Load third LoRA if provided
        if lora_name_3 and lora_name_3 != "None":
            model, clip, freefuse_data, _ = self._load_single_lora(
                model, clip, lora_name_3, adapter_name, location_text,
                strength_model_3, strength_clip_3, concept_text_3, freefuse_data
            )

        # Load fourth LoRA if provided
        if lora_name_4 and lora_name_4 != "None":
            model, clip, freefuse_data, _ = self._load_single_lora(
                model, clip, lora_name_4, adapter_name, location_text,
                strength_model_4, strength_clip_4, concept_text_4, freefuse_data
            )

        # Load fifth LoRA if provided
        if lora_name_5 and lora_name_5 != "None":
            model, clip, freefuse_data, _ = self._load_single_lora(
                model, clip, lora_name_5, adapter_name, location_text,
                strength_model_5, strength_clip_5, concept_text_5, freefuse_data
            )

        # Load sixth LoRA if provided
        if lora_name_6 and lora_name_6 != "None":
            model, clip, freefuse_data, _ = self._load_single_lora(
                model, clip, lora_name_6, adapter_name, location_text,
                strength_model_6, strength_clip_6, concept_text_6, freefuse_data
            )

        # Debug output
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
    "FreeFuse6LoraLoader": FreeFuse6LoraLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeFuse6LoraLoader": "FreeFuse 6-LoRA Stacked Loader",
}
