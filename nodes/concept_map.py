"""
FreeFuse Concept Map Node

Maps adapter names to concept text and computes token positions for similarity map collection.
Supports both Flux (T5 tokenizer) and SDXL (CLIP tokenizer) models.
"""

from ..freefuse_core.token_utils import (
    find_concept_positions,
    find_background_positions,
    detect_model_type,
    compute_token_position_maps,
)


class FreeFuseConceptMap:
    """
    Define mapping between LoRA adapter names and concept text.
    
    The concept text must appear verbatim in your prompt.
    This node prepares concept definitions - token positions are computed
    later when prompt is available in FreeFuseTokenPositions node.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "adapter_name_1": ("STRING", {"default": "character1"}),
                "concept_text_1": ("STRING", {
                    "default": "a woman with red hair",
                    "multiline": True
                }),
            },
            "optional": {
                "adapter_name_2": ("STRING", {"default": ""}),
                "concept_text_2": ("STRING", {"default": "", "multiline": True}),
                "adapter_name_3": ("STRING", {"default": ""}),
                "concept_text_3": ("STRING", {"default": "", "multiline": True}),
                "adapter_name_4": ("STRING", {"default": ""}),
                "concept_text_4": ("STRING", {"default": "", "multiline": True}),
                "enable_background": ("BOOLEAN", {"default": True}),
                "background_text": ("STRING", {"default": "", "multiline": True}),
                "freefuse_data": ("FREEFUSE_DATA",),
            }
        }
    
    RETURN_TYPES = ("FREEFUSE_DATA",)
    RETURN_NAMES = ("freefuse_data",)
    FUNCTION = "create_map"
    CATEGORY = "FreeFuse"
    
    def create_map(self, adapter_name_1, concept_text_1,
                   adapter_name_2="", concept_text_2="",
                   adapter_name_3="", concept_text_3="",
                   adapter_name_4="", concept_text_4="",
                   enable_background=True, background_text="",
                   freefuse_data=None):
        
        # Start with existing data or create new
        if freefuse_data is None:
            data = {"adapters": [], "concepts": {}, "settings": {}}
        else:
            data = dict(freefuse_data)
            data["concepts"] = dict(data.get("concepts", {}))
            data["settings"] = dict(data.get("settings", {}))
        
        # Add concept mappings
        pairs = [
            (adapter_name_1, concept_text_1),
            (adapter_name_2, concept_text_2),
            (adapter_name_3, concept_text_3),
            (adapter_name_4, concept_text_4),
        ]
        
        for name, text in pairs:
            name = name.strip()
            text = text.strip()
            if name and text:
                data["concepts"][name] = text
        
        data["settings"]["enable_background"] = enable_background
        data["settings"]["background_text"] = background_text.strip()
        
        return (data,)


class FreeFuseTokenPositions:
    """
    Compute token positions for FreeFuse concepts.
    
    This node takes the CLIP model and prompt to compute actual token positions
    for each concept. Works with both Flux (T5) and SDXL (CLIP) models.
    
    The token positions tell FreeFuse which tokens in the prompt correspond
    to each LoRA adapter's concept, enabling spatial-aware mask generation.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "forceInput": True}),
                "freefuse_data": ("FREEFUSE_DATA",),
            },
            "optional": {
                "filter_meaningless": ("BOOLEAN", {"default": True}),
                "filter_single_char": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("FREEFUSE_DATA",)
    RETURN_NAMES = ("freefuse_data",)
    FUNCTION = "compute_positions"
    CATEGORY = "FreeFuse"
    
    def compute_positions(self, clip, prompt, freefuse_data,
                          filter_meaningless=True, filter_single_char=True):
        
        # Copy data to avoid mutation
        data = dict(freefuse_data)
        data["concepts"] = dict(data.get("concepts", {}))
        data["settings"] = dict(data.get("settings", {}))
        
        concepts = data.get("concepts", {})
        settings = data.get("settings", {})
        
        if not concepts:
            print("[FreeFuseTokenPositions] Warning: No concepts defined")
            data["token_pos_maps"] = {}
            return (data,)
        
        # Detect model type
        model_type = detect_model_type(clip)
        print(f"[FreeFuseTokenPositions] Detected model type: {model_type}")
        
        # Compute token positions for each concept
        token_pos_maps = find_concept_positions(
            clip=clip,
            prompts=prompt,
            concepts=concepts,
            filter_meaningless=filter_meaningless,
            filter_single_char=filter_single_char,
            model_type=model_type,
        )
        
        # Handle background
        enable_background = settings.get("enable_background", True)
        background_text = settings.get("background_text", "")
        
        if enable_background:
            bg_positions = find_background_positions(
                clip=clip,
                prompt=prompt,
                background_text=background_text if background_text else None,
                model_type=model_type,
            )
            if bg_positions:
                token_pos_maps["__background__"] = [bg_positions]
                print(f"[FreeFuseTokenPositions] Background positions: {bg_positions}")
        
        # Log results
        for name, positions_list in token_pos_maps.items():
            if name != "__background__":
                print(f"[FreeFuseTokenPositions] {name}: positions = {positions_list}")
        
        data["token_pos_maps"] = token_pos_maps
        data["model_type"] = model_type
        data["prompt"] = prompt
        
        return (data,)


class FreeFuseConceptMapSimple:
    """
    All-in-one node that defines concepts AND computes token positions.
    
    Combines FreeFuseConceptMap + FreeFuseTokenPositions into a single node.
    Convenient for simple workflows with 2-4 concepts.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "forceInput": True}),
                "adapter_name_1": ("STRING", {"default": "character1"}),
                "concept_text_1": ("STRING", {
                    "default": "a woman with red hair",
                    "multiline": True
                }),
            },
            "optional": {
                "adapter_name_2": ("STRING", {"default": ""}),
                "concept_text_2": ("STRING", {"default": "", "multiline": True}),
                "adapter_name_3": ("STRING", {"default": ""}),
                "concept_text_3": ("STRING", {"default": "", "multiline": True}),
                "adapter_name_4": ("STRING", {"default": ""}),
                "concept_text_4": ("STRING", {"default": "", "multiline": True}),
                "enable_background": ("BOOLEAN", {"default": True}),
                "background_text": ("STRING", {"default": "", "multiline": True}),
                "filter_meaningless": ("BOOLEAN", {"default": True}),
                "filter_single_char": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("FREEFUSE_DATA",)
    RETURN_NAMES = ("freefuse_data",)
    FUNCTION = "process"
    CATEGORY = "FreeFuse"
    
    def process(self, clip, prompt, adapter_name_1, concept_text_1,
                adapter_name_2="", concept_text_2="",
                adapter_name_3="", concept_text_3="",
                adapter_name_4="", concept_text_4="",
                enable_background=True, background_text="",
                filter_meaningless=True, filter_single_char=True):
        
        # Build concepts dict
        concepts = {}
        pairs = [
            (adapter_name_1, concept_text_1),
            (adapter_name_2, concept_text_2),
            (adapter_name_3, concept_text_3),
            (adapter_name_4, concept_text_4),
        ]
        
        for name, text in pairs:
            name = name.strip()
            text = text.strip()
            if name and text:
                concepts[name] = text
        
        if not concepts:
            print("[FreeFuseConceptMapSimple] Warning: No valid concepts defined")
            return ({
                "adapters": [],
                "concepts": {},
                "settings": {},
                "token_pos_maps": {},
            },)
        
        # Detect model type
        model_type = detect_model_type(clip)
        print(f"[FreeFuseConceptMapSimple] Detected model type: {model_type}")
        
        # Compute token positions
        token_pos_maps = find_concept_positions(
            clip=clip,
            prompts=prompt,
            concepts=concepts,
            filter_meaningless=filter_meaningless,
            filter_single_char=filter_single_char,
            model_type=model_type,
        )
        
        # Handle background
        if enable_background:
            bg_positions = find_background_positions(
                clip=clip,
                prompt=prompt,
                background_text=background_text.strip() if background_text else None,
                model_type=model_type,
            )
            if bg_positions:
                token_pos_maps["__background__"] = [bg_positions]
                print(f"[FreeFuseConceptMapSimple] Background positions: {bg_positions}")
        
        # Log results
        for name, positions_list in token_pos_maps.items():
            if name != "__background__":
                print(f"[FreeFuseConceptMapSimple] {name}: positions = {positions_list}")
        
        data = {
            "adapters": [],
            "concepts": concepts,
            "settings": {
                "enable_background": enable_background,
                "background_text": background_text.strip(),
            },
            "token_pos_maps": token_pos_maps,
            "model_type": model_type,
            "prompt": prompt,
        }
        
        return (data,)
