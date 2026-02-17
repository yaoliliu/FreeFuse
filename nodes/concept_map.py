"""
FreeFuse Concept Map Node

Maps adapter names to concept text and computes token positions for similarity map collection.
Supports both Flux (T5 tokenizer) and SDXL (CLIP tokenizer) models.
"""

from typing import Dict, List, Union

from ..freefuse_core.token_utils import (
    find_concept_positions,
    find_background_positions,
    detect_model_type,
    LUMINA2_SYSTEM_PROMPT,
)


def _format_prompt_preview(prompt: Union[str, List[str]], max_len: int = 220) -> str:
    """Format prompt for error/warning messages."""
    if isinstance(prompt, list):
        prompt_text = " | ".join(str(p) for p in prompt)
    else:
        prompt_text = str(prompt)
    prompt_text = prompt_text.replace("\n", " ").strip()
    if len(prompt_text) <= max_len:
        return prompt_text
    return prompt_text[:max_len] + "..."


def _format_concept_preview(text: str, max_len: int = 90) -> str:
    """Format concept text preview for compact error messages."""
    text = str(text).replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _normalize_positions_list(positions_data) -> List[List[int]]:
    """Normalize token position payload to a list-of-lists format."""
    if not isinstance(positions_data, list):
        return []
    if positions_data and all(isinstance(x, int) for x in positions_data):
        # Backward compatibility for flat list format.
        return [positions_data]
    return [pos for pos in positions_data if isinstance(pos, list)]


def _raise_if_subject_positions_empty(
    concepts: Dict[str, str],
    token_pos_maps: Dict[str, List[List[int]]],
    prompt: Union[str, List[str]],
) -> None:
    """
    Enforce FreeFuse requirement:
    each subject concept (adapter) must map to at least one prompt token.
    """
    missing = []

    for adapter_name, concept_text in concepts.items():
        positions_per_prompt = _normalize_positions_list(token_pos_maps.get(adapter_name, []))
        missing_prompt_indices = []

        if not positions_per_prompt:
            missing_prompt_indices = ["all"]
        else:
            for idx, positions in enumerate(positions_per_prompt):
                if not positions:
                    missing_prompt_indices.append(str(idx + 1))

        if missing_prompt_indices:
            missing.append((adapter_name, concept_text, ",".join(missing_prompt_indices)))

    if not missing:
        return

    missing_lines = "\n".join(
        f"  - {name} (concept_text='{_format_concept_preview(text)}', prompt_index={indices})"
        for name, text, indices in missing
    )
    prompt_preview = _format_prompt_preview(prompt)
    raise ValueError(
        "[FreeFuse] Subject adapter token positions are empty.\n"
        "Each subject concept_text must appear verbatim in the main prompt.\n"
        "You must ensure the subject prompt appears in the main prompt.\n"
        f"Missing subject adapters:\n{missing_lines}\n"
        "Example:\n"
        "  concept_text A: \"<detailed A>\"\n"
        "  concept_text B: \"<detailed B>\"\n"
        "  main prompt: \"<detailed A> is hugging <detailed B>\"\n"
        f"Main prompt preview: \"{prompt_preview}\""
    )


def _warn_if_background_text_missing(
    background_text: str,
    bg_positions: List[int],
    prompt: Union[str, List[str]],
    source: str,
) -> None:
    """Warn only when explicit background_text is configured but not detected."""
    bg_text = (background_text or "").strip()
    if not bg_text or bg_positions:
        return

    print(
        f"[{source}] Warning: background_text was provided but no token positions were detected. "
        "FreeFuse can still run, but background separation may be weaker. "
        f"background_text='{bg_text}', main prompt preview='{_format_prompt_preview(prompt)}'"
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
                "prompt": ("STRING", {"multiline": True}),
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
        
        # Detect model type (prefer model-side hint stored in freefuse_data)
        model_type_hint = data.get("model_type")
        model_type = detect_model_type(clip=clip, model_type_hint=model_type_hint)
        print(f"[FreeFuseTokenPositions] Detected model type: {model_type}")
        
        # For Z-Image (Lumina2), pass system prompt so token positions
        # align with the conditioning produced by CLIPTextEncodeLumina2
        system_prompt = LUMINA2_SYSTEM_PROMPT if model_type == 'z_image' else None
        
        # Compute token positions for each concept
        token_pos_maps = find_concept_positions(
            clip=clip,
            prompts=prompt,
            concepts=concepts,
            filter_meaningless=filter_meaningless,
            filter_single_char=filter_single_char,
            model_type=model_type,
            system_prompt=system_prompt,
        )

        # Only enforce hard error for subject LoRAs.
        # If adapter list is unavailable, fall back to all concept entries.
        adapter_names = {
            adapter.get("name")
            for adapter in data.get("adapters", [])
            if isinstance(adapter, dict) and adapter.get("name")
        }
        subject_concepts = {
            name: text
            for name, text in concepts.items()
            if (not adapter_names) or (name in adapter_names)
        }
        _raise_if_subject_positions_empty(subject_concepts, token_pos_maps, prompt)
        
        # Handle background
        enable_background = settings.get("enable_background", True)
        background_text = settings.get("background_text", "")
        
        if enable_background:
            bg_positions = find_background_positions(
                clip=clip,
                prompt=prompt,
                background_text=background_text if background_text else None,
                model_type=model_type,
                system_prompt=system_prompt,
            )
            if bg_positions:
                token_pos_maps["__background__"] = [bg_positions]
                print(f"[FreeFuseTokenPositions] Background positions: {bg_positions}")
            else:
                _warn_if_background_text_missing(
                    background_text=background_text,
                    bg_positions=bg_positions,
                    prompt=prompt,
                    source="FreeFuseTokenPositions",
                )
        
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
                "prompt": ("STRING", {"multiline": True}),
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
        
        # Detect model type (clip-only fallback for simple workflow node)
        model_type = detect_model_type(clip=clip)
        print(f"[FreeFuseConceptMapSimple] Detected model type: {model_type}")
        
        # For Z-Image (Lumina2), pass system prompt so token positions
        # align with the conditioning produced by CLIPTextEncodeLumina2
        system_prompt = LUMINA2_SYSTEM_PROMPT if model_type == 'z_image' else None
        
        # Compute token positions
        token_pos_maps = find_concept_positions(
            clip=clip,
            prompts=prompt,
            concepts=concepts,
            filter_meaningless=filter_meaningless,
            filter_single_char=filter_single_char,
            model_type=model_type,
            system_prompt=system_prompt,
        )

        _raise_if_subject_positions_empty(concepts, token_pos_maps, prompt)
        
        # Handle background
        if enable_background:
            bg_text = background_text.strip() if background_text else ""
            bg_positions = find_background_positions(
                clip=clip,
                prompt=prompt,
                background_text=bg_text if bg_text else None,
                model_type=model_type,
                system_prompt=system_prompt,
            )
            if bg_positions:
                token_pos_maps["__background__"] = [bg_positions]
                print(f"[FreeFuseConceptMapSimple] Background positions: {bg_positions}")
            else:
                _warn_if_background_text_missing(
                    background_text=bg_text,
                    bg_positions=bg_positions,
                    prompt=prompt,
                    source="FreeFuseConceptMapSimple",
                )
        
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
