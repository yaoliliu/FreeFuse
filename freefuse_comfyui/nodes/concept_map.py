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
            # Check if concept_text might have newlines - if so, it's probably intentional
            if '\n' in concept_text:
                print(f"[FreeFuse] Note: Concept '{adapter_name}' contains newlines - skipping validation")
                continue
            missing_prompt_indices = ["all"]
        else:
            for idx, positions in enumerate(positions_per_prompt):
                if not positions:
                    missing_prompt_indices.append(str(idx + 1))

        if missing_prompt_indices:
            missing.append((adapter_name, concept_text, ",".join(missing_prompt_indices)))

    if not missing:
        return

    # If we're here, there are real missing concepts without newlines
    missing_lines = "\n".join(
        f"  - {name} (concept_text='{_format_concept_preview(text)}', prompt_index={indices})"
        for name, text, indices in missing
    )
    prompt_preview = _format_prompt_preview(prompt)
    
    # Just warn, don't raise error
    print(
        "[FreeFuse] WARNING: Some subject adapter token positions are empty.\n"
        "Each subject concept_text should appear verbatim in the main prompt.\n"
        f"Missing subject adapters:\n{missing_lines}\n"
        f"Main prompt preview: \"{prompt_preview}\"\n"
        "Continuing anyway - mask generation may be affected."
    )
    # Don't raise ValueError


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
    
    # Add this helper function near the top of the file, after the other helper functions
def _format_token_map(concepts: Dict[str, str], token_pos_maps: Dict[str, List[List[int]]]) -> str:
    """
    Format token positions into a human-readable string.
    
    Args:
        concepts: Dictionary mapping adapter names to concept text
        token_pos_maps: Dictionary mapping adapter names to token positions
    
    Returns:
        Formatted string showing each adapter and its token positions
    """
    lines = []
    for name, positions_list in token_pos_maps.items():
        if name == "__background__":
            display = "BACKGROUND"
        else:
            concept = concepts.get(name, "")
            # Truncate concept text if too long
            if len(concept) > 40:
                concept = concept[:37] + "..."
            display = f"{name}[{concept}]"
        
        if positions_list and positions_list[0]:
            positions = positions_list[0]
            lines.append(f"{display}: {positions}")
        else:
            lines.append(f"{display}: []")
    
    return "\n".join(lines)


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
    
    This node takes the CLIP model and user prompt, appends the collected captions,
    and computes token positions for all concepts.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                # 👇 Text injection that goes BEFORE everything
                "injection_text": ("STRING", {
                    "multiline": True,
                    "default": "You are an assistant designed to generate superior images.",
                    "placeholder": "Text injected at the very beginning...",
                    "tooltip": "This text will be inserted before everything else"
                }),
                # 👇 This is the user's text input field
                "user_text": ("STRING", {
                    "multiline": True,
                    "default": "A picture of",
                    "placeholder": "Enter your main prompt here...",
                    "tooltip": "This text will have the captions appended to it"
                }),
                "freefuse_data": ("FREEFUSE_DATA",),
            },
            "optional": {
                "filter_meaningless": ("BOOLEAN", {"default": True}),
                "filter_single_char": ("BOOLEAN", {"default": True}),
            }
        }
    
    # 👇 FOUR outputs now: freefuse_data, user_prompt, combined_prompt, token_map
    RETURN_TYPES = ("FREEFUSE_DATA", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("freefuse_data", "user_prompt", "combined_prompt", "token_map")
    FUNCTION = "compute_positions"
    CATEGORY = "FreeFuse"
    
    def compute_positions(self, clip, injection_text, user_text, freefuse_data,
                          filter_meaningless=True, filter_single_char=True):
        
        # Copy data to avoid mutation
        data = dict(freefuse_data)
        data["concepts"] = dict(data.get("concepts", {}))
        data["settings"] = dict(data.get("settings", {}))
        
        concepts = data.get("concepts", {})
        settings = data.get("settings", {})

        # 📝 COLLECT AND FORMAT CAPTIONS
        formatted_lines = []
        
        # Get adapters data which contains location_text
        adapters = data.get("adapters", [])
        adapter_info = {adapter.get("name"): adapter for adapter in adapters if adapter.get("name")}
        
        # Add all LoRA concept texts - just name + concept
        for name, text in concepts.items():
            if name != "__background__" and text and text.strip():
                formatted_lines.append(f"{name} {text.strip()}")
                print(f"   Adding for {name}: {name} {text[:30]}...")
        
        # Add background text from settings if enabled
        enable_background = settings.get("enable_background", True)
        background_text = settings.get("background_text", "")
        
        if enable_background and background_text and background_text.strip():
            formatted_lines.append(background_text.strip())
        
        # 🔥 IMPORTANT: Join with NEWLINES, not spaces!
        captions_text = "\n".join(formatted_lines)
        
        # 🔥 BUILD USER PROMPT: user_text + captions (all with spaces)
        user_parts = []
        if user_text:
            user_parts.append(user_text.strip())
        if captions_text:
            user_parts.append(captions_text.strip())
        user_prompt = " ".join(user_parts)
        
        # 🔥 BUILD COMBINED PROMPT: injection_text + user_text + captions
        combined_parts = []
        if injection_text:
            combined_parts.append(injection_text.strip())
        if user_text:
            combined_parts.append(user_text.strip())
        if captions_text:
            combined_parts.append(captions_text.strip())
        combined_prompt = " ".join(combined_parts)

        if not concepts:
            print("[FreeFuseTokenPositions] Warning: No concepts defined")
            data["token_pos_maps"] = {}
            return (data, user_prompt, combined_prompt, "No concepts defined")
        
        # Detect model type
        model_type_hint = data.get("model_type")
        model_type = detect_model_type(clip=clip, model_type_hint=model_type_hint)
        print(f"[FreeFuseTokenPositions] Detected model type: {model_type}")

        # For Z-Image (Lumina2) and Qwen-Image, pass system prompt
        system_prompt = LUMINA2_SYSTEM_PROMPT if model_type in ('z_image', 'qwen_image') else None

        # 🔥 PROCESS THE USER PROMPT (not including injection text)
        token_pos_maps = find_concept_positions(
            clip=clip,
            prompts=user_prompt,
            concepts=concepts,
            filter_meaningless=filter_meaningless,
            filter_single_char=filter_single_char,
            model_type=model_type,
            system_prompt=system_prompt,
        )

        # Validate positions
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
        _raise_if_subject_positions_empty(subject_concepts, token_pos_maps, user_prompt)
        
        # Handle background
        if enable_background:
            bg_positions = find_background_positions(
                clip=clip,
                prompt=user_prompt,
                background_text=background_text if background_text else None,
                model_type=model_type,
                system_prompt=system_prompt,
            )
            if bg_positions:
                token_pos_maps["__background__"] = [bg_positions]
                print(f"[FreeFuseTokenPositions] Background positions: {bg_positions}")
        
        # Log results
        for name, positions_list in token_pos_maps.items():
            if name != "__background__":
                print(f"[FreeFuseTokenPositions] {name}: positions = {positions_list}")
        
        # 👇 Generate token map string using helper function
        token_map_string = _format_token_map(concepts, token_pos_maps)
        
        data["token_pos_maps"] = token_pos_maps
        data["model_type"] = model_type
        data["injection_text"] = injection_text
        data["user_text"] = user_text
        data["captions_text"] = captions_text
        data["user_prompt"] = user_prompt
        data["combined_prompt"] = combined_prompt
        
        return (data, user_prompt, combined_prompt, token_map_string)
        
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

        # For Z-Image (Lumina2) and Qwen-Image, pass system prompt so token positions
        # align with the conditioning produced by CLIPTextEncodeLumina2
        system_prompt = LUMINA2_SYSTEM_PROMPT if model_type in ('z_image', 'qwen_image') else None
        
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
