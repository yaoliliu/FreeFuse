"""
FreeFuse Concept Map Node

Maps adapter names to concept text and computes token positions for similarity map collection.
Supports both Flux (T5 tokenizer) and SDXL (CLIP tokenizer) models.
"""

from typing import Dict, List, Union
import re

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
    newline_issues = []

    for adapter_name, concept_text in concepts.items():
        positions_per_prompt = _normalize_positions_list(token_pos_maps.get(adapter_name, []))
        missing_prompt_indices = []

        if not positions_per_prompt:
            # Check if concept_text has newlines
            if '\n' in concept_text:
                newline_issues.append((adapter_name, concept_text))
                continue
            missing_prompt_indices = ["all"]
        else:
            for idx, positions in enumerate(positions_per_prompt):
                if not positions:
                    missing_prompt_indices.append(str(idx + 1))

        if missing_prompt_indices:
            missing.append((adapter_name, concept_text, ",".join(missing_prompt_indices)))

    if newline_issues:
        newline_lines = "\n".join(
            f"  - {name} (contains newlines: '{_format_concept_preview(text)}')"
            for name, text in newline_issues
        )
        print(
            "[FreeFuse] NOTE: Some concepts contain newlines.\n"
            "Newlines are preserved in FreeFuse data but are replaced with spaces for token matching.\n"
            f"Concepts with newlines:\n{newline_lines}\n"
        )

    if not missing:
        return

    # If we're here, there are real missing concepts
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
    

def _format_token_map(concepts: Dict[str, str], token_pos_maps: Dict[str, List[List[int]]]) -> str:
    """
    Format token positions into a human-readable string.
    """
    lines = []
    
    # Check if background exists in token_pos_maps
    has_background = "__background__" in token_pos_maps
    
    for name, positions_list in token_pos_maps.items():
        if name == "__background__":
            display = "BACKGROUND"
        else:
            concept = concepts.get(name, "")
            if len(concept) > 40:
                concept = concept[:37] + "..."
            display = f"{name}[{concept}]"
        
        if positions_list and positions_list[0]:
            positions = positions_list[0]
            lines.append(f"{display}: {positions}")
        else:
            lines.append(f"{display}: []")
    
    # Optional: Add a note if background was requested but not found
    if not has_background:
        # You could check settings here if needed
        pass
        
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
    
    RETURN_TYPES = ("FREEFUSE_DATA", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("freefuse_data", "user_prompt", "combined_prompt", "token_map")
    FUNCTION = "compute_positions"
    CATEGORY = "FreeFuse"
    
    def _normalize_text(self, text: str) -> str:
        """
        Clean text for tokenizer:
        1. Replace newlines and carriage returns with spaces
        2. Collapse multiple spaces into single space
        3. Strip leading/trailing whitespace
        """
        if not text or not isinstance(text, str):
            return ""
        # Step 1: Replace newlines and carriage returns with spaces
        text = text.replace('\n', ' ').replace('\r', ' ')
        # Step 2: Collapse multiple spaces and strip
        return re.sub(r'\s+', ' ', text).strip()
    
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
        
        # 🔥 FIX: Add background text from settings if enabled
        enable_background = settings.get("enable_background", True)
        background_text = settings.get("background_text", "")
        
        if enable_background and background_text and background_text.strip():
            # Add background as-is without a prefix
            formatted_lines.append(background_text.strip())
            print(f"   Adding background: {background_text[:30]}...")
        
        # Join with NEWLINES for internal storage
        captions_text = "\n".join(formatted_lines)
        
        # 🔥 BUILD USER PROMPT: user_text + captions
        user_parts = []
        if user_text:
            user_parts.append(user_text.strip())
        if captions_text:
            user_parts.append(captions_text.strip())
        user_prompt_raw = " ".join(user_parts)
        
        # 🔥 BUILD COMBINED PROMPT: injection_text + user_text + captions
        combined_parts = []
        if injection_text:
            combined_parts.append(injection_text.strip())
        if user_text:
            combined_parts.append(user_text.strip())
        if captions_text:
            combined_parts.append(captions_text.strip())
        combined_prompt_raw = " ".join(combined_parts)

        if not concepts:
            print("[FreeFuseTokenPositions] Warning: No concepts defined")
            data["token_pos_maps"] = {}
            # Still return cleaned versions for display
            return (data, 
                    self._normalize_text(user_prompt_raw), 
                    self._normalize_text(combined_prompt_raw), 
                    "No concepts defined")
        
        # Detect model type
        model_type_hint = data.get("model_type")
        model_type = detect_model_type(clip=clip, model_type_hint=model_type_hint)
        print(f"[FreeFuseTokenPositions] Detected model type: {model_type}")

        # 🔥 NEW: Detect transformer block count for LTX-Video
        # Note: CLIP object doesn't have diffusion_model, so we can't detect blocks here
        # Block count will be detected by the similarity extractor which has the MODEL
        if model_type == "ltx_video":
            print(f"[FreeFuseTokenPositions] LTX-Video detected - block count will be detected by similarity extractor")

        # For Z-Image (Lumina2) and Qwen-Image, pass system prompt
        system_prompt = LUMINA2_SYSTEM_PROMPT if model_type in ('z_image', 'qwen_image') else None
        
        # 🔥 PREPARE CONCEPTS FOR TOKEN MATCHING
        concepts_for_matching = {}
        concepts_with_newlines = []
        
        for name, text in concepts.items():
            if name == "__background__":
                concepts_for_matching[name] = text
            else:
                cleaned = self._normalize_text(text)
                concepts_for_matching[name] = cleaned
                if text != cleaned and '\n' in text:
                    concepts_with_newlines.append(name)
        
        if concepts_with_newlines:
            print(f"[FreeFuseTokenPositions] Normalized {len(concepts_with_newlines)} concepts with newlines")
        
        # 🔥 NORMALIZE THE PROMPTS FOR PROCESSING AND DISPLAY
        normalized_user_prompt = self._normalize_text(user_prompt_raw)
        normalized_combined_prompt = self._normalize_text(combined_prompt_raw)
        
        print(f"\n[FreeFuseTokenPositions] Final combined prompt (cleaned):")
        print(f"  \"{normalized_combined_prompt}\"\n")
        
        # 🔥 PROCESS WITH NORMALIZED TEXT
        token_pos_maps = find_concept_positions(
            clip=clip,
            prompts=normalized_user_prompt,  # Use normalized for matching
            concepts=concepts_for_matching,
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
        _raise_if_subject_positions_empty(subject_concepts, token_pos_maps, user_prompt_raw)
        
        # Handle background
        if enable_background:
            bg_positions = find_background_positions(
                clip=clip,
                prompt=normalized_user_prompt,
                background_text=background_text if background_text else None,
                model_type=model_type,
                system_prompt=system_prompt,
            )
            if bg_positions:
                token_pos_maps["__background__"] = [bg_positions]
                print(f"[FreeFuseTokenPositions] Background positions: {bg_positions}")
            else:
                print(f"[FreeFuseTokenPositions] No background positions found for: '{background_text}'")
        
        # Log results
        for name, positions_list in token_pos_maps.items():
            if name != "__background__":
                print(f"[FreeFuseTokenPositions] {name}: positions = {positions_list}")
        
        # Generate token map string
        token_map_string = _format_token_map(concepts, token_pos_maps)
        
        data["token_pos_maps"] = token_pos_maps
        data["model_type"] = model_type
        data["injection_text"] = injection_text
        data["user_text"] = user_text
        data["captions_text"] = captions_text
        data["user_prompt_raw"] = user_prompt_raw
        data["combined_prompt_raw"] = combined_prompt_raw
        data["user_prompt"] = normalized_user_prompt
        data["combined_prompt"] = normalized_combined_prompt
        data["concepts_normalized"] = concepts_for_matching
        
        # 👇 Return NORMALIZED prompts for display
        return (data, normalized_user_prompt, normalized_combined_prompt, token_map_string)
        

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
    
    def _normalize_text(self, text: str) -> str:
        """Helper to normalize text."""
        if not text or not isinstance(text, str):
            return ""
        text = text.replace('\n', ' ').replace('\r', ' ')
        return re.sub(r'\s+', ' ', text).strip()
    
    def process(self, clip, prompt, adapter_name_1, concept_text_1,
                adapter_name_2="", concept_text_2="",
                adapter_name_3="", concept_text_3="",
                adapter_name_4="", concept_text_4="",
                enable_background=True, background_text="",
                filter_meaningless=True, filter_single_char=True):
        
        # Build concepts dict (keep original)
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
                "normalized_prompt": "",
            },)
        
        # Detect model type
        model_type = detect_model_type(clip=clip)
        print(f"[FreeFuseConceptMapSimple] Detected model type: {model_type}")

        # For Z-Image (Lumina2) and Qwen-Image, pass system prompt
        system_prompt = LUMINA2_SYSTEM_PROMPT if model_type in ('z_image', 'qwen_image') else None
        
        # 🔥 NORMALIZE EVERYTHING
        normalized_prompt = self._normalize_text(prompt)
        normalized_concepts = {}
        for name, text in concepts.items():
            normalized_concepts[name] = self._normalize_text(text)
        
        print(f"\n[FreeFuseConceptMapSimple] Final prompt (cleaned):")
        print(f"  \"{normalized_prompt}\"\n")
        
        # Compute token positions using normalized text
        token_pos_maps = find_concept_positions(
            clip=clip,
            prompts=normalized_prompt,
            concepts=normalized_concepts,
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
                prompt=normalized_prompt,
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
            "concepts": concepts,  # Keep original
            "settings": {
                "enable_background": enable_background,
                "background_text": background_text.strip(),
            },
            "token_pos_maps": token_pos_maps,
            "model_type": model_type,
            "prompt": prompt,  # Keep original
            "normalized_prompt": normalized_prompt,  # Add cleaned version
            "concepts_normalized": normalized_concepts,
        }
        
        return (data,)


# Export node mappings - MAKE SURE THESE ARE AT THE BOTTOM
NODE_CLASS_MAPPINGS = {
    "FreeFuseConceptMap": FreeFuseConceptMap,
    "FreeFuseTokenPositions": FreeFuseTokenPositions,
    "FreeFuseConceptMapSimple": FreeFuseConceptMapSimple,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeFuseConceptMap": "FreeFuse Concept Map",
    "FreeFuseTokenPositions": "FreeFuse Token Positions",
    "FreeFuseConceptMapSimple": "FreeFuse Concept Map (Simple)",
}
