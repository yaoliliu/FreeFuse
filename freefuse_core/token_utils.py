"""
FreeFuse Token Position Utilities

Unified token position finding for both Flux (T5) and SDXL (CLIP) models in ComfyUI.

T5 tokenizer (Flux): Uses offset_mapping (character position mapping)
CLIP tokenizer (SDXL): Uses sliding window token matching
"""

from typing import Dict, List, Union, Optional, Any


# Default Lumina2 system prompt used by CLIPTextEncodeLumina2 in ComfyUI.
# This must be prepended to the user prompt for Z-Image token position finding
# so that positions align with the actual prompt_embeds tensor.
LUMINA2_SYSTEM_PROMPT = (
    "You are an assistant designed to generate superior images with the superior "
    "degree of image-text alignment based on textual prompts or user prompts."
)

# Flux2.Klein should match diffusers apply_chat_template(..., enable_thinking=False).
KLEIN_NO_THINK_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"


# Stopwords and punctuation to filter (shared between both methods)
STOPWORDS = {
    # Articles
    'a', 'an', 'the',
    # Conjunctions
    'and', 'or', 'but', 'nor', 'so', 'yet',
    # Prepositions
    'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'as',
    'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'under', 'over',
    # Pronouns
    'it', 'its', 'this', 'that', 'these', 'those', 'their', 'his', 'her',
    'my', 'your', 'our',
    # Auxiliaries/be verbs
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'has', 'have', 'had', 'having',
    # Other common stopwords
    'which', 'who', 'whom', 'whose', 'where', 'when', 'while',
}

PUNCTUATION = {
    ',', '.', '!', '?', ';', ':', '"', "'", '(', ')', '[', ']',
    '{', '}', '-', '–', '—', '/', '\\', '...', '..'
}

MEANINGLESS_TOKENS = STOPWORDS | PUNCTUATION


def clean_token_text(token_text: str) -> str:
    """Clean token text by removing special prefixes/suffixes."""
    # Remove T5's ▁ prefix (U+2581), CLIP's </w> suffix, and GPT-2's Ġ marker.
    # NOTE: Do NOT strip ASCII '_' (U+005F) — it is real content in tokens
    # like '_A' (Qwen3 tokenization of "Jinx_Arcane").
    cleaned = token_text.replace('▁', '')
    cleaned = cleaned.replace('</w>', '').replace('Ġ', '')
    return cleaned.strip().lower()


def is_meaningless_token(token_text: str, check_single_char: bool = True) -> bool:
    """Check if a token is meaningless (stopword, punctuation, etc.)."""
    cleaned = clean_token_text(token_text)
    if not cleaned:
        return True
    if check_single_char and len(cleaned) == 1:
        return True
    return cleaned in MEANINGLESS_TOKENS


def _normalize_model_type(model_type: Optional[str]) -> Optional[str]:
    """Normalize user-provided model type aliases."""
    if not isinstance(model_type, str):
        return None

    value = model_type.strip().lower()
    if not value or value in {"auto", "unknown", "none", "null"}:
        return None

    aliases = {
        "zimage": "z_image",
        "z-image": "z_image",
        "lumina2": "z_image",
        "nextdit": "z_image",
        "flux1": "flux",
        "flux2.klein": "flux2",
        "flux2_klein": "flux2",
        "sd1.5": "sd1",
        "sd2": "sd1",
        "sd2.x": "sd1",
    }
    return aliases.get(value, value)


def _extract_model_core(model):
    """Extract inner model object from a ComfyUI model patcher."""
    if model is None:
        return None
    return getattr(model, "model", model)


def detect_model_type_from_model(model) -> str:
    """
    Detect model type from UNet/model object instead of text encoder.

    Args:
        model: ComfyUI model patcher or inner model object

    Returns:
        'flux', 'flux2', 'sdxl', 'z_image', or 'unknown'
    """
    core_model = _extract_model_core(model)
    if core_model is None:
        return "unknown"

    model_cls = core_model.__class__.__name__.lower()
    if "nextdit" in model_cls or "lumina" in model_cls:
        return "z_image"
    if "flux2" in model_cls:
        return "flux2"
    if "flux" in model_cls:
        return "flux"

    has_flux_block_layout = False
    dm = getattr(core_model, "diffusion_model", None)
    if dm is not None:
        dm_cls = dm.__class__.__name__.lower()
        if "nextdit" in dm_cls or "lumina" in dm_cls:
            return "z_image"
        if "flux2" in dm_cls:
            return "flux2"
        if "flux" in dm_cls:
            return "flux"
        if hasattr(dm, "double_blocks") and hasattr(dm, "single_blocks"):
            # Could be Flux1 or Flux2; defer final decision to config hints below.
            has_flux_block_layout = True
        # Lumina2/NextDiT style: layered stack without Flux double/single blocks
        if hasattr(dm, "layers") and not hasattr(dm, "double_blocks"):
            return "z_image"

    model_cfg = getattr(core_model, "model_config", None)
    unet_cfg = getattr(model_cfg, "unet_config", None)
    if isinstance(unet_cfg, dict):
        image_model = str(unet_cfg.get("image_model", "")).lower()
        if image_model == "flux2":
            return "flux2"
        if image_model == "flux":
            return "flux"
        if image_model == "lumina2":
            return "z_image"

    if has_flux_block_layout:
        return "flux"

    return "unknown"


def detect_model_type(clip=None, model=None, model_type_hint: Optional[str] = None) -> str:
    """
    Detect model type using model-first strategy.
    
    Args:
        clip: ComfyUI CLIP object (fallback only)
        model: ComfyUI model patcher or model object (preferred)
        model_type_hint: Explicit override or hint from workflow data
        
    Returns:
        'flux', 'flux2', 'sdxl', 'z_image', 'qwen3', 'sd1', or 'unknown'
    """
    normalized_hint = _normalize_model_type(model_type_hint)
    if normalized_hint is not None:
        return normalized_hint

    model_type = detect_model_type_from_model(model)
    if model_type != "unknown":
        return model_type

    if clip is None:
        return "unknown"

    cond_stage_model = getattr(clip, "cond_stage_model", None)
    cond_stage_name = cond_stage_model.__class__.__name__.lower() if cond_stage_model is not None else ""
    if "nextdit" in cond_stage_name or "zimage" in cond_stage_name or "lumina" in cond_stage_name:
        return "z_image"
    if "flux2" in cond_stage_name:
        return "flux2"
    if "flux" in cond_stage_name:
        return "flux"

    tokenizer = getattr(clip, "tokenizer", None)
    if tokenizer is None:
        return "unknown"

    clip_name = str(getattr(tokenizer, "clip_name", "")).lower()
    clip_key = getattr(tokenizer, "clip", None)
    clip_key_lower = clip_key.lower() if isinstance(clip_key, str) else ""

    # Qwen3 tokenizer family can be used by both Z-Image and Flux2-Klein.
    # Without model context, treat it as generic qwen3.
    tokenizer_attr_names = set(vars(tokenizer).keys())
    if (
        any("qwen3" in name.lower() for name in tokenizer_attr_names)
        or "qwen3" in clip_name
        or "qwen3" in clip_key_lower
    ):
        return "qwen3"

    if hasattr(tokenizer, "t5xxl"):
        return "flux"
    if hasattr(tokenizer, "clip_g"):
        return "sdxl"
    if hasattr(tokenizer, "clip_l"):
        if clip_name in {"l", "clip_l"} or clip_key_lower in {"l", "clip_l"}:
            return "sd1"
        return "sdxl"

    # SD1-style dynamic clip key, e.g. tokenizer.clip="clip_l"
    if isinstance(clip_key, str) and hasattr(tokenizer, clip_key):
        return "sd1"

    if hasattr(tokenizer, "tokenizer"):
        return "sd1"

    return "unknown"


def _extract_nested_tokenizer(tokenizer_obj):
    """
    Return underlying HF tokenizer from wrapper objects when available.
    """
    if tokenizer_obj is None:
        return None
    if hasattr(tokenizer_obj, "tokenizer"):
        return tokenizer_obj.tokenizer
    return tokenizer_obj


def _resolve_sd1_subtokenizer(tokenizer):
    """
    Resolve SD1-style dynamic tokenizer wrapper:
    tokenizer.clip is a string key to the actual sub-tokenizer object.
    """
    clip_key = getattr(tokenizer, "clip", None)
    if isinstance(clip_key, str) and hasattr(tokenizer, clip_key):
        return _extract_nested_tokenizer(getattr(tokenizer, clip_key))
    return None


def _resolve_qwen3_tokenizer(tokenizer):
    """
    Resolve qwen3 tokenizer from multiple ComfyUI wrappers.
    """
    for key in ("qwen3_4b", "qwen3_8b"):
        if hasattr(tokenizer, key):
            resolved = _extract_nested_tokenizer(getattr(tokenizer, key))
            if resolved is not None:
                return resolved

    clip_name = str(getattr(tokenizer, "clip_name", "")).lower()
    clip_key = getattr(tokenizer, "clip", None)
    if "qwen3" in clip_name and hasattr(tokenizer, clip_name):
        resolved = _extract_nested_tokenizer(getattr(tokenizer, clip_name))
        if resolved is not None:
            return resolved
    if isinstance(clip_key, str) and "qwen3" in clip_key.lower() and hasattr(tokenizer, clip_key):
        resolved = _extract_nested_tokenizer(getattr(tokenizer, clip_key))
        if resolved is not None:
            return resolved

    return None


def get_tokenizer_for_model(clip, model_type: str = None):
    """
    Get the appropriate tokenizer object from CLIP.
    
    Args:
        clip: ComfyUI CLIP object
        model_type: Optional override ('flux', 'flux2', 'sdxl', 'sd1', 'z_image', 'qwen3')
        
    Returns:
        The underlying tokenizer object
    """
    if model_type is None:
        model_type = detect_model_type(clip=clip)
    else:
        model_type = _normalize_model_type(model_type) or "unknown"

    tokenizer = getattr(clip, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Could not find clip.tokenizer on CLIP object.")

    if model_type in ("z_image", "flux2", "qwen3"):
        resolved = _resolve_qwen3_tokenizer(tokenizer)
        if resolved is not None:
            return resolved

    if model_type == "flux":
        if hasattr(tokenizer, "t5xxl"):
            resolved = _extract_nested_tokenizer(tokenizer.t5xxl)
            if resolved is not None:
                return resolved

    if model_type in ("sdxl", "sd1"):
        if hasattr(tokenizer, "clip_l"):
            resolved = _extract_nested_tokenizer(tokenizer.clip_l)
            if resolved is not None:
                return resolved
        resolved = _resolve_sd1_subtokenizer(tokenizer)
        if resolved is not None:
            return resolved

    # Best-effort fallback chain
    for key in ("t5xxl", "clip_l", "clip_g"):
        if hasattr(tokenizer, key):
            resolved = _extract_nested_tokenizer(getattr(tokenizer, key))
            if resolved is not None:
                return resolved

    resolved = _resolve_sd1_subtokenizer(tokenizer)
    if resolved is not None:
        return resolved

    direct = _extract_nested_tokenizer(tokenizer)
    if direct is not tokenizer and direct is not None:
        return direct

    available_attrs = sorted(vars(tokenizer).keys()) if hasattr(tokenizer, "__dict__") else []
    raise ValueError(
        "Could not resolve tokenizer for model_type='{}' (tokenizer_class='{}', attrs={}).".format(
            model_type, tokenizer.__class__.__name__, available_attrs
        )
    )


def find_concept_positions_t5(
    tokenizer,
    prompts: Union[str, List[str]],
    concepts: Dict[str, str],
    filter_meaningless: bool = True,
    filter_single_char: bool = True,
) -> Dict[str, List[List[int]]]:
    """
    Find token positions for T5-based models (Flux) using offset_mapping.
    
    This method uses character-to-token position mapping which handles
    tokenizer context differences correctly.
    
    Args:
        tokenizer: T5TokenizerFast instance
        prompts: Single prompt or list of prompts
        concepts: Dict mapping concept name to concept text
        filter_meaningless: Whether to filter stopwords/punctuation
        filter_single_char: Whether to filter single-char tokens
        
    Returns:
        Dict mapping concept name to list of position lists (one per prompt)
    """
    if isinstance(prompts, str):
        prompts = [prompts]
    
    # Tokenize each prompt with offset_mapping
    prompt_data_list = []
    for prompt in prompts:
        # T5TokenizerFast supports offset_mapping
        encoded = tokenizer(
            prompt,
            padding=False,
            truncation=False,
            return_offsets_mapping=True,
            return_tensors=None,  # Return lists, not tensors
        )
        
        prompt_ids = encoded['input_ids']
        offset_mapping = encoded['offset_mapping']  # [(start, end), ...]
        
        prompt_data_list.append({
            'text': prompt,
            'ids': prompt_ids,
            'offset_mapping': offset_mapping
        })
    
    # Find positions for each concept
    concept_pos_map = {}
    for concept_name, concept_text in concepts.items():
        concept_pos_map[concept_name] = []
        
        for prompt_data in prompt_data_list:
            positions = []
            positions_with_text = []
            prompt_text = prompt_data['text']
            prompt_ids = prompt_data['ids']
            offset_mapping = prompt_data['offset_mapping']
            
            # Find all occurrences of concept_text in prompt
            search_start = 0
            while True:
                char_start = prompt_text.find(concept_text, search_start)
                if char_start == -1:
                    break
                char_end = char_start + len(concept_text)
                
                # Find tokens that overlap with this character range
                for token_idx, (token_start, token_end) in enumerate(offset_mapping):
                    if token_end > char_start and token_start < char_end:
                        if token_idx not in positions:
                            positions.append(token_idx)
                            # Decode token for filtering
                            token_text = tokenizer.decode(
                                [prompt_ids[token_idx]], 
                                skip_special_tokens=False
                            )
                            positions_with_text.append((token_idx, token_text))
                
                search_start = char_start + 1
            
            # Apply filtering
            if filter_meaningless and positions_with_text:
                filtered_positions = [
                    pos for pos, text in positions_with_text
                    if not is_meaningless_token(text, check_single_char=filter_single_char)
                ]
                
                # Fallback if all filtered
                if not filtered_positions:
                    non_punct = [
                        pos for pos, text in positions_with_text
                        if clean_token_text(text) not in PUNCTUATION
                    ]
                    if non_punct:
                        filtered_positions = [non_punct[0]]
                    elif positions_with_text:
                        filtered_positions = [positions_with_text[0][0]]
                
                positions = filtered_positions
            
            positions.sort()
            concept_pos_map[concept_name].append(positions)
    
    return concept_pos_map


def _collect_token_ids_recursive(payload, flat_ids: List[int]) -> None:
    """Recursively collect token ids from nested ComfyUI token structures."""
    if payload is None:
        return

    if isinstance(payload, dict):
        for value in payload.values():
            _collect_token_ids_recursive(value, flat_ids)
        return

    if isinstance(payload, (int, float)):
        flat_ids.append(int(payload))
        return

    if isinstance(payload, (tuple, list)):
        # Most ComfyUI entries are (token_id, weight[, word_id]).
        is_token_tuple = (
            payload
            and isinstance(payload[0], (int, float))
            and len(payload) <= 3
            and (len(payload) == 1 or isinstance(payload[1], (int, float)))
        )
        if is_token_tuple:
            flat_ids.append(int(payload[0]))
            return
        for item in payload:
            _collect_token_ids_recursive(item, flat_ids)


def _flatten_chunked_token_ids(token_weight_pairs) -> List[int]:
    """
    Flatten ComfyUI token-weight pairs into a single token id list.

    Supports both CLIP-style chunking and Qwen/Klein dict outputs.
    """
    chunks = token_weight_pairs
    if isinstance(token_weight_pairs, dict):
        chunks = None
        # Prefer known text branches first.
        for key in ("l", "qwen3_8b", "qwen3_4b", "qwen3", "t5xxl", "clip_l", "g"):
            if key in token_weight_pairs:
                chunks = token_weight_pairs[key]
                break
        if chunks is None:
            # Fallback to first available payload.
            chunks = next(iter(token_weight_pairs.values()), [])

    flat_ids: List[int] = []
    _collect_token_ids_recursive(chunks, flat_ids)
    return flat_ids


def find_concept_positions_clip(
    clip,
    tokenizer,
    prompts: Union[str, List[str]],
    concepts: Dict[str, str],
    filter_meaningless: bool = True,
    filter_single_char: bool = True,
) -> Dict[str, List[List[int]]]:
    """
    Find token positions for CLIP-based models (SDXL, SD1.x).

    IMPORTANT: For SDXL in ComfyUI, tokenization is chunked into 77-token blocks
    with BOS/EOS inserted for each chunk. The cross-attention key/value tensor
    has shape (B, heads, img_len, N*77) where N = number of chunks. So token
    positions must refer to indices in the *flattened chunked* stream.

    When a concept text spans a chunk boundary, the contiguous concept token
    sequence is interrupted by EOS+BOS tokens. This function handles that by:
    1. Building a mapping from "content-only index" -> "chunked position"
    2. Matching concept tokens against the content-only stream
    3. Returning the corresponding chunked positions

    Args:
        clip: ComfyUI CLIP object (needed for chunked tokenization)
        tokenizer: CLIPTokenizer instance
        prompts: Single prompt or list of prompts
        concepts: Dict mapping concept name to concept text
        filter_meaningless: Whether to filter stopwords/punctuation
        filter_single_char: Whether to filter single-char tokens

    Returns:
        Dict mapping concept name to list of position lists (one per prompt)
    """
    if isinstance(prompts, str):
        prompts = [prompts]

    # Detect BOS/EOS token IDs for this tokenizer
    bos_id = getattr(tokenizer, 'bos_token_id', 49406)
    eos_id = getattr(tokenizer, 'eos_token_id', 49407)
    if bos_id is None:
        bos_id = 49406
    if eos_id is None:
        eos_id = 49407

    concept_pos_map = {}
    for concept_name, concept_text in concepts.items():
        concept_pos_map[concept_name] = []

        # Tokenize concept text (without special tokens)
        concept_tokens = tokenizer.encode(concept_text, add_special_tokens=False)

        for prompt in prompts:
            # Get chunked tokenization from ComfyUI (this is what cross-attention sees)
            try:
                token_weight_pairs = clip.tokenize(prompt)
                flat_ids = _flatten_chunked_token_ids(token_weight_pairs)
            except Exception:
                flat_ids = tokenizer.encode(prompt, add_special_tokens=True)

            # Build content-only stream with position mapping.
            # Each chunk has: [BOS, content..., EOS, padding...]
            # We extract only content tokens and remember their positions
            # in the flattened chunked stream.
            content_ids: List[int] = []     # content token IDs only
            content_pos: List[int] = []     # corresponding position in flat_ids

            # Determine which positions are BOS/EOS/padding per chunk
            # Chunks are 77 tokens each
            chunk_size = 77
            num_chunks = (len(flat_ids) + chunk_size - 1) // chunk_size

            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * chunk_size
                chunk_end = min(chunk_start + chunk_size, len(flat_ids))

                for pos_in_chunk, global_pos in enumerate(range(chunk_start, chunk_end)):
                    tid = flat_ids[global_pos]
                    if pos_in_chunk == 0 and tid == bos_id:
                        continue  # Skip BOS
                    if tid == eos_id:
                        # EOS marks end of content in this chunk; skip rest
                        break
                    if tid == 0:
                        continue  # Skip padding (clip_g uses 0 for padding)
                    content_ids.append(tid)
                    content_pos.append(global_pos)

            # Now do sliding window match on the content-only stream
            positions = []
            positions_with_text = []
            concept_len = len(concept_tokens)
            content_len = len(content_ids)

            if concept_len > 0:
                for i in range(content_len - concept_len + 1):
                    match = all(
                        content_ids[i + j] == concept_tokens[j]
                        for j in range(concept_len)
                    )
                    if match:
                        for j in range(concept_len):
                            chunked_pos = content_pos[i + j]
                            if chunked_pos not in positions:
                                positions.append(chunked_pos)
                                token_text = tokenizer.decode(
                                    [content_ids[i + j]],
                                    skip_special_tokens=False
                                )
                                positions_with_text.append((chunked_pos, token_text))

            # Apply filtering
            if filter_meaningless and positions_with_text:
                filtered_positions = [
                    pos for pos, text in positions_with_text
                    if not is_meaningless_token(text, check_single_char=filter_single_char)
                ]
                
                # Fallback if all filtered
                if not filtered_positions:
                    non_punct = [
                        pos for pos, text in positions_with_text
                        if clean_token_text(text) not in PUNCTUATION
                    ]
                    if non_punct:
                        filtered_positions = [non_punct[0]]
                    elif positions_with_text:
                        filtered_positions = [positions_with_text[0][0]]
                
                positions = filtered_positions
            
            positions.sort()
            concept_pos_map[concept_name].append(positions)
    
    return concept_pos_map


def find_concept_positions_qwen3(
    clip,
    tokenizer,
    prompts: Union[str, List[str]],
    concepts: Dict[str, str],
    filter_meaningless: bool = True,
    filter_single_char: bool = True,
    system_prompt: Optional[str] = None,
    llama_template: Optional[str] = None,
) -> Dict[str, List[List[int]]]:
    """
    Find token positions for Z-Image (Qwen3) models.
    
    Z-Image uses Qwen3 tokenizer with chat template wrapping. This function:
    1. Optionally prepends the Lumina2 system prompt (to match CLIPTextEncodeLumina2)
    2. Passes the text through clip.tokenize() which applies the llama chat template
    3. Builds a concat_text by decoding each token and tracks positions
    4. Finds concept text in concat_text and maps back to token positions
    
    IMPORTANT - Embedding alignment:
    In ComfyUI, CLIPTextEncodeLumina2 prepends a system prompt before calling
    clip.tokenize(). If system_prompt is provided, this function prepends it
    in the same way so that the returned token positions correspond to the
    correct positions in the prompt_embeds tensor.
    
    Args:
        clip: ComfyUI CLIP object
        tokenizer: Qwen3 tokenizer instance
        prompts: Single prompt or list of prompts
        concepts: Dict mapping concept name to concept text
        filter_meaningless: Whether to filter stopwords/punctuation
        filter_single_char: Whether to filter single-char tokens
        system_prompt: System prompt to prepend (for ComfyUI Lumina2 alignment).
                       When provided, constructs "{system_prompt} <Prompt Start> {prompt}"
                       to match CLIPTextEncodeLumina2 behavior.
                       Use LUMINA2_SYSTEM_PROMPT for the standard Lumina2 prompt.
        llama_template: Optional explicit chat template passed to clip.tokenize().
                        Use this to align Flux2.Klein with no-<think> templating.
        
    Returns:
        Dict mapping concept name to list of position lists (one per prompt)
    """
    if isinstance(prompts, str):
        prompts = [prompts]
    
    # Build prompt data with template wrapping (matching ZImageTokenizer behavior)
    prompt_data_list = []
    for prompt in prompts:
        # For ComfyUI Z-Image: CLIPTextEncodeLumina2 prepends a system prompt
        # before calling clip.tokenize(). We must do the same so that token
        # positions align with the actual prompt_embeds tensor.
        if system_prompt:
            tokenize_prompt = f"{system_prompt} <Prompt Start> {prompt}"
        else:
            tokenize_prompt = prompt
        
        # Default Qwen chat template (used by Z-Image) unless caller overrides.
        template_to_use = llama_template or "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        wrapped_text = template_to_use.format(tokenize_prompt)
        
        # Tokenize using the ComfyUI clip's tokenize method
        # This gives us the actual token sequence used during generation
        try:
            token_kwargs = {}
            if llama_template is not None:
                token_kwargs["llama_template"] = llama_template
            token_weight_pairs = clip.tokenize(tokenize_prompt, **token_kwargs)
            token_ids = _flatten_chunked_token_ids(token_weight_pairs)
            if not token_ids:
                raise ValueError("clip.tokenize returned no token ids")
        except Exception as e:
            # Fallback: tokenize directly
            print(f"[FreeFuse] Warning: clip.tokenize failed, using direct tokenization: {e}")
            encoded = tokenizer(wrapped_text, add_special_tokens=True)
            token_ids = encoded['input_ids'] if hasattr(encoded, 'keys') else encoded.input_ids
            if hasattr(token_ids, 'tolist'):
                token_ids = token_ids.tolist()
        
        # Decode each token individually to build char offsets
        token_texts = [tokenizer.decode([tid]) for tid in token_ids]
        
        # Filter out padding tokens (0 or eos_token_id at end)
        # For Z-Image, we usually don't have padding but keep all non-zero tokens
        active_indices = list(range(len(token_ids)))
        active_token_texts = token_texts
        
        # Reconstruct the concatenated decoded string and track spans
        concat_text = ""
        token_spans = []  # (start_in_concat, end_in_concat) per token
        for tt in active_token_texts:
            start = len(concat_text)
            concat_text += tt
            token_spans.append((start, len(concat_text)))
        
        prompt_data_list.append({
            "raw": prompt,
            "wrapped": wrapped_text,
            "active_indices": active_indices,
            "active_token_texts": active_token_texts,
            "concat_text": concat_text,
            "token_spans": token_spans,
        })
    
    # Find positions for each concept
    concept_pos_map = {}
    for concept_name, concept_text in concepts.items():
        concept_pos_map[concept_name] = []
        
        for pd in prompt_data_list:
            positions = []
            positions_with_text = []
            
            # Find concept_text inside concat_text (case-sensitive)
            search_start = 0
            while True:
                idx = pd["concat_text"].find(concept_text, search_start)
                if idx == -1:
                    break
                c_start, c_end = idx, idx + len(concept_text)
                
                for tok_i, (ts, te) in enumerate(pd["token_spans"]):
                    if te > c_start and ts < c_end and tok_i not in positions:
                        positions.append(tok_i)
                        positions_with_text.append(
                            (tok_i, pd["active_token_texts"][tok_i])
                        )
                search_start = idx + 1
            
            # Filter meaningless tokens
            if filter_meaningless and positions_with_text:
                filtered_positions = [
                    pos for pos, text in positions_with_text
                    if not is_meaningless_token(text, check_single_char=filter_single_char)
                ]
                
                # Fallback if all filtered
                if not filtered_positions:
                    non_punct = [
                        pos for pos, text in positions_with_text
                        if clean_token_text(text) not in PUNCTUATION
                    ]
                    if non_punct:
                        filtered_positions = non_punct[:1]
                    elif positions_with_text:
                        filtered_positions = [positions_with_text[0][0]]
                
                positions = filtered_positions
            
            positions.sort()
            concept_pos_map[concept_name].append(positions)
    
    return concept_pos_map


def find_concept_positions(
    clip,
    prompts: Union[str, List[str]],
    concepts: Dict[str, str],
    filter_meaningless: bool = True,
    filter_single_char: bool = True,
    model_type: str = None,
    system_prompt: Optional[str] = None,
) -> Dict[str, List[List[int]]]:
    """
    Unified function to find concept token positions for any supported model.
    
    Automatically detects model type (Flux/SDXL) and uses the appropriate method.
    
    Args:
        clip: ComfyUI CLIP object
        prompts: Single prompt or list of prompts
        concepts: Dict mapping concept name (adapter name) to concept text
                 Example: {'lora_a': 'a woman with red hair', 'lora_b': 'a man in suit'}
        filter_meaningless: Whether to filter stopwords/punctuation tokens
        filter_single_char: Whether to filter single-character tokens
        model_type: Optional override ('flux', 'flux2', 'sdxl', 'sd1', 'z_image', 'qwen3')
        system_prompt: System prompt for Z-Image/Lumina2 alignment (see
                       find_concept_positions_qwen3 for details).
        
    Returns:
        Dict with structure:
        {
            'lora_a': [[pos1, pos2, ...], ...],  # One list per prompt
            'lora_b': [[pos1, pos2, ...], ...],
        }
        
    Example:
        >>> concepts = {'char1': 'Harry Potter', 'char2': 'Ron Weasley'}
        >>> prompt = "Harry Potter and Ron Weasley at Hogwarts"
        >>> pos_maps = find_concept_positions(clip, prompt, concepts)
        >>> # pos_maps['char1'][0] = [token indices for 'Harry Potter']
        >>> # pos_maps['char2'][0] = [token indices for 'Ron Weasley']
    """
    if model_type is None:
        model_type = detect_model_type(clip=clip)
    else:
        model_type = _normalize_model_type(model_type) or model_type
    
    tokenizer = get_tokenizer_for_model(clip, model_type)
    
    if model_type in ('z_image', 'flux2', 'qwen3'):
        qwen_template = KLEIN_NO_THINK_TEMPLATE if model_type == "flux2" else None
        return find_concept_positions_qwen3(
            clip, tokenizer, prompts, concepts,
            filter_meaningless=filter_meaningless,
            filter_single_char=filter_single_char,
            system_prompt=system_prompt,
            llama_template=qwen_template,
        )
    elif model_type == 'flux':
        return find_concept_positions_t5(
            tokenizer, prompts, concepts,
            filter_meaningless=filter_meaningless,
            filter_single_char=filter_single_char
        )
    elif model_type in ('sdxl', 'sd1'):
        return find_concept_positions_clip(
            clip, tokenizer, prompts, concepts,
            filter_meaningless=filter_meaningless,
            filter_single_char=filter_single_char
        )
    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            "Use 'flux', 'flux2', 'sdxl', 'z_image', 'qwen3', or 'sd1'."
        )


def find_eos_position_t5(
    clip,
    prompt: str,
    max_sequence_length: int = 512,
) -> int:
    """
    Find the EOS token position for T5 (Flux).
    
    T5 adds EOS token at the end of actual content, followed by padding.
    This is useful for background extraction in FreeFuse.
    
    Args:
        clip: ComfyUI CLIP object
        prompt: The prompt string
        max_sequence_length: Max sequence length for reference
        
    Returns:
        Index of the first EOS token, or -1 if not found
    """
    model_type = detect_model_type(clip)
    if model_type != 'flux':
        return -1  # Only T5 has explicit EOS in this manner
    
    tokenizer = get_tokenizer_for_model(clip, model_type)
    
    # Get EOS token ID
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        return -1
    
    # Tokenize and find EOS
    encoded = tokenizer(prompt, return_tensors=None)
    input_ids = encoded['input_ids']
    
    for i, token_id in enumerate(input_ids):
        if token_id == eos_token_id:
            return i
    
    return -1


def find_eos_position_qwen3(
    clip,
    prompt: str,
    model_type: Optional[str] = None,
) -> int:
    """
    Find the EOS token position for Qwen3-family tokenizers (Flux2-Klein).

    Args:
        clip: ComfyUI CLIP object
        prompt: Prompt string
        model_type: Optional explicit model type

    Returns:
        Index of the first EOS token, or -1 if not found.
    """
    if model_type is None:
        model_type = detect_model_type(clip=clip)
    else:
        model_type = _normalize_model_type(model_type) or model_type

    if model_type not in {"flux2", "qwen3"}:
        return -1

    tokenizer = get_tokenizer_for_model(clip, model_type)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        return -1

    try:
        token_kwargs = {}
        if model_type == "flux2":
            token_kwargs["llama_template"] = KLEIN_NO_THINK_TEMPLATE
        token_weight_pairs = clip.tokenize(prompt, **token_kwargs)
        input_ids = _flatten_chunked_token_ids(token_weight_pairs)
        if not input_ids:
            raise ValueError("empty token stream")
    except Exception:
        encoded = tokenizer(prompt, return_tensors=None, add_special_tokens=True)
        input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded.input_ids
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()

    for i, token_id in enumerate(input_ids):
        if int(token_id) == int(eos_token_id):
            return i

    return -1


def find_background_positions(
    clip,
    prompt: str,
    background_text: str = None,
    model_type: str = None,
    system_prompt: Optional[str] = None,
) -> List[int]:
    """
    Find token positions for background region.
    
    For Flux/Flux2: Uses EOS position if no background_text provided
    For SDXL/Z-Image: Uses the provided background_text or returns empty
    
    Args:
        clip: ComfyUI CLIP object
        prompt: The prompt string
        background_text: Optional background description text
        model_type: Optional model type override
        system_prompt: System prompt for Z-Image/Lumina2 alignment
        
    Returns:
        List of token positions for background
    """
    if model_type is None:
        model_type = detect_model_type(clip=clip)
    
    if background_text:
        # Use provided background text
        pos_map = find_concept_positions(
            clip, prompt, {'__bg__': background_text},
            filter_meaningless=True,
            filter_single_char=True,
            model_type=model_type,
            system_prompt=system_prompt,
        )
        return pos_map.get('__bg__', [[]])[0]
    
    # For Flux/Flux2, use EOS as background anchor when explicit text is absent.
    if model_type == 'flux':
        eos_pos = find_eos_position_t5(clip, prompt)
        if eos_pos >= 0:
            return [eos_pos]
    elif model_type in ('flux2', 'qwen3'):
        eos_pos = find_eos_position_qwen3(clip, prompt, model_type=model_type)
        if eos_pos >= 0:
            return [eos_pos]
    
    return []


# Convenience function for ComfyUI nodes
def compute_token_position_maps(
    clip,
    prompt: str,
    concepts: Dict[str, str],
    enable_background: bool = True,
    background_text: str = None,
    system_prompt: Optional[str] = None,
) -> Dict[str, List[List[int]]]:
    """
    Compute token position maps for FreeFuse, ready to use in sampling.
    
    This is the main entry point for ComfyUI nodes.
    
    Args:
        clip: ComfyUI CLIP object
        prompt: The generation prompt
        concepts: Dict mapping adapter names to concept descriptions
        enable_background: Whether to include background positions
        background_text: Optional background description (for explicit background)
        system_prompt: System prompt for Z-Image/Lumina2 alignment
        
    Returns:
        freefuse_token_pos_maps: Dict ready for FreeFuse sampling
    """
    # Compute concept positions
    token_pos_maps = find_concept_positions(
        clip, prompt, concepts, system_prompt=system_prompt,
    )
    
    # Add background if enabled
    if enable_background:
        bg_positions = find_background_positions(
            clip, prompt, background_text, system_prompt=system_prompt,
        )
        if bg_positions:
            token_pos_maps['__background__'] = [bg_positions]
    
    return token_pos_maps
