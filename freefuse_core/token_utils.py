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


def detect_model_type(clip) -> str:
    """
    Detect whether the CLIP model is for Flux (T5), SDXL, or Z-Image (Qwen3).
    
    Args:
        clip: ComfyUI CLIP object
        
    Returns:
        'flux', 'sdxl', 'z_image', or 'unknown'
    """
    tokenizer = clip.tokenizer
    
    # Check for Z-Image (Qwen3) tokenizer
    if hasattr(tokenizer, 'qwen3_4b'):
        return 'z_image'
    
    # Check for Flux tokenizer (has t5xxl attribute)
    if hasattr(tokenizer, 't5xxl'):
        return 'flux'
    
    # Check for SDXL tokenizer (has clip_l and clip_g, or just clip_l)
    if hasattr(tokenizer, 'clip_l') or hasattr(tokenizer, 'clip_g'):
        return 'sdxl'
    
    # Fallback: check for single clip attribute (SD1.x style)
    if hasattr(tokenizer, 'clip'):
        return 'sd1'
    
    return 'unknown'


def get_tokenizer_for_model(clip, model_type: str = None):
    """
    Get the appropriate tokenizer object from CLIP.
    
    Args:
        clip: ComfyUI CLIP object
        model_type: Optional override ('flux', 'sdxl', 'sd1', 'z_image')
        
    Returns:
        The underlying tokenizer object
    """
    if model_type is None:
        model_type = detect_model_type(clip)
    
    tokenizer = clip.tokenizer
    
    if model_type == 'z_image':
        # Z-Image uses Qwen3 tokenizer
        return tokenizer.qwen3_4b.tokenizer
    elif model_type == 'flux':
        # Flux uses T5 tokenizer for the main text encoding
        return tokenizer.t5xxl.tokenizer
    elif model_type == 'sdxl':
        # SDXL uses CLIP-L tokenizer
        if hasattr(tokenizer, 'clip_l'):
            return tokenizer.clip_l.tokenizer
        elif hasattr(tokenizer, 'clip'):
            return tokenizer.clip.tokenizer
    elif model_type == 'sd1':
        if hasattr(tokenizer, 'clip'):
            return tokenizer.clip.tokenizer
        elif hasattr(tokenizer, 'clip_l'):
            return tokenizer.clip_l.tokenizer
    
    # Direct access fallback
    if hasattr(tokenizer, 'tokenizer'):
        return tokenizer.tokenizer
    
    raise ValueError(f"Could not find tokenizer for model type: {model_type}")


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


def _flatten_chunked_token_ids(token_weight_pairs) -> List[int]:
    """
    Flatten ComfyUI chunked token-weight pairs into a single token id list.

    Supports:
    - SDXL: dict with keys "l"/"g", each a list of chunks
    - SD1/SD2: list of chunks directly
    Each chunk is a list of (token_id, weight[, word_id]) tuples.
    """
    if isinstance(token_weight_pairs, dict):
        if "l" in token_weight_pairs:
            chunks = token_weight_pairs["l"]
        else:
            # Fallback to first entry in dict
            chunks = next(iter(token_weight_pairs.values()))
    else:
        chunks = token_weight_pairs

    flat_ids: List[int] = []
    for chunk in chunks:
        for item in chunk:
            if isinstance(item, (tuple, list)):
                flat_ids.append(int(item[0]))
            else:
                flat_ids.append(int(item))
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
        
        # Apply Z-Image's llama template (from comfy/text_encoders/z_image.py)
        # Template: "<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        llama_template = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        wrapped_text = llama_template.format(tokenize_prompt)
        
        # Tokenize using the ComfyUI clip's tokenize method
        # This gives us the actual token sequence used during generation
        try:
            token_weight_pairs = clip.tokenize(tokenize_prompt)
            # For Z-Image, the tokens are directly in qwen3_4b key
            if 'qwen3_4b' in token_weight_pairs:
                chunks = token_weight_pairs['qwen3_4b']
            else:
                chunks = token_weight_pairs
            
            # Flatten chunks to get token IDs
            token_ids = []
            for chunk in chunks:
                for item in chunk:
                    if isinstance(item, (tuple, list)):
                        token_ids.append(int(item[0]))
                    else:
                        token_ids.append(int(item))
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
        model_type: Optional override ('flux', 'sdxl', 'sd1')
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
        model_type = detect_model_type(clip)
    
    tokenizer = get_tokenizer_for_model(clip, model_type)
    
    if model_type == 'z_image':
        return find_concept_positions_qwen3(
            clip, tokenizer, prompts, concepts,
            filter_meaningless=filter_meaningless,
            filter_single_char=filter_single_char,
            system_prompt=system_prompt,
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
        raise ValueError(f"Unsupported model type: {model_type}. Use 'flux', 'sdxl', 'z_image', or 'sd1'.")


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


def find_background_positions(
    clip,
    prompt: str,
    background_text: str = None,
    model_type: str = None,
    system_prompt: Optional[str] = None,
) -> List[int]:
    """
    Find token positions for background region.
    
    For Flux: Uses EOS position if no background_text provided
    For SDXL: Uses the provided background_text or returns empty
    
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
        model_type = detect_model_type(clip)
    
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
    
    # For Flux, can use EOS position as background anchor
    if model_type == 'flux':
        eos_pos = find_eos_position_t5(clip, prompt)
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
