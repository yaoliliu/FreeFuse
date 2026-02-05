"""
FreeFuse Token Position Utilities

Unified token position finding for both Flux (T5) and SDXL (CLIP) models in ComfyUI.

T5 tokenizer (Flux): Uses offset_mapping (character position mapping)
CLIP tokenizer (SDXL): Uses sliding window token matching
"""

from typing import Dict, List, Union, Optional, Any


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
    # Remove T5's ▁ prefix, CLIP's </w> suffix, and other markers
    cleaned = token_text.replace('▁', '').replace('_', '')
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
    Detect whether the CLIP model is for Flux (T5) or SDXL.
    
    Args:
        clip: ComfyUI CLIP object
        
    Returns:
        'flux' or 'sdxl' or 'unknown'
    """
    tokenizer = clip.tokenizer
    
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
        model_type: Optional override ('flux', 'sdxl', 'sd1')
        
    Returns:
        The underlying tokenizer object
    """
    if model_type is None:
        model_type = detect_model_type(clip)
    
    tokenizer = clip.tokenizer
    
    if model_type == 'flux':
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


def find_concept_positions_clip(
    tokenizer,
    prompts: Union[str, List[str]],
    concepts: Dict[str, str],
    filter_meaningless: bool = True,
    filter_single_char: bool = True,
) -> Dict[str, List[List[int]]]:
    """
    Find token positions for CLIP-based models (SDXL, SD1.x) using sliding window matching.
    
    This method matches token ID sequences since CLIP tokenizers may not support
    offset_mapping consistently.
    
    Args:
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
    
    concept_pos_map = {}
    for concept_name, concept_text in concepts.items():
        concept_pos_map[concept_name] = []
        
        # Tokenize concept text (without special tokens)
        concept_tokens = tokenizer.encode(concept_text, add_special_tokens=False)
        
        for prompt in prompts:
            # Tokenize prompt (with special tokens)
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
            prompt_token_texts = [tokenizer.decode([tid]) for tid in prompt_tokens]
            
            positions = []
            positions_with_text = []
            
            # Sliding window match
            concept_len = len(concept_tokens)
            prompt_len = len(prompt_tokens)
            
            if concept_len > 0:
                for i in range(prompt_len - concept_len + 1):
                    match = all(
                        prompt_tokens[i + j] == concept_tokens[j]
                        for j in range(concept_len)
                    )
                    if match:
                        for j in range(concept_len):
                            pos = i + j
                            if pos not in positions:
                                positions.append(pos)
                                positions_with_text.append((pos, prompt_token_texts[pos]))
            
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


def find_concept_positions(
    clip,
    prompts: Union[str, List[str]],
    concepts: Dict[str, str],
    filter_meaningless: bool = True,
    filter_single_char: bool = True,
    model_type: str = None,
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
    
    if model_type == 'flux':
        return find_concept_positions_t5(
            tokenizer, prompts, concepts,
            filter_meaningless=filter_meaningless,
            filter_single_char=filter_single_char
        )
    elif model_type in ('sdxl', 'sd1'):
        return find_concept_positions_clip(
            tokenizer, prompts, concepts,
            filter_meaningless=filter_meaningless,
            filter_single_char=filter_single_char
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Use 'flux', 'sdxl', or 'sd1'.")


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
            model_type=model_type
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
        
    Returns:
        freefuse_token_pos_maps: Dict ready for FreeFuse sampling
    """
    # Compute concept positions
    token_pos_maps = find_concept_positions(clip, prompt, concepts)
    
    # Add background if enabled
    if enable_background:
        bg_positions = find_background_positions(clip, prompt, background_text)
        if bg_positions:
            token_pos_maps['__background__'] = [bg_positions]
    
    return token_pos_maps
