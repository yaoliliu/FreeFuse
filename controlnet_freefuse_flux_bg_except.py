"""
Main script using FreeFuseFluxAttnProcessor with ControlNet support.

This script extracts concept embeddings directly from encoder_hidden_states using
position maps, avoiding the token symmetry problem caused by separate encoding.
It also supports ControlNet for structural conditioning.
"""
import os
import torch
from diffusers.models.transformers.transformer_flux import FluxTransformerBlock, FluxAttention
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel as orgFluxTransformer2DModel
from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel
from diffusers.utils import load_image
from src.pipeline.freefuse_controlnet_flux_direct_extract_attn_bais_pipeline import FreeFuseFluxDirectExtractAttnBaisPipeline
from src.attn_processor.freefuse_attn_processor import FreeFuseFluxAttnProcessor
from src.models.freefuse_transformer_flux import FluxTransformer2DModel, FreeFuseFluxTransformerBlock, FreeFuseFluxAttention
from src.tuner.freefuse_lora_layer import FreeFuseLinear
from peft.tuners.lora.layer import LoraLayer, Linear
from diffusers import PipelineQuantizationConfig, TorchAoConfig
from torchao.quantization import autoquant
from diffusers import GGUFQuantizationConfig
from transformers import T5EncoderModel, BitsAndBytesConfig


def prepare_prompt_embeds(pipe, prompt, prompt_2=None, negative_prompt=None, num_images_per_prompt=1, max_sequence_length=512):
    with torch.no_grad():
        prompt_2 = prompt_2 or prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

        # We only use the pooled prompt output from the CLIPTextModel
        pooled_prompt_embeds = pipe._get_clip_prompt_embeds(
            prompt=prompt,
            device=pipe.text_encoder_2.device,
            num_images_per_prompt=num_images_per_prompt,
        )
        prompt_embeds = pipe._get_t5_prompt_embeds(
            prompt=prompt_2,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            device=pipe.text_encoder_2.device,
        )

        pooled_negative_prompt_embeds = None
        negative_prompt_embeds = None
        if negative_prompt is not None:
            negative_prompt_2 = negative_prompt or negative_prompt
            negative_prompt_2 = [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            pooled_negative_prompt_embeds = pipe._get_clip_prompt_embeds(
                prompt=negative_prompt,
                device=pipe.text_encoder_2.device,
                num_images_per_prompt=num_images_per_prompt,
            )
            negative_prompt_embeds = pipe._get_t5_prompt_embeds(
                prompt=negative_prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=pipe.text_encoder_2.device,
            )
    
    return pooled_prompt_embeds, prompt_embeds, pooled_negative_prompt_embeds, negative_prompt_embeds


def find_concept_positions(pipe, prompts, concepts, filter_meaningless=True, filter_single_char=True):
    """
    找到每个concept的token ids在prompt的token ids中的位置。
    使用字符位置映射的方式来处理tokenizer上下文差异问题。
    
    Args:
        pipe: 包含tokenizer_2的pipeline对象
        prompts: 单个prompt字符串或prompt字符串列表
        concepts: 字典，键为concept名称（如adapter name），值为concept描述文本
                  例如: {'lora_a': 'cat', 'lora_b': 'dog'}
        filter_meaningless: 是否过滤无意义的停用词/标点token（默认True）
        filter_single_char: 是否过滤只有一个字符的token（默认True）
    
    Returns:
        concept_pos_map: 字典，结构如下:
            {
                'lora_a': [[pos_1], [pos_2], ...],  # 每个子列表对应一个prompt
                'lora_b': [[pos_1], [pos_2], ...],
                ...
            }
            其中 pos_x 是一个位置列表，表示该concept在对应prompt中出现的token位置
    """
    # 定义停用词和标点符号集合（小写）
    # 这些token在InfoNCE loss中会干扰梯度优化方向
    # 使用.lower()比较，因此首字母大写形式（如'A', 'The'）也会被过滤
    STOPWORDS = {
        # 冠词
        'a', 'an', 'the',
        # 连词
        'and', 'or', 'but', 'nor', 'so', 'yet',
        # 介词
        'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under', 'over',
        # 代词
        'it', 'its', 'this', 'that', 'these', 'those', 'their', 'his', 'her', 'my', 'your', 'our',
        # 助词/be动词
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'has', 'have', 'had', 'having',
        # 其他常见停用词
        'which', 'who', 'whom', 'whose', 'where', 'when', 'while',
    }
    
    # 标点符号
    PUNCTUATION = {',', '.', '!', '?', ';', ':', '"', "'", '(', ')', '[', ']', '{', '}', '-', '–', '—', '/', '\\', '...', '..'}
    
    # 合并停用词和标点
    MEANINGLESS_TOKENS = STOPWORDS | PUNCTUATION
    
    def is_meaningless_token(token_text, check_single_char=True):
        """判断token是否为无意义token"""
        # 去除T5 tokenizer的前缀标记（如▁）
        cleaned = token_text.replace('▁', '').replace('_', '').strip().lower()
        # 空字符串也视为无意义
        if not cleaned:
            return True
        # 单字符过滤（可选）
        if check_single_char and len(cleaned) == 1:
            return True
        return cleaned in MEANINGLESS_TOKENS
    
    # 确保prompts是列表形式
    if isinstance(prompts, str):
        prompts = [prompts]
    
    # 对每个prompt进行tokenize，同时获取offset_mapping（字符位置到token位置的映射）
    prompt_data_list = []
    for prompt in prompts:
        prompt_inputs = pipe.tokenizer_2(
            prompt,
            padding=False,
            truncation=False,
            return_length=False,
            return_overflowing_tokens=False,
            return_offsets_mapping=True,  # 获取字符偏移映射
            return_tensors="pt",
        )
        # 获取ids列表和offset_mapping（去除batch维度）
        prompt_ids = prompt_inputs.input_ids[0].tolist()
        offset_mapping = prompt_inputs.offset_mapping[0].tolist()  # [(start, end), ...]
        prompt_data_list.append({
            'text': prompt,
            'ids': prompt_ids,
            'offset_mapping': offset_mapping
        })
    
    # 找到每个concept在每个prompt中的位置
    concept_pos_map = {}
    for concept_name, concept_text in concepts.items():
        concept_pos_map[concept_name] = []
        
        for prompt_data in prompt_data_list:
            positions = []
            positions_with_text = []  # 记录位置和对应的token文本，用于过滤和fallback
            prompt_text = prompt_data['text']
            prompt_ids = prompt_data['ids']
            offset_mapping = prompt_data['offset_mapping']
            
            # 在原始文本中查找concept文本的所有出现位置
            search_start = 0
            while True:
                # 查找concept在prompt中的字符位置
                char_start = prompt_text.find(concept_text, search_start)
                if char_start == -1:
                    break
                char_end = char_start + len(concept_text)
                
                # 根据字符位置找到对应的token位置
                for token_idx, (token_start, token_end) in enumerate(offset_mapping):
                    # 如果token的字符范围与concept的字符范围有重叠，则记录该token位置
                    if token_end > char_start and token_start < char_end:
                        if token_idx not in positions:
                            positions.append(token_idx)
                            # 解码token获取文本
                            token_text = pipe.tokenizer_2.decode([prompt_ids[token_idx]], skip_special_tokens=False)
                            positions_with_text.append((token_idx, token_text))
                
                # 继续搜索下一个出现位置
                search_start = char_start + 1
            
            # 过滤无意义token
            if filter_meaningless and positions_with_text:
                filtered_positions = [
                    pos for pos, text in positions_with_text 
                    if not is_meaningless_token(text, check_single_char=filter_single_char)
                ]
                
                # Fallback机制：如果全部被过滤掉了，保留第一个有意义的token
                # 如果全是停用词，退而求其次保留第一个非标点的token
                # 如果全是标点，保留第一个token
                if not filtered_positions:
                    # 尝试找非标点的token
                    non_punct_positions = [
                        pos for pos, text in positions_with_text
                        if text.replace('▁', '').replace('_', '').strip() not in PUNCTUATION
                    ]
                    if non_punct_positions:
                        filtered_positions = [non_punct_positions[0]]
                    elif positions_with_text:
                        # 实在没有，保留第一个
                        filtered_positions = [positions_with_text[0][0]]
                
                positions = filtered_positions
            
            # 排序位置列表
            positions.sort()
            concept_pos_map[concept_name].append(positions)
    
    return concept_pos_map


def get_prompt_token_texts(pipe, prompt, max_sequence_length=512):
    """
    Get the token texts for the full prompt.
    This is used for visualization to show which tokens correspond to which concepts.
    
    Args:
        pipe: Pipeline with tokenizer_2
        prompt: The prompt string
        max_sequence_length: Max sequence length for tokenization
        
    Returns:
        token_texts: List of decoded token strings for the entire prompt
    """
    prompt_inputs = pipe.tokenizer_2(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )
    prompt_ids = prompt_inputs.input_ids[0].tolist()
    
    # Decode each token to get its text representation
    token_texts = [pipe.tokenizer_2.decode([token_id], skip_special_tokens=False) for token_id in prompt_ids]
    
    return token_texts


def find_first_eos_index(pipe, prompt, max_sequence_length=512):
    """
    Find the index of the first EOS token in the tokenized prompt.
    This can be used for background masking in FreeFuse.
    
    Args:
        pipe: Pipeline with tokenizer_2
        prompt: The prompt string
        max_sequence_length: Max sequence length for tokenization
        
    Returns:
        eos_index: Index of the first EOS token, or None if not found
    """
    prompt_inputs = pipe.tokenizer_2(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )
    prompt_ids = prompt_inputs.input_ids[0].tolist()
    
    # Find the EOS token ID for T5 tokenizer (usually 1)
    eos_token_id = pipe.tokenizer_2.eos_token_id
    
    # Find the first occurrence of EOS token
    for idx, token_id in enumerate(prompt_ids):
        if token_id == eos_token_id:
            return idx
    
    return None


def main():
    # ============================
    # ControlNet Configuration
    # ============================
    use_controlnet = True  # Set to True to enable ControlNet
    controlnet_model_path = "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0"  # or your local path
    control_image_path = "optional_condition/pose.png"  # Set this to your canny edge image path, e.g., "path/to/canny_image.png"
    # control_image_path = "optional_condition/depth.png"  # Set this to your canny edge image path, e.g., "path/to/canny_image.png"
    controlnet_conditioning_scale = 0.7  # Adjust this value (0.0-1.0) for control strength
    
    # Load ControlNet model if enabled
    controlnet = None
    control_image = None
    if use_controlnet and control_image_path is not None:
        print(f"Loading ControlNet from {controlnet_model_path}...")
        controlnet = FluxControlNetModel.from_pretrained(
            controlnet_model_path,
            torch_dtype=torch.bfloat16
        )
        print(f"Loading control image from {control_image_path}...")
        control_image = load_image(control_image_path)
    elif use_controlnet and control_image_path is None:
        print("Warning: use_controlnet=True but control_image_path is None. ControlNet will not be applied.")
        print("Set control_image_path to a canny edge image to use ControlNet.")

    # Load pipeline
    print("Loading FreeFuse pipeline...")
    quantization = False
    if quantization:
        ckpt_path = (
            "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q5_0.gguf"
        )
        
        # 使用 bitsandbytes 8-bit 量化加载 T5 
        # 8-bit quantization: ~10GB VRAM savings vs full precision T5-XXL
        # bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # 如果需要更进一步节省显存，可以使用 4-bit 量化 (会有轻微质量损失):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",  # "nf4" or "fp4"
        )
        
        text_encoder_2 = T5EncoderModel.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            subfolder="text_encoder_2",
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        transformer = orgFluxTransformer2DModel.from_single_file(
            ckpt_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
        )
        pipe = FreeFuseFluxDirectExtractAttnBaisPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            transformer=transformer,
            controlnet=controlnet,
            text_encoder_2=text_encoder_2,
            torch_dtype=torch.bfloat16,
        )

        # Change transformer class to our custom class
        pipe.transformer.__class__ = FluxTransformer2DModel
        
        pipe.to('cuda')
        # pipe.enable_model_cpu_offload()
    else:
        pipe = FreeFuseFluxDirectExtractAttnBaisPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", 
            controlnet=controlnet,  # Pass controlnet here
            torch_dtype=torch.bfloat16
        )
        # Change transformer class to our custom class
        pipe.transformer.__class__ = FluxTransformer2DModel

        pipe.to('cuda')

    # Load LoRA weights

    pipe.load_lora_weights("loras/daiyu_lin_flux.safetensors", adapter_name="daiyu")
    pipe.load_lora_weights("loras/harry_potter_flux.safetensors", adapter_name="harry")
    pipe.set_adapters(["daiyu", "harry"], [0.9, 0.9])


    current_processors = pipe.transformer.attn_processors

    # Set up attention processors
    processor_dict = {}
    for name in current_processors.keys():
        if 'single_transformer_blocks' in name:
            processor_dict[name] = current_processors[name]
        elif "transformer_blocks.18.attn" in name and "single" not in name:
            # Use DirectExtract processor for the block that computes concept sim maps
            processor_dict[name] = FreeFuseFluxAttnProcessor()
            processor_dict[name].cal_concept_sim_map = True
        else:
            processor_dict[name] = FreeFuseFluxAttnProcessor()

    # Apply processor dict
    pipe.transformer.set_attn_processor(processor_dict)

    # Replace Linear lora layers with FreeFuseLinearLayer
    for parent in pipe.transformer.modules():
        for name, module in parent.named_modules():
            if isinstance(module, LoraLayer) and isinstance(module, Linear):
                FreeFuseLinear.init_from_lora_linear(module)
        
    # Replace transformer blocks with FreeFuseFluxTransformerBlock
    for parent in pipe.transformer.modules():
        for name, module in parent.named_modules():
            if isinstance(module, FluxTransformerBlock):
                module.__class__ = FreeFuseFluxTransformerBlock
            if isinstance(module, FluxAttention):
                module.__class__ = FreeFuseFluxAttention

    # Define concept map - this maps each LoRA adapter to its concept text
    # The concept text should appear in the prompt
    concept_map = {
        "daiyu": "daiyu_lin, a young East Asian photorealistic style woman in traditional Chinese hanfu dress, elaborate black updo hairstyle adorned with delicate white floral hairpins and ornaments, dangling red tassel earrings, soft pink and red color palette, gentle smile with knowing expression",
        "harry": "harry potter, an European photorealistic style teenage wizard boy with messy black hair, round wire-frame glasses, and bright green eyes, wearing a white shirt, burgundy and gold striped tie, and dark robes"
    }


    # Build prompt containing all concepts
    clip_prompt = "A picture of two distinct characters"
    t5_prompt = "A picture of two characters with snow falling in the background: daiyu_lin, a young East Asian photorealistic style woman in traditional Chinese hanfu dress, elaborate black updo hairstyle adorned with delicate white floral hairpins and ornaments, dangling red tassel earrings, soft pink and red color palette, gentle smile with knowing expression is performing a traditional Chinese salute and harry potter, an European photorealistic style teenage wizard boy with messy black hair, round wire-frame glasses, and bright green eyes, wearing a white shirt, burgundy and gold striped tie, and dark robes is making a V sign."
    negtive_prompt = ""

    pooled_prompt_embeds, prompt_embeds, pooled_negative_prompt_embeds, negative_prompt_embeds = prepare_prompt_embeds(pipe, clip_prompt,t5_prompt, negative_prompt=negtive_prompt)

    # Find concept positions in the prompt
    freefuse_token_pos_maps = find_concept_positions(pipe, t5_prompt, concept_map)
    pipe.transformer.set_freefuse_token_pos_maps(freefuse_token_pos_maps)
    
    # Get token texts for visualization
    prompt_token_texts = get_prompt_token_texts(pipe, t5_prompt)
    
    # Create concept_token_texts dict for visualization
    # This maps each lora_name to the full prompt's token texts
    concept_token_texts = {lora_name: prompt_token_texts for lora_name in concept_map.keys()}
    
    # === Background mask extraction options (choose ONE) ===
    # Option 1: Use EOS token (automatic, no user input needed)
    # eos_token_index = find_first_eos_index(pipe, t5_prompt)
    # print(f"EOS token index: {eos_token_index}")
    # background_token_positions = None  # Not using this option
    
    # Option 2: Use user-defined background concept (uncomment to use)
    # Define what text in the prompt describes the background
    background_concept = "snow falling in the background"  # Adjust based on your prompt
    background_token_positions = find_concept_positions(pipe, t5_prompt, {"__bg__": background_concept})["__bg__"][0]
    print(f"Background token positions: {background_token_positions}")
    eos_token_index = None  # Don't use EOS when using background_token_positions
    
    seed = 42

    # Generate image
    # Note: We pass freefuse_token_pos_maps through joint_attention_kwargs
    # We don't need to pass concept_embeds_map anymore!
    print("Starting image generation...")
    if control_image is not None:
        print(f"Using ControlNet with conditioning scale {controlnet_conditioning_scale}")

    generator = torch.Generator("cuda").manual_seed(seed)
    image = pipe(
        prompt_embeds=prompt_embeds.to('cuda'), 
        pooled_prompt_embeds=pooled_prompt_embeds.to('cuda'), 
        negative_prompt_embeds=negative_prompt_embeds.to('cuda'),
        negative_pooled_prompt_embeds=pooled_negative_prompt_embeds.to('cuda'),
        # concept_embeds_map is no longer needed!
        concept_embeds_map=None,  # Explicitly set to None
        height=1024, 
        width=1024, 
        generator=generator, 
        # debug_save_path='debug_direct_extract',
        joint_attention_kwargs={
            'freefuse_token_pos_maps': freefuse_token_pos_maps,
            'concept_token_texts': concept_token_texts,
            # Background detection options (use ONE):
            'eos_token_index': eos_token_index,  # Option 1: EOS-based background
            'background_token_positions': background_token_positions,  # Option 2: user-defined background
        },
        guidance_scale=3,
        true_cfg_scale=0,
        # ControlNet parameters
        control_image=control_image,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
    ).images[0]
    
    output_filename = f"flux_controlnet.png" if control_image is not None else "flux.png"
    image.save(output_filename)
    print(f"Image saved to {output_filename}")


if __name__ == "__main__":
    main()
