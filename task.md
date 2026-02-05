# FreeFuse ComfyUI Integration - Fix Bypass Mode for Flux Fused Weights

## 问题概述

FreeFuse 的 ComfyUI 集成中，LoRA 无法在 Flux 模型上生效。生成的图像显示普通人物而非 LoRA 训练的角色（如 Harry Potter）。

## 根本原因

**ComfyUI 的 `load_bypass_lora_for_models()` 不支持 Flux 模型的 fused QKV weights。**

### 技术细节

1. **Flux 模型使用 fused QKV weights**：
   - ComfyUI 模型 key: `diffusion_model.double_blocks.0.img_attn.qkv.weight`
   - 这个权重包含了 Q、K、V 三个投影矩阵，拼接在一起

2. **diffusers 格式的 LoRA 使用分离的 Q/K/V**：
   - LoRA key: `transformer.transformer_blocks.0.attn.to_q.lora_A.weight`
   - 分别有 `to_q`, `to_k`, `to_v` 的 LoRA 权重

3. **ComfyUI 的 key mapping 使用 tuple 格式来处理这种映射**：
   ```python
   # 在 comfy/utils.py 的 flux_to_diffusers() 函数中
   key_map["{}to_q.{}".format(k, end)] = (qkv, (0, 0, hidden_size))
   key_map["{}to_k.{}".format(k, end)] = (qkv, (0, hidden_size, hidden_size))
   key_map["{}to_v.{}".format(k, end)] = (qkv, (0, hidden_size * 2, hidden_size))
   ```
   
   tuple 格式: `(target_key, (dim, offset, size))`
   - `target_key`: 目标权重的 key
   - `dim`: 切片的维度
   - `offset`: 起始偏移
   - `size`: 切片大小

4. **`load_lora()` 返回 tuple key**：
   ```python
   # load_lora 返回的 patch_dict 的 key 是:
   ('diffusion_model.double_blocks.0.img_attn.qkv.weight', (0, 0, 3072))
   ('diffusion_model.double_blocks.0.img_attn.qkv.weight', (0, 3072, 3072))
   ('diffusion_model.double_blocks.0.img_attn.qkv.weight', (0, 6144, 3072))
   ```

5. **Bug 位置 - `load_bypass_lora_for_models()` 在 comfy/sd.py 第 106-195 行**：
   ```python
   for key, adapter in bypass_patches.items():
       if key in model_sd_keys:  # BUG: key 是 tuple，model_sd_keys 是 string set
           manager.add_adapter(key, adapter, strength=strength_model)
   ```
   
   这里直接用 tuple 去检查是否在 `model_sd_keys`（字符串集合）中，永远不会匹配。

6. **正常的 `add_patches()` 如何处理 tuple key（comfy/model_patcher.py 第 581-670 行）**：
   ```python
   def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
       for k in patches:
           offset = None
           if isinstance(k, str):
               key = k
           else:
               offset = k[1]  # 提取 offset 信息
               key = k[0]     # 提取真正的 key 字符串
           
           if key in model_sd:  # 用字符串 key 检查
               # ... 处理 patch
               if offset is not None:
                   weight = weight.narrow(offset[0], offset[1], offset[2])
   ```

## 需要修复的代码

### 1. 修改 `comfy/sd.py` 的 `load_bypass_lora_for_models()` 函数

**问题代码（第 152-162 行）**:
```python
manager = comfy.weight_adapter.BypassInjectionManager()
model_sd_keys = set(new_modelpatcher.model.state_dict().keys())

for key, adapter in bypass_patches.items():
    if key in model_sd_keys:
        manager.add_adapter(key, adapter, strength=strength_model)
        k.add(key)
    else:
        logging.warning(f"[BypassLoRA] Adapter key not in model state_dict: {key}")
```

**需要改成**:
```python
manager = comfy.weight_adapter.BypassInjectionManager()
model_sd_keys = set(new_modelpatcher.model.state_dict().keys())

for key, adapter in bypass_patches.items():
    # 处理 tuple key (用于 fused weights 如 Flux 的 QKV)
    offset = None
    if isinstance(key, str):
        actual_key = key
    else:
        actual_key = key[0]
        offset = key[1]  # (dim, start, size)
    
    if actual_key in model_sd_keys:
        manager.add_adapter(actual_key, adapter, strength=strength_model, offset=offset)
        k.add(key)
    else:
        logging.warning(f"[BypassLoRA] Adapter key not in model state_dict: {actual_key}")
```

### 2. 修改 `comfy/weight_adapter/bypass.py` 的 `BypassInjectionManager.add_adapter()` 方法

需要让 `add_adapter()` 接受 `offset` 参数，并在创建 hook 时传递给 `BypassForwardHook`。

### 3. 修改 `comfy/weight_adapter/bypass.py` 的 `BypassForwardHook` 类

需要在 forward hook 中处理 offset：
- 当有 offset 时，LoRA 的输出应该只应用到 fused weight 输出的特定部分
- 例如，对于 `to_q` 的 LoRA，只修改输出的前 3072 维

## 关键文件位置

| 文件 | 描述 |
|------|------|
| `ComfyUI/comfy/sd.py` | `load_bypass_lora_for_models()` 函数，第 106-195 行 |
| `ComfyUI/comfy/model_patcher.py` | `add_patches()` 参考实现，第 581-670 行 |
| `ComfyUI/comfy/weight_adapter/bypass.py` | `BypassInjectionManager` 和 `BypassForwardHook` |
| `ComfyUI/comfy/utils.py` | `flux_to_diffusers()` key mapping，第 595-700 行 |
| `ComfyUI/comfy/lora.py` | `load_lora()` 函数，第 37-95 行 |

## 验证方法

1. **标准 LoRA 模式已验证可以工作**：使用 `load_lora_for_models()` + `add_patches()` 能正确应用 Flux LoRA

2. **测试文件**: `/root/FreeFuse/freefuse_comfyui/tests/test_flux_e2e.py`

3. **测试命令**:
   ```bash
   cd /root/FreeFuse/ComfyUI
   .venv/bin/python -m pytest ../freefuse_comfyui/tests/test_flux_e2e.py -v
   ```

4. **验证 bypass 模式是否工作的方法**:
   - 检查是否还有 `[BypassLoRA] Adapter key not in model state_dict` 警告
   - 生成的图像应该显示 Harry Potter 和 Daiyu 角色

## 测试用的 LoRA

- Harry Potter: `/root/FreeFuse/loras/harry_potter_flux.safetensors`
- Daiyu: `/root/FreeFuse/loras/daiyu_lin_flux.safetensors`

## 补充信息

### Flux 模型的 hidden_size

从 key mapping 可以看出 hidden_size = 3072:
- to_q: offset (0, 0, 3072)
- to_k: offset (0, 3072, 3072)  
- to_v: offset (0, 6144, 3072)

### BypassForwardHook 的当前实现

hook 通过以下方式工作：
```python
# 原始: output = layer(input)
# Bypass: output = layer(input) + lora_output
```

对于 fused weights，需要改成：
```python
# 原始: output = qkv_layer(input)  # output shape: [..., 9216]
# 对于 to_q 的 LoRA (offset = (0, 0, 3072)):
#   output[:, :, 0:3072] += lora_q_output
# 对于 to_k 的 LoRA (offset = (0, 3072, 3072)):
#   output[:, :, 3072:6144] += lora_k_output
# 对于 to_v 的 LoRA (offset = (0, 6144, 3072)):
#   output[:, :, 6144:9216] += lora_v_output
```

### 注意事项

1. 一个 fused weight 可能有多个 LoRA adapter（分别对应 Q、K、V）
2. 需要确保同一个 key 的多个 adapter 都能正确注册和执行
3. `BypassInjectionManager` 可能需要用 `defaultdict(list)` 来存储同一个 key 的多个 adapter
