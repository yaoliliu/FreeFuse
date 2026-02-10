# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import math
import warnings
from contextlib import contextmanager
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import svd_lowrank
from transformers.pytorch_utils import Conv1D

from peft.tuners._buffer_dict import BufferDict
from peft.tuners.tuners_utils import BaseTunerLayer, _get_in_out_features, check_adapters_to_merge
from peft.utils.integrations import (
    dequantize_module_weight,
    gather_params_ctx,
    get_bnb_param_type,
    skip_init_on_device,
)
from peft.utils.other import transpose
from peft.utils.warning import PeftWarning

from peft.tuners.lora.config import ArrowConfig, LoraConfig
from peft.tuners.lora.layer import LoraLayer, Linear


VARIANT_KWARG_KEYS = ["alora_offsets"]



# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class FreeFuseLinear(nn.Module, LoraLayer):
    @classmethod
    def init_from_lora_linear(cls, instance):
        """将 LoraLinear 的实例原地升级为 FreeFuseLinear 的实例"""
        if not isinstance(instance, LoraLayer) and isinstance(instance, Linear):
            raise TypeError("Can only upgrade from LoraLayer")
            
        # 1. 修改类型
        instance.__class__ = cls

        # 2. 初始化 FreeFuse 特有的数据
        instance.freefuse_masks = None
        instance.freefuse_token_pos_maps = None
        instance.mask_enabled = True
        
        # 返回升级后的对象（虽然是原地修改，但返回便于链式调用）
        return instance

    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        use_alora: bool = False,
        arrow_config: ArrowConfig = None,
        use_bdlora=None,
        lora_bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            use_alora=use_alora,
            lora_bias=lora_bias,
            arrow_config=arrow_config,
            use_bdlora=use_bdlora,
            **kwargs,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
        self.freefuse_mask = None
        self.freefuse_token_pos_maps = None
        self.mask_enabled = True

    def resolve_lora_variant(
        self, *, arrow_config: ArrowConfig, use_dora: bool, use_alora: bool, use_bdlora=None, **kwargs
    ) -> Optional[LoraVariant]:
        if arrow_config is not None:
            from .variants import ArrowLinearVariant

            return ArrowLinearVariant()

        if use_bdlora is not None:
            from .variants import BdLoraLinearVariant

            return BdLoraLinearVariant()

        if not use_dora and not use_alora:
            return None

        from .variants import ALoraLinearVariant, DoraLinearVariant

        if use_alora:
            return ALoraLinearVariant()
        else:
            return DoraLinearVariant()

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weight = base_layer.weight.data.clone()
                    orig_dtype = orig_weight.dtype
                    if active_adapter not in self.lora_variant:  # vanilla LoRA
                        delta_weight = self.get_delta_weight(active_adapter)
                        orig_weight += delta_weight.to(orig_dtype)
                    else:
                        orig_weight = self.lora_variant[active_adapter].merge_safe(self, active_adapter, orig_weight)

                    if not torch.isfinite(orig_weight).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weight

                    if self.lora_bias[active_adapter]:
                        if getattr(base_layer, "bias", None) is None:
                            raise RuntimeError(
                                "Impossible to merge LoRA with `lora_bias=True` because the base layer has no bias."
                            )
                        new_bias = base_layer.bias + self.lora_B[active_adapter].bias * self.scaling[active_adapter]
                        if not torch.isfinite(new_bias).all():
                            raise ValueError(
                                f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                            )
                        base_layer.bias.data = new_bias.to(orig_dtype)

                else:
                    if active_adapter not in self.lora_variant:  # vanilla LoRA
                        delta_weight = self.get_delta_weight(active_adapter)
                        base_layer.weight.data += delta_weight
                    else:
                        self.lora_variant[active_adapter].merge_unsafe(self, active_adapter, base_layer.weight)

                    if self.lora_bias[active_adapter]:
                        if getattr(base_layer, "bias", None) is None:
                            raise RuntimeError(
                                "Impossible to merge LoRA with `lora_bias=True` because the base layer has no bias."
                            )
                        base_layer.bias.data += self.lora_B[active_adapter].bias * self.scaling[active_adapter]

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """

        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                weight = self.get_base_layer().weight
                if active_adapter not in self.lora_variant:  # vanilla LoRA
                    orig_dtype = weight.dtype
                    delta_weight = self.get_delta_weight(active_adapter)
                    weight.data -= delta_weight.to(orig_dtype)
                else:
                    unmerged = self.lora_variant[active_adapter].unmerge(self, active_adapter, weight)
                    weight.data = unmerged

                if self.lora_bias[active_adapter]:
                    self.get_base_layer().bias.data -= self.lora_B[active_adapter].bias * self.scaling[active_adapter]

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)
        variant_kwargs = {k: kwargs.pop(k, None) for k in VARIANT_KWARG_KEYS}  # don't pass these to base_layer

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **variant_kwargs, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            lora_A_keys = self.lora_A.keys()
            for active_adapter in self.active_adapters:
                if active_adapter not in lora_A_keys:
                    continue

                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = self._cast_input_dtype(x, lora_A.weight.dtype)
                if active_adapter not in self.lora_variant:  # vanilla LoRA
                    lora_result = lora_B(lora_A(dropout(x))) * scaling
                    if self.freefuse_masks is not None and self.mask_enabled and active_adapter in self.freefuse_masks.keys():
                        lora_result = lora_result * self.freefuse_masks[active_adapter]
                    if self.freefuse_token_pos_maps is not None and self.mask_enabled:
                        for cur_active_adapter in self.active_adapters:
                            if active_adapter != cur_active_adapter and cur_active_adapter in self.freefuse_token_pos_maps.keys():
                                poses = self.freefuse_token_pos_maps[cur_active_adapter]
                                for i, pos in enumerate(poses):
                                    lora_result[i, pos, :] = 0
                    result = result + lora_result.to(torch_result_dtype)
                else:
                    print("Warning: 我们还没有实现针对lora variant的freefuse，然而这里却发生了调用")
                    result = self.lora_variant[active_adapter].forward(
                        self,
                        active_adapter=active_adapter,
                        x=x,
                        result=result,
                        **variant_kwargs,
                        **kwargs,
                    )

            result = result.to(torch_result_dtype)

        return result

    def set_freefuse_token_pos_maps(self, masks: Dict[str, torch.Tensor]):
        self.freefuse_token_pos_maps = masks
        
    def set_freefuse_masks(self, masks: Dict[str, torch.Tensor]):
        self.freefuse_masks = masks

    def enable_masks(self):
        self.mask_enabled = True

    def disable_masks(self):
        self.mask_enabled = False

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


def convert_peft_lora_to_freefuse_lora(model: nn.Module, lora_names: list = None) -> None:
    """
    Convert PEFT LoRA layers to FreeFuseLinear layers in-place.
    
    This function traverses the model and upgrades any LoraLayer instances
    to FreeFuseLinear, enabling spatial masking for FreeFuse.
    
    Args:
        model: The model containing LoRA layers
        lora_names: Optional list of LoRA adapter names to convert.
                   If None, converts all LoRA layers.
    """
    from peft.tuners.lora.layer import LoraLayer, Linear as LoraLinear
    
    modules_to_convert = []
    
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer) and not isinstance(module, FreeFuseLinear):
            modules_to_convert.append((name, module))
    
    for name, module in modules_to_convert:
        FreeFuseLinear.init_from_lora_linear(module)

