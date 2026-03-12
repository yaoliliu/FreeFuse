"""
Helpers for logging tensor statistics safely.

ComfyUI extensions such as ComfyUI-GGUF can propagate ``torch.Tensor``
subclasses (for example ``GGMLTensor``). Formatting those tensors directly
with specs like ``:.6f`` raises ``TypeError`` even for scalar results.
"""

from typing import Any

import torch


def tensor_scalar_to_float(value: Any) -> float:
    """Convert a scalar-like tensor or numeric value to a Python float."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected a scalar tensor, got shape={tuple(value.shape)}")
        return value.item()
    return float(value)


def format_tensor_stats(
    tensor: torch.Tensor,
    *,
    include_shape: bool = False,
    include_mean: bool = False,
) -> str:
    """Format common tensor stats without relying on tensor ``__format__``."""
    parts = []
    if include_shape:
        parts.append(f"shape={tuple(tensor.shape)}")
    parts.append(f"min={tensor_scalar_to_float(tensor.min()):.6f}")
    parts.append(f"max={tensor_scalar_to_float(tensor.max()):.6f}")
    if include_mean:
        parts.append(f"mean={tensor_scalar_to_float(tensor.mean()):.6f}")
    return ", ".join(parts)
