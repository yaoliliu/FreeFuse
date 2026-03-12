#!/usr/bin/env python
"""
Regression tests for safe tensor-stat formatting.
"""

import importlib.util
import os
import unittest

import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FREEFUSE_COMFYUI_DIR = os.path.dirname(SCRIPT_DIR)
MODULE_PATH = os.path.join(FREEFUSE_COMFYUI_DIR, "freefuse_core", "tensor_debug.py")

SPEC = importlib.util.spec_from_file_location("tensor_debug", MODULE_PATH)
tensor_debug = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tensor_debug)

format_tensor_stats = tensor_debug.format_tensor_stats
tensor_scalar_to_float = tensor_debug.tensor_scalar_to_float


class FakeTensor(torch.Tensor):
    """Minimal tensor subclass that reproduces scalar formatting failures."""

    @staticmethod
    def __new__(cls, value):
        return torch.Tensor._make_subclass(cls, value, value.requires_grad)


class TestTensorDebug(unittest.TestCase):
    def test_scalar_subclass_is_convertible(self):
        value = FakeTensor(torch.tensor(1.25, dtype=torch.float32))

        self.assertEqual(tensor_scalar_to_float(value), 1.25)

    def test_stats_format_accepts_tensor_subclass(self):
        tensor = FakeTensor(torch.arange(6, dtype=torch.float32).reshape(2, 3))

        stats = format_tensor_stats(tensor, include_shape=True, include_mean=True)

        self.assertEqual(
            stats,
            "shape=(2, 3), min=0.000000, max=5.000000, mean=2.500000",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
