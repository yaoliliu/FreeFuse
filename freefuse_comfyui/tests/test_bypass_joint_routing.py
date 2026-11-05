#!/usr/bin/env python3
"""Real bypass-hook regressions; run with COMFYUI_PATH=/path/to/ComfyUI (CPU only)."""

import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch


class JointRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        comfy = os.environ.get("COMFYUI_PATH")
        if not comfy or not (Path(comfy) / "comfy").is_dir():
            raise unittest.SkipTest("Set COMFYUI_PATH to a native ComfyUI checkout")
        sys.path.insert(0, comfy)
        previous_argv = sys.argv
        try:
            sys.argv = ["main.py", "--cpu"]
            import comfy.options
            comfy.options.enable_args_parsing()
            path = Path(__file__).resolve().parents[1] / "freefuse_core/bypass_lora_loader.py"
            spec = importlib.util.spec_from_file_location("_freefuse_bypass_joint_test", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cls.hook_class = module.MultiAdapterBypassForwardHook
        finally:
            sys.argv = previous_argv

    def hook(self, key):
        hook = self.hook_class(torch.nn.Linear(2, 2), module_key=key)
        hook.set_masks({"left": torch.tensor([[0., 1., 0., 1.]])},
                       latent_size=(1, 4), txt_len=3)
        return hook

    def test_krea2_attention_and_mlp_preserve_text_then_image_alignment(self):
        expected = torch.tensor([1., 1., 1., 0., 1., 0., 1.])
        for prefix in ("blocks.3.", "diffusion_model.blocks.3."):
            for layer in ("attn.wq", "attn.wk", "attn.wv", "attn.wo", "attn.gate",
                          "mlp.gate", "mlp.up", "mlp.down"):
                for suffix in ("", ".weight"):
                    with self.subTest(key=prefix + layer + suffix):
                        hook = self.hook(prefix + layer + suffix)
                        mask = hook._get_mask_for_sequence("left", 7, torch.device("cpu"), torch.float32)
                        torch.testing.assert_close(mask, expected, rtol=0, atol=0)
                        self.assertEqual(hook._get_mask_type(), "img_with_text")

    def test_anima_mlp_keeps_image_only_alignment(self):
        expected = torch.tensor([0., 1., 0., 1.])
        for prefix in ("blocks.3.", "diffusion_model.blocks.3."):
            for layer in ("mlp.layer1", "mlp.layer2", "self_attn.q_proj", "cross_attn.q_proj"):
                for suffix in ("", ".weight"):
                    with self.subTest(key=prefix + layer + suffix):
                        hook = self.hook(prefix + layer + suffix)
                        mask = hook._get_mask_for_sequence("left", 4, torch.device("cpu"), torch.float32)
                        torch.testing.assert_close(mask, expected, rtol=0, atol=0)
                        self.assertEqual(hook._get_mask_type(), "img_only")

    def test_krea2_mlp_routes_two_low_rank_deltas_to_their_image_regions(self):
        # Two independent rank-one LoRAs write distinct output channels.
        down = torch.tensor([[1., 0.]])
        expected = torch.tensor([[[1., 2.]] * 3 + [[1., 0.]] * 2 + [[0., 2.]] * 2])
        for layer in ("gate", "up", "down"):
            with self.subTest(layer=layer):
                module = torch.nn.Linear(2, 2, bias=False)
                module.weight.data.zero_()
                hook = self.hook_class(module, f"diffusion_model.blocks.3.mlp.{layer}")
                hook.original_forward = module.forward
                masks = {"left": torch.tensor([[1., 1., 0., 0.]]),
                         "right": torch.tensor([[0., 0., 1., 1.]])}
                hook.set_masks(masks, latent_size=(1, 4), txt_len=3)
                for name, up in (("left", torch.tensor([[1.], [0.]])),
                                 ("right", torch.tensor([[0.], [2.]]))):
                    adapter = SimpleNamespace(
                        weights=(up, down),
                        h=lambda x, base, up=up: torch.nn.functional.linear(
                            torch.nn.functional.linear(x, down), up),
                        g=lambda x: x,
                    )
                    hook.adapters.append({"adapter": adapter, "offset": None, "adapter_name": name})
                x = torch.ones(1, 7, 2)
                torch.testing.assert_close(hook._bypass_forward(x), expected, rtol=0, atol=0)
                hook.disable_lora()
                torch.testing.assert_close(hook._bypass_forward(x), torch.zeros_like(expected))
                hook.enable_lora()
                hook.clear_masks()
                torch.testing.assert_close(hook._bypass_forward(x), torch.tensor([1., 2.]).expand_as(expected))

    def test_other_sequence_layouts_keep_their_existing_domains(self):
        spatial = torch.tensor([0., 1., 0., 1.])
        for key, length, expected in (
            ("diffusion_model.double_blocks.0.img_mlp.0", 4, spatial),
            ("diffusion_model.layers.3.proj.ffn", 7, torch.cat((spatial, torch.ones(3)))),
        ):
            with self.subTest(key=key):
                actual = self.hook(key)._get_mask_for_sequence("left", length, torch.device("cpu"), torch.float32)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        hook = self.hook("diffusion_model.blocks.3.mlp.gate")
        hook.packed_mask_slice = (3, 7)
        actual = hook._get_mask_for_sequence("left", 9, torch.device("cpu"), torch.float32)
        torch.testing.assert_close(actual, torch.cat((torch.ones(3), spatial, torch.ones(2))), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
