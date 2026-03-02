#!/usr/bin/env python
"""
Lightweight tests for FreeFuseMaskTap / FreeFuseMaskReassemble.
"""

import importlib.util
import os
import tempfile

import torch
from PIL import Image


def _load_module(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _module():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    freefuse_dir = os.path.dirname(script_dir)
    path = os.path.join(freefuse_dir, "nodes", "mask_tap.py")
    return _load_module("freefuse_mask_tap_for_test", path)


def _unwrap_result(value):
    if isinstance(value, dict) and "result" in value:
        return value["result"]
    return value


def test_tap_order_from_freefuse_data():
    mod = _module()
    tap = mod.FreeFuseMaskTap()

    mask_bank = {
        "masks": {
            "b": torch.ones(8, 8),
            "a": torch.zeros(8, 8),
        }
    }
    freefuse_data = {
        "adapters": [
            {"name": "a"},
            {"name": "b"},
        ]
    }

    out = _unwrap_result(tap.tap_masks(mask_bank=mask_bank, freefuse_data=freefuse_data))
    slot_names = out[1]
    edited_bank = out[0]
    slot_mask_images = out[2]

    assert slot_names.splitlines()[0] == "00:a"
    assert slot_names.splitlines()[1] == "01:b"
    assert torch.allclose(edited_bank["masks"]["a"], torch.zeros(8, 8))
    assert torch.allclose(edited_bank["masks"]["b"], torch.ones(8, 8))
    assert slot_mask_images.shape == (10, 8, 8, 3)
    assert slot_mask_images.device.type == "cpu"


def test_reassemble_preserve_and_resize():
    mod = _module()
    reassemble = mod.FreeFuseMaskReassemble()

    original = {
        "masks": {
            "a": torch.zeros(1, 8, 8),
            "b": torch.ones(8, 8),
        },
        "metadata": {"adapter_names": ["a", "b"]},
    }

    edited_small = torch.full((4, 4), 0.5, dtype=torch.float32)
    out, = reassemble.reassemble_masks(
        original_mask_bank=original,
        slot_names="00:a\n01:b",
        clamp_01=True,
        mask_00=edited_small,
        # mask_01 omitted -> keep original
    )

    masks = out["masks"]
    assert masks["a"].shape == (1, 8, 8)
    assert torch.allclose(masks["a"], torch.full((1, 8, 8), 0.5), atol=1e-5)
    assert torch.allclose(masks["b"], torch.ones(8, 8))


def test_tap_override_from_mask_image_ref():
    mod = _module()
    tap = mod.FreeFuseMaskTap()

    with tempfile.TemporaryDirectory() as td:
        # RGBA alpha checker:
        # alpha=0 -> expected mask=1
        # alpha=255 -> expected mask=0
        img = Image.new("RGBA", (8, 8), (0, 0, 0, 255))
        px = img.load()
        for y in range(8):
            for x in range(8):
                px[x, y] = (0, 0, 0, 0 if x < 4 else 255)
        path = os.path.join(td, "edited.png")
        img.save(path)

        mask_bank = {"masks": {"a": torch.zeros(8, 8)}}
        out = _unwrap_result(tap.tap_masks(mask_bank=mask_bank, mask_image_00=path))
        edited_bank = out[0]
        slot_mask_images = out[2]
        edited = edited_bank["masks"]["a"]

        assert edited.shape == (8, 8)
        assert torch.allclose(edited[:, :4], torch.ones(8, 4), atol=1e-6)
        assert torch.allclose(edited[:, 4:], torch.zeros(8, 4), atol=1e-6)
        assert torch.allclose(edited_bank["masks"]["a"][:, :4], torch.ones(8, 4), atol=1e-6)
        assert torch.allclose(edited_bank["masks"]["a"][:, 4:], torch.zeros(8, 4), atol=1e-6)
        assert slot_mask_images.shape == (10, 8, 8, 3)
        assert torch.allclose(slot_mask_images[0, :, :4, 0], torch.ones(8, 4), atol=1e-6)
        assert torch.allclose(slot_mask_images[0, :, 4:, 0], torch.zeros(8, 4), atol=1e-6)


def test_tap_slot_images_cpu_with_cuda_source_when_available():
    if not torch.cuda.is_available():
        return

    mod = _module()
    tap = mod.FreeFuseMaskTap()
    mask_bank = {"masks": {"a": torch.full((8, 8), 0.25, device="cuda")}}
    out = _unwrap_result(tap.tap_masks(mask_bank=mask_bank))
    slot_mask_images = out[2]

    assert slot_mask_images.device.type == "cpu"
    assert torch.allclose(slot_mask_images[0, :, :, 0], torch.zeros((8, 8)), atol=1e-6)


def test_tap_override_prefers_visible_grayscale_and_nearest_resize():
    mod = _module()
    tap = mod.FreeFuseMaskTap()

    with tempfile.TemporaryDirectory() as td:
        # Alpha is constant here, so RGB grayscale must drive the loaded mask.
        img = Image.new("RGBA", (2, 2), (0, 0, 0, 255))
        px = img.load()
        px[0, 0] = (255, 255, 255, 255)  # -> 1
        px[1, 0] = (0, 0, 0, 255)        # -> 0
        px[0, 1] = (127, 127, 127, 255)  # -> 0 (below 0.5)
        px[1, 1] = (200, 200, 200, 255)  # -> 1
        path = os.path.join(td, "edited_rgb.png")
        img.save(path)

        mask_bank = {"masks": {"a": torch.zeros(4, 4)}}
        out = _unwrap_result(tap.tap_masks(mask_bank=mask_bank, mask_image_00=path))
        edited = out[0]["masks"]["a"]

        assert edited.shape == (4, 4)
        # Nearest resize from 2x2 -> 4x4 should produce clear 2x2 quadrants.
        assert torch.allclose(edited[:2, :2], torch.ones(2, 2), atol=1e-6)
        assert torch.allclose(edited[:2, 2:], torch.zeros(2, 2), atol=1e-6)
        assert torch.allclose(edited[2:, :2], torch.zeros(2, 2), atol=1e-6)
        assert torch.allclose(edited[2:, 2:], torch.ones(2, 2), atol=1e-6)


def test_tap_ignores_default_slot_or_blank_refs():
    mod = _module()
    tap = mod.FreeFuseMaskTap()

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "freefuse-masktap-slot-legacy-00.png")
        Image.new("L", (8, 8), color=255).save(path)

        original_mask = torch.zeros(8, 8)
        mask_bank = {"masks": {"a": original_mask.clone()}}
        out = _unwrap_result(tap.tap_masks(mask_bank=mask_bank, mask_image_00=path))
        edited = out[0]["masks"]["a"]

        # Default slot refs should not be treated as user edits.
        assert torch.allclose(edited, original_mask, atol=1e-6)


def test_tap_user_edit_ref_requires_matching_phase1_seed():
    mod = _module()
    tap = mod.FreeFuseMaskTap()

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "freefuse-masktap-edit-s77-case.png")
        Image.new("L", (8, 8), color=255).save(path)

        base = torch.zeros(8, 8)
        bank = {
            "masks": {"a": base.clone()},
            "metadata": {"phase1_seed": 77},
        }
        out = _unwrap_result(tap.tap_masks(mask_bank=bank, mask_image_00=path))
        edited = out[0]["masks"]["a"]
        assert torch.allclose(edited, torch.ones(8, 8), atol=1e-6)

        bank_mismatch = {
            "masks": {"a": base.clone()},
            "metadata": {"phase1_seed": 88},
        }
        out_mismatch = _unwrap_result(tap.tap_masks(mask_bank=bank_mismatch, mask_image_00=path))
        edited_mismatch = out_mismatch[0]["masks"]["a"]
        assert torch.allclose(edited_mismatch, base, atol=1e-6)


def test_tap_legacy_untagged_edit_ref_ignored_when_phase1_seed_present():
    mod = _module()
    tap = mod.FreeFuseMaskTap()

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "freefuse-masktap-edit-legacy.png")
        Image.new("L", (8, 8), color=255).save(path)

        base = torch.zeros(8, 8)
        bank = {
            "masks": {"a": base.clone()},
            "metadata": {"phase1_seed": 77},
        }
        out = _unwrap_result(tap.tap_masks(mask_bank=bank, mask_image_00=path))
        edited = out[0]["masks"]["a"]
        assert torch.allclose(edited, base, atol=1e-6)


def test_ui_editor_image_contains_visible_rgb_and_alpha_mask():
    mod = _module()
    original_folder_paths = getattr(mod, "folder_paths", None)

    try:
        with tempfile.TemporaryDirectory() as td:
            class _FakeFolderPaths:
                @staticmethod
                def get_temp_directory():
                    return td

            mod.folder_paths = _FakeFolderPaths()
            refs = mod._build_mask_editor_ui_images(
                [torch.tensor([[0.0, 0.5], [1.0, 0.25]], dtype=torch.float32)]
            )
            assert refs and isinstance(refs[0], dict)
            path = os.path.join(td, refs[0]["filename"])
            with Image.open(path) as img:
                px = img.convert("RGBA").load()
                r00, g00, b00, a00 = px[0, 0]
                r10, g10, b10, a10 = px[1, 0]
                r01, g01, b01, a01 = px[0, 1]
                r11, g11, b11, a11 = px[1, 1]

            # UI image should be binary: mask=1 => white RGB + transparent alpha.
            assert (r00, g00, b00, a00) == (0, 0, 0, 255)
            assert (r10, g10, b10, a10) == (255, 255, 255, 0)
            assert (r01, g01, b01, a01) == (255, 255, 255, 0)
            assert (r11, g11, b11, a11) == (0, 0, 0, 255)
    finally:
        mod.folder_paths = original_folder_paths


def run_all_tests():
    test_tap_order_from_freefuse_data()
    test_reassemble_preserve_and_resize()
    test_tap_override_from_mask_image_ref()
    test_tap_slot_images_cpu_with_cuda_source_when_available()
    test_tap_override_prefers_visible_grayscale_and_nearest_resize()
    test_tap_ignores_default_slot_or_blank_refs()
    test_tap_user_edit_ref_requires_matching_phase1_seed()
    test_tap_legacy_untagged_edit_ref_ignored_when_phase1_seed_present()
    test_ui_editor_image_contains_visible_rgb_and_alpha_mask()
    print("All mask tap/reassemble tests passed.")


if __name__ == "__main__":
    run_all_tests()
