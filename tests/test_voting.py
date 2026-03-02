#!/usr/bin/env python
"""
Lightweight tests for FreeFuse range collection and voting aggregation.
"""

import os
import sys
import importlib.util

import torch


def _find_comfyui_dir(start_dir: str) -> str:
    cur = os.path.abspath(start_dir)
    for _ in range(10):
        if os.path.isdir(os.path.join(cur, "comfy")) and os.path.isfile(os.path.join(cur, "main.py")):
            return cur
        sibling = os.path.join(cur, "ComfyUI")
        if os.path.isdir(sibling) and os.path.isdir(os.path.join(sibling, "comfy")):
            return sibling
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    raise FileNotFoundError("Could not locate ComfyUI directory")


script_dir = os.path.dirname(os.path.abspath(__file__))
freefuse_comfyui_dir = os.path.dirname(script_dir)
repo_root = os.path.dirname(freefuse_comfyui_dir)
comfyui_dir = _find_comfyui_dir(repo_root)
if comfyui_dir not in sys.path:
    sys.path.insert(0, comfyui_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
os.chdir(comfyui_dir)


def _load_module(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _to_sim_map(values):
    """Build (B, img_len, 1) similarity tensor."""
    return torch.tensor(values, dtype=torch.float32).view(1, -1, 1)


class _DummyHookHandle:
    def __init__(self):
        self.removed = False

    def remove(self):
        self.removed = True


class _DummyLayer:
    def __init__(self):
        self.pre_hooks = []

    def register_forward_pre_hook(self, fn, with_kwargs=False):
        self.pre_hooks.append((fn, with_kwargs))
        return _DummyHookHandle()


class _DummyDiffusionModel:
    def __init__(self, n_single=0, n_layers=0):
        self.single_blocks = [object() for _ in range(n_single)]
        self.layers = [_DummyLayer() for _ in range(n_layers)]


class _DummyInnerModel:
    def __init__(self, diffusion_model):
        self.diffusion_model = diffusion_model


class _DummyModelPatcher:
    def __init__(self, diffusion_model):
        self.model = _DummyInnerModel(diffusion_model)
        self.patch_replace_calls = []
        self.double_block_patch = None

    def set_model_patch_replace(self, fn, model_part, block_kind, block_index):
        self.patch_replace_calls.append((model_part, block_kind, int(block_index), fn))

    def set_model_double_block_patch(self, fn):
        self.double_block_patch = fn


def test_state_collect_range():
    attn_module = _load_module(
        "freefuse_attention_replace_for_test",
        os.path.join(freefuse_comfyui_dir, "freefuse_core", "attention_replace.py"),
    )
    FreeFuseState = attn_module.FreeFuseState

    state = FreeFuseState()
    state.phase = "collect"
    state.collect_step = 3
    state.collect_block = 2
    state.collect_block_end = 4

    assert state.is_collect_step(3, 2)
    assert state.is_collect_step(3, 3)
    assert state.is_collect_step(3, 4)
    assert not state.is_collect_step(3, 1)
    assert not state.is_collect_step(3, 5)
    assert not state.is_collect_step(2, 3)


def test_consensus_similarity_maps_majority():
    voting_module = _load_module(
        "freefuse_voting_for_test",
        os.path.join(freefuse_comfyui_dir, "freefuse_core", "voting.py"),
    )
    create_consensus_similarity_maps = voting_module.create_consensus_similarity_maps

    # 2x2 latent => img_len=4
    # Block winners:
    # b0: a a b b
    # b1: a b b b
    # b2: a b a b
    # majority => a b b b
    block_maps = {
        0: {
            "a": _to_sim_map([1, 1, 0, 0]),
            "b": _to_sim_map([0, 0, 1, 1]),
        },
        1: {
            "a": _to_sim_map([1, 0, 0, 0]),
            "b": _to_sim_map([0, 1, 1, 1]),
        },
        2: {
            "a": _to_sim_map([1, 0, 1, 0]),
            "b": _to_sim_map([0, 1, 0, 1]),
        },
    }

    out = create_consensus_similarity_maps(
        all_blocks_similarity_maps=block_maps,
        concept_names=["a", "b"],
        latent_h=2,
        latent_w=2,
        device=torch.device("cpu"),
    )

    assert set(out.keys()) == {"a", "b"}
    assert out["a"].shape == (1, 4, 1)
    assert out["b"].shape == (1, 4, 1)

    winner_a = out["a"].view(-1).tolist()
    winner_b = out["b"].view(-1).tolist()
    assert winner_a == [1.0, 0.0, 0.0, 0.0]
    assert winner_b == [0.0, 1.0, 1.0, 1.0]


def test_range_patch_routing_flux2_and_zimage():
    attn_module = _load_module(
        "freefuse_attention_replace_for_test_range_routing",
        os.path.join(freefuse_comfyui_dir, "freefuse_core", "attention_replace.py"),
    )
    FreeFuseState = attn_module.FreeFuseState
    apply_freefuse_replace_patches = attn_module.apply_freefuse_replace_patches

    # Flux2: range should register multiple single_block patch_replace callbacks.
    flux2_diffusion = _DummyDiffusionModel(n_single=6)
    flux2_model = _DummyModelPatcher(flux2_diffusion)
    flux2_state = FreeFuseState()
    flux2_state.collect_block = 1
    flux2_state.collect_block_end = 3
    apply_freefuse_replace_patches(
        model=flux2_model,
        state=flux2_state,
        model_type="flux2",
        flux2_collect_blocks=[1, 2, 3],
    )
    flux2_single_blocks = [idx for _, kind, idx, _ in flux2_model.patch_replace_calls if kind == "single_block"]
    assert flux2_single_blocks == [1, 2, 3]

    # Z-Image: range should install one double_block patch and pre-hooks on each target layer.
    z_diffusion = _DummyDiffusionModel(n_layers=8)
    z_model = _DummyModelPatcher(z_diffusion)
    z_state = FreeFuseState()
    z_state.collect_block = 2
    z_state.collect_block_end = 4
    apply_freefuse_replace_patches(
        model=z_model,
        state=z_state,
        model_type="z_image",
        z_image_collect_blocks=[2, 3, 4],
    )
    assert callable(z_model.double_block_patch)
    assert len(z_diffusion.layers[2].pre_hooks) == 1
    assert len(z_diffusion.layers[3].pre_hooks) == 1
    assert len(z_diffusion.layers[4].pre_hooks) == 1
    assert len(z_diffusion.layers[1].pre_hooks) == 0


def run_all_tests():
    test_state_collect_range()
    test_consensus_similarity_maps_majority()
    test_range_patch_routing_flux2_and_zimage()
    print("All voting tests passed.")


if __name__ == "__main__":
    run_all_tests()
