#!/usr/bin/env python3
"""Native bypass routing regressions. Set COMFYUI_PATH; CPU only."""
import ast
from contextlib import nullcontext
import copy
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from test_bypass_joint_routing import JointRoutingTests


class TokenMaskingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        JointRoutingTests.setUpClass()
        cls.Hook = JointRoutingTests.hook_class
        # The existing standalone loader intentionally does not register a package.
        cls.globals = cls.Hook.set_masks.__globals__
        cls.Manager = cls.globals['OffsetBypassInjectionManager']
        cls.context = staticmethod(cls.globals.get('bypass_token_mask_context', lambda p: nullcontext()))
        path = Path(__file__).resolve().parents[1] / 'nodes/mask_applicator.py'
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FreeFuseMaskApplicator')
        # Run the actual node class, avoiding unrelated extension-startup imports.
        tree.body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node]
        namespace = dict(torch=torch, logging=logging,
                         OffsetBypassInjectionManager=cls.Manager,
                         MultiAdapterBypassForwardHook=cls.Hook,
                         bypass_token_mask_context=cls.context,
                         detect_model_type=lambda **kw: kw['model'].kind)
        exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), namespace)
        cls.Applicator = namespace['FreeFuseMaskApplicator']

    def fixture(self, kind, key, maps=None, multiple=True):
        module = torch.nn.Linear(2, 2, bias=False)
        module.weight.data.zero_()
        hook = self.Hook(module, key)
        hook.original_forward = module.forward
        module.forward = hook._bypass_forward
        down = torch.tensor([[1., 0.]])
        for name, up in [('a', torch.tensor([[1.], [0.]])), ('b', torch.tensor([[0.], [2.]]))]:
            adapter = SimpleNamespace(weights=(up, down),
                h=lambda x, base, up=up: torch.nn.functional.linear(torch.nn.functional.linear(x, down), up),
                g=lambda x: x)
            hook.adapters.append(dict(adapter=adapter, adapter_name=name, offset=None))
        manager = self.Manager()
        manager.hooks = [hook]
        options = {'freefuse_data': {'txt_len': 3}}
        if multiple:
            options['freefuse_bypass_managers'] = [{'manager': manager}]
        else:
            options['freefuse_bypass_manager'] = manager
        model = SimpleNamespace(kind=kind, model_options={'transformer_options': options})
        model.clone = lambda: model
        model.set_model_unet_function_wrapper = lambda wrapper: model.model_options.update(model_function_wrapper=wrapper)
        data = dict(model_type=kind, adapters=[dict(name='a'), dict(name='b')],
                    token_pos_maps=maps or {'a': [[0]], 'b': [[1]], '__background__': [[2]]})
        masks = {'masks': {'a': torch.tensor([[1., 1., 0., 0.]]), 'b': torch.tensor([[0., 0., 1., 1.]])}}
        return model, module, hook, manager, data, masks

    def run_node(self, fixture, shape, enabled=True, branches=(0,), text_length=3):
        model, module, hook, manager, data, masks = fixture
        self.Applicator().apply_masks(model, masks, data, enable_token_masking=enabled, enable_attention_bias=False)
        batch = len(branches) if '.layerwise_blocks.' in hook.module_key else shape[0]
        params = dict(input=torch.ones(batch, 1, 1, 1), timestep=torch.zeros(batch),
                      cond_or_uncond=list(branches),
                      c={'c_crossattn': torch.zeros(batch, text_length, 2),
                         'transformer_options': {'cond_or_uncond': list(branches)}})
        wrapper = model.model_options.get('model_function_wrapper')
        call = lambda *a, **kw: module(torch.ones(shape))
        return wrapper(call, params) if wrapper else call()

    def test_joint_and_text_only_domains_across_models(self):
        cases = [
            ('krea2', 'blocks.3.mlp.gate', 7, 0),
            ('krea2', 'txtfusion.refiner_blocks.0.attn.wq', 3, 0),
            ('flux', 'single_blocks.0.linear1', 7, 0),
            ('flux', 'double_blocks.0.txt_attn.qkv', 3, 0),
            ('flux2', 'single_blocks.0.linear1', 7, 0),
            ('flux2', 'double_blocks.0.txt_mlp.0', 3, 0),
            ('z_image', 'layers.0.attention.qkv', 7, 0),
            ('z_image', 'context_refiner.0.attention.qkv', 3, 0),
            ('moss_image', 'layers.0.proj.ffn', 7, 4),
            ('moss_image', 'text_embedder.linear_1', 3, 0),
            ('anima', 'blocks.0.cross_attn.k_proj', 3, 0),
            ('sdxl', 'input_blocks.4.1.transformer_blocks.0.attn2.to_k', 3, 0),
            ('sdxl', 'down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_v', 3, 0),
        ]
        for kind, key, seq, start in cases:
            for multiple in (True, False):
                with self.subTest(kind=kind, key=key, multiple=multiple):
                    fixture = self.fixture(kind, 'diffusion_model.' + key, multiple=multiple)
                    off = self.run_node(fixture, (1, seq, 2), False)
                    on = self.run_node(fixture, (1, seq, 2))
                    expected = off.clone()
                    expected[0, start:start+3] = torch.tensor([[1., 0.], [0., 2.], [1., 2.]])
                    torch.testing.assert_close(on, expected, rtol=0, atol=0)

    def test_krea2_layerwise_fusion_masks_flattened_token_axis(self):
        fixture = self.fixture('krea2', 'diffusion_model.txtfusion.layerwise_blocks.0.mlp.gate')
        actual = self.run_node(fixture, (6, 12, 2), branches=(1, 0))
        expected = torch.tensor([1., 2.]).expand(6, 12, 2).clone()
        expected[3] = torch.tensor([1., 0.])
        expected[4] = torch.tensor([0., 2.])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_cfg_and_multiple_prompt_positions_are_not_flattened_together(self):
        maps = {'a': [[0], [1]], 'b': [[1], [0]], '__background__': [[2], [2]]}
        fixture = self.fixture('krea2', 'diffusion_model.blocks.3.mlp.gate', maps)
        actual = self.run_node(fixture, (4, 7, 2), branches=(1, 0))
        expected = actual.clone()
        expected[:2,:3] = torch.tensor([[1.,2.]]*3)
        expected[2,:3] = torch.tensor([[1.,0.],[0.,2.],[1.,2.]])
        expected[3,:3] = torch.tensor([[0.,2.],[1.,0.],[1.,2.]])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_shared_overlapping_and_out_of_range_positions(self):
        maps = {'a': [[0,2]], 'b': [[1,2,3,999]], '__background__': [[3]]}
        fixture = self.fixture('flux2', 'diffusion_model.double_blocks.0.txt_attn.qkv', maps)
        actual = self.run_node(fixture, (1, 4, 2), text_length=4)
        torch.testing.assert_close(actual[0], torch.tensor([[1.,0.],[0.,2.],[1.,2.],[1.,2.]]), rtol=0, atol=0)

    def test_toggle_clear_and_clone_preserve_lifecycle(self):
        fixture = self.fixture('krea2', 'diffusion_model.blocks.3.mlp.gate')
        on = self.run_node(fixture, (1,7,2))
        off = self.run_node(fixture, (1,7,2), False)
        torch.testing.assert_close(off[0,:3], torch.tensor([[1.,2.]]*3))
        self.run_node(fixture, (1,7,2))
        cloned = copy.deepcopy(fixture[2])
        cloned.original_forward = fixture[2].original_forward
        with self.context({'input': torch.ones(1,1,1,1), 'cond_or_uncond': [0],
                           'c': {'c_crossattn': torch.zeros(1,3,2)}}):
            torch.testing.assert_close(cloned._bypass_forward(torch.ones(1,7,2)), on)
        fixture[3].clear_masks()
        torch.testing.assert_close(fixture[1](torch.ones(1,7,2)), torch.tensor([1.,2.]).expand(1,7,2))

    def test_image_only_and_h3_segments_are_not_text_masked(self):
        for kind,key in [('flux','double_blocks.0.img_attn.qkv'),
                         ('anima','blocks.0.self_attn.q_proj'),
                         ('z_image','noise_refiner.0.attention.qkv')]:
            with self.subTest(kind=kind):
                fixture = self.fixture(kind,'diffusion_model.'+key)
                off = self.run_node(fixture,(1,4,2),False)
                on = self.run_node(fixture,(1,4,2))
                torch.testing.assert_close(on,off,rtol=0,atol=0)
        fixture = self.fixture('krea2','diffusion_model.blocks.3.mlp.gate')
        fixture[2].packed_mask_slice=(3,7)
        off=self.run_node(fixture,(1,9,2),False)
        on=self.run_node(fixture,(1,9,2))
        torch.testing.assert_close(on,off,rtol=0,atol=0)

    def test_pending_masks_survive_clone_and_hook_rebuild(self):
        fixture = self.fixture('krea2', 'diffusion_model.blocks.0.mlp.gate')
        manager = self.Manager()
        for entry in fixture[2].adapters:
            manager.add_adapter('diffusion_model.blocks.0.mlp.gate.weight', entry['adapter'],
                                adapter_name=entry['adapter_name'])
        manager.set_masks(fixture[5]['masks'], (1,4), 3, fixture[4]['token_pos_maps'], 'krea2')
        cloned = copy.deepcopy(manager)
        root = torch.nn.Module()
        root.diffusion_model = torch.nn.Module()
        block = torch.nn.Module()
        block.mlp = torch.nn.Module()
        block.mlp.gate = torch.nn.Linear(2,2,bias=False)
        block.mlp.gate.weight.data.zero_()
        root.diffusion_model.blocks = torch.nn.ModuleList([block])
        injection = cloned.create_injections(root)[0]
        patcher = SimpleNamespace(model=root,model_options={'transformer_options':{'freefuse_bypass_manager':cloned}})
        injection.inject(patcher)
        try:
            with self.context({'input': torch.ones(1,1,1,1), 'cond_or_uncond':[0],
                               'c':{'c_crossattn':torch.zeros(1,3,2)}}):
                actual = block.mlp.gate(torch.ones(1,7,2))
            torch.testing.assert_close(actual[0,:3],torch.tensor([[1.,0.],[0.,2.],[1.,2.]]))
        finally:
            injection.eject(patcher)
        self.assertIsNone(cloned.hooks[0].original_forward)
        self.assertIs(block.mlp.gate.forward.__self__, block.mlp.gate)

    def test_wrapper_composes_and_restores_context_after_errors(self):
        fixture = self.fixture('flux', 'diffusion_model.single_blocks.0.linear1')
        model = fixture[0]
        calls = []
        def previous(fn, params):
            calls.append('previous')
            return fn(params['input'], params['timestep'], **params['c'])
        model.model_options['model_function_wrapper'] = previous
        self.run_node(fixture,(1,7,2))
        wrapper = model.model_options['model_function_wrapper']
        self.Applicator._install_token_mask_wrapper(model)
        self.assertIs(wrapper,model.model_options['model_function_wrapper'])
        self.assertEqual(calls,['previous'])
        params = {'input':torch.ones(1,1,1,1),'timestep':torch.zeros(1),
                  'cond_or_uncond':[1],'c':{'c_crossattn':torch.zeros(1,3,2)}}
        def fail(*args,**kwargs):
            raise RuntimeError('expected test failure')
        with self.assertRaisesRegex(RuntimeError,'expected test failure'):
            wrapper(fail,params)
        self.assertIsNone(self.globals['_token_mask_context'].get())


if __name__ == '__main__':
    unittest.main()
