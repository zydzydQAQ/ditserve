"""Reproducible patch application must preserve official scheduling/transport."""
import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from typing import List, Optional, Union

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
import external_tetriserve_port as port
import external_tetriserve as source


RUNNER = '''from tetriserve.xfuser import xFuserFluxPipeline, xFuserStableDiffusion3Pipeline
class FluidServeRunner:
    MODEL_REGISTRY = {
        "black-forest-labs/FLUX.1-dev": xFuserFluxPipeline,
        "black-forest-labs/FLUX.1-schnell": xFuserFluxPipeline,
        "stabilityai/stable-diffusion-3-medium-diffusers": xFuserStableDiffusion3Pipeline,
    }
    def preprocess(self, reqs):
        for idx in range(1):
            if idx:
                self.pipeline.dummy_inference_prologue()
            else:
                tensors = self.pipeline.inference_prologue(
                    num_inference_steps=reqs.origin_steps[0],
                    generator=torch.Generator(device="cuda").manual_seed(reqs.input_config.seed),
                )
        latents = torch.cat(latents_list, dim=0)
        return latents
    def generate(self, reqs, latent_future_dict):
        from tetriserve.distributed.globals import set_seq_parallel_pg
        set_seq_parallel_pg(1, 1, 0, [0], 1)
        t_preprocess_start = time.perf_counter()
        return self.preprocess(reqs)
'''


class TetriPortTests(unittest.TestCase):
    def fixture(self, directory):
        root=Path(directory)/'official'
        files={name: '# unchanged official mechanism fixture\n' for name in port.PRESERVED}
        files.update({
            'tetriserve/scheduler/perf_model.py': 'import pandas as pd\n',
            'tetriserve/__init__.py': 'SUPPORTED_MODEL_DICT = {}\n',
            'tetriserve/xfuser/__init__.py': 'from legacy_model import Pipeline\n',
            'tetriserve/config/args.py': 'class ServerArgs:\n    def __post_init__(self):\n        pass\n',
            'tetriserve/xfuser/config/config.py': '''class InputConfig:
    prompt: str = ""
    def __post_init__(self):
        if isinstance(self.prompt, list):
            assert (
                len(self.prompt) == len(self.negative_prompt) or len(self.negative_prompt) == 0
            )
''',
            'tetriserve/server/api_server.py': '''class RequestInput:
    guidance_scale: float = 0.0
async def generate(request):
    return InputConfig(
            prompt=request.prompt,
    )
''',
            'tetriserve/engine/runner.py': RUNNER,
        })
        for name,text in files.items():
            path=root/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(text)
        manifest=Path(directory)/'source.json'
        record={'schema':'ditserve.official-source-archive.v1','repository':source.REPOSITORY,
                'commit':source.COMMIT,'archive_sha256':source.ARCHIVE_SHA256,
                'files':{name:port.sha((root/name).read_bytes()) for name in files}}
        manifest.write_text(json.dumps(record))
        return root,manifest,port.sha(manifest.read_bytes())

    def test_plan_pins_original_and_shared_adapters_without_execution(self):
        with TemporaryDirectory() as directory:
            args=self.fixture(directory)
            result,patches,added=port.plan(*args)
            self.assertEqual(len(patches),7)
            self.assertEqual(set(added),set(port.ADAPTER_FILES))
            self.assertFalse(result['model_execution_verified'])
            self.assertFalse(result['multi_gpu_numerical_verified'])
            self.assertFalse(result['qualification_eligible'])
            self.assertEqual(result['allows'],[])
            for name in port.PRESERVED:
                self.assertNotIn(name,patches)
                self.assertEqual(result['preserved_native_mechanisms'][name],result['files'][name])
            runner=patches['tetriserve/engine/runner.py'].decode()
            self.assertIn('activate_request(self.pipeline, reqs, latent_future_dict)',runner)
            self.assertIn('validate_transferred_batch(',runner)
            self.assertIn('generator=torch.Generator(device="cpu")',runner)
            self.assertNotIn('device="cuda"',runner)
            self.assertIn('num_inference_steps=reqs.origin_steps[idx]',runner)
            api=patches['tetriserve/server/api_server.py'].decode()
            self.assertIn('negative_prompt=request.negative_prompt',api)
            self.assertIn('guidance_scale=request.guidance_scale',api)

    def test_frozen_input_config_broadcasts_scalar_negative_for_list_prompts(self):
        root = Path(__file__).resolve().parents[3] / '.local/ditserve-kairos-rewrite-runtime/external-systems/20261103/TetriServe'
        if not root.is_dir():
            self.skipTest('frozen official source needed for real InputConfig regression')
        manifest = root.with_name('TetriServe.source.json')
        _, patches, _ = port.plan(root, manifest,
            '2d0d9f9efaf4b3749063458d56c74b0ef79ef72de22096b39889e798e792d027')
        tree = ast.parse(patches['tetriserve/xfuser/config/config.py'])
        config_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'InputConfig')
        # Execute the real upstream dataclass after the explicit patch, without
        # importing GPU packages merely to exercise its request normalization.
        namespace = dict(dataclass=dataclass, Optional=Optional, Union=Union, List=List)
        exec(compile(ast.Module(body=[config_class], type_ignores=[]), '<official InputConfig>', 'exec'), namespace)
        for prompts in ('fox', ['fox'], ['fox', 'hare']):
            result = namespace['InputConfig'](prompt=prompts, negative_prompt='blurry, low quality')
            expected = [prompts] if isinstance(prompts, str) else prompts
            self.assertEqual(result.prompt, expected)
            self.assertEqual(result.batch_size, len(expected))
            self.assertEqual(result.negative_prompt, 'blurry, low quality')
            self.assertEqual(result.guidance_scale, 4.5)
        with self.assertRaises(AssertionError):
            namespace['InputConfig'](prompt=['fox', 'hare'], negative_prompt=['one'])

    def test_apply_creates_new_verified_tree_without_modifying_source(self):
        with TemporaryDirectory() as directory:
            args=self.fixture(directory)
            old={p.relative_to(args[0]).as_posix():p.read_bytes() for p in args[0].rglob('*') if p.is_file()}
            output=Path(directory)/'derived'
            result=port.apply(*args,output)
            manifest=json.loads((output/'PORT_MANIFEST.json').read_bytes())
            self.assertEqual(result['manifest_sha256'],port.sha((output/'PORT_MANIFEST.json').read_bytes()))
            for name,expected in manifest['files'].items():
                self.assertEqual(port.sha((output/name).read_bytes()),expected)
            self.assertEqual(old,{p.relative_to(args[0]).as_posix():p.read_bytes() for p in args[0].rglob('*') if p.is_file()})
            with self.assertRaises(FileExistsError): port.apply(*args,output)
            self.assertEqual(result['manifest_sha256'],port.sha((output/'PORT_MANIFEST.json').read_bytes()))

    def test_original_tree_destination_and_ancestor_are_rejected(self):
        with TemporaryDirectory() as directory:
            args=self.fixture(directory)
            for output in (args[0],args[0]/'nested',Path(directory)):
                with self.assertRaisesRegex(ValueError,'disjoint'):
                    port.apply(*args,output)

    def test_source_mutation_bad_manifest_and_ambiguous_anchor_fail_before_output(self):
        with TemporaryDirectory() as directory:
            args=self.fixture(directory)
            output=Path(directory)/'derived'
            (args[0]/'tetriserve/engine/runner.py').write_text(RUNNER+'\n# changed\n')
            with self.assertRaisesRegex(ValueError,'SHA256 differs'):
                port.apply(*args,output)
            self.assertFalse(output.exists())
        with self.assertRaisesRegex(ValueError,'not unique'):
            port.replace_once('anchor anchor','anchor','new','unit')
        with self.assertRaisesRegex(ValueError,'not unique'):
            port.replace_once('missing','anchor','new','unit')


if __name__ == '__main__':
    unittest.main()
