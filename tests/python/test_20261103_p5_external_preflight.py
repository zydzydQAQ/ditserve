"""External preflight preserves unknowns, pinned identity and qualification limits."""

import contextlib
import copy
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
import p5_external_preflight as p5


def write_json(path, value):
    raw = (json.dumps(value, indent=2) + '\n').encode()
    path.write_bytes(raw)
    return p5.digest(raw)


class ExternalPreflightTests(unittest.TestCase):
    def fixture(self, root):
        # Test data are isolated miniature snapshots; no dependency on the
        # gitignored official downloads or on locally installed frameworks.
        plan = p5.read_json(p5.DEFAULT_PLAN.read_bytes())
        sources = root / 'sources'
        for project in plan['projects']:
            directory = sources / project['id']
            directory.mkdir(parents=True)
            sections = {'project': ['requires-python = ' + json.dumps(project['requires_python'])]}
            documents = {'README.md': '\n'.join(i['documented_text'] for i in project['interfaces']) + '\n'}
            for group in project['dependency_groups']:
                if group['file'].endswith('.toml'):
                    sections.setdefault(group['section'], []).append(group['key'] + ' = [\n' +
                        ''.join('    ' + json.dumps(req) + ',\n' for req in group['requirements']) + ']')
                else:
                    documents[group['file']] = '\n'.join(['-r ' + f for f in group['includes']] + group['requirements']) + '\n'
            documents['pyproject.toml'] = '\n'.join('[' + key + ']\n' + '\n'.join(lines) for key, lines in sections.items())
            for relative, text in documents.items():
                path = directory / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            self.repin_project(project, sources)
        runtime = root / 'runtime-facts.json'
        facts = dict(schema='ditserve.runtime-audit.v1', run_id=plan['runtime_snapshot']['run_id'], python='3.10.20',
            packages={'torch': '2.10.0+cu126', 'diffusers': '0.36.0', 'transformers': '4.57.6',
                      'cache-dit': '1.5.0', 'accelerate': '1.12.0', 'safetensors': '0.8.0', 'numpy': '2.2.6'})
        plan['runtime_snapshot']['sha256'] = write_json(runtime, facts)
        return plan, sources, runtime

    @staticmethod
    def repin_project(project, sources, change=None):
        root = sources / project['id']
        for relative, ref in project['files'].items():
            raw = (root / relative).read_bytes()
            ref.update(sha256=p5.digest(raw), bytes=len(raw))
        source = dict(repository=project['repository'], commit=project['commit'], files=project['files'],
                      project={'requires-python': project['requires_python']})
        if change:
            change(source)
        project['source_manifest_sha256'] = write_json(root / 'source.json', source)

    @staticmethod
    def repin_runtime(plan, runtime, change):
        facts = p5.read_json(runtime.read_bytes())
        change(facts)
        plan['runtime_snapshot']['sha256'] = write_json(runtime, facts)

    def test_known_conflicts_unknowns_and_no_external_measurements(self):
        with TemporaryDirectory() as directory:
            plan, sources, runtime = self.fixture(Path(directory))
            result = p5.evaluate(plan, sources, runtime)
            projects = {p['id']: p for p in result['projects']}
            self.assertEqual({r['package'] for r in projects['TetriServe']['known_conflicts']}, {'numpy', 'diffusers'})
            self.assertEqual({r['package'] for r in projects['GF-DiT']['known_conflicts']}, {'cache-dit'})
            self.assertEqual(projects['xDiT']['known_conflicts'], [])
            self.assertEqual(projects['xDiT']['runnable'], 'unverified')
            self.assertTrue(all(p['unknowns'] for p in projects.values()))
            self.assertEqual(result['external_measurements'], [])
            self.assertEqual(result['status'], 'preflight_only')
            self.assertEqual(result['allows'], [])
            self.assertFalse(result['p5_qualified'])
            self.assertEqual(result, p5.evaluate(plan, sources, runtime))

    def test_omitted_package_is_unknown_even_if_known_version_would_conflict(self):
        with TemporaryDirectory() as directory:
            plan, sources, runtime = self.fixture(Path(directory))
            self.repin_runtime(plan, runtime, lambda v: v['packages'].pop('diffusers'))
            result = p5.evaluate(plan, sources, runtime)
            for project in result['projects']:
                row = next(r for r in project['checks'] if r['package'] == 'diffusers')
                self.assertEqual(row['status'], 'unknown')
                self.assertEqual(row['reason'], 'package_not_observed_in_partial_inventory')
                self.assertNotIn('observed_version', row)

    def test_constraints_handle_numeric_versions_local_build_and_unknown_markers(self):
        cases = [
            ('torch>=2.4.1', {'torch': '2.10.0+cu126'}, 'satisfies_declared_constraint'),
            ('torch==2.10', {'torch': '2.10.0+cu126'}, 'satisfies_declared_constraint'),
            ('cache_dit>=1.5,<1.6', {'cache-dit': '1.6.0'}, 'conflict'),
            ('numpy<2', {'numpy': '2.2.6'}, 'conflict'),
            ('numpy<2', {'numpy': '2.0.0rc1'}, 'unknown'),
            ('numpy<2; python_version<"3.11"', {'numpy': '2.2.6'}, 'unknown'),
            ('numpy~=1.26', {'numpy': '2.2.6'}, 'unknown'),
            ('numpy', {'numpy': 'unavailable'}, 'unknown'),
            ('torch @ git+https://example.invalid/repo', {'torch': '2.10.0'}, 'unknown'),
            ('imageio[ffmpeg]>=2.37.2', {'imageio': '2.37.2'}, 'satisfies_declared_constraint'),
        ]
        for requirement, inventory, expected in cases:
            with self.subTest(requirement=requirement):
                self.assertEqual(p5.dependency_check(requirement, inventory)['status'], expected)
        self.assertEqual(p5.dependency_check(cases[-1][0], cases[-1][1])['extras_status'], 'not_audited')

    def test_raw_official_file_tampering_is_rejected(self):
        with TemporaryDirectory() as directory:
            plan, sources, runtime = self.fixture(Path(directory))
            path = sources / 'TetriServe/pyproject.toml'
            path.write_text(path.read_text().replace('numpy<2', 'numpy>=2'))
            with self.assertRaisesRegex(ValueError, 'source hash/size differs'):
                p5.evaluate(plan, sources, runtime)

    def test_source_manifest_tampering_is_rejected(self):
        with TemporaryDirectory() as directory:
            plan, sources, runtime = self.fixture(Path(directory))
            path = sources / 'xDiT/source.json'
            path.write_text(path.read_text() + ' ')
            with self.assertRaisesRegex(ValueError, 'source hash/size differs'):
                p5.evaluate(plan, sources, runtime)

    def test_rehashed_manifest_with_different_commit_is_rejected(self):
        with TemporaryDirectory() as directory:
            plan, sources, runtime = self.fixture(Path(directory))
            self.repin_project(plan['projects'][0], sources, lambda v: v.update(commit='0' * 40))
            with self.assertRaisesRegex(ValueError, 'commit/repository/inventory differs'):
                p5.evaluate(plan, sources, runtime)

    def test_url_must_bind_exact_repository_and_commit(self):
        with TemporaryDirectory() as directory:
            plan, sources, runtime = self.fixture(Path(directory))
            project = plan['projects'][0]
            project['files']['README.md']['url'] = project['files']['README.md']['url'].replace(project['commit'], 'main')
            self.repin_project(project, sources)
            with self.assertRaisesRegex(ValueError, 'source URL'):
                p5.evaluate(plan, sources, runtime)

    def test_declared_requirements_are_read_from_the_bound_file(self):
        with TemporaryDirectory() as directory:
            plan, sources, runtime = self.fixture(Path(directory))
            plan['projects'][0]['dependency_groups'][0]['requirements'].remove('numpy<2')
            with self.assertRaisesRegex(ValueError, 'dependency declarations differ'):
                p5.evaluate(plan, sources, runtime)

    def test_unpinned_requirements_include_is_rejected(self):
        with TemporaryDirectory() as directory:
            plan, sources, runtime = self.fixture(Path(directory))
            project = plan['projects'][1]
            group = next(g for g in project['dependency_groups'] if g['file'] == 'requirements/cuda.txt')
            group['includes'] = ['unreviewed.txt']
            path = sources / 'GF-DiT/requirements/cuda.txt'
            path.write_text(path.read_text().replace('common.txt', 'unreviewed.txt'))
            self.repin_project(project, sources)
            with self.assertRaisesRegex(ValueError, 'include is not pinned/audited'):
                p5.evaluate(plan, sources, runtime)

    def test_path_traversal_and_escaping_symlinks_are_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan, sources, runtime = self.fixture(root)
            with self.assertRaisesRegex(ValueError, 'invalid pinned source path'):
                p5.checked_file(sources, '../runtime-facts.json', plan['runtime_snapshot']['sha256'])
            (sources / 'TetriServe').rename(root / 'outside')
            (sources / 'TetriServe').symlink_to(root / 'outside', target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'escapes its snapshot root'):
                p5.evaluate(plan, sources, runtime)

    def test_runtime_hash_and_run_identity_are_required(self):
        with TemporaryDirectory() as directory:
            plan, sources, runtime = self.fixture(Path(directory))
            runtime.write_text(runtime.read_text() + ' ')
            with self.assertRaisesRegex(ValueError, 'runtime inventory snapshot identity'):
                p5.evaluate(plan, sources, runtime)
            self.repin_runtime(plan, runtime, lambda v: v.update(run_id='another-job'))
            with self.assertRaisesRegex(ValueError, 'runtime inventory snapshot identity'):
                p5.evaluate(plan, sources, runtime)

    def test_duplicate_package_aliases_and_json_keys_are_rejected(self):
        with TemporaryDirectory() as directory:
            plan, sources, runtime = self.fixture(Path(directory))
            self.repin_runtime(plan, runtime, lambda v: v['packages'].update({'cache_dit': '1.3.0'}))
            with self.assertRaisesRegex(ValueError, 'duplicate or invalid runtime package identity'):
                p5.evaluate(plan, sources, runtime)
        with self.assertRaisesRegex(ValueError, 'duplicate JSON key'):
            p5.read_json('{"numpy":"1.26", "numpy":"2.2.6"}')

    def test_protocol_and_documented_interfaces_cannot_drift(self):
        with TemporaryDirectory() as directory:
            plan, sources, runtime = self.fixture(Path(directory))
            changed = copy.deepcopy(plan)
            changed['protocol_sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'formal protocol identity'):
                p5.evaluate(changed, sources, runtime)
            plan['projects'][0]['interfaces'][0]['documented_text'] = 'made_up_launcher'
            with self.assertRaisesRegex(ValueError, 'interface is not documented'):
                p5.evaluate(plan, sources, runtime)

    def test_claim_plan_cannot_grant_qualification_or_overwrite_results(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan, sources, runtime = self.fixture(root)
            plan.update(allows=['P5'], p5_qualified=True, status='passed')
            path = root / 'plan.json'
            write_json(path, plan)
            output = root / 'summary.json'
            args = ['--plan', str(path), '--sources-root', str(sources), '--runtime-facts', str(runtime), '--output', str(output)]
            self.assertEqual(p5.main(args), 0)
            result = p5.read_json(output.read_bytes())
            self.assertFalse(result['p5_qualified'])
            self.assertEqual(result['allows'], [])
            self.assertEqual(result['status'], 'preflight_only')
            original = output.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                p5.main(args)
            self.assertEqual(original, output.read_bytes())

    def test_completion_plan_preserves_all_three_external_validity_dimensions(self):
        plan = p5.read_json(p5.DEFAULT_PLAN.read_bytes())
        steps = {row['id']: row for row in plan['completion_plan']}
        self.assertEqual(set(steps['review_and_qualification']['depends_on']),
                         {'failures_and_statistics', 'additional_model_family', 'hardware_and_cross_node_scale'})
        self.assertTrue(all(row['status'] == 'pending' for row in steps.values()))
        self.assertEqual(steps['exact_inputs']['frozen_values']['inference_seeds'], [301, 302, 303])
        self.assertEqual(steps['exact_inputs']['frozen_values']['requests_per_cell'], 200)
        self.assertFalse(plan['execution_policy']['build_image'])
        self.assertFalse(plan['execution_policy']['install'])


if __name__ == '__main__':
    unittest.main()
