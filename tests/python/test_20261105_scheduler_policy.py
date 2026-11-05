"""P2 resolver registries mirror the native policy registry of each reviewed policy.cpp revision.

CPU-only source checks. The current registry is compared against the compiled C++
registry (simulation-tests --policy-catalog); the legacy registry against a verbatim
copy of the resolver that was reviewed with 6429a94's policy.cpp.
"""

import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2_benefit as p2
import p2_contrasts as contrasts
from native_serving import FORM_DEGREES

SIMULATION_TESTS = Path(os.environ.get('DITSERVE_SIMULATION_TESTS', ROOT / '.cache/build/benchmarks/simulation-tests'))
KNOWN = ('ditserve', 'cfg2-first', 'abl-no-block', 'abl-no-prune', 'abl-no-mindeg', 'abl-first-fit',
         'edf-cfg2', 'block-cfg2', 'xdit', 'tetriserve', 'sbp-esp', 'edf-bestfit')
OPTION_SETS = (
    {'sp_transport': 'export', 'forms': ['sp1', 'cfg2', 'sp2', 'cfg2sp2']},
    {'sp_transport': 'export', 'forms': ['cfg2'], 'doorbell': True, 'partition_image_gpus': 2, 'late_short_ms': 500},
    {'sp_transport': 'export', 'forms': ['sp1', 'cfg2'], 'reserve_image_gpus': 1, 'demote_late': True,
     'prune_dominated': False, 'round_steps': 1},
    {'sp_transport': 'export', 'forms': ['sp1'], 'partition_image_gpus': 6, 'image_batch': 2},
)


def reviewed_6429a94_resolver(policy, options, gpus):
    """Verbatim resolver of p2_contrasts.py at 7ee8beb, reviewed against 6429a94's policy.cpp."""
    if (options.get('sp_transport') != 'export' or options.get('pin_started', False)
            or options.get('tiered_memory', False) or gpus not in (2, 4, 6, 8)):
        raise ValueError('unsupported: P2 resolver covers unpinned export transport on 2/4/6/8 GPUs only')
    policy = 'xdit-cfg2' if policy == 'xdit' else policy
    known = {'ditserve', 'cfg2-first', 'abl-no-block', 'abl-no-prune', 'abl-no-mindeg', 'abl-first-fit',
             'edf-cfg2', 'block-cfg2', 'xdit-cfg2', 'tetriserve', 'sbp-esp', 'edf-bestfit'}
    if policy not in known:
        raise ValueError('policy has no audited exact resolver: ' + policy)
    ours = policy in {'ditserve', 'cfg2-first', 'abl-no-block', 'abl-no-prune', 'abl-no-mindeg', 'abl-first-fit'}
    fixed = policy in {'edf-cfg2', 'block-cfg2', 'xdit-cfg2'}
    blocks = policy in {'ditserve', 'abl-no-prune', 'abl-no-mindeg', 'abl-first-fit', 'block-cfg2'}
    default = dict(elastic=not fixed, blocks=blocks, fixed_form='cfg2' if fixed else None,
        prune_dominated=ours and policy != 'abl-no-prune', min_video_degree=2 if ours and policy != 'abl-no-mindeg' else 1,
        widen_efficiency=.9 if ours else 0., image_placement='pair_fit' if ours and policy != 'abl-first-fit' else 'first',
        order='fcfs' if policy == 'xdit-cfg2' else 'edf', hold_diffuse=policy == 'xdit-cfg2',
        demote_late=policy != 'xdit-cfg2', rounds=policy == 'tetriserve',
        deadline_dp=policy in {'tetriserve', 'sbp-esp'}, best_fit=policy == 'edf-bestfit',
        lazy_preemption=True, widen_remaining_work=True, doorbell=False, check_blocks=1,
        quantum_blocks=4, quantum_ms=0., pin_started=False, round_steps=5,
        image_batching=policy == 'sbp-esp')
    overridable = {'prune_dominated', 'min_video_degree', 'widen_efficiency', 'image_placement', 'order',
                   'hold_diffuse', 'demote_late', 'lazy_preemption', 'widen_remaining_work', 'doorbell',
                   'check_blocks', 'quantum_blocks', 'quantum_ms', 'pin_started', 'round_steps'}
    default.update({key: options[key] for key in overridable if key in options})
    default['forms'] = ['cfg2'] if fixed else options.get('forms')
    if (not isinstance(default['forms'], list) or not default['forms']
            or any(form not in FORM_DEGREES for form in default['forms'])):
        raise ValueError('unsupported or implicit parallel form space')
    default['reserve_image_gpus'] = options.get('reserve_image_gpus',
        options.get('partition_image_gpus', max(1, gpus // 4)) if policy == 'xdit-cfg2' else 0)
    default['strict_partition'] = options.get('strict_partition', policy == 'xdit-cfg2')
    reserve = default['reserve_image_gpus']
    if type(reserve) is not int or reserve < 0 or reserve >= gpus or fixed and gpus - reserve < 2:
        raise ValueError('unsupported baseline: image partition leaves no complete fixed CFG2 video group')
    default['doorbell_effective'] = blocks and default['doorbell']
    default['image_batching_effective'] = default['image_batching'] and 'image_batch' not in options
    default['shared_options'] = {key: value for key, value in options.items() if key not in overridable
                                 and key not in {'forms', 'reserve_image_gpus', 'partition_image_gpus', 'strict_partition'}}
    return default


def outcome(resolve, *args):
    try:
        return resolve(*args)
    except ValueError as error:
        return 'ValueError: ' + str(error)


class RegistryVersionTest(unittest.TestCase):
    def test_legacy_registry_reproduces_the_reviewed_6429a94_resolutions(self):
        for policy, options, gpus in itertools.product(KNOWN, OPTION_SETS, (2, 4, 6, 8)):
            with self.subTest(policy=policy, options=options, gpus=gpus):
                self.assertEqual(
                    outcome(contrasts.resolved_options, policy, copy.deepcopy(options), gpus, contrasts.LEGACY_POLICY_SOURCE),
                    outcome(reviewed_6429a94_resolver, policy, copy.deepcopy(options), gpus))
        self.assertEqual(contrasts.canonical_policy('xdit', contrasts.LEGACY_POLICY_SOURCE), 'xdit-cfg2')
        # Every reviewed native bundle executed 6429a94's policy.cpp.
        for name, pins in contrasts.AUDITED_NATIVE_BUNDLES.items():
            with self.subTest(bundle=name):
                audit = contrasts.audit_native_sources(dict(pins))
                self.assertEqual(audit['policy_source_sha256'], contrasts.LEGACY_POLICY_SOURCE)

    def test_current_registry_digest_is_the_working_tree_policy_source(self):
        # Tripwire: editing policy.cpp requires reviewing CURRENT_POLICY_SOURCE's registry.
        digest = hashlib.sha256((ROOT / 'src/ditserve/control/policy.cpp').read_bytes()).hexdigest()
        self.assertEqual(contrasts.CURRENT_POLICY_SOURCE, digest)
        self.assertNotEqual(contrasts.CURRENT_POLICY_SOURCE, contrasts.LEGACY_POLICY_SOURCE)

    def test_unknown_policy_sources_fail_closed(self):
        for source in ('f' * 64, None, ''):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, 'no reviewed P2 policy registry'):
                contrasts.resolved_options('ditserve', dict(OPTION_SETS[0]), 8, source)
        pins = {path: (digest,) for path, digest in contrasts.AUDITED_NATIVE_BUNDLES['97f0713'].items()}
        native = dict(contrasts.AUDITED_NATIVE_BUNDLES['97f0713'])
        self.assertEqual(contrasts.audit_native_sources(native, pins)['policy_source_sha256'],
                         contrasts.LEGACY_POLICY_SOURCE)
        pins['src/ditserve/control/policy.cpp'] = ('e' * 64,)
        native['src/ditserve/control/policy.cpp'] = 'e' * 64
        with self.assertRaisesRegex(ValueError, 'no reviewed P2 policy registry'):
            contrasts.audit_native_sources(native, pins)


@unittest.skipUnless(SIMULATION_TESTS.exists(), 'compiled simulation-tests binary required (cmake --build .cache/build)')
class CurrentRegistryMirrorsNativeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dump = json.loads(subprocess.run([str(SIMULATION_TESTS), '--policy-catalog'], check=True,
                                         capture_output=True, text=True, timeout=60).stdout)
        cls.catalog = {row['id']: row for row in dump['catalog']}
        cls.planner = dump['planner']

    def native(self, policy):
        canonical = contrasts.canonical_policy(policy)
        traits = self.catalog[canonical]
        planner = self.planner[canonical]
        return dict(elastic=traits['elastic'], blocks=traits['blocks'], fixed_form=traits['fixed_form'] or None,
                    prune_dominated=traits['prune_dominated'], min_video_degree=traits['min_video_degree'],
                    widen_efficiency=traits['widen'], image_placement=traits['image_placement'], order=traits['order'],
                    hold_diffuse=traits['hold_diffuse'], demote_late=traits['demote_late'], rounds=traits['rounds'],
                    deadline_dp=traits['deadline_dp'], best_fit=traits['best_fit'], image_batching=traits['image_batching'],
                    round_steps=traits['round_steps'], late_short_ms=planner['late_short_ms'],
                    round_hold=traits['rounds'] and traits['round_steps'] > 1)

    def test_every_known_policy_resolves_to_the_compiled_registry(self):
        self.assertEqual(contrasts.canonical_policy('xdit'), 'xdit-sp1')
        for policy in KNOWN:
            with self.subTest(policy=policy):
                resolved = contrasts.resolved_options(policy, dict(OPTION_SETS[0]), 8)
                expected = self.native(policy)
                self.assertEqual({key: resolved[key] for key in expected}, expected)
                # The partition default and its clamp follow cluster_plan.
                divisor = self.catalog[contrasts.canonical_policy(policy)]['partition_divisor']
                fixed = expected['fixed_form']
                share = max(1, 8 // divisor) if divisor else 0
                if fixed and share:
                    share = min(share, 8 - FORM_DEGREES[fixed])
                self.assertEqual(resolved['reserve_image_gpus'], share)
                self.assertEqual(resolved['strict_partition'], divisor > 0)

    def test_baseline_traits_changed_by_7ee8beb_match_the_native_registry(self):
        xdit = contrasts.resolved_options('xdit', dict(OPTION_SETS[0]), 8)
        self.assertEqual((xdit['fixed_form'], xdit['forms'], xdit['hold_diffuse'], xdit['order'], xdit['demote_late']),
                         ('sp1', ['sp1'], True, 'fcfs', False))
        self.assertFalse(contrasts.resolved_options('tetriserve', dict(OPTION_SETS[0]), 8)['prune_dominated'])
        self.assertFalse(contrasts.resolved_options('edf-bestfit', dict(OPTION_SETS[0]), 8)['demote_late'])
        self.assertEqual(contrasts.resolved_options('sbp-esp', dict(OPTION_SETS[0]), 8)['late_short_ms'], 0.)
        self.assertEqual(self.planner['sbp-esp']['late_short_ms'], 0.)

    def test_round_hold_matches_the_native_planner(self):
        # Native probe: the free GPU of a video two steps into a five-step round is kept from an image.
        for policy in KNOWN:
            canonical = contrasts.canonical_policy(policy)
            with self.subTest(policy=policy):
                resolved = contrasts.resolved_options(policy, dict(OPTION_SETS[0]), 8)
                held = resolved['round_hold'] or resolved['hold_diffuse']
                self.assertEqual(held, self.planner[canonical]['mid_round_gpu_held'])
        self.assertTrue(contrasts.resolved_options('tetriserve', dict(OPTION_SETS[0]), 8)['round_hold'])
        self.assertFalse(contrasts.resolved_options('tetriserve', dict(OPTION_SETS[2]), 8)['round_hold'])


class ContrastResolutionTest(unittest.TestCase):
    def phase(self, policy, gpus=8, **options):
        return dict(policy=policy, gpus=list(range(gpus)), options=dict(OPTION_SETS[0], **options),
                    profile=dict(classes=dict(video=dict(kind='video'), image=dict(kind='image'))))

    def test_system_xdit_audits_the_single_gpu_arm_that_runs(self):
        result = contrasts.validate_contrast('system-xdit', self.phase('xdit'), self.phase('ditserve'))
        self.assertEqual(result['reference']['forms'], ['sp1'])
        self.assertEqual(result['reference']['fixed_form'], 'sp1')
        self.assertIn('fixed_form', result['differing_fields'])
        # Two GPUs with one image GPU still leave an sp1 video GPU; reserving both does not.
        two = contrasts.validate_contrast('system-xdit', self.phase('xdit', 2, partition_image_gpus=1),
                                          self.phase('ditserve', 2, partition_image_gpus=1))
        self.assertEqual(two['reference']['reserve_image_gpus'], 1)
        with self.assertRaisesRegex(ValueError, 'no complete fixed SP1'):
            contrasts.validate_contrast('system-xdit', self.phase('xdit', 2, reserve_image_gpus=2),
                                        self.phase('ditserve', 2, reserve_image_gpus=2))
        # A class needing two GPUs cannot run as the declared sp1 form.
        wide = [self.phase('xdit'), self.phase('ditserve')]
        for phase in wide:
            phase['profile']['classes']['video']['min_degree'] = 2
        with self.assertRaisesRegex(ValueError, 'fixed SP1 form'):
            contrasts.validate_contrast('system-xdit', *wide)
        legacy = contrasts.validate_contrast('system-xdit', self.phase('xdit'), self.phase('ditserve'),
                                             contrasts.LEGACY_POLICY_SOURCE)
        self.assertEqual(legacy['reference']['forms'], ['cfg2'])

    def test_shared_planner_defaults_add_no_spurious_differing_fields(self):
        for name in ('system-edf-cfg2', 'system-cfg2-first'):
            with self.subTest(contrast=name):
                definition = contrasts.CONTRASTS[name]
                result = contrasts.validate_contrast(name, self.phase(definition['reference']), self.phase('ditserve'))
                self.assertNotIn('late_short_ms', result['differing_fields'])
                self.assertNotIn('round_hold', result['differing_fields'])
        sbp = contrasts.validate_contrast('system-sbp-esp', self.phase('sbp-esp'), self.phase('ditserve'))
        self.assertIn('late_short_ms', sbp['differing_fields'])
        tetri = contrasts.validate_contrast('system-tetriserve', self.phase('tetriserve'), self.phase('ditserve'))
        self.assertIn('round_hold', tetri['differing_fields'])
        self.assertIn('prune_dominated', tetri['differing_fields'])
        edf = contrasts.validate_contrast('system-edf-bestfit', self.phase('edf-bestfit'), self.phase('ditserve'))
        self.assertIn('demote_late', edf['differing_fields'])
        # An explicit shared option wins over every policy default.
        same = contrasts.validate_contrast('system-sbp-esp', self.phase('sbp-esp', late_short_ms=750),
                                           self.phase('ditserve', late_short_ms=750))
        self.assertNotIn('late_short_ms', same['differing_fields'])
        self.assertEqual(same['reference']['late_short_ms'], 750)
        # Ablations still isolate exactly their mechanism under the current registry.
        lazy = contrasts.validate_contrast('lazy', self.phase('ditserve', lazy_preemption=False, doorbell=True),
                                           self.phase('ditserve', lazy_preemption=True, doorbell=True))
        self.assertEqual(lazy['differing_fields'], ['lazy_preemption'])

    def test_evidence_resolves_with_the_registry_the_arms_executed(self):
        audit = lambda source: {'native_source_audit': {'policy_source_sha256': source}}
        current = dict(benchmark=dict(p2=dict(policy_source_sha256=contrasts.CURRENT_POLICY_SOURCE)))
        arms = {arm: dict(config=current) for arm in ('reference', 'candidate')}
        self.assertEqual(p2.resolution_source(arms, {'reference': audit(contrasts.LEGACY_POLICY_SOURCE), 'candidate': {}}),
                         dict(sha256=contrasts.LEGACY_POLICY_SOURCE, basis='audited run native source'))
        self.assertEqual(p2.resolution_source(arms, {'reference': {}, 'candidate': {}})['sha256'],
                         contrasts.CURRENT_POLICY_SOURCE)
        older = {arm: dict(config=dict(benchmark=dict(p2=dict()))) for arm in ('reference', 'candidate')}
        self.assertEqual(p2.resolution_source(older, {'reference': {}, 'candidate': {}})['sha256'],
                         contrasts.LEGACY_POLICY_SOURCE)
        with self.assertRaisesRegex(ValueError, 'different policy.cpp'):
            p2.resolution_source(arms, {'reference': audit(contrasts.LEGACY_POLICY_SOURCE),
                                        'candidate': audit(contrasts.CURRENT_POLICY_SOURCE)})


if __name__ == '__main__':
    unittest.main()
