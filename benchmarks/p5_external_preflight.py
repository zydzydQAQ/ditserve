#!/usr/bin/env python3
"""Offline review of pinned external declarations, never a P5 stage receipt.

Uses only the standard library; does not import external frameworks or execute
their launchers. Version checks cover numeric releases and simple comparisons,
not dependency resolution, extras, markers, CUDA ABI or actual model support.
An unobserved package is unknown, not evidence that it is absent.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


DEFAULT_PLAN = Path(__file__).resolve().parents[1] / 'configs/benchmarks/p5-external-v1.json'
DEFAULT_PROTOCOL = DEFAULT_PLAN.with_name('formal-mixed-v1.json')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical_hash(value):
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':'),
                             ensure_ascii=False, allow_nan=False).encode())


def read_json(raw):
    def reject(value):
        raise ValueError('nonfinite JSON: ' + value)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(raw, parse_constant=reject, object_pairs_hook=unique)


def checked_file(root, name, sha256, size=None):
    relative = PurePosixPath(name)
    if (not name or relative.is_absolute() or '..' in relative.parts or '\\' in name
            or str(relative) != name or not re.fullmatch(r'[0-9a-f]{64}', sha256)):
        raise ValueError('invalid pinned source path/hash: ' + name)
    root = Path(root).resolve()
    path = root.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError('source escapes its snapshot root: ' + name)
    raw = path.read_bytes()
    if digest(raw) != sha256 or size is not None and len(raw) != size:
        raise ValueError('source hash/size differs: ' + name)
    return raw


def toml_section(text, name):
    match = re.search(r'(?ms)^\[' + re.escape(name) + r'\]\s*\n(.*?)(?=^\[|\Z)', text)
    if not match:
        raise ValueError('missing declared TOML section: ' + name)
    return match.group(1)


def string_array(text, section, key):
    """Read the simple string arrays used by these pinned files, fail closed.

    Deliberately not a general TOML parser; no escaped/multiline strings or
    arbitrary expressions are accepted. Brackets inside quoted extras work.
    """
    body = toml_section(text, section)
    match = re.search(r'(?m)^\s*' + re.escape(key) + r'\s*=\s*\[', body)
    if not match:
        raise ValueError('missing declared TOML array: ' + key)
    cursor, result, need_comma = match.end(), [], False
    while cursor < len(body):
        token = re.match(r'\s+|#[^\n]*', body[cursor:])
        if token:
            cursor += token.end()
            continue
        if body[cursor] == ']':
            return result
        if body[cursor] == ',' and need_comma:
            need_comma = False
            cursor += 1
            continue
        token = re.match(r'''(["'])([^"'\\\n]*)\1''', body[cursor:])
        if not token or need_comma:
            raise ValueError('unsupported pinned TOML array syntax: ' + key)
        result.append(token.group(2))
        cursor += token.end()
        need_comma = True
    raise ValueError('unterminated pinned TOML array: ' + key)


def normalize(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def numeric_version(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d+(?:\.\d+)*(?:\+[a-zA-Z0-9]+(?:[-_.][a-zA-Z0-9]+)*)?', value):
        return None
    result = tuple(int(v) for v in value.split('+')[0].split('.'))
    while len(result) > 1 and result[-1] == 0:
        result = result[:-1]
    return result


def dependency_check(requirement, inventory):
    result = dict(requirement=requirement, status='unknown')
    match = re.fullmatch(r'([A-Za-z0-9][A-Za-z0-9_.-]*)(?:\[([A-Za-z0-9_,.-]+)\])?\s*(.*)', requirement)
    if not match:
        return dict(result, reason='unsupported_requirement_syntax')
    name, extras, constraints = match.groups()
    name = normalize(name)
    result['package'] = name
    if extras:
        result.update(extras=extras.split(','), extras_status='not_audited')
    if name not in inventory:
        return dict(result, reason='package_not_observed_in_partial_inventory')
    observed = inventory[name]
    result['observed_version'] = observed
    version = numeric_version(observed)
    if version is None:
        return dict(result, reason='unsupported_or_unknown_observed_version')
    if not constraints:
        return dict(result, status='observed_unconstrained', reason='no_declared_version_bound')
    for item in constraints.split(','):
        spec = re.fullmatch(r'\s*(==|!=|>=|<=|>|<)\s*(\d+(?:\.\d+)*)\s*', item)
        if spec is None:
            return dict(result, reason='unsupported_specifier_marker_or_direct_reference')
    conflicts = []
    for item in constraints.split(','):
        op, bound = re.fullmatch(r'\s*(==|!=|>=|<=|>|<)\s*(\d+(?:\.\d+)*)\s*', item).groups()
        expected = numeric_version(bound)
        width = max(len(version), len(expected))
        left, right = version + (0,) * (width - len(version)), expected + (0,) * (width - len(expected))
        accepted = {'==': left == right, '!=': left != right, '>=': left >= right,
                    '<=': left <= right, '>': left > right, '<': left < right}[op]
        if not accepted:
            conflicts.append(op + bound)
    if conflicts:
        return dict(result, status='conflict', violated_constraints=conflicts)
    return dict(result, status='satisfies_declared_constraint')


def verify_source(project, sources_root):
    name = project['id']
    # Project directory and every file must remain below the supplied snapshot.
    source_raw = checked_file(sources_root, name + '/source.json', project['source_manifest_sha256'])
    source = read_json(source_raw)
    if (not re.fullmatch(r'[0-9a-f]{40}', project['commit'])
            or source.get('commit') != project['commit']
            or source.get('repository') != project['repository']
            or source.get('files') != project['files']):
        raise ValueError('official source commit/repository/inventory differs: ' + name)
    files = {}
    for relative, reference in project['files'].items():
        expected_url = 'https://raw.githubusercontent.com/' + project['repository'] + '/' + project['commit'] + '/' + relative
        if reference.get('url') != expected_url:
            raise ValueError('source URL does not bind the pinned repository/commit: ' + name)
        files[relative] = checked_file(sources_root, name + '/' + relative,
            reference['sha256'], reference['bytes']).decode('utf-8')
    pyproject = files['pyproject.toml']
    python = project['requires_python']
    if (source.get('project', {}).get('requires-python') != python
            or not re.search(r'''(?m)^requires-python\s*=\s*(["'])''' + re.escape(python) + r'''\1\s*$''',
                             toml_section(pyproject, 'project'))):
        raise ValueError('Python version declaration differs: ' + name)
    for group in project['dependency_groups']:
        text = files[group['file']]
        if group['file'].endswith('.toml'):
            declared = string_array(text, group['section'], group['key'])
        else:
            lines = [line.split('#', 1)[0].strip() for line in text.splitlines()]
            declared = [line for line in lines if line and not line.startswith('-r ')]
            includes = [line[3:].strip() for line in lines if line.startswith('-r ')]
            if includes != group.get('includes', []):
                raise ValueError('requirements include declarations differ: ' + name)
            for included in includes:
                target = str(PurePosixPath(group['file']).parent / included)
                if target not in files or not any(g['file'] == target for g in project['dependency_groups']):
                    raise ValueError('requirements include is not pinned/audited: ' + target)
        if declared != group['requirements']:
            raise ValueError('selected dependency declarations differ: ' + name + '/' + group['file'])
    prefix = 'https://github.com/' + project['repository'] + '/blob/' + project['commit'] + '/'
    for interface in project['interfaces']:
        if (not interface['url'].startswith(prefix)
                or interface['documented_text'] not in files['README.md']):
            raise ValueError('interface is not documented in the pinned README: ' + name)
    return files


def evaluate(plan, sources_root, runtime_path, protocol_path=DEFAULT_PROTOCOL):
    if plan.get('schema') != 'ditserve.p5-preflight-plan.v1':
        raise ValueError('unsupported P5 preflight plan')
    protocol_raw = Path(protocol_path).read_bytes()
    protocol = read_json(protocol_raw)
    if (protocol.get('schema') != 'ditserve.formal-protocol.v1'
            or canonical_hash(protocol) != plan['protocol_sha256']):
        raise ValueError('frozen formal protocol identity differs')
    runtime_raw = Path(runtime_path).read_bytes()
    runtime = read_json(runtime_raw)
    if (digest(runtime_raw) != plan['runtime_snapshot']['sha256']
            or runtime.get('schema') != 'ditserve.runtime-audit.v1'
            or runtime.get('run_id') != plan['runtime_snapshot']['run_id']):
        raise ValueError('runtime inventory snapshot identity differs')
    inventory = {}
    if not isinstance(runtime.get('packages'), dict):
        raise ValueError('runtime packages inventory must be a mapping')
    for name, version in runtime['packages'].items():
        key = normalize(name)
        if key in inventory or not isinstance(version, str):
            raise ValueError('duplicate or invalid runtime package identity: ' + name)
        inventory[key] = version
    if isinstance(runtime.get('python'), str):
        inventory['python'] = runtime['python']
    if len({p['id'] for p in plan['projects']}) != len(plan['projects']):
        raise ValueError('duplicate external project identity')
    projects = []
    for project in plan['projects']:
        verify_source(project, sources_root)
        checks = [dict(dependency_check('python' + project['requires_python'], inventory),
                       source='pyproject.toml', activation='base')]
        for group in project['dependency_groups']:
            for requirement in group['requirements']:
                checks.append(dict(dependency_check(requirement, inventory),
                    source=group['file'], activation=group['activation']))
        conflicts = [row for row in checks if row['status'] == 'conflict']
        unknowns = [row for row in checks if row['status'] == 'unknown' or row.get('extras_status') == 'not_audited']
        projects.append(dict(id=project['id'], repository=project['repository'], commit=project['commit'],
            source_manifest_sha256=project['source_manifest_sha256'], source_files=project['files'],
            declaration_status='known_conflicts' if conflicts else 'no_known_conflict_in_audited_subset',
            runnable='unverified', checks=checks, known_conflicts=conflicts, unknowns=unknowns,
            interfaces=copy.deepcopy(project['interfaces']), applicability=copy.deepcopy(project['applicability']),
            unaudited=copy.deepcopy(project['unaudited'])))
    return dict(schema='ditserve.p5-preflight-review.v1', status='preflight_only', stage='P5',
        allows=[], p5_qualified=False, external_measurements=[],
        plan_sha256=canonical_hash(plan), protocol_sha256=canonical_hash(protocol),
        protocol_file_sha256=digest(protocol_raw), runtime_snapshot=dict(
            sha256=digest(runtime_raw), run_id=runtime['run_id'], packages=runtime['packages'],
            python=runtime.get('python'), inventory_scope='partial; omission is not absence',
            models_are_not_manifest_verification=True), projects=projects,
        claim_limits=copy.deepcopy(plan['claim_limits']),
        completion_plan=copy.deepcopy(plan['completion_plan']),
        execution_policy=copy.deepcopy(plan['execution_policy']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
    parser.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument('--sources-root', type=Path, required=True)
    parser.add_argument('--runtime-facts', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='new summary JSON; never overwrite an existing result')
    args = parser.parse_args(argv)
    try:
        raw = args.plan.read_bytes()
        result = evaluate(read_json(raw), args.sources_root, args.runtime_facts, args.protocol)
        result['plan_file_sha256'] = digest(raw)
        payload = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n'
        if args.output:
            with args.output.open('x', encoding='utf-8') as stream:
                stream.write(payload)
        else:
            print(payload, end='')
    except (ValueError, KeyError, OSError, TypeError) as exc:
        parser.error(str(exc))
    # Successful inspection is never a compatibility/P5 pass; consumers must
    # use the explicit review schema, preflight_only status and empty allows.
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
