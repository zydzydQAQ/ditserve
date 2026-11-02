"""Keep parsed configuration binding separate from native JSON byte identity.

The runtime hashes nlohmann::json::dump(), whose floating-point spelling can
differ from Python json.dumps() for the same value. Callers must verify their
raw evidence/config references and runtime source independently. This helper
does not recompute or authenticate native serializer bytes.
"""

import re

from stage_gates import canonical_hash


def config_identity(record, expected, *, require_native=True):
    """Bind parsed content and validate the opaque native digest separately.

    Only historical P2 records may omit the native digest. That is reported as
    missing, never filled from a Python hash. A present but malformed digest
    is rejected even in that compatibility mode.
    """
    actual = record.get('config')
    parsed_hash = canonical_hash(expected)
    if actual != expected or canonical_hash(actual) != parsed_hash:
        raise ValueError('run parsed configuration differs from frozen inputs')
    native_hash = record.get('config_sha256')
    if 'config_sha256' not in record and not require_native:
        native_hash = None
    elif not isinstance(native_hash, str) or re.fullmatch(r'[0-9a-f]{64}', native_hash) is None:
        raise ValueError('run lacks a valid native config SHA256')
    return dict(parsed_config_sha256=parsed_hash, native_config_sha256=native_hash,
                native_digest_recomputed=False,
                scope='parsed content binding and native digest consistency; raw bytes/source verified by caller')


def match_native_config(identity, owner, *, required=True):
    """A session, charge or cutoff receipt must use its record's native SHA."""
    if not isinstance(owner, dict):
        raise ValueError('native config identity owner must be an object')
    if 'config_sha256' not in owner and not required:
        return
    native_hash = identity['native_config_sha256']
    if native_hash is None or owner.get('config_sha256') != native_hash:
        raise ValueError('session/receipt native config hash differs from its run')
