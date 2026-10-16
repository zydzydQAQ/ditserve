"""Finite, explicitly identified cache policies for the CFG experiment.

The logical policy and its execution implementation have different identities.
Resolving a policy does not qualify it: callers must bind quality and cost
evidence to the returned checksum as well as the current frozen protocol.
"""

import copy

from ..base import digest


POLICY_SCHEMA = "ditserve.research.cfg.policy.v1"
POLICIES = (
    dict(id="off", cache=False, threshold=0.0, max_continuous=1, cfg_mode="independent"),
    dict(id="independent_008", cache=True, threshold=0.08, max_continuous=1, cfg_mode="independent"),
    dict(id="independent_004", cache=True, threshold=0.04, max_continuous=1, cfg_mode="independent"),
    dict(id="paired_008", cache=True, threshold=0.08, max_continuous=1, cfg_mode="paired"),
)
IMPLEMENTATIONS = ("native", "oracle", "split_prefix")


def validate_catalog(config):
    """Reject unknown, reordered, missing, or modified logical policies."""
    recipes = config.get("recipes") if isinstance(config, dict) else None
    if recipes != list(POLICIES):
        raise ValueError("CFG policy catalog differs from the four fixed logical policies")
    for recipe in recipes:
        if (type(recipe["cache"]) is not bool or type(recipe["max_continuous"]) is not int
                or type(recipe["threshold"]) not in (int, float)):
            raise ValueError("invalid CFG policy field types")
    return copy.deepcopy(recipes)


def resolve_policy(config, policy_id, implementation=None):
    """Return an independent validated dictionary, including implementation SHA.

    ``split_prefix`` is a describable implementation, not permission to run it.
    Its caller must enforce the current protocol's prototype continuation gate.
    No legacy ``t008`` alias is accepted in this new protocol.
    """
    catalog = validate_catalog(config)
    matches = [row for row in catalog if row["id"] == policy_id]
    if len(matches) != 1:
        raise ValueError("unknown CFG policy: " + str(policy_id))
    policy = matches[0]
    paired = policy["cfg_mode"] == "paired"
    if implementation is None:
        implementation = "oracle" if paired else "native"
    if (implementation not in IMPLEMENTATIONS
            or paired and implementation == "native"
            or not paired and implementation != "native"):
        raise ValueError("CFG policy does not support implementation: " + str(implementation))
    result = dict(policy, schema_version=POLICY_SCHEMA, policy_id=policy_id,
                  implementation=implementation)
    result["sha256"] = digest(result)
    return result
