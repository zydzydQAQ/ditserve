"""Paper display names for native policies.

Standard library only: paper_results.py imports this without workers/ (ditserve)
on sys.path. native_serving.py re-exports both dictionaries.
"""

# Paper tables (see docs/experiments.md and src/ditserve/control/policy.cpp, the authoritative
# registry). Preprint-only methods are named by method; TetriServe (ASPLOS'26) and xDiT by system.
POLICY_DISPLAY = {
    "xdit": "xDiT", "tetriserve": "TetriServe (re-impl.)", "sbp-esp": "SBP-ESP", "edf-bestfit": "EDF-BF",
    "edf-cfg2": "EDF-CFG2", "cfg2-first": "CFG2-First", "partition": "Static-Partition",
    "static-cfg2": "Static-CFG2", "static-cfg2sp2": "Static-CFG2SP2",
    "static-cfg2sp3": "Static-CFG2SP3", "static-cfg2sp4": "Static-CFG2SP4",
    "ditserve": "DitServe", "ditserve-approx": "DitServe-Approx",
    "fold-cfg2sp3": "CFG2SP3 fold only", "approx-cfg2sp3": "CFG2SP3 fold + fission",
    "abl-no-block": "DitServe w/o block preemption", "abl-no-prune": "DitServe w/o form pruning",
    "abl-no-mindeg": "DitServe w/o CFG-pair admission", "abl-first-fit": "DitServe w/o pair-fit placement",
    "ditserve-v1": "DitServe-v1 (knapsack)", "block-cfg2": "Block-cfg2", "block-cfg2sp3": "Block-cfg2sp3",
}
IN_FRAMEWORK_MECHANISM_DISPLAY = {
    "edf-cfg2": "EDF-CFG2",
    "static-cfg2sp2": "Static-CFG2SP2",
    "static-cfg2sp3": "Static-CFG2SP3",
    "static-cfg2sp4": "Static-CFG2SP4",
    "cfg2-first": "CFG2-First",
    "tetriserve": "TetriServe-policy",
    "sbp-esp": "SBP-ESP",
    "edf-bestfit": "EDF-BF",
    "ditserve": "DitServe",
}
