#!/usr/bin/env python3
"""Validate the deployed absolute-rate compiler with frozen production inputs."""

import copy
import hashlib
import json
import math
import os
import pathlib
import subprocess
import sys

RUN_DIR = pathlib.Path(sys.argv[1])
RUNTIME = pathlib.Path(os.environ["DITSERVE_KAIROS_RUNTIME_ROOT"])
sys.path[:0] = [str(RUNTIME / "benchmarks"), os.environ["DITSERVE_PACKAGE_ROOT"]]

import formal_benchmark as formal

matrix_path = pathlib.Path(os.environ["DITSERVE_RATE_MATRIX_PATH"])
matrix = json.loads(matrix_path.read_text())
bundle_path = pathlib.Path(os.environ["DITSERVE_REUSE_BUNDLES"])
bundle = formal.read_json(bundle_path)
manifests = formal.read_json(os.environ["DITSERVE_MODEL_MANIFESTS"])
protocol = formal.protocol(RUNTIME / "configs" / "benchmarks" / "formal-mixed-v2.json")
pools = {
    kind: formal.read_json(pathlib.Path("/shared/data/DITSERVE/datasets") / (protocol["datasets"][kind]["kind"] + ".json"))
    for kind in ("image", "video")
}
assert matrix["commit"] == os.environ["DITSERVE_TEST_REVISION"]
assert hashlib.sha256(bundle_path.read_bytes()).hexdigest() == matrix["calibration_bundle"]["file_sha256"]
assert bundle["sha256"] == matrix["calibration_bundle"]["sealed_sha256"]
assert bundle["model_manifests"] == manifests
fixed = matrix["fixed"]
uuids = bundle["gpu_uuids"]


def build(**kwargs):
    options = dict(
        split="test",
        seed=fixed["seed"],
        video_fraction=fixed["video_fraction"],
        pattern=fixed["pattern"],
        suite="sweep",
        slo_scale=fixed["slo_scale"],
        resolution_distribution=fixed["resolution_distribution"],
    )
    options.update(kwargs)
    return formal.serving(protocol, "wan22", uuids, bundle, pools, manifests, **options)


source_anchor = matrix["source_anchor"]
source = build(load=source_anchor["load"])
source_manifest = source["benchmark"]["trace"]["manifest"]
assert math.isclose(source_manifest["r0_rpm"], source_anchor["r0_rpm"], rel_tol=0.0, abs_tol=1e-12)
assert math.isclose(source_manifest["rate_rpm"], source_anchor["rate_rpm"], rel_tol=0.0, abs_tol=1e-9)
assert source_manifest["workload_sha256"] == source_anchor["workload_sha256"]
reference_rows = source["phases"][0]["matrix"]["/workload"][0]
results = []
for cell in matrix["cells"]:
    assert isinstance(cell["rate_rpm"], int)
    config = build(rate_rpm=cell["rate_rpm"])
    benchmark = config["benchmark"]
    manifest = benchmark["trace"]["manifest"]
    absolute = benchmark["absolute_rate"]
    rows = config["phases"][0]["matrix"]["/workload"][0]
    assert math.isclose(manifest["rate_rpm"], cell["rate_rpm"], rel_tol=1e-12, abs_tol=1e-9)
    assert math.isclose(benchmark["load"], cell["expected_load"], rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(absolute["r0_rpm"], matrix["r0_rpm"], rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(absolute["anchor_multiplier"], cell["anchor_multiplier"], rel_tol=1e-12, abs_tol=1e-12)
    assert benchmark["requested_policies"] == fixed["policies"]
    assert len(config["phases"][0]["matrix"]["/policy"]) == len(fixed["policies"])
    identity = lambda row: {key: row[key] for key in (
        "request_id", "kind", "class", "spec", "service_ms", "slo_ms", "dataset_sample_id"
    )}
    assert [identity(row) for row in rows] == [identity(row) for row in reference_rows]
    results.append({
        "tag": cell["tag"],
        "rate_rpm": manifest["rate_rpm"],
        "load": benchmark["load"],
        "workload_sha256": manifest["workload_sha256"],
        "watchdog_seconds": benchmark["time_planning"]["watchdog_seconds"],
    })
assert len({row["workload_sha256"] for row in results}) == len(matrix["cells"])

validation_cell = matrix["cells"][0]
launcher_text = (RUNTIME / "benchmarks" / "kairos_formal_benchmark.sh").read_text()
function_start = launcher_text.index("run_formal_serve()")
snippet_start = launcher_text.index("import copy, json, pathlib, sys", function_start)
snippet_end = launcher_text.index("\nPY\n}", snippet_start)
snippet = launcher_text[snippet_start:snippet_end]
wrapper_output = RUN_DIR / f"wrapper-{validation_cell['tag']}.json"
materialized = copy.deepcopy({key: value for key, value in bundle.items() if key != "sha256"})
materialized.update(
    active_gpu_uuids=list(uuids),
    origin_gpu_uuids=list(uuids),
    calibration_match_mode="exact_uuid",
)
materialized = formal.seal(materialized)
wrapper_bundle = RUN_DIR / "wrapper-bundle.json"
wrapper_bundle.write_text(json.dumps(materialized, indent=2) + "\n")
command = [
    sys.executable,
    "-c",
    snippet,
    str(RUNTIME / "benchmarks"),
    str(RUNTIME / "configs" / "benchmarks" / "formal-mixed-v2.json"),
    "wan22",
    str(wrapper_bundle),
    "/shared/data/DITSERVE/datasets",
    os.environ["DITSERVE_MODEL_MANIFESTS"],
    str(wrapper_output),
    "--rate-rpm",
    str(validation_cell["rate_rpm"]),
    "--video-fraction",
    "0.5",
    "--slo-scale",
    "1.5",
    "--resolution-distribution",
    "uniform",
    "--pattern",
    "poisson",
    "--suite",
    "sweep",
]
completed = subprocess.run(
    command,
    env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
    capture_output=True,
    text=True,
)
if completed.returncode:
    raise RuntimeError("wrapper rate_rpm contract failed:\n" + completed.stdout + completed.stderr)
wrapper = json.loads(wrapper_output.read_text())
direct_cell = next(row for row in results if row["tag"] == validation_cell["tag"])
assert math.isclose(wrapper["benchmark"]["trace"]["manifest"]["rate_rpm"], validation_cell["rate_rpm"], rel_tol=1e-12)
assert wrapper["benchmark"]["trace"]["manifest"]["workload_sha256"] == direct_cell["workload_sha256"]
assert wrapper["benchmark"]["absolute_rate"]["requested_rate_rpm"] == validation_cell["rate_rpm"]
assert wrapper["benchmark"]["formal_protocol"]["calibration_match_mode"] == "exact_uuid"

result = {
    "schema": "ditserve.rate-rpm-contract.v1",
    "status": "passed",
    "commit": matrix["commit"],
    "protocol_sha256": matrix["protocol_sha256"],
    "bundle_sha256": bundle["sha256"],
    "source_anchor_workload_sha256": source_manifest["workload_sha256"],
    "wrapper_rate_rpm": validation_cell["rate_rpm"],
    "wrapper_workload_sha256": wrapper["benchmark"]["trace"]["manifest"]["workload_sha256"],
    "rate_points": len(results),
    "policies_per_point": len(fixed["policies"]),
    "cells": results,
    "scope": "CPU-only frozen-input config and trace contract; serving performance pending",
}
with (RUN_DIR / "rate-rpm-contract.json").open("x") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
print(json.dumps(result))
