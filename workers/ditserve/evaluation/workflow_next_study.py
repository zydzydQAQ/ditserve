"""Frozen run03 inputs. Preparation and selection never initialize CUDA."""

import copy
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import time

from .workflow_budget import atomic_json
from .workflow_serving_study import binding, digest

SCHEMA = "ditserve.workflow.next-study.v1"
CAPS = dict(calibration=1800, cpu=4500, mixed=7200, sp=9000,
            confirmation=3600, extension=1800, reserve=900)
DEFAULT_ROOT = Path("/mnt/ssd0/zyd/ditserve-data/workflow")
PROMPTS = [
    "A red bicycle leaning against a blue wall in warm afternoon light.",
    "A small sailboat glides across a calm lake in soft morning light.",
    "A ceramic teapot steaming on a wooden kitchen table.",
    "A paper lantern floating over a quiet night market.",
]


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def save(study):
    atomic_json(Path(study["root"]) / "study.json", study)


def container_path(study, path):
    return "/data/workflow/" + study["run_id"] + "/" + str(Path(path).relative_to(study["root"]))


def snapshot(repo, root):
    """Copy mutable development files; builds are added before sealing."""
    repo, root = Path(repo), Path(root)
    root.mkdir(parents=True, exist_ok=False)
    for name in ("workers", "configs", "docker", "tests", "src", "include", "cmd", "benchmarks"):
        shutil.copytree(repo / name, root / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
    shutil.copy2(repo / "CMakeLists.txt", root / "CMakeLists.txt")
    shutil.copytree(repo / ".cache/deps/nlohmann3.11.3", root / ".cache/deps/nlohmann3.11.3")


def seal(source):
    source = Path(source)
    files = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        # Intermediate CMake files are not executed by experiments.
        if ".build" in path.relative_to(source).parts:
            continue
        row = binding(path)
        row["path"] = str(path.relative_to(source))
        files.append(row)
    if not any(row["path"] == "bin/ditserve" for row in files):
        raise ValueError("seal requires the binary built from this snapshot")
    return dict(source=str(source), source_files=files, source_digest=digest(files))


def prepare(repo, root, image, inventory, revision, model_root):
    root, repo = Path(root), Path(repo)
    if root.parent.name != "workflow":
        raise ValueError("study root must be DATA_ROOT/workflow/RUN_ID")
    if (root / "study.json").exists():
        raise FileExistsError(root / "study.json")
    if len(inventory) != 2 or any("RTX 4090" not in row["name"] for row in inventory):
        raise ValueError("run03 requires the two verified RTX 4090 GPUs")
    models = []
    for name in ("wan2.1-t2v-1.3b", "sana-sprint-0.6b"):
        model_files = []
        for path in sorted((Path(model_root) / name).rglob("*")):
            if path.is_file():
                st = path.stat()
                row = dict(path=str(path), bytes=st.st_size, mtime_ns=st.st_mtime_ns)
                if st.st_size <= 16 * 1024**2:
                    row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                model_files.append(row)
        if not model_files or not (Path(model_root) / name / "model_index.json").is_file():
            raise ValueError("missing required local model: " + name)
        models.extend(model_files)
    study = dict(schema=SCHEMA, run_id=root.name, root=str(root), image=image,
                 source=revision["source"], source_digest=revision["source_digest"],
                 source_files=revision["source_files"], source_revisions=[revision],
                 created_unix=time.time(), budget_seconds=28800, family_caps_seconds=CAPS,
                 gpu_mapping=inventory, model_bindings=models, model_root=str(model_root),
                 status="prepared", active_job=None, jobs=[], decisions=[], selections={},
                 assumptions=dict(semantic="unverified", transport="host_file_v1",
                                  sp_threshold=0.05, warmups=1, steps=20,
                                  measurement_budget="cumulative task wall seconds; GPU seconds reported separately"))
    root.mkdir(parents=True, exist_ok=True)
    save(study)
    return study


def environment(study, devices=(0, 1)):
    env = read(Path(study["source"]) / "configs/env/rtx4090-2x24g.json")
    env["name"] = "run03-" + "-".join(map(str, devices))
    env["gpus"]["count"] = len(devices)
    env["gpus"]["uuids"] = [study["gpu_mapping"][d]["uuid"] for d in devices]
    env["cost_profile"] = None
    return env


def _worker(kind, rank, physical, cores):
    mask = list(range(60 * physical, 60 * physical + cores)) if kind != "sana" else list(range(60 * physical + 16, 60 * physical + 20))
    config = dict(cpu_affinity=mask, cpu_threads=len(mask))
    if kind == "wan":
        config["options"] = dict(execution="block")
    elif kind == "model":
        config["attention"] = "math"
    return dict(backend="sana" if kind == "sana" else "wan", gpu=rank,
                model="sana-sprint-0.6b" if kind == "sana" else "wan2.1-t2v-1.3b",
                config=config)


def explicit_study(template, devices=(0, 1), cores=8):
    value = copy.deepcopy(template)
    workers, expanded = {}, {}
    for name, worker in value["workers"].items():
        if worker.get("per_gpu"):
            expanded[name] = []
            for rank, physical in enumerate(devices):
                key = name + str(rank)
                item = copy.deepcopy(worker)
                item.pop("per_gpu")
                item["gpu"] = rank
                kind = "sana" if worker["backend"] == "sana" else "wan"
                controls = _worker(kind, rank, physical, cores)["config"]
                item.setdefault("config", {}).update({k: controls[k] for k in ("cpu_affinity", "cpu_threads")})
                workers[key] = item
                expanded[name].append(key)
        else:
            item = copy.deepcopy(worker)
            physical = devices[item.get("gpu", 0)]
            controls = _worker("sana" if worker["backend"] == "sana" else "wan", item.get("gpu", 0), physical, cores)["config"]
            item.setdefault("config", {}).update({k: controls[k] for k in ("cpu_affinity", "cpu_threads")})
            workers[name] = item
    value["workers"] = workers
    for phase in value["phases"]:
        if "workers" in phase:
            phase["workers"] = [actual for name in phase["workers"] for actual in expanded.get(name, [name])]
        if isinstance(phase.get("gpus"), dict):
            slot = phase["gpus"]
            phase["gpus"] = [dict(video=expanded[slot["video"]][i], image=expanded[slot["image"]][i])
                             for i in range(len(devices))]
        if "warmup" in phase:
            warmups = []
            for item in phase["warmup"]:
                for name in expanded.get(item["worker"], [item["worker"]]):
                    warmups.append(dict(item, worker=name))
            phase["warmup"] = warmups
    return value


def add_job(study, identifier, family, config, *, module=None, kind=None, devices=(0, 1),
            requires=(), estimate=600, timeout=1800, native=False, metadata=None):
    config = copy.deepcopy(config)
    if native:
        config["experiment"] = "run03-" + identifier
        for phase in config["phases"]:
            phase.setdefault("controller_cpu_affinity", [59])
            phase.setdefault("artifact_dependencies", [
                dict(closed=False, reason="run03 numerical diagnosis and review pending")])
        config["budget"] = dict(limit_gpu_seconds=timeout * len(devices),
                                watchdog_seconds=timeout, repair_reserve_gpu_seconds=0, carry="none")
    existing = [row for row in study["jobs"] if row.get("logical_id", row["id"]) == identifier
                and not row.get("superseded")]
    if existing:
        if len(existing) != 1:
            raise ValueError("multiple current blocks: " + identifier)
        row = existing[0]
        frozen = read(row["input"])
        if digest(frozen) != row["configuration_sha256"]:
            raise ValueError("frozen input contents changed: " + identifier)
        if native:
            frozen["experiment"] = config["experiment"]
        if (frozen != config or row["family"] != family or row["devices"] != list(devices)
                or row["requires"] != list(requires)):
            raise ValueError("cannot silently replace frozen block configuration: " + identifier)
        return row
    root = Path(study["root"])
    path, output = root / "inputs" / (identifier + ".json"), root / "measurements" / identifier
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or output.exists():
        raise FileExistsError("new task must not overwrite input/output: " + identifier)
    if native:
        env_path = root / "inputs" / (identifier + ".env.json")
        atomic_json(env_path, environment(study, devices))
        command = ["/workspace/bin/ditserve", "--root", "/data", "--package-root", "/workspace/workers",
                   "--python", "/opt/venv/bin/python", "run", "--config", container_path(study, path),
                   "--env", container_path(study, env_path), "--allow-gpu", "--run-id", identifier,
                   "--keep-raw",
                   "--work-dir", container_path(study, output),
                   "--results-root", container_path(study, root / "native-results"),
                   "--ledger", container_path(study, root / "native-ledgers" / (identifier + ".json"))]
    else:
        command = ["/opt/venv/bin/python", "-m", "ditserve.evaluation." + module]
        if kind:
            command += ["--kind", kind]
        command += ["--config", container_path(study, path), "--out", container_path(study, output)]
    atomic_json(path, config)
    row = dict(id=identifier, logical_id=identifier, block=identifier, family=family, input=str(path),
               output=str(output), configuration_sha256=digest(config), devices=list(devices),
               command=command, requires=list(requires), estimated_seconds=estimate,
               timeout_seconds=timeout, minimum_budget_seconds=120,
               source=study["source"], source_digest=study["source_digest"],
               input_bindings=[binding(path)], metadata=metadata or {})
    if native:
        row.update(command_type="native", environment_input=str(env_path),
                   gpu_mapping=[study["gpu_mapping"][d] for d in devices])
        row["input_bindings"].append(binding(env_path))
    study["jobs"].append(row)
    save(study)
    return row


def sp_diagnosis_jobs(study):
    """Independent controls; retain frozen inputs and never qualify elastic SP."""
    originals = [job for job in study["jobs"]
                 if job.get("logical_id", job["id"]) == "sp-diagnostic" and not job.get("superseded")]
    if len(originals) != 1:
        raise ValueError("SP diagnosis requires one frozen sp-diagnostic input")
    original = originals[0]
    base = read(original["input"])
    if digest(base) != original["configuration_sha256"]:
        raise ValueError("frozen input contents changed: sp-diagnostic")
    native = dict(base, variants=["cache_dit", "baseline"], extended_diagnostics=False,
                  localize_modes=[], timeout_seconds=2900)
    linear = dict(base, variants=["full_rows_linear", "no_reduced_precision"],
                  extended_diagnostics=True, localize_modes=["full_rows_linear", "no_reduced_precision"],
                  timeout_seconds=3500)
    # add_job compares the complete frozen configuration before reusing a job.
    # Its source binding stays intact even after the controller revision changes.
    return [
        add_job(study, "sp-native-control", "sp", native, module="workflow_serving_sp",
                requires=["acceptance"], estimate=1600, timeout=3000,
                metadata=dict(role="diagnostic_processor_control", deployment_candidate_mode="cache_dit",
                    diagnostic_control_mode="baseline",
                    purpose="Compare native pinned cache-dit processor against the previous custom baseline on unchanged FP16 recipe; not performance evidence",
                    previous_diagnostic="sp-diagnostic")),
        add_job(study, "sp-linear-controls", "sp", linear, module="workflow_serving_sp",
                requires=["acceptance"], estimate=1800, timeout=3600,
                metadata=dict(role="diagnostic_linear_controls", diagnostic_only=True,
                    performance_evidence=False, previous_diagnostic="sp-diagnostic",
                    purpose="Localize sequence-shape GEMM and FP16 reduction differences; does not qualify elastic SP")),
    ]


def sp_linear_replay_job(study, *, identifier="sp-linear-replay", candidates=None,
                         microbenchmark=None, estimate=90, timeout=360):
    """Bind completed reference captures for an independent, single-GPU replay."""
    from .workflow_next_report import current_job, evidence

    source_name = "sp-linear-controls"
    bound = {}

    def unavailable(reason):
        decision = dict(job=identifier, decision="diagnostic_source_unavailable",
                        source_job=source_name, reason=str(reason))
        if decision not in study["decisions"]:
            study["decisions"].append(decision)
        save(study)
        return None

    def bind(path):
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError("missing or symlinked replay source: " + str(path))
        before = path.stat()
        sha = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                sha.update(chunk)
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError("replay source changed while hashing: " + str(path))
        row = dict(path=str(path), bytes=after.st_size, sha256=sha.hexdigest())
        bound[str(path)] = row
        return row

    def data_file(path):
        row = bind(path)
        return dict(path=container_path(study, path), sha256=row["sha256"])

    try:
        source = current_job(study, source_name)
        ev = evidence(study, source)
        if not ev["guarded"] or ev["result"].get("status") not in ("completed", "qualification_failed"):
            return unavailable("source task has not completed with verified output")
        base = read(source["input"])
        if digest(base) != source["configuration_sha256"]:
            return unavailable("source configuration differs from its frozen digest")
        if base.get("seeds") != [3501, 3502] or base.get("frames") != [17, 33]:
            return unavailable("source does not cover the frozen two-seed, 17/33-frame matrix")
        bind(ev["path"])
        bind(source["input"])
        tensors = ("to_q_input", "to_q", "feed_forward_input",
                   "feed_forward_first_projection", "feed_forward_activation",
                   "feed_forward_second_projection")
        branches = ["cond", "uncond"]
        cases = []
        for seed in base["seeds"]:
            for frames in base["frames"]:
                captures = []
                for rank in (0, 1):
                    directory = (Path(source["output"]) / f"rank{rank}" / f"seed{seed}" / f"f{frames}" /
                                 "localization/reference/step00/block00-qkv")
                    path = directory / "capture.json"
                    manifest = read(path, {})
                    if (manifest.get("status") != "completed" or manifest.get("rank") != rank
                            or manifest.get("target_step") != 0 or manifest.get("target_block") != 0
                            or manifest.get("scheduler_steps") != 20
                            or manifest.get("scope") != "selected_block_qkv"
                            or manifest.get("comparison_layout") !=
                            "SP2 local sequence vs matching SP1 contiguous sequence shard"):
                        return unavailable("missing or incomplete reference capture: " + str(path))
                    rows = {row["key"]: row for row in manifest.get("layers", [])}
                    if len(rows) != len(manifest.get("layers", [])):
                        return unavailable("duplicate reference tensor keys: " + str(path))
                    files = {}
                    for branch in branches:
                        for tensor in tensors:
                            key = f"{branch}.block00.{tensor}"
                            row = rows.get(key, {})
                            array = directory / (key + ".npy")
                            if (row.get("path") != container_path(study, array)
                                    or row.get("block") != 0 or row.get("branch") != branch
                                    or row.get("tensor") != tensor
                                    or not row.get("hash", {}).get("sha256")):
                                return unavailable("missing or mismatched reference tensor: " + str(array))
                            files[key] = data_file(array)
                    captures.append(dict(rank=rank, **data_file(path), files=files))
                cases.append(dict(seed=seed, frames=frames, captures=captures))
        transformer = Path(study["model_root"]) / "wan2.1-t2v-1.3b/transformer"
        index_path = transformer / "diffusion_pytorch_model.safetensors.index.json"
        index = read(index_path, {})
        modules = ("blocks.0.attn1.to_q", "blocks.0.ffn.net.0.proj", "blocks.0.ffn.net.2")
        tensor_names = [module + "." + suffix for module in modules for suffix in ("weight", "bias")]
        shards = {index["weight_map"][name] for name in tensor_names}
        if any(Path(name).name != name for name in shards):
            return unavailable("model index contains an unsafe shard path")
        model_index = dict(path="/models/wan2.1-t2v-1.3b/transformer/" + index_path.name,
                           sha256=bind(index_path)["sha256"])
        model_shards = {name: dict(path="/models/wan2.1-t2v-1.3b/transformer/" + name,
                                  sha256=bind(transformer / name)["sha256"]) for name in sorted(shards)}
        config = dict(schema="ditserve.workflow.linear-replay-config.v1", device="cuda:0",
                      gpu_uuid=study["gpu_mapping"][1]["uuid"], cpu_affinity=list(range(60, 68)),
                      cpu_threads=8, repeats=3, zero_padded=True,
                      operators=["to_q", "ffn.net.0.proj", "ffn.net.2"], branches=branches,
                      cases=cases, model_index=model_index, model_shards=model_shards)
        if candidates is not None:
            config["candidates"] = list(candidates)
        if microbenchmark is not None:
            config["microbenchmark"] = copy.deepcopy(microbenchmark)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return unavailable(error)
    existing = any(job.get("logical_id", job["id"]) == identifier and not job.get("superseded")
                   for job in study["jobs"])
    job = add_job(study, identifier, "sp", config, module="workflow_linear_replay",
                  devices=(1,), requires=["acceptance"], estimate=estimate, timeout=timeout,
                  metadata=dict(role="diagnostic_linear_replay", diagnostic_only=True,
                      performance_evidence=False, deployment_qualification=False,
                      source_job=source["id"], source_digest=source["source_digest"],
                      source_configuration_sha256=source["configuration_sha256"],
                      source_output_sha256=ev["launch"]["output_sha256"], model_tensors=tensor_names,
                      purpose="Replay captured native SP1 Linear operands; does not qualify elastic SP"))
    bindings = [binding(job["input"])] + [bound[path] for path in sorted(bound)]
    if existing and job["input_bindings"] != bindings:
        raise ValueError("cannot silently replace frozen replay source bindings")
    job["input_bindings"] = bindings
    save(study)
    return job


def initial_jobs(study):
    root = Path(study["source"])
    acceptance = explicit_study(read(root / "configs/studies/acceptance.json"))
    add_job(study, "acceptance", "calibration", acceptance, native=True, estimate=360, timeout=1500)
    cost = explicit_study(read(root / "configs/studies/cost_profile.json"), devices=(1,))
    # Only the two shapes admitted to this round. Preserve complete state groups.
    for phase in cost["phases"]:
        # Memory reset/sample actions belong to the surrounding shape too.
        actions, current_group = [], []
        for action in phase["actions"]:
            if action["op"] == "memory" and action.get("reset_peak") and current_group:
                if not any("81" in a.get("state_id", "") for a in current_group):
                    actions.extend(current_group)
                current_group = []
            current_group.append(action)
        if not any("81" in a.get("state_id", "") for a in current_group):
            actions.extend(current_group)
        phase["actions"] = actions
    add_job(study, "cost", "calibration", cost, devices=(1,), native=True,
            requires=["acceptance"], estimate=150, timeout=900)
    add_job(study, "sp-diagnostic", "sp",
            dict(model_path="/models/wan2.1-t2v-1.3b", devices=["cuda:0", "cuda:1"],
                 cpu_threads=8, cpu_masks={"cuda:0": list(range(8)), "cuda:1": list(range(60, 68))},
                 frames=[17, 33], steps=20, seeds=[3501, 3502], seed=3501, repeats=1,
                 variants=["baseline"], group_timeout_seconds=60, timeout_seconds=2900, prefixes=[1, 4, 20],
                 extended_diagnostics=True, localize_modes=["baseline"]),
            module="workflow_serving_sp", requires=["acceptance"], estimate=1200, timeout=3000)
    # Each screening job has all six arms in independent children. Reversing
    # the order on repeat 2 avoids confounding core count with time.
    for repeat, seed in enumerate((3101, 3102)):
        arms = [(c, d) for c in (8, 16) for d in (0, 1, 2)]
        if repeat:
            arms.reverse()
        sequence = []
        for cores, depth in arms:
            sequence.append(dict(id=f"c{cores}-d{depth}",
                module="workflow_serving_encoder", kind="prefetch",
                config=dict(model_path="/models/wan2.1-t2v-1.3b", device="cuda:1",
                            cpu_mask=list(range(60, 60 + cores)), cpu_threads=cores,
                            frames=[17, 33], steps=20, seed=seed, requests=4,
                            prompt_cases=["shortenglish", "chinese", "truncation"],
                            repeats=1, warmups=1, depth=depth)))
        add_job(study, f"cpu-screen-r{repeat}", "cpu", dict(kind="sequence", arms=sequence),
                module="workflow_next_job", devices=(1,), requires=["acceptance"],
                estimate=1300, timeout=2400, metadata=dict(repeat=repeat, seed=seed))
    elastic_gate_jobs(study)
    save(study)


def profile_from_cost(study):
    from .cost_profile import measured_classes
    job = next(row for row in study["jobs"] if row["logical_id"] == "cost" and not row.get("superseded"))
    path = Path(job["output"]) / "run.json"
    record = read(path, {})
    if record.get("status") != "completed":
        raise ValueError("complete measured cost profile required")
    classes = measured_classes(path, steps=20)
    if set(classes) != {"wan1.3b-480x832x17", "wan1.3b-480x832x33", "sana-1024"}:
        raise ValueError("cost profile missing an admitted class")
    prior = read(Path(study["source"]) / "configs/profiles/rtx4090-sp1-local.json")
    return dict(provenance="measured", classes=classes, links=prior["links"],
                sp_efficiency=1., resize_ms=0.,
                source=dict(cost=binding(path), links=prior["source"],
                            scope="new SP1 costs; inherited separately measured link, no SP efficiency measurement"))


def service_ms(profile, name, include_encode=True):
    c = profile["classes"][name]
    if c["kind"] == "image":
        return c["service_ms"]
    return c.get("prepare_ms", 0) + 20 * c["step_compute_ms"] + c["finish_ms"] + (c.get("encode_ms", 0) if include_encode else 0)


def trajectory(profile, seed, *, count=40, videos_only=False, pattern="poisson",
               load=.8, gpus=2, saturation=False, lambda0=None):
    """Generate once; both arms consume the exact same immutable list."""
    rng = random.Random(seed)
    if videos_only:
        names = ["wan1.3b-480x832x17", "wan1.3b-480x832x33"] * (count // 2)
    elif count == 16:
        names = ["sana-1024"] * 12 + ["wan1.3b-480x832x17"] * 2 + ["wan1.3b-480x832x33"] * 2
    else:
        if count % 10:
            raise ValueError("mixed workload count must be divisible by ten")
        names = ["sana-1024"] * (count * 8 // 10) + ["wan1.3b-480x832x17"] * (count // 10) + ["wan1.3b-480x832x33"] * (count // 10)
    rng.shuffle(names)
    capacity = lambda0 or gpus * 1000 / statistics.mean(service_ms(profile, name) for name in names)
    active_clock = 0.
    rows = []
    for i, name in enumerate(names):
        if i and not saturation:
            active_clock += rng.expovariate(load * capacity) * 1000 * (.2 if pattern == "bursty" else 1.)
        arrival = (math.floor(active_clock / 16000) * 80000 + active_clock % 16000) if pattern == "bursty" else active_clock
        spec = dict(prompt=PROMPTS[i % len(PROMPTS)], seed=seed * 100 + i)
        kind = profile["classes"][name]["kind"]
        if kind == "video":
            frames = 17 if name.endswith("x17") else 33
            spec.update(height=480, width=832, frames=frames, steps=20, guidance_scale=5., negative_prompt="")
        rows.append(dict(request_id=f"s{seed}-r{i}", kind=kind, **{"class": name},
                         arrival_ms=arrival, deadline_ms=arrival + (2. if videos_only else 2.5) * service_ms(profile, name),
                         spec=spec))
    return rows


def cluster_config(profile, rows, arms, *, cores=8, depth=1, devices=(0, 1), elastic=False):
    workers = {}
    for rank, physical in enumerate(devices):
        for kind in ("wan", "sana"):
            workers[kind + str(rank)] = _worker(kind, rank, physical, cores)
    phases = []
    for arm in arms:
        arm = dict(arm)
        label = arm.pop("id")
        policy = arm.pop("policy", "static_sp1")
        arm_cores, arm_depth = arm.pop("cores", cores), arm.pop("depth", depth)
        # CPU-count comparisons use per-arm worker definitions, applied by
        # the native worker_overrides extension before any process is started.
        overrides = {}
        for rank, physical in enumerate(devices):
            overrides["wan" + str(rank)] = _worker("wan", rank, physical, arm_cores)["config"]
        phase = dict(id=label, kind="cluster",
                     gpus=[dict(video="wan" + str(i), image="sana" + str(i)) for i in range(len(devices))],
                     policy=policy, options=dict(quantum_ms=arm.pop("quantum_ms", 300),
                                                late_short_ms=arm.pop("late_short_ms", 2000),
                                                chunked_decode=arm.pop("chunked_decode", True), **arm),
                     profile=copy.deepcopy(profile), workload=copy.deepcopy(rows),
                     text_prefetch_depth=arm_depth, output_mode="check",
                     worker_overrides=overrides, simulate=not elastic,
                     admission_wait_seconds=0, admission_poll_seconds=1,
                     gates=[dict(metric="/cells/0/metrics/completed", op="eq", value=len(rows), scope="performance"),
                            dict(metric="/cells/0/metrics/failed", op="eq", value=0, scope="performance")],
                     warmup=[dict(worker="wan" + str(i), spec=dict(prompt=PROMPTS[0], seed=7,
                         height=480, width=832, frames=33, steps=2, guidance_scale=5., negative_prompt=""),
                         quantum="step") for i in range(len(devices))] +
                     [dict(worker="sana" + str(i), spec=dict(prompt=PROMPTS[0], seed=7)) for i in range(len(devices))])
        if elastic:
            phase["real_sp"] = dict(enabled=True, degrees=[1, 2], transport="host_file_v1", timeout_seconds=60)
            phase["note"] = "Mechanism study: SP efficiency and host_file migration scheduler costs are uncalibrated; no optimal scheduling claim."
        phases.append(phase)
    return dict(schema="ditserve.study.v1", experiment="run03", workers=workers, phases=phases)


def measured_jobs(study, profile):
    # Separate baseline saturated workloads calibrate both workload families.
    for videos_only, label, count in ((True, "video", 8), (False, "mixed", 20)):
        rows = trajectory(profile, 3000, count=count, videos_only=videos_only, saturation=True)
        config = cluster_config(profile, rows, [dict(id=f"repeat{r}", policy="static_sp1") for r in range(3)])
        add_job(study, "calibration-" + label, "calibration", config, native=True,
                requires=["cost"], estimate=300, timeout=1200, metadata=dict(calibration=label))


def calibration_capacity(result):
    values = []
    for phase in result.get("phases", []):
        for cell in phase.get("cells", []):
            rows = cell.get("requests", [])
            if rows and all(row.get("status") == "completed" for row in rows) and cell.get("elapsed_ms", 0) > 0:
                values.append(len(rows) * 1000 / cell["elapsed_ms"])
    if len(values) != 3:
        raise ValueError("three complete saturation observations required")
    return statistics.median(values)


def native_followups(study, profile, capacities):
    chosen_depth = study["selections"].get("screen_depth", {}).get("depth", 1)
    for contrast in ("depth", "cores"):
        for repeat, seed in enumerate((3201, 3202, 3203)):
            arms = ([dict(id="reference", cores=8, depth=1), dict(id="candidate", cores=8, depth=2)]
                    if contrast == "depth" else
                    [dict(id="reference", cores=8, depth=chosen_depth),
                     dict(id="candidate", cores=16, depth=chosen_depth)])
            if repeat % 2:
                arms.reverse()
            rows = trajectory(profile, seed, count=16, videos_only=True, lambda0=capacities["video"])
            add_job(study, f"cpu-online-{contrast}-r{repeat}", "cpu",
                    cluster_config(profile, rows, arms), native=True,
                    requires=["calibration-video", "cpu-screen-r0", "cpu-screen-r1"],
                    estimate=550, timeout=1800,
                    metadata=dict(comparison="cpu-" + contrast, repeat=repeat, trace_sha256=digest(rows)))


def quantum_screen_job(study, profile, capacities):
    cpu = study["selections"].get("cpu", dict(cores=8, depth=1))
    rows = trajectory(profile, 3300, count=16, pattern="bursty", gpus=1,
                      lambda0=capacities["mixed"] / 2)
    add_job(study, "quantum-screen", "mixed", cluster_config(profile, rows,
            [dict(id="step"), dict(id="block100", policy="block_elastic", quantum_ms=100),
             dict(id="block300", policy="block_elastic", quantum_ms=300)], devices=(1,),
             cores=cpu["cores"], depth=cpu["depth"]),
            native=True, devices=(1,), requires=["calibration-mixed"], estimate=500, timeout=1800)


def mixed_jobs(study, profile, capacity, quantum, *, confirmation=False):
    cpu = study["selections"].get("cpu", dict(cores=8, depth=1))
    seeds = (3401, 3402) if confirmation else (3301, 3302, 3303)
    for pattern in ("poisson", "bursty"):
        for repeat, seed in enumerate(seeds):
            rows = trajectory(profile, seed, pattern=pattern, lambda0=capacity)
            arms = [dict(id="step"), dict(id="block", policy="block_elastic", quantum_ms=quantum)]
            if repeat % 2:
                arms.reverse()
            add_job(study, f"{'confirm' if confirmation else 'mixed'}-{pattern}-s{seed}",
                    "confirmation" if confirmation else "mixed",
                    cluster_config(profile, rows, arms, cores=cpu["cores"], depth=cpu["depth"]),
                    native=True, requires=["quantum-screen"], estimate=550, timeout=1800,
                    metadata=dict(comparison="mixed", pattern=pattern, seed=seed,
                                  confirmation=confirmation, trace_sha256=digest(rows)))


def ablation_jobs(study, profile, capacity, quantum):
    cpu = study["selections"].get("cpu", dict(cores=8, depth=1))
    rows = trajectory(profile, 3304, count=16, pattern="bursty", lambda0=capacity)
    for name, change in (("late-short", dict(late_short_ms=0)),
                         ("decode", dict(chunked_decode=False))):
        arms = [dict(id="reference", policy="block_elastic", quantum_ms=quantum),
                dict(id="candidate", policy="block_elastic", quantum_ms=quantum, **change)]
        add_job(study, "ablation-" + name, "mixed",
                cluster_config(profile, rows, arms, cores=cpu["cores"], depth=cpu["depth"]),
                native=True, requires=["mixed-bursty-s3301", "mixed-bursty-s3302", "mixed-bursty-s3303"],
                estimate=350, timeout=1200, metadata=dict(comparison="ablation", variable=name,
                                                        trace_sha256=digest(rows)))


def elastic_gate_jobs(study):
    for stage, requires in (("transfer", ["acceptance"]), ("pause", ["sp-diagnostic", "elastic-transfer"]),
                            ("resize", ["elastic-pause"]), ("interleave", ["elastic-resize"])):
        add_job(study, "elastic-" + stage, "sp",
                dict(stage=stage, model_path="/models/wan2.1-t2v-1.3b",
                     devices=["cuda:0", "cuda:1"], frames=[17, 33], seeds=[3501, 3502], steps=20,
                     schedule=[dict(step=5, degree=2), dict(step=10, degree=1)], group_timeout_seconds=60,
                     timeout_seconds=1400, pause_blocks=[23, 14, 23], cpu_threads=8,
                     cpu_masks={"cuda:0": list(range(8)), "cuda:1": list(range(60, 68))}),
                module="workflow_elastic_validation", requires=requires, estimate=600, timeout=1500)


def elastic_jobs(study, profile, capacity):
    elastic_gate_jobs(study)
    rows = trajectory(profile, 3503, count=16, pattern="bursty", lambda0=capacity)
    cpu = study["selections"].get("cpu", dict(cores=8, depth=1))
    quantum = study["selections"].get("quantum", {}).get("quantum_ms", 300)
    add_job(study, "elastic-online", "sp", cluster_config(profile, rows,
            [dict(id="step", policy="step_elastic"),
             dict(id="step-reserved", policy="step_elastic", reserve_image_gpus=1),
             dict(id="block", policy="block_elastic", quantum_ms=quantum)],
             elastic=True, cores=cpu["cores"], depth=cpu["depth"]),
            native=True, requires=["elastic-interleave"], estimate=600, timeout=1800,
            metadata=dict(comparison="elastic", trace_sha256=digest(rows)))


def elastic_repeats(study, profile, capacity):
    cpu = study["selections"].get("cpu", dict(cores=8, depth=1))
    quantum = study["selections"].get("quantum", {}).get("quantum_ms", 300)
    for index, seed in enumerate((3504, 3505)):
        rows = trajectory(profile, seed, count=16, pattern="bursty", lambda0=capacity)
        arms = [dict(id="step", policy="step_elastic"),
                dict(id="step-reserved", policy="step_elastic", reserve_image_gpus=1),
                dict(id="block", policy="block_elastic", quantum_ms=quantum)]
        arms = arms[1:] + arms[:1] if index == 0 else arms[2:] + arms[:2]
        add_job(study, f"elastic-online-s{seed}", "sp",
                cluster_config(profile, rows, arms, elastic=True, cores=cpu["cores"], depth=cpu["depth"]),
                native=True, requires=["elastic-online"], estimate=600, timeout=1800,
                metadata=dict(comparison="elastic", seed=seed, trace_sha256=digest(rows)))


def extension_jobs(study):
    common = dict(model_path="/models/wan2.1-t2v-1.3b", device="cuda:1",
                  cpu_threads=8, cpu_masks={"cuda:1": list(range(60, 68))},
                  height=480, width=832, steps=20, seed=3601, frame_pairs=[[17, 17]],
                  reuse_streams=True, trim_after_references=True, release_b_before_a_decode=True)
    add_job(study, "overlap-lifecycle", "extension",
            dict(common, repeats=1, memory_diagnostics=True), module="workflow_serving_overlap",
            devices=(1,), requires=["acceptance"], estimate=200, timeout=600)
    add_job(study, "overlap-timing", "extension",
            dict(common, repeats=3, memory_diagnostics=False), module="workflow_serving_overlap",
            devices=(1,), requires=["overlap-lifecycle"], estimate=400, timeout=900)
    row = add_job(study, "overlap-trace", "extension",
                  dict(common, repeats=1, memory_diagnostics=False), module="workflow_serving_overlap",
                  devices=(1,), requires=["overlap-timing"], estimate=250, timeout=900)
    if not row.get("profiling"):
        root = Path(study["root"])
        (root / "traces").mkdir(exist_ok=True)
        capture = "ditserve.workflow/P4/stage_pair|" + json.dumps(dict(
            frames=17, mode="serial", peer_frames=17, purpose="timed", repeat=0,
            request_id="P4-timed-a17-b17-r0-serial", variant="P4"), sort_keys=True, separators=(",", ":"))
        row["command"] = [
            "/data/tools/nsight-workflow/extracted/opt/nvidia/nsight-systems/2024.6.2/target-linux-x64/nsys",
            "profile", "--trace=cuda,nvtx,osrt", "--sample=none", "--cpuctxsw=none",
            "--capture-range=nvtx", "--capture-range-end=none", "--kill=none",
            "--env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0", "--nvtx-capture=" + capture,
            "--gpu-metrics-devices=1", "--gpu-metrics-frequency=1000",
            "--output", container_path(study, root / "traces" / row["id"]),
        ] + row["command"]
        row.update(profiling=True, counters=True)
    # If P4 cannot retain the required headroom, use one bounded SP1
    # capacity observation. This never certifies online co-residency or speedup.
    actions = []
    for index in range(2):
        sid = f"capacity{index}"
        actions.append(dict(op="prepare", state_id=sid, spec=dict(prompt=PROMPTS[index], seed=3601 + index,
            height=480, width=832, frames=81, steps=20, guidance_scale=5., negative_prompt="")))
        actions.extend(dict(op="advance", state_id=sid, quantum="step") for _ in range(20))
        actions.extend([dict(op="finish", state_id=sid, output_mode="check"),
                        dict(op="release", state_id=sid), dict(op="memory", label="capacity")])
    config = dict(schema="ditserve.study.v1", workers={"wan": _worker("wan", 0, 1, 8)},
                  phases=[dict(id="capacity", kind="operations", worker="wan", actions=actions,
                               admission_wait_seconds=0)])
    config["workers"]["wan"]["config"]["memory_fraction"] = .95
    add_job(study, "capacity-81", "extension", config, native=True, devices=(1,),
            requires=["acceptance"], estimate=400, timeout=900,
            metadata=dict(scope="isolated Wan capacity only, memory fraction .95; no co-residency, throughput or A100 scaling claim"))
    save(study)


def retry_block(study, identifier):
    if study.get("active_job") or read(Path(study["root"]) / "budget.json", {}).get("active"):
        raise RuntimeError("cannot retry an active study")
    old = next(row for row in study["jobs"] if row["id"] == identifier and not row.get("superseded"))
    stem = old["logical_id"]
    attempt = 2
    while any(row["id"] == f"{stem}-attempt{attempt}" for row in study["jobs"]):
        attempt += 1
    new_id = f"{stem}-attempt{attempt}"
    config = read(old["input"])
    # Rewrite only this job's artifact namespace, never old evidence.
    config = json.loads(json.dumps(config).replace("/measurements/" + identifier + "/",
                                                   "/measurements/" + new_id + "/"))
    if old.get("command_type") == "native":
        new = add_job(study, new_id, old["family"], config, native=True, devices=old["devices"],
                      requires=old["requires"], estimate=old["estimated_seconds"], timeout=old["timeout_seconds"],
                      metadata=old["metadata"])
    else:
        module = old["command"][old["command"].index("-m") + 1].split(".")[-1]
        kind = old["command"][old["command"].index("--kind") + 1] if "--kind" in old["command"] else None
        new = add_job(study, new_id, old["family"], config, module=module, kind=kind, devices=old["devices"],
                      requires=old["requires"], estimate=old["estimated_seconds"], timeout=old["timeout_seconds"],
                      metadata=old["metadata"])
    new["logical_id"], old["superseded"] = stem, new_id
    if old.get("profiling"):
        prefix = old["command"][:old["command"].index("/opt/venv/bin/python")]
        new["command"] = [value.replace("/traces/" + identifier, "/traces/" + new_id)
                          for value in prefix] + new["command"]
        new.update(profiling=True, counters=True)
    save(study)
    return new
