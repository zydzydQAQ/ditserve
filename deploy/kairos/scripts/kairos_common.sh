# Sourced by the Kairos acceptance scripts; runs inside the fixed runtime image (venv and toolchain).
set -euo pipefail
: "${DITSERVE_RUN_ID:?set DITSERVE_RUN_ID}"
OUTPUT=/shared/output/DITSERVE
DATA=/shared/data/DITSERVE
MODELS=$DATA/models
# Wan2.2-TI2V-5B is the default. wan selects the separately verified Wan2.1-T2V-1.3B snapshot.
WAN_MODEL=${WAN_MODEL:-wan2.2-ti2v-5b}
case "$WAN_MODEL" in
  wan) WAN_SUFFIX= ;;
  wan2.2-ti2v-5b) WAN_SUFFIX=-5b ;;
  *) echo "unknown WAN_MODEL $WAN_MODEL"; exit 1 ;;
esac
CODE=/shared/code/DITSERVE
PACKAGE=/opt/ditserve/share/ditserve
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
RUN_DIR=$OUTPUT/acceptance/$DITSERVE_RUN_ID
# Jobsrun as root; keep outputs removable by the login user.
umask 0000
# Every job selects the write-once build published by tools/deploy_build.sh; never fall back to
# an image's built-in source. If deploy_build.sh staged .src-<commit> for in-cluster compilation,
# the first job container compiles C++, runs ctest + doctor in /tmp, and atomically publishes builds/<commit>.
: "${DITSERVE_BUILD:?set DITSERVE_BUILD to a published commit}"
BUILD=$CODE/builds/$DITSERVE_BUILD
if [ ! -f "$BUILD/BUILD_INFO.json" ]; then
  SRC_STAGE=$CODE/builds/.src-$DITSERVE_BUILD
  if [ -f "$SRC_STAGE/SOURCE_INFO.json" ]; then
    echo "== lazy cluster compile for DITSERVE_BUILD=$DITSERVE_BUILD on $(hostname)"
    python3 - "$CODE/builds" "$DITSERVE_BUILD" "${DITSERVE_EXECUTION_IMAGE:-}" <<'PY'
import datetime, hashlib, json, os, pathlib, shutil, subprocess, sys, time

builds_root = pathlib.Path(sys.argv[1]).resolve()
short = sys.argv[2]
exec_image = sys.argv[3]
target_dir = builds_root / short
info_file = target_dir / "BUILD_INFO.json"
src_dir = builds_root / f".src-{short}"
lock_dir = builds_root / f".lock-{short}"

def check_shared_storage():
    out = subprocess.check_output(["df", "-BG", "/shared"], text=True).splitlines()[-1].split()
    used, pct = int(out[2].rstrip("G")), int(out[4].rstrip("%"))
    if used > 805 or pct >= 81:
        raise RuntimeError(f"shared storage stop line reached before build: used={used}G pct={pct}%")

deadline = time.monotonic() + 300.0
acquired = False
while not info_file.is_file():
    try:
        os.mkdir(lock_dir)
        acquired = True
        break
    except FileExistsError:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for concurrent cluster build lock {lock_dir}")
        time.sleep(2.0)

if not acquired:
    sys.exit(0)

pid = os.getpid()
build_tmp = pathlib.Path(f"/tmp/ditserve-build-{short}-{pid}")
install_tmp = pathlib.Path(f"/tmp/ditserve-install-{short}-{pid}")
nfs_tmp = builds_root / f".tmp-{short}-{pid}"

try:
    if info_file.is_file():
        sys.exit(0)
    check_shared_storage()
    src_info = json.loads((src_dir / "SOURCE_INFO.json").read_bytes())
    commit = src_info["commit"]
    subject = src_info["subject"]
    runtime_image = exec_image or src_info.get("runtime_image", "")
    if not commit.startswith(short):
        raise ValueError(f"staged SOURCE_INFO commit {commit} does not match {short}")
    json_header = src_dir / ".cache/deps/nlohmann3.11.3/nlohmann/json.hpp"
    if hashlib.sha256(json_header.read_bytes()).hexdigest() != src_info["json_header_sha256"]:
        raise ValueError("staged nlohmann/json.hpp SHA256 mismatch")

    build_tmp.mkdir(parents=True, exist_ok=True)
    install_tmp.mkdir(parents=True, exist_ok=True)
    jobs = str(min(8, max(2, os.cpu_count() or 4)))
    builder_name = "cluster-cmake-install"

    cmake_bin = shutil.which("cmake")
    ctest_bin = shutil.which("ctest")
    cmake_ok = False
    if cmake_bin and ctest_bin:
        cfg = subprocess.run(
            [cmake_bin, "-S", str(src_dir), "-B", str(build_tmp), "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_TESTING=ON"],
            capture_output=True, text=True
        )
        if cfg.returncode == 0:
            subprocess.run([cmake_bin, "--build", str(build_tmp), "--parallel", jobs], check=True)
            subprocess.run([ctest_bin, "--test-dir", str(build_tmp), "--output-on-failure"], check=True)
            subprocess.run([cmake_bin, "--install", str(build_tmp), "--prefix", str(install_tmp), "--strip"], check=True)
            cmake_ok = True

    if not cmake_ok:
        builder_name = "cluster-gxx-install"
        gxx = shutil.which("g++")
        if not gxx:
            raise RuntimeError("neither cmake nor g++ available in container")
        shim_inc = build_tmp / "shim_include"
        if not pathlib.Path("/usr/include/openssl/sha.h").is_file():
            (shim_inc / "openssl").mkdir(parents=True, exist_ok=True)
            (shim_inc / "openssl" / "sha.h").write_text(
                "#pragma once\n#include <cstddef>\nextern \"C\" {\n"
                "#define SHA_LBLOCK 16\n#define SHA256_CBLOCK (SHA_LBLOCK*4)\n#define SHA256_DIGEST_LENGTH 32\n"
                "typedef unsigned int SHA_LONG;\n"
                "typedef struct SHA256state_st { SHA_LONG h[8]; SHA_LONG Nl, Nh; SHA_LONG data[SHA_LBLOCK]; unsigned int num, md_len; } SHA256_CTX;\n"
                "int SHA256_Init(SHA256_CTX *c);\nint SHA256_Update(SHA256_CTX *c, const void *data, size_t len);\n"
                "int SHA256_Final(unsigned char *md, SHA256_CTX *c);\n"
                "unsigned char *SHA256(const unsigned char *d, size_t n, unsigned char *md);\n}\n"
            )
        libcrypto = "-lcrypto"
        for cand in ("/usr/lib/x86_64-linux-gnu/libcrypto.so", "/usr/lib/x86_64-linux-gnu/libcrypto.so.3", "/lib/x86_64-linux-gnu/libcrypto.so.3"):
            if pathlib.Path(cand).exists():
                libcrypto = cand
                break

        inc_flags = [
            "-I", str(src_dir / "include"),
            "-I", str(src_dir / "benchmarks/include"),
            "-isystem", str(src_dir / ".cache/deps/nlohmann3.11.3"),
        ]
        if shim_inc.exists():
            inc_flags.extend(["-isystem", str(shim_inc)])
        cxx_flags = [
            "-std=c++17", "-O3", "-DNDEBUG", "-Wall", "-Wextra", "-Wpedantic", "-ffp-contract=off", "-pthread",
            f'-DDITSERVE_PROJECT_ROOT="{src_dir}"', '-DDITSERVE_VERSION="0.4.0"',
            f'-DCONTROL_BASELINE_PATH="{src_dir}/tests/fixtures/control-baseline.json"',
        ]

        core_srcs = [
            "src/ditserve/types.cpp",
            "src/ditserve/runtime/random.cpp",
            "src/ditserve/runtime/process.cpp",
            "src/ditserve/control/budget.cpp",
            "src/ditserve/control/dispatch.cpp",
            "src/ditserve/control/cluster.cpp",
            "src/ditserve/control/policy.cpp",
        ]
        bench_src = "benchmarks/src/cluster_sim.cpp"
        app_srcs = ["src/ditserve/control/study.cpp", "src/ditserve/interface/command.cpp"]

        objs = {}
        for rel in [*core_srcs, bench_src, *app_srcs, "cmd/main.cpp"]:
            obj = build_tmp / (rel.replace("/", "_") + ".o")
            subprocess.run([gxx, *cxx_flags, *inc_flags, "-c", str(src_dir / rel), "-o", str(obj)], check=True)
            objs[rel] = str(obj)

        core_objs = [objs[r] for r in core_srcs]
        bench_objs = [*core_objs, objs[bench_src]]
        app_objs = [*bench_objs, *(objs[r] for r in app_srcs)]

        bin_dir = install_tmp / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        ditserve_bin = bin_dir / "ditserve"
        subprocess.run([gxx, *cxx_flags, objs["cmd/main.cpp"], *app_objs, libcrypto, "-pthread", "-o", str(ditserve_bin)], check=True)
        strip_bin = shutil.which("strip")
        if strip_bin:
            subprocess.run([strip_bin, str(ditserve_bin)], check=True)
        (bin_dir / "ditserve-core").symlink_to("ditserve")

        tests = [
            ("supervision-tests", "tests/core/test_supervision.cpp", core_objs, []),
            ("control-tests", "tests/core/test_control.cpp", core_objs, []),
            ("command-tests", "tests/core/test_command.cpp", app_objs, [libcrypto]),
            ("simulation-tests", "benchmarks/tests/test_cluster.cpp", bench_objs, []),
            ("controller-decisions-tests", "benchmarks/tests/test_controller_decisions.cpp", core_objs, []),
        ]
        for tname, trel, tobjs, tlibs in tests:
            texe = build_tmp / tname
            subprocess.run([gxx, *cxx_flags, *inc_flags, str(src_dir / trel), *tobjs, *tlibs, "-pthread", "-o", str(texe)], check=True)
            subprocess.run([str(texe)], check=True)

        share_root = install_tmp / "share" / "ditserve"
        py_root = share_root / "python"
        py_root.mkdir(parents=True, exist_ok=True)

        def copy_filtered(s: pathlib.Path, d: pathlib.Path, suffixes: tuple[str, ...], extra_names: tuple[str, ...] = ()):
            for p in sorted(s.rglob("*")):
                if "__pycache__" in p.parts or not p.is_file():
                    continue
                if p.suffix in suffixes or p.name in extra_names:
                    t = d / p.relative_to(s)
                    t.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, t)

        copy_filtered(src_dir / "workers" / "ditserve", py_root / "ditserve", (".py",))
        copy_filtered(src_dir / "workers" / "xfuser", py_root / "xfuser", (".py",), ("LICENSE.txt",))
        copy_filtered(src_dir / "workers" / "gfc", py_root / "gfc", (".py",), ("LICENSE", "VENDORED.md"))
        for folder in ("include", "src", "cmd", "benchmarks"):
            copy_filtered(src_dir / folder, share_root / folder, (".cpp", ".hpp", ".py"), ("CMakeLists.txt",))
        shutil.copy2(src_dir / "CMakeLists.txt", share_root / "CMakeLists.txt")
        shutil.copytree(src_dir / "configs", share_root / "configs")

    # Container runtime smoke check
    smoke_env = os.environ.copy()
    smoke_env["PYTHONDONTWRITEBYTECODE"] = "1"
    smoke_env["DITSERVE_PACKAGE_ROOT"] = str(install_tmp / "share/ditserve/python")
    subprocess.run([str(install_tmp / "bin/ditserve"), "--version"], env=smoke_env, check=True)
    subprocess.run([str(install_tmp / "bin/ditserve"), "doctor", "--python", "/opt/venv/bin/python"],
                   env=smoke_env, stdout=subprocess.DEVNULL, check=True)

    core_sha = hashlib.sha256((install_tmp / "bin/ditserve").read_bytes()).hexdigest()
    all_files = [p for p in install_tmp.rglob("*") if p.is_file()]
    built_utc = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    build_info = {
        "commit": commit,
        "subject": subject,
        "built_utc": built_utc,
        "builder_image": builder_name,
        "runtime_image": runtime_image,
        "tested_runtime_image": runtime_image,
        "core_sha256": core_sha,
        "runtime_smoke": "passed_cluster",
        "local_checks": ["cmake", "ctest", "installed_doctor"],
        "files": len(all_files),
        "bytes": sum(p.stat().st_size for p in all_files),
    }
    (install_tmp / "BUILD_INFO.json").write_text(json.dumps(build_info, indent=1) + "\n")

    check_shared_storage()
    if nfs_tmp.exists():
        shutil.rmtree(nfs_tmp)
    shutil.copytree(install_tmp, nfs_tmp, symlinks=True)
    if hashlib.sha256((nfs_tmp / "bin/ditserve").read_bytes()).hexdigest() != core_sha:
        raise RuntimeError("NFS copy of bin/ditserve failed SHA256 verification")
    subprocess.run(["chmod", "-R", "a+rX", str(nfs_tmp)], check=True)
    if not target_dir.exists():
        os.rename(nfs_tmp, target_dir)
    else:
        shutil.rmtree(nfs_tmp)
finally:
    shutil.rmtree(build_tmp, ignore_errors=True)
    shutil.rmtree(install_tmp, ignore_errors=True)
    if nfs_tmp.exists():
        shutil.rmtree(nfs_tmp, ignore_errors=True)
    if acquired and lock_dir.exists():
        try:
            os.rmdir(lock_dir)
        except OSError:
            pass
PY
  fi
fi
[ -f "$BUILD/BUILD_INFO.json" ] || { echo "no deployed build $DITSERVE_BUILD under $CODE/builds"; exit 1; }
PACKAGE=$BUILD/share/ditserve
export PATH=$BUILD/bin:$PATH DITSERVE_PACKAGE_ROOT=$PACKAGE/python
# The image points DITSERVE_ROOT, HOME and caches at /data, which does not outlive the job.
export DITSERVE_ROOT=$OUTPUT HOME=$OUTPUT
export HF_HOME=$OUTPUT/cache/huggingface TORCH_HOME=$MODELS/torch
export TORCHINDUCTOR_CACHE_DIR=$OUTPUT/cache/torchinductor TRITON_CACHE_DIR=$OUTPUT/cache/triton
export MPLCONFIGDIR=$OUTPUT/cache/matplotlib
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Budget ledgers stay on container-local disk: result finalization takes LOCK_EX on the
# ledger's .lock through a read-only handle, which NFS rejects with EBADF. Copied back on exit.
LEDGERS=/tmp/ditserve-ledgers
# Ephemeral doorbell mmap files and step-boundary state transfers stay in container-local RAM
# (/dev/shm) rather than NFS (/shared/output) to avoid NFS attribute/page cache lag.
if [ -d /dev/shm ] && [ -w /dev/shm ]; then
  export DITSERVE_TRANSFER_DIR=/dev/shm/ditserve-transfer-$DITSERVE_RUN_ID
else
  export DITSERVE_TRANSFER_DIR=/tmp/ditserve-transfer-$DITSERVE_RUN_ID
fi
mkdir -p "$RUN_DIR/ledgers" "$LEDGERS" "$DITSERVE_TRANSFER_DIR" "$HF_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR"
# On exit also make this run's work directory readable from the login node (the controller writes
# run.json 0600 as root), so a failed cell can be diagnosed without the archive.
trap 'rm -rf "$DITSERVE_TRANSFER_DIR" 2>/dev/null || true; cp -a "$LEDGERS/." "$RUN_DIR/ledgers/" || true; chmod -R a+rX "$OUTPUT"/.cache/results-work/*/"$DITSERVE_RUN_ID" "$RUN_DIR" 2>/dev/null || true' EXIT
exec > >(tee -a "$RUN_DIR/$(basename "$0" .sh).log") 2>&1
echo "== $(basename "$0") run=$DITSERVE_RUN_ID host=$(hostname) $(date -Is)"
echo "build=$DITSERVE_BUILD core=$(sha256sum "$(command -v ditserve)" | cut -c1-16) package=$PACKAGE"
cp "$BUILD/BUILD_INFO.json" "$RUN_DIR/BUILD_INFO.json"
ditserve --version

kairos_calibrate_native_profile() {
  local model model_name calibration_id experiment cal_dir
  case "$WAN_MODEL" in
    wan) model=wan21; model_name=Wan2.1-T2V-1.3B ;;
    wan2.2-ti2v-5b) model=wan22; model_name=Wan2.2-TI2V-5B ;;
    *) echo "unknown WAN_MODEL $WAN_MODEL"; return 1 ;;
  esac
  calibration_id=$DITSERVE_RUN_ID-calibration
  experiment=kairos-$model-native-calibration
  cal_dir=$OUTPUT/acceptance/$calibration_id
  mkdir -p "$cal_dir"
  python3 "$SCRIPTS/kairos_runtime_audit.py"
  PYTHONPATH=$PACKAGE/python python3 "$PACKAGE/benchmarks/native_study.py" --model "$model" \
    --image-batch-calibration "${DITSERVE_IMAGE_BATCH:-1}" --output "$cal_dir/study.json"
  ditserve run --config "$cal_dir/study.json" --run-id "$calibration_id" --allow-gpu \
    --ledger "$LEDGERS/$experiment.json"
  ditserve results verify "$OUTPUT/results/$experiment/$calibration_id"
  PYTHONPATH=$PACKAGE/python python3 "$PACKAGE/benchmarks/fit_native_profile.py" \
    --run "$OUTPUT/.cache/results-work/$experiment/$calibration_id/run.json" \
    --runtime-facts "$RUN_DIR/runtime-facts.json" \
    --model "$model_name" \
    --output "$RUN_DIR/profile.json"
  sha256sum "$RUN_DIR/profile.json"
}
