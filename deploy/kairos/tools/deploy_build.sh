#!/bin/bash
# Stage one DITSERVE commit to the Kairos code dir as builds/.src-<commit>/ without
# compiling on the local machine (saving local WSL memory). The first cluster job running
# with DITSERVE_BUILD=<commit> automatically compiles C++, runs ctest + doctor inside the
# cluster runtime container (/tmp), and atomically publishes write-once builds/<commit>/
# (bin/ditserve + share/ditserve + BUILD_INFO.json).
#   usage: deploy_build.sh [COMMIT] [--mode cluster|local] [--runtime-smoke cluster|local] [--sync-info]
# An existing builds/<commit> is never replaced: a result names its build, so the build must not change.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=${DITSERVE_REPO:-$(cd "$HERE/../../.." && pwd)}
STAGE=${DITSERVE_STAGE_DIR:-$REPO/.cache/kairos-builds}  # local metadata/share mirror of published builds
BUILDER=local-cmake-install
RUNTIME=192.168.1.114:30500/jjjkairos/ditserve:external-serving-vllm-cu126-sm80-20261104-r3
JSON_SHA=9bea4c8066ef4a1c206b2be5a36302f8926f7fdc6087af5d20b417d0cf103ea6
REMOTE_HOST=jjjkairos
REMOTE_DIR=/data/share/jjjkairos/code/DITSERVE/builds
SSH=(ssh -o ControlMaster=auto "-o ControlPath=/run/user/$(id -u)/ditserve-ssh-%C" -o ControlPersist=15m -o ClearAllForwardings=yes -o BatchMode=yes)

ref=HEAD
mode=cluster
smoke=cluster
sync_only=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --mode) mode=${2:?missing mode (cluster|local)}; shift 2 ;;
    --runtime-smoke) smoke=${2:?missing runtime-smoke mode}; shift 2 ;;
    --sync-info) sync_only=1; shift ;;
    --*) echo "unknown option: $1" >&2; exit 2 ;;
    *) ref=$1; shift ;;
  esac
done
case "$mode" in cluster|local) ;; *) echo 'mode must be cluster or local' >&2; exit 2 ;; esac
case "$smoke" in local|cluster) ;; *) echo 'runtime-smoke must be local or cluster' >&2; exit 2 ;; esac

storage_gate() {
  "${SSH[@]}" "$REMOTE_HOST" 'df -h /data'
  storage=$("${SSH[@]}" "$REMOTE_HOST" 'df -BG /data')
  python3 -c 'import sys; row=sys.stdin.read().splitlines()[-1].split(); used=int(row[2].rstrip("G")); pct=int(row[4].rstrip("%")); sys.exit(0 if used<=790 and pct<80 else "shared storage stop line reached")' <<< "$storage"
}

commit=$(git -C "$REPO" rev-parse --verify "$ref^{commit}")
sha=${commit:0:7}
subject=$(git -C "$REPO" log -1 --format=%s "$commit")

if [ "$sync_only" = 1 ]; then
  mkdir -p "$STAGE/$sha"
  "${SSH[@]}" "$REMOTE_HOST" "cat $REMOTE_DIR/$sha/BUILD_INFO.json" > "$STAGE/$sha/BUILD_INFO.json.tmp"
  mv "$STAGE/$sha/BUILD_INFO.json.tmp" "$STAGE/$sha/BUILD_INFO.json"
  cat "$STAGE/$sha/BUILD_INFO.json"
  exit 0
fi

if [ "$mode" = local ] && [ "$smoke" = local ]; then
  docker image inspect "$RUNTIME" >/dev/null || {
    echo "existing runtime image is unavailable; publication stopped (no pull/build)" >&2
    exit 1
  }
fi

storage_gate
if "${SSH[@]}" "$REMOTE_HOST" "test -f $REMOTE_DIR/$sha/BUILD_INFO.json" 2>/dev/null; then
  echo "build $sha already published on cluster; not replacing it"
  mkdir -p "$STAGE/$sha"
  if [ ! -f "$STAGE/$sha/BUILD_INFO.json" ]; then
    "${SSH[@]}" "$REMOTE_HOST" "cat $REMOTE_DIR/$sha/BUILD_INFO.json" > "$STAGE/$sha/BUILD_INFO.json.tmp" && mv "$STAGE/$sha/BUILD_INFO.json.tmp" "$STAGE/$sha/BUILD_INFO.json"
  fi
  cat "$STAGE/$sha/BUILD_INFO.json"
  exit 0
fi

if [ "$mode" = cluster ]; then
  if "${SSH[@]}" "$REMOTE_HOST" "test -f $REMOTE_DIR/.src-$sha/SOURCE_INFO.json" 2>/dev/null; then
    echo "source .src-$sha already staged on cluster; first job with DITSERVE_BUILD=$sha will compile and publish $REMOTE_DIR/$sha"
    "${SSH[@]}" "$REMOTE_HOST" "cat $REMOTE_DIR/.src-$sha/SOURCE_INFO.json" 2>/dev/null
    exit 0
  fi

  src_tmp=$(mktemp -d)
  trap 'rm -rf "$src_tmp"' EXIT
  git -C "$REPO" archive "$commit" | tar -x -C "$src_tmp"
  header=$REPO/.cache/deps/nlohmann3.11.3/nlohmann/json.hpp
  [ -f "$header" ] || header=$REPO/deploy/kairos/deps/nlohmann3.11.3/nlohmann/json.hpp
  echo "$JSON_SHA  $header" | sha256sum -c --quiet
  mkdir -p "$src_tmp/.cache/deps/nlohmann3.11.3/nlohmann"
  cp "$header" "$src_tmp/.cache/deps/nlohmann3.11.3/nlohmann/"

  echo "== stage source and local share mirror for $sha (no local C++ compile)"
  mkdir -p "$STAGE/$sha"
  python3 -I - "$src_tmp" "$STAGE/$sha" "$commit" "$sha" "$subject" "$RUNTIME" "$JSON_SHA" <<'EOF'
import datetime, hashlib, json, pathlib, shutil, sys

src_dir = pathlib.Path(sys.argv[1]).resolve()
stage_dir = pathlib.Path(sys.argv[2]).resolve()
commit, sha, subject, runtime, json_sha = sys.argv[3:]

# Populate local share/ditserve mirror (pure Python/config/source files) so local
# evidence relocation tools can verify protocol/source paths without compiling C++ locally.
share_root = stage_dir / "share" / "ditserve"
if share_root.exists():
    shutil.rmtree(share_root)
py_root = share_root / "python"
py_root.mkdir(parents=True, exist_ok=True)

def copy_filtered(src: pathlib.Path, dst: pathlib.Path, suffixes: tuple[str, ...], extra_names: tuple[str, ...] = ()):
    for path in sorted(src.rglob("*")):
        if "__pycache__" in path.parts or not path.is_file():
            continue
        if path.suffix in suffixes or path.name in extra_names:
            target = dst / path.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)

copy_filtered(src_dir / "workers" / "ditserve", py_root / "ditserve", (".py",))
copy_filtered(src_dir / "workers" / "xfuser", py_root / "xfuser", (".py",), ("LICENSE.txt",))
copy_filtered(src_dir / "workers" / "gfc", py_root / "gfc", (".py",), ("LICENSE", "VENDORED.md"))
for folder in ("include", "src", "cmd", "benchmarks"):
    copy_filtered(src_dir / folder, share_root / folder, (".cpp", ".hpp", ".py"), ("CMakeLists.txt",))
shutil.copy2(src_dir / "CMakeLists.txt", share_root / "CMakeLists.txt")
shutil.copytree(src_dir / "configs", share_root / "configs")

files = {}
total_bytes = 0
for path in sorted(src_dir.rglob("*")):
    if path.is_file() and path.name != "SOURCE_INFO.json":
        rel = path.relative_to(src_dir).as_posix()
        raw = path.read_bytes()
        files[rel] = hashlib.sha256(raw).hexdigest()
        total_bytes += len(raw)

manifest_raw = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
staged_utc = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
source_info = {
    "schema": "ditserve.cluster-source-stage.v1",
    "commit": commit,
    "short_commit": sha,
    "subject": subject,
    "staged_utc": staged_utc,
    "build_mode": "cluster_lazy_compile",
    "runtime_image": runtime,
    "json_header_sha256": json_sha,
    "source_manifest_sha256": manifest_sha,
    "files": len(files),
    "bytes": total_bytes,
}
raw_info = (json.dumps(source_info, indent=1) + "\n").encode()
(src_dir / "SOURCE_INFO.json").write_bytes(raw_info)
(stage_dir / "SOURCE_INFO.json").write_bytes(raw_info)
EOF

  info_sha=$(sha256sum "$src_tmp/SOURCE_INFO.json" | cut -d" " -f1)
  echo "== upload source .src-$sha to cluster"
  storage_gate
  "${SSH[@]}" "$REMOTE_HOST" "mkdir -p $REMOTE_DIR && rm -rf $REMOTE_DIR/.src-tmp-$sha"
  storage_gate
  rsync -rlpt --chmod=Dgo=rx,Fgo=rX -e "${SSH[*]}" "$src_tmp/" "$REMOTE_HOST:$REMOTE_DIR/.src-tmp-$sha/"
  storage_gate
  "${SSH[@]}" "$REMOTE_HOST" "set -e; test ! -e $REMOTE_DIR/.src-$sha; mv $REMOTE_DIR/.src-tmp-$sha $REMOTE_DIR/.src-$sha;
    echo '$info_sha  $REMOTE_DIR/.src-$sha/SOURCE_INFO.json' | sha256sum -c --quiet && echo 'staged $REMOTE_DIR/.src-$sha (will compile in cluster container on first run)'"
  exit 0
fi

# Legacy --mode local path
[ ! -e "$STAGE/$sha/BUILD_INFO.json" ] || { echo "local stage $STAGE/$sha exists but is not published; inspect it first"; exit 1; }
src=$STAGE/.src-$sha
mkdir -p "$src" "$STAGE/$sha"
git -C "$REPO" archive "$commit" | tar -x -C "$src"
header=$REPO/.cache/deps/nlohmann3.11.3/nlohmann/json.hpp
[ -f "$header" ] || header=$REPO/deploy/kairos/deps/nlohmann3.11.3/nlohmann/json.hpp
echo "$JSON_SHA  $header" | sha256sum -c --quiet
mkdir -p "$src/.cache/deps/nlohmann3.11.3/nlohmann" && cp "$header" "$src/.cache/deps/nlohmann3.11.3/nlohmann/"

echo "== build and test $sha (local mode)"
build_tmp=$(mktemp -d)
trap 'rm -rf "$build_tmp"' EXIT
cmake_extra=()
[ -n "${OPENSSL_ROOT_DIR:-}" ] && cmake_extra+=("-DOPENSSL_ROOT_DIR=$OPENSSL_ROOT_DIR")
cmake -S "$src" -B "$build_tmp" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON "${cmake_extra[@]}" >/dev/null
cmake --build "$build_tmp" --parallel 4 >/dev/null
ctest --test-dir "$build_tmp" --output-on-failure | tail -3
cmake --install "$build_tmp" --prefix "$STAGE/$sha" --strip >/dev/null

echo "== smoke test local install"
DITSERVE_PACKAGE_ROOT="$STAGE/$sha/share/ditserve/python" "$STAGE/$sha/bin/ditserve" --version
DITSERVE_PACKAGE_ROOT="$STAGE/$sha/share/ditserve/python" "$STAGE/$sha/bin/ditserve" doctor >/dev/null && echo doctor ok

if [ "$smoke" = local ]; then
  echo "== smoke in existing runtime image"
  docker run --rm --pull=never --network=none \
  --entrypoint /bin/bash -e PYTHONDONTWRITEBYTECODE=1 \
  -e DITSERVE_PACKAGE_ROOT=/review/share/ditserve/python \
  -v "$STAGE/$sha:/review:ro" "$RUNTIME" -c \
  '/review/bin/ditserve --version && /review/bin/ditserve doctor --python /opt/venv/bin/python >/dev/null'
else
  echo "== authorized cluster validation: candidate pending 1x1 acceptance; no local image smoke"
fi

core=$(sha256sum "$STAGE/$sha/bin/ditserve" | cut -d" " -f1)
python3 -I - "$STAGE/$sha/BUILD_INFO.json" "$commit" "$subject" \
  "$BUILDER" "$RUNTIME" "$core" "$STAGE/$sha" "$smoke" <<'EOF'
import datetime, json, pathlib, sys
path, commit, subject, builder, runtime, core, root, smoke = sys.argv[1:]
files = [p for p in pathlib.Path(root).rglob("*") if p.is_file()]
built = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
info = {"commit": commit, "subject": subject, "built_utc": built,
        "builder_image": builder, "runtime_image": runtime, "core_sha256": core,
        "runtime_smoke": "passed_local" if smoke == "local" else "pending_cluster",
        "local_checks": ["cmake", "ctest", "installed_doctor"],
        "files": len(files), "bytes": sum(p.stat().st_size for p in files)}
if smoke == "local": info["tested_runtime_image"] = runtime
json.dump(info, open(path, "x"), indent=1)
EOF
[[ $src == "$STAGE"/.src-??????? ]] && { rm -r -- "$src" || echo "warning: could not remove $src"; }

echo "== publish $sha"
storage_gate
"${SSH[@]}" "$REMOTE_HOST" "mkdir -p $REMOTE_DIR"
storage_gate
rsync -rlpt --chmod=Dgo=rx,Fgo=rX -e "${SSH[*]}" "$STAGE/$sha/" "$REMOTE_HOST:$REMOTE_DIR/.tmp-$sha/"
storage_gate
"${SSH[@]}" "$REMOTE_HOST" "set -e; test ! -e $REMOTE_DIR/$sha; mv $REMOTE_DIR/.tmp-$sha $REMOTE_DIR/$sha;
  echo '$core  $REMOTE_DIR/$sha/bin/ditserve' | sha256sum -c --quiet && echo published $REMOTE_DIR/$sha"
