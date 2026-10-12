"""Resolve metadata and download the Wan commit pinned in configs/v100.json."""

from ..util import ROOT, CONFIGS, atomic_json

import argparse
import fnmatch
import json
from pathlib import Path


ALLOW_PATTERNS = ["model_index.json", "scheduler/*", "tokenizer/*", "text_encoder/*", "transformer/*", "vae/*"]


def build_metadata(info, model_id, revision):
    """Keep Hub-declared LFS SHA256 separate from non-LFS Git blob IDs."""
    if info.sha != revision:
        raise ValueError("Hub returned a different commit than configs/v100.json")
    siblings = sorted(info.siblings or [], key=lambda sibling: sibling.rfilename)
    files = []
    for sibling in siblings:
        if not any(fnmatch.fnmatchcase(sibling.rfilename, pattern) for pattern in ALLOW_PATTERNS):
            continue
        size = sibling.size
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("Hub did not provide a valid file size: " + sibling.rfilename)
        lfs = sibling.lfs
        sha256 = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        if sha256 is not None and (not isinstance(sha256, str) or len(sha256) != 64 or
                                   any(character not in "0123456789abcdefABCDEF" for character in sha256)):
            raise ValueError("Hub returned an invalid LFS SHA256: " + sibling.rfilename)
        files.append({"file": sibling.rfilename, "bytes": size, "sha256": sha256})
    if not files or "model_index.json" not in {file["file"] for file in files}:
        raise ValueError("Pinned snapshot is missing model component metadata")
    revision_record = {"id": model_id, "sha": info.sha, "files": [sibling.rfilename for sibling in siblings]}
    files_record = {"repo": model_id, "revision": info.sha, "files": files,
                    "total_bytes": sum(file["bytes"] for file in files)}
    return revision_record, files_record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIGS / "v100.json")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/huggingface/hub")
    parser.add_argument("--environment-dir", type=Path, default=ROOT / ".cache/runs/v100/environment")
    args = parser.parse_args(argv)
    from huggingface_hub import HfApi, snapshot_download

    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_id, revision = config["model_id"], config["revision"]
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("configs/v100.json must contain model_id")
    if (not isinstance(revision, str) or len(revision) != 40 or
            any(character not in "0123456789abcdef" for character in revision)):
        raise ValueError("configs/v100.json revision must be a full lowercase commit SHA")
    info = HfApi().model_info(repo_id=model_id, revision=revision, files_metadata=True)
    revision_record, files_record = build_metadata(info, model_id, revision)
    environment = args.environment_dir
    environment.mkdir(parents=True, exist_ok=True)
    for name, record in (("model-revision.json", revision_record), ("model-files.json", files_record)):
        atomic_json(environment / name, record)
    path = snapshot_download(
        repo_id=model_id, revision=revision,
        cache_dir=str(args.cache_dir), max_workers=4,
        allow_patterns=ALLOW_PATTERNS,
    )
    (environment / "model-path.txt").write_text(str(path) + "\n", encoding="utf-8")
    print(path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
