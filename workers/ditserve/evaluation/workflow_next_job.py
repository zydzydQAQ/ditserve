"""Run all arms of a screening block, each in its own clean Python process."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from .workflow_budget import atomic_json


def run(config, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    result = dict(schema="ditserve.workflow.next-block.v1", status="running",
                  config=config, arms=[], started_unix=time.time())
    atomic_json(out / "run.json", result)
    try:
        ids = set()
        for arm in config["arms"]:
            name = arm["id"]
            if not name.replace("-", "").isalnum() or name in ids:
                raise ValueError("invalid or duplicate arm identifier")
            ids.add(name)
            path = out / (name + ".json")
            atomic_json(path, arm["config"])
            command = [sys.executable, "-m", "ditserve.evaluation." + arm["module"],
                       "--kind", arm["kind"], "--config", str(path), "--out", str(out / name)]
            with (out / (name + ".stdout.log")).open("w") as stdout, (out / (name + ".stderr.log")).open("w") as stderr:
                child = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
            target = out / name / "run.json"
            measured = json.loads(target.read_text()) if target.exists() else None
            result["arms"].append(dict(id=name, command=command, returncode=child.returncode,
                output=str(target), output_sha256=hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else None,
                result=measured))
            atomic_json(out / "run.json", result)
            if child.returncode or not measured or measured.get("status") != "completed":
                raise RuntimeError("screening arm failed: " + name)
        result["status"] = "completed"
    except Exception as error:
        result.update(status="failed", error=str(error))
    result["finished_unix"] = time.time()
    atomic_json(out / "run.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(json.loads(args.config.read_text()), args.out)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
