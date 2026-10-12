"""Validated CPU media conversion shared by both experiment families."""

import argparse
import hashlib
from pathlib import Path

from ..util import atomic_json, read_json


def video_frames(raw, np, expected=None, allow_uint8=False):
    if raw.ndim != 5 or raw.shape[0] != 1 or raw.shape[-1] != 3 or any(dim <= 0 for dim in raw.shape):
        raise ValueError("expected [1,T,H,W,3] array")
    if expected and (list(raw.shape) != expected["shape"] or str(raw.dtype) != expected["dtype"]):
        raise ValueError("NPY shape/dtype differs from recorded output")
    is_float = np.issubdtype(raw.dtype, np.floating)
    if not is_float and not (allow_uint8 and raw.dtype == np.uint8):
        raise ValueError("unexpected dtype; floating-point [0,1] output is required")
    digest, frames = hashlib.sha256(), []
    for frame in raw[0]:
        if not np.isfinite(frame).all() or (is_float and (frame.min() < 0 or frame.max() > 1)):
            raise ValueError("nonfinite or out-of-range pixels; no clipping/rescaling")
        digest.update(frame.tobytes(order="C"))
        frames.append(np.rint(frame * 255).astype(np.uint8) if is_float else frame)
    if expected and digest.hexdigest() != expected["sha256"]:
        raise ValueError("NPY pixel SHA256 differs from recorded output")
    return np.stack(frames)


def export_v100(spec):
    import numpy as np
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    root = Path(spec["output"])
    result = {"status": "completed", "fps": 16, "timing_scope": "outside profiling",
              "quality_status": "manual_review_pending", "videos": [], "contact_sheets": [], "errors": []}
    groups = {}
    for item in spec["items"]:
        try:
            frames = video_frames(np.load(item["artifact"], allow_pickle=False), np, allow_uint8=True)
            folder = root / item["stage"] / (item["prompt_id"] + "-seed" + str(item["seed"]))
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / (item["arm"] + ".mp4")
            imageio.mimwrite(str(path), frames, fps=16, codec="libx264", quality=8, macro_block_size=1)
            result["videos"].append(dict(item, video=str(path), frame_count=len(frames), fps=16))
            indices = np.linspace(0, len(frames) - 1, num=min(6, len(frames)), dtype=int).tolist()
            samples = [(index, Image.fromarray(frames[index]).resize((256, max(1, round(frames.shape[1] * 256 / frames.shape[2]))))) for index in indices]
            groups.setdefault((item["stage"], item["prompt_id"], item["seed"]), {})[item["arm"]] = samples
        except Exception as exc:
            result["errors"].append({"item": item, "type": type(exc).__name__, "message": str(exc)})
    for key, arms in groups.items():
        if not all(arm in arms for arm in ("none", "cache-dit")):
            continue
        try:
            stage, prompt_id, seed = key
            if [i for i, image in arms["none"]] != [i for i, image in arms["cache-dit"]]:
                raise ValueError("Mismatched frame indices for paired contact sheet")
            height = max(image.height for samples in arms.values() for index, image in samples)
            sheet = Image.new("RGB", (256 * len(arms["none"]), 44 + 2 * (height + 26)), "white")
            draw = ImageDraw.Draw(sheet)
            draw.text((8, 7), "%s / %s / seed=%s / %s; same frame indices" % (stage, prompt_id, seed, spec["dtype"]), fill="black")
            for row, arm in enumerate(("none", "cache-dit")):
                top = 44 + row * (height + 26)
                for column, (index, image) in enumerate(arms[arm]):
                    draw.text((column * 256 + 4, top), "%s | frame %d" % (arm, index), fill="black")
                    sheet.paste(image, (column * 256, top + 24))
            path = root / stage / (prompt_id + "-seed" + str(seed)) / "paired-contact-sheet.png"
            sheet.save(str(path))
            result["contact_sheets"].append({"stage": stage, "prompt_id": prompt_id, "seed": seed, "path": str(path)})
        except Exception as exc:
            result["errors"].append({"pair": list(key), "type": type(exc).__name__, "message": str(exc)})
    if result["errors"]:
        result["status"] = "partial_failure"
    atomic_json(spec["report"], result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    args = parser.parse_args(argv)
    return 0 if export_v100(read_json(args.spec))["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
