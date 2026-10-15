"""Bounded CPU previews of recorded experiment artifacts, outside measured work."""

import hashlib
import json
from pathlib import Path
import re

from ..gpu.media import video_frames
from ..research.common import atomic_json, file_digest


STAGES = {1: ("profile", "correctness", "preemption", "online"),
          2: ("screen", "qualify", "pressure", "actions", "state_cost"),
          3: ("correctness", "profile", "update", "multisession")}


def _stage(parts, number):
    for part in reversed(parts):
        for stage in STAGES.get(number, ()):
            if re.search(r"(?<![a-z])" + re.escape(stage) + r"(?![a-z])", str(part).lower()):
                return stage
    return None


def _direction(parts):
    for part in parts:
        match = re.fullmatch(r"direction([123])", str(part))
        if match:
            return int(match.group(1))
    return None


def _failure(value):
    exact = value.get("cached_reference_exact")
    if (value.get("failed_quality") is True or
            any(value.get(key) is False for key in ("quality_passed", "passed", "urgent_reference_exact")) or
            isinstance(exact, list) and any(v is False for v in exact)):
        return 2
    return 1 if value.get("status") in ("failed", "incomplete") else 0


def _reported_shape(value):
    """Use explicit report metadata, never filenames or decoded array sizes."""
    keys = ("frames", "height", "width")
    if isinstance(value, dict) and all(type(value.get(key)) is int and value[key] > 0 for key in keys):
        return {key: value[key] for key in keys}
    return None


def _shape_key(shape):
    return tuple(shape[key] for key in ("frames", "height", "width")) if shape else ()


def _references(output):
    """Discover referenced arrays only; never treat unreferenced files as results."""
    found, errors, unavailable = {}, [], []
    for report in sorted(output.rglob("*.json")):
        relative = report.relative_to(output)
        if "sources" in relative.parts or "media" in relative.parts:
            continue
        try:
            data = json.loads(report.read_text())
        except (OSError, ValueError) as error:
            errors.append(dict(report=str(report), error=str(error)))
            continue
        def visit(value, direction=None, failed=0, stage=None, pointer="$", reported_shape=None):
            if isinstance(value, dict):
                number = value.get("direction")
                if type(number) is int and number in (1, 2, 3):
                    direction = number
                if value.get("phase") in STAGES.get(direction, ()):
                    stage = value["phase"]
                failed = max(failed, _failure(value))
                reported_shape = _reported_shape(value.get("shape")) or reported_shape
                if value.get("raw_available") is False:
                    unavailable.append(dict(direction=direction, stage=stage or _stage(relative.parts, direction),
                        report=str(report), pointer=pointer, kind="raw_not_retained",
                        reason=value.get("reason", "report explicitly records raw_available=false")))
                for key, nested in value.items():
                    visit(nested, direction, failed, stage, pointer + "." + key, reported_shape)
            elif isinstance(value, list):
                for index, nested in enumerate(value):
                    visit(nested, direction, failed, stage, pointer + "[%d]" % index, reported_shape)
            elif isinstance(value, str) and Path(value).suffix.lower() in (".npy", ".npz"):
                candidate = Path(value)
                if not candidate.is_absolute():
                    # Current writers use absolute paths. Accommodate older
                    # report-local and run-root-relative artifact references.
                    choices = (report.parent / candidate, output / candidate, candidate)
                    candidate = next((path for path in choices if path.is_file()), choices[0])
                candidate = candidate.resolve()
                number = direction or _direction(candidate.parts)
                if number is None or "sources" in candidate.parts:
                    return
                key = (number, str(candidate))
                item = found.setdefault(key, dict(direction=number, source=str(candidate),
                    failure_priority=False, priority=0, referenced_by=[], reported_shape=None,
                    stage=_stage(candidate.parts, number) or stage or _stage(relative.parts, number) or "unknown"))
                if item["reported_shape"] is None and reported_shape is not None:
                    item["reported_shape"] = reported_shape
                if item["stage"] == "unknown":
                    item["stage"] = stage or _stage(relative.parts, number) or "unknown"
                item["priority"] = max(item["priority"], failed)
                item["failure_priority"] = bool(item["priority"])
                if str(report) not in item["referenced_by"]:
                    item["referenced_by"].append(str(report))
        visit(data, _direction(relative.parts))
    return sorted(found.values(), key=lambda row: (row["direction"], -row["priority"], row["source"])), errors, unavailable


def _clip_key(item):
    return (item["source"], item.get("member"))


def _groups(candidates):
    """Pair only artifacts actually referenced by reports; never invent outputs."""
    by_path = {(row["direction"], row["source"]): row for row in candidates}
    used, groups = set(), []
    def add(rows, kind, missing=()):
        number = rows[0]["direction"]
        keys = sorted((_clip_key(row) for row in rows), key=str)
        group_id = hashlib.sha256(repr((number, keys)).encode()).hexdigest()[:16]
        shapes = [row["reported_shape"] for row in rows if row.get("reported_shape")]
        reported_shape = shapes[0] if shapes and all(shape == shapes[0] for shape in shapes) else None
        groups.append(dict(group_id=group_id, direction=number, stage=rows[0]["stage"], kind=kind,
            priority=max(row["priority"] for row in rows), reported_shape=reported_shape,
            clips=rows, missing_members=list(missing)))
        used.update((number, row["source"]) for row in rows)
    def siblings(item, names):
        parent = Path(item["source"]).parent
        rows, missing = [], []
        for name in names:
            path = str(parent / name)
            row = by_path.get((item["direction"], path))
            if row is None:
                missing.append(dict(source=path, reason="counterpart not referenced or not retained"))
            else:
                rows.append(dict(row, member=None))
        return rows, missing
    for item in candidates:
        source = Path(item["source"])
        if source.suffix.lower() == ".npz":
            add([dict(item, member=member) for member in ("output", "reference_output")], "full_pair")
    # State-cost trials live in separate per-action directories. Every action
    # compares with its own round's keep and the shared fresh-off reference.
    # Resolve exclusively through the report reference index, even when an
    # expected counterpart happens to exist on disk without a report reference.
    for item in candidates:
        source = Path(item["source"])
        if item["direction"] != 2 or item["stage"] != "state_cost" or source.suffix.lower() != ".npy":
            continue
        match = re.fullmatch(r"round(\d+)-(keep|full_cpu|flush)", source.parent.name)
        if not match or source.name != match[2] + ".npy":
            continue
        root = source.parent.parent
        paths = [source, root / ("round%s-keep" % match[1]) / "keep.npy", root / "reference-off" / "off.npy"]
        rows, missing = [], []
        for path in dict.fromkeys(paths):
            row = by_path.get((2, str(path)))
            if row is None:
                missing.append(dict(source=str(path), reason="state_cost counterpart not referenced or not retained"))
            else:
                rows.append(dict(row, member=None))
        add(rows, "full_pair" if not missing else "incomplete_pair", missing)
    # D2 action failures can need both an uncached and a cached reference.
    # References shared by repeated inputs are charged once against preview quota.
    for item in candidates:
        source = Path(item["source"])
        if item["direction"] != 2 or source.suffix.lower() != ".npy":
            continue
        match = re.fullmatch(r"candidate-\d+-(.+)", source.stem)
        if match:
            rows, missing = siblings(item, [source.name, "reference-off-" + match[1] + ".npy",
                                             "reference-paired_008-" + match[1] + ".npy"])
            add(rows, "full_pair" if not missing else "incomplete_pair", missing)
        elif source.stem.startswith("urgent-candidate-"):
            rows, missing = siblings(item, [source.name, source.name.replace("urgent-candidate-", "urgent-reference-", 1)])
            add(rows, "full_pair" if not missing else "incomplete_pair", missing)
    for item in candidates:
        key, source = (item["direction"], item["source"]), Path(item["source"])
        if key in used:
            continue
        if item["direction"] == 2 and item["stage"] == "state_cost":
            # A lone fresh-off reference (or an unrecognized cost layout) is
            # explicitly incomplete; it is not an off/paired_008 quality pair.
            add([dict(item, member=None)], "incomplete_pair", [dict(
                role="state_cost_round_comparison",
                reason="no recognized referenced round candidate/keep comparison for this state_cost artifact")])
        elif item["direction"] == 2 and source.stem in ("off", "paired_008"):
            rows, missing = siblings(item, ["off.npy", "paired_008.npy"])
            add(rows, "full_pair" if not missing else "incomplete_pair", missing)
        elif item["direction"] == 3 and (source.stem.endswith("-reference") or
                (3, str(source.with_name(source.stem + "-reference.npy"))) in by_path):
            stem = source.stem[:-10] if source.stem.endswith("-reference") else source.stem
            rows, missing = siblings(item, [stem + ".npy", stem + "-reference.npy"])
            add(rows, "full_pair" if not missing else "incomplete_pair", missing)
        else:
            add([dict(item, member=None)], "chunk" if "chunk" in source.stem.lower() else "unpaired")
    return groups


def _selection(groups, limit):
    """Keep pairs whole and rotate stages and reported shapes, larger first."""
    selected, seen = [], {number: set() for number in (1, 2, 3)}
    for number in (1, 2, 3):
        shape_counts = {}
        def shape_bucket(group):
            return (group["priority"], group["stage"], _shape_key(group["reported_shape"]))
        def order(group):
            stages = STAGES[number]
            shape = group["reported_shape"]
            volume = shape["frames"] * shape["height"] * shape["width"] if shape else 0
            return (-group["priority"], stages.index(group["stage"]) if group["stage"] in stages else len(stages),
                    shape_counts.get(shape_bucket(group), 0), -volume,
                    group["clips"][0]["source"], group["group_id"])
        for kinds in (("full_pair",), ("incomplete_pair", "unpaired"), ("chunk",)):
            pending = sorted([g for g in groups if g["direction"] == number and g["kind"] in kinds], key=order)
            # Round-robin stages: the first pass takes one best group from each
            # stage before a second sample from a stage can consume the quota.
            # Within the same failure level and stage, rotate reported shapes
            # before repeating one, starting with the largest reported volume.
            while pending:
                stages_seen, deferred = set(), []
                for group in sorted(pending, key=order):
                    if group["stage"] in stages_seen:
                        deferred.append(group)
                        continue
                    stages_seen.add(group["stage"])
                    new = {_clip_key(row) for row in group["clips"]} - seen[number]
                    if len(seen[number]) + len(new) <= limit:
                        selected.append(group)
                        seen[number].update(new)
                        bucket = shape_bucket(group)
                        shape_counts[bucket] = shape_counts.get(bucket, 0) + 1
                pending = deferred
    return selected


def _frames(source, member, direction):
    import numpy as np
    if member is None:
        raw = np.load(source, allow_pickle=False, mmap_mode="r")
    else:
        with np.load(source, allow_pickle=False) as archive:
            raw = archive[member]
    shape, dtype = list(raw.shape), str(raw.dtype)
    if direction == 3 and raw.ndim == 4:
        raw = raw[None, ...]
    frames = video_frames(raw, np, allow_uint8=True)
    digest = hashlib.sha256()
    for frame in raw[0]:
        digest.update(frame.tobytes(order="C"))
    return frames, dict(shape=shape, dtype=dtype, pixel_sha256=digest.hexdigest())


def _write_mp4(path, frames):
    import imageio.v2 as imageio
    imageio.mimwrite(str(path), frames, fps=16, codec="libx264", quality=8, macro_block_size=1)


def _contact_sheet(path, frames, title):
    import numpy as np
    from PIL import Image, ImageDraw
    indices = np.linspace(0, len(frames) - 1, 6, dtype=int).tolist()
    width = 256
    height = max(1, round(frames.shape[1] * width / frames.shape[2]))
    sheet = Image.new("RGB", (width * 3, 36 + (height + 26) * 2), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title[:95], fill="black")
    for position, index in enumerate(indices):
        x, y = position % 3 * width, 36 + position // 3 * (height + 26)
        draw.text((x + 4, y), "frame %d / %d" % (index, len(frames)), fill="black")
        frame = Image.fromarray(frames[index]).resize((width, height))
        sheet.paste(frame, (x, y + 24))
    sheet.save(path)
    return indices


def export_media(output, limit=8):
    """Export at most ``limit`` referenced videos per direction, failures first.

    NPZ output/reference_output each count as one video. Original arrays are
    read-only. Preview conversion is CPU-only and cannot establish quality,
    semantic freshness, blinded-review results, or measured serving latency.
    """
    if type(limit) is not int or not 1 <= limit <= 8:
        raise ValueError("media limit must be an integer from 1 to 8 per direction")
    output = Path(output).resolve()
    if not output.is_dir():
        raise ValueError("run output directory does not exist")
    destination = output / "media"
    destination.mkdir(parents=True, exist_ok=True)
    manifest = dict(schema_version="ditserve.directions.media.v2", status="incomplete",
                    output=str(output), fps=16, limit_per_direction=limit,
                    timing_scope="CPU offline export; excluded from original measured latency",
                    quality_status="manual_review_pending_not_blinded_or_quality_qualified",
                    originals_modified=False,
                    selection="whole complete pairs first; failure priority; stage and reported-shape round-robin, larger reported volume first; chunks last",
                    coverage_scope="bounded preview only; not exhaustive heldout/action videos or all scientific conclusions",
                    contact_sheet_sampling="six evenly spaced indices; repeats when fewer than six frames",
                    videos=[], errors=[], counts={"direction%d" % number: 0 for number in (1, 2, 3)})
    manifest_path = destination / "manifest.json"
    def save():
        unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        manifest["manifest_sha256"] = hashlib.sha256(json.dumps(unsigned, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        atomic_json(manifest_path, manifest)
    candidates, errors, unavailable = _references(output)
    manifest["errors"].extend(errors)
    groups = _groups(candidates)
    selected = _selection(groups, limit)
    selected_ids = {g["group_id"] for g in selected}
    manifest["raw_unavailable_records"] = unavailable
    manifest["raw_unavailable_scope"] = ("Explicit raw_available=false report occurrences may duplicate the same sample; "
        "unreferenced or never-retained outputs cannot be enumerated or reconstructed by this exporter.")
    manifest["missing_pair_counterparts"] = [dict(group_id=g["group_id"], direction=g["direction"],
        stage=g["stage"], **member) for g in groups for member in g["missing_members"]]
    manifest["missing_referenced_sources"] = [row["source"] for row in candidates if not Path(row["source"]).is_file()]
    manifest["selection_groups"] = [dict(group_id=g["group_id"], direction=g["direction"], stage=g["stage"],
        kind=g["kind"], reported_shape=g["reported_shape"], video_count=len(g["clips"]), selected=g["group_id"] in selected_ids,
        sources=[dict(source=r["source"], member=r["member"]) for r in g["clips"]],
        reason="selected_whole_group" if g["group_id"] in selected_ids else "bounded_quota_group_not_split") for g in groups]
    attempted = {number: 0 for number in (1, 2, 3)}
    exported, attempted_keys = {}, {number: set() for number in (1, 2, 3)}
    save()
    for group in selected:
        number, completed, prepared = group["direction"], [], []
        new = [row for row in group["clips"] if (number, _clip_key(row)) not in exported]
        try:
            # Validate every new member before writing any preview in this
            # group. A bad paired member never leaves a successful half-pair
            # in the manifest.
            for item in new:
                source, member = Path(item["source"]), item["member"]
                key = _clip_key(item)
                if key not in attempted_keys[number]:
                    attempted_keys[number].add(key)
                    attempted[number] += 1
                source_sha = file_digest(source)
                frames, original = _frames(source, member, number)
                prepared.append((item, source_sha, frames, original))
            for item, source_sha, frames, original in prepared:
                source, member = Path(item["source"]), item["member"]
                record = dict(item, group_id=group["group_id"])
                identity = hashlib.sha256((str(source) + ":" + str(member)).encode()).hexdigest()[:12]
                stem = re.sub(r"[^A-Za-z0-9_.-]", "_", source.stem)[:48]
                name = "%s-%s-%s-%s" % (group["stage"], stem, member or "output", identity)
                folder = destination / ("direction%d" % number)
                folder.mkdir(parents=True, exist_ok=True)
                video, sheet = folder / (name + ".mp4"), folder / (name + ".png")
                _write_mp4(video, frames)
                indices = _contact_sheet(sheet, frames, "direction%d | %s" % (number, name))
                record.update(original, source_sha256=source_sha, frame_count=len(frames), fps=16,
                              video=dict(path=str(video), sha256=file_digest(video)),
                              contact_sheet=dict(path=str(sheet), sha256=file_digest(sheet), frame_indices=indices))
                completed.append(record)
            for record in completed:
                exported[(number, _clip_key(record))] = record
                manifest["videos"].append(record)
                manifest["counts"]["direction%d" % number] += 1
        except Exception as error:
            manifest["errors"].append(dict(group_id=group["group_id"], direction=number,
                sources=[row["source"] for row in group["clips"]], whole_group_unqualified=True,
                error_type=type(error).__name__, error=str(error)))
        save()
    manifest["attempted_per_direction"] = {"direction%d" % key: value for key, value in attempted.items()}
    manifest["referenced_arrays"] = len(candidates)
    manifest["selection_summary"] = {}
    for number in (1, 2, 3):
        available = {_clip_key(row) for g in groups if g["direction"] == number for row in g["clips"]}
        chosen = {_clip_key(row) for g in selected if g["direction"] == number for row in g["clips"]}
        produced = {_clip_key(row) for row in manifest["videos"] if row["direction"] == number}
        manifest["selection_summary"]["direction%d" % number] = dict(
            referenced_videos=len(available), selected_videos=len(chosen), exported_videos=len(produced),
            unselected_videos=len(available - chosen), selected_but_not_exported=len(chosen - produced),
            referenced_stages=sorted({g["stage"] for g in groups if g["direction"] == number}),
            selected_stages=sorted({g["stage"] for g in selected if g["direction"] == number}))
    manifest["status"] = "incomplete" if manifest["errors"] or manifest["missing_referenced_sources"] else "passed" if manifest["videos"] else "blocked"
    if not candidates:
        manifest["reason"] = "no referenced array artifacts are currently available for preview"
    elif not selected:
        manifest["reason"] = "quota cannot fit any whole artifact group; paired videos were not split"
    save()
    return manifest
