"""CPU-only, blinded reconstruction of actual D3 presentation windows.

Only the public directory is for reviewers. The adjacent private key preserves
all event denominators and provenance; it must not accompany the review bundle.
No conditioning tag is interpreted as evidence of semantic success.
"""

import bisect
import csv
import hashlib
import json
import math
import random
import re
from collections import OrderedDict
from pathlib import Path

from ..gpu.media import video_frames
from ..util import atomic_json


FPS = 16
SHUFFLE_SEED = 13013


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("invalid " + name)
    return float(value)


class FrameSources:
    """Bounded CPU cache of validated, immutable original chunk arrays."""

    def __init__(self, report_path, capacity=4):
        self.report_path, self.capacity = report_path, capacity
        self.cache = OrderedDict()

    def load(self, artifact):
        import numpy as np
        path = Path(artifact["path"])
        if not path.is_absolute():
            local = self.report_path.parent / path
            path = local if local.is_file() else path.resolve()
        path = path.resolve()
        stat = path.stat()
        identity = (str(path), stat.st_size, stat.st_mtime_ns, artifact.get("sha256"))
        if identity in self.cache:
            self.cache.move_to_end(identity)
            return self.cache[identity]
        digest = _hash(path)
        if artifact.get("sha256") and digest != artifact["sha256"]:
            raise ValueError("source artifact file SHA256 mismatch: " + str(path))
        raw = np.load(path, allow_pickle=False, mmap_mode="r")
        if raw.ndim != 4:
            raise ValueError("D3 chunk must have shape [T,H,W,3]")
        frames = video_frames(raw[None], np, allow_uint8=True)
        if artifact.get("frames") is not None and len(frames) != artifact["frames"]:
            raise ValueError("D3 chunk frame count differs from report")
        after = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("source artifact changed while reading")
        value = frames, dict(path=str(path), sha256=digest, shape=list(raw.shape), dtype=str(raw.dtype))
        self.cache[identity] = value
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return value


def reconstruct_window(sample, session_id, arrival_s, window_seconds, pre_seconds, sources, shape):
    """Return presentation indices; no generated-but-unpresented frame is used."""
    end_observation = _finite(sample["observation_end_s"], "observation_end_s")
    start, end = arrival_s - pre_seconds, min(arrival_s + window_seconds, end_observation)
    if end <= start:
        raise ValueError("event has no observable window")
    player = sample.get("playback", {}).get(session_id, {})
    if player.get("fps", FPS) != FPS:
        raise ValueError("presentation FPS differs from the frozen 16fps protocol")
    presentations = player.get("presentations", [])
    presentations = sorted(presentations, key=lambda row: _finite(row["presented_s"], "presented_s"))
    if any(row["presented_s"] < 0 for row in presentations):
        raise ValueError("negative presentation timestamp")
    times = [row["presented_s"] for row in presentations]
    ticks = [start + index / FPS for index in range(math.ceil((end - start) * FPS))]
    ticks = [instant for instant in ticks if instant < end]
    selected = [bisect.bisect_right(times, instant) - 1 for instant in ticks]
    chunks = {}
    for row in sample.get("chunks", []):
        if row["session_id"] == session_id:
            if row["block"] in chunks:
                raise ValueError("duplicate session chunk")
            chunks[row["block"]] = row
    references, provenance = {}, {}
    height, width = shape
    for index in sorted(set(selected) - {-1}):
        row = presentations[index]
        match = re.fullmatch(re.escape(session_id) + r"-chunk-(\d+)-(\d+)", row["frame_id"])
        if not match or int(match[1]) != row["chunk"]:
            raise ValueError("presentation frame_id does not match its chunk")
        chunk, frame_index = int(match[1]), int(match[2])
        artifact = chunks[chunk]["artifact"]
        frames, identity = sources.load(artifact)
        if frame_index >= len(frames):
            raise ValueError("presentation frame index is outside source chunk")
        if tuple(frames.shape[1:3]) != (height, width):
            raise ValueError("presentation dimensions disagree with the experiment")
        references[index] = (artifact, frame_index)
        provenance[identity["path"]] = identity
    return dict(start_s=start, end_s=end, arrival_s=arrival_s, ticks=ticks,
                selected=selected, references=references, sources=list(provenance.values()),
                presentations=presentations, shape=(height, width),
                censored=end < arrival_s + window_seconds,
                no_presented_frames=not any(index >= 0 for index in selected),
                black_frames=sum(index < 0 for index in selected),
                repeated_presentation_frames=sum(left >= 0 and left == right
                    for left, right in zip(selected, selected[1:])),
                encoded_duration_s=len(ticks) / FPS,
                sampling_rule="latest actual presentation at or before each 16fps tick; hold through stalls")


def window_frames(window, sources):
    import numpy as np
    black = np.zeros((*window["shape"], 3), dtype=np.uint8)
    for selected in window["selected"]:
        if selected < 0:
            yield black
        else:
            artifact, index = window["references"][selected]
            yield sources.load(artifact)[0][index]


def _write_media(public, anonymous_id, window, sources):
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw

    path, temporary = public / (anonymous_id + ".mp4"), public / (anonymous_id + ".tmp.mp4")
    count = len(window["ticks"])
    contact_indices = set(np.linspace(0, count - 1, min(9, count), dtype=int).tolist())
    contact_indices.add(min(range(count), key=lambda index: abs(window["ticks"][index] - window["arrival_s"])))
    contacts = []
    try:
        with imageio.get_writer(str(temporary), fps=FPS, codec="libx264", quality=8, macro_block_size=1) as writer:
            for index, frame in enumerate(window_frames(window, sources)):
                relative = window["ticks"][index] - window["arrival_s"]
                # The copy and timestamp strip leave source pixels untouched.
                picture = Image.fromarray(frame)
                annotated = Image.new("RGB", (picture.width, picture.height + 24), "black")
                annotated.paste(picture, (0, 24))
                ImageDraw.Draw(annotated).text((5, 6), "%s   t=%+.3fs" % (anonymous_id, relative), fill="white")
                writer.append_data(np.asarray(annotated))
                if index in contact_indices:
                    thumbnail = picture.resize((256, max(1, round(picture.height * 256 / picture.width))))
                    contacts.append((relative, thumbnail))
        temporary.replace(path)
        thumb_height = contacts[0][1].height
        columns = min(5, len(contacts))
        sheet = Image.new("RGB", (columns * 256, 28 + math.ceil(len(contacts) / columns) * (thumb_height + 24)), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((6, 7), anonymous_id + "   t=0: requested prompt arrival", fill="black")
        for index, (relative, picture) in enumerate(contacts):
            x, y = (index % columns) * 256, 28 + (index // columns) * (thumb_height + 24)
            draw.text((x + 5, y + 4), "t=%+.3fs" % relative, fill="black")
            sheet.paste(picture, (x, y + 24))
        sheet_path = public / (anonymous_id + "-contact.png")
        sheet.save(sheet_path)
        return dict(video=path.name, contact_sheet=sheet_path.name), dict(video_sha256=_hash(path), contact_sheet_sha256=_hash(sheet_path))
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def export_blind_windows(output, window_seconds=8, pre_seconds=1):
    """Export every recorded event from output/direction3/update/report.json."""
    window_seconds, pre_seconds = _finite(window_seconds, "window_seconds"), _finite(pre_seconds, "pre_seconds")
    if window_seconds <= 0 or pre_seconds < 0:
        raise ValueError("window_seconds must be positive and pre_seconds nonnegative")
    root = Path(output).resolve()
    report_path = root / "direction3/update/report.json"
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    events = []
    for sample_index, sample in enumerate(report["samples"]):
        for session_id, records in sample["updates"].items():
            ordered = sorted(enumerate(records), key=lambda pair: (pair[1]["arrival_s"], pair[1]["version"], pair[0]))
            previous = None
            for ordinal, event in ordered:
                events.append((sample_index, session_id, ordinal, event, previous, [row for _, row in ordered]))
                previous = event["prompt"]
    random.Random(SHUFFLE_SEED).shuffle(events)
    destination = report_path.parent / "blind-windows"
    destination.mkdir()  # Refuse to overwrite an existing review or its ratings.
    public = destination / "public"
    public.mkdir()
    manifest = dict(schema_version="ditserve.blind.public.v1", fps=FPS, items=[])
    key = dict(schema_version="ditserve.blind.private.v1", shuffle_seed=SHUFFLE_SEED,
               source_report=str(report_path), source_report_sha256=hashlib.sha256(report_bytes).hexdigest(),
               window_seconds=window_seconds, pre_seconds=pre_seconds, fps=FPS,
               event_denominator=len(events), semantic_status="unrated", items=[])
    sources = FrameSources(report_path)
    options = report.get("effective_config", {})
    shape = (int(options.get("height", 480)), int(options.get("width", 832)))
    failures = 0
    for index, (sample_index, sid, ordinal, event, previous, timeline) in enumerate(events):
        anonymous_id = "W%05d" % (index + 1)
        sample = report["samples"][sample_index]
        public_row = dict(id=anonymous_id, desired_prompt=event["prompt"], previous_prompt=previous,
                          video=None, contact_sheet=None, assessment_status="unrated")
        private_row = dict(id=anonymous_id, sample_index=sample_index, session_id=sid,
                           event_ordinal=ordinal, original_event=event,
                           sample_identity={name: sample.get(name) for name in ("unit", "policy", "buffer_chunks", "seed", "pattern", "session_policy")},
                           observation_end_s=sample.get("observation_end_s"),
                           source_artifacts=[row.get("artifact") for row in sample.get("chunks", [])
                                             if row.get("session_id") == sid])
        try:
            arrival = _finite(event["arrival_s"], "arrival_s")
            private_row.update(requested_start_s=arrival - pre_seconds, requested_end_s=arrival + window_seconds)
            start = arrival - pre_seconds
            end = min(arrival + window_seconds, _finite(sample["observation_end_s"], "observation_end_s"))
            public_row["timeline"] = dict(start_relative_s=start - arrival,
                end_relative_s=end - arrival, frame_interval_s=1 / FPS,
                prompt_requests=[dict(relative_s=row["arrival_s"] - arrival, prompt=row["prompt"])
                                 for row in timeline if start <= row["arrival_s"] < end])
            private_row.update(start_s=start, end_s=end, censored=end < arrival + window_seconds)
            window = reconstruct_window(sample, sid, arrival, window_seconds, pre_seconds, sources, shape)
            media, hashes = _write_media(public, anonymous_id, window, sources)
            public_row.update(media)
            private_row.update({name: value for name, value in window.items()
                                if name not in ("references", "presentations")})
            private_row.update(hashes, media_status="no_presented_frames" if window["no_presented_frames"] else "rendered")
            private_row["selected_frame_ids"] = [None if choice < 0 else window["presentations"][choice]["frame_id"] for choice in window["selected"]]
        except Exception as error:
            failures += 1
            private_row.update(media_status="failed", error_type=type(error).__name__, error=str(error))
            public_row["media_status"] = "unavailable"
        manifest["items"].append(public_row)
        key["items"].append(private_row)
        atomic_json(public / "manifest.json", manifest)
        atomic_json(destination / "private-key.json", key)
    with (public / "ratings.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "reviewer", "target_visible", "first_target_visible_relative_s", "transition_coherence", "notes"])
        writer.writeheader()
        writer.writerows({"id": item["id"]} for item in manifest["items"])
    (public / "README.txt").write_text(
        "Review each anonymous clip against its desired prompt. t=0 is the request arrival.\n"
        "The previous prompt is the preceding request; the timeline shows other requests in the window.\n"
        "Playback is reconstructed at 16 fps from actual presentation timestamps. Stalls hold the previous frame; before the first presentation the screen is black.\n"
        "Leave unavailable or unjudgeable items explicitly noted. Do not delete rows. All score cells begin empty.\n"
        "Judge visible content, and record the first visible target time relative to t=0 when possible. No semantic result has been assigned automatically.\n")
    if not events:
        atomic_json(public / "manifest.json", manifest)
        atomic_json(destination / "private-key.json", key)
    result = dict(status="partial_failure" if failures else "completed", event_denominator=len(events),
                  media_failures=failures, semantic_status="unrated", public_directory=str(public),
                  private_key=str(destination / "private-key.json"), source_report_sha256=key["source_report_sha256"])
    atomic_json(destination / "export-report.json", result)
    return result
