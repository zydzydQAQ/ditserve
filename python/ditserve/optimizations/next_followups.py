"""Run only the conditional extensions supported by completed paired evidence."""
from pathlib import Path

from ..research.common import atomic_json, file_digest, read_json


def read_admission(binding):
    path = Path(binding["path"])
    if file_digest(path) != binding["sha256"]:
        raise ValueError("conditional admission changed")
    value = read_json(path)
    evidence = value["trigger_evidence"]
    bindings = evidence.get("immutable_bindings", [])
    if (evidence.get("vae_trigger") or evidence.get("encoder_trigger")) and not bindings:
        raise ValueError("conditional trigger lacks immutable supporting evidence")
    for source in bindings:
        if file_digest(source["path"]) != source["sha256"]:
            raise ValueError("conditional trigger source changed: " + source["path"])
    return value


def _skip(name, trigger, evidence, *, unavailable=False):
    reason = ("trigger observed but V100 was not idle within the resource window" if unavailable else
              "no complete authorized baseline evidence" if not evidence.get("admitted_sources") else
              "predeclared bottleneck trigger absent in completed baseline traces")
    return dict(status="stopped", candidate=name, triggered=bool(trigger),
        decision="未验证" if unavailable or not evidence.get("admitted_sources") else "简单策略已足够",
        local_qualified=False, online_qualified=False, online_evaluation_complete=False,
        gpu_work_started=False, reason=reason,
        scope="conditional branch only; absence of this trigger is not universal evidence of no benefit")


def run_followups(context):
    from .next_conditional import run_video_vae
    from .next_encoder import run_encoder
    admission = context.conditional_admission
    evidence = admission["trigger_evidence"]
    path = Path(context.output) / "conditional.json"
    record = dict(schema_version="ditserve.optimizations.next_conditional.v1", status="running",
        decision="未验证", local_qualified=False, online_qualified=False,
        trigger_evidence=evidence, admitted_devices=admission["devices"], active_component=None,
        scope="independent conditional comparisons against eager paired_008 K4; no combined default is qualified",
        adaptive_k=dict(status="stopped", decision="未验证",
            reason="no newly measured conflicting K1/K4/K8 tradeoff; fixed K4 retained"),
        cache_memory=dict(status="stopped", decision="未验证",
            reason="no demonstrated paused-state capacity boundary in this request-at-a-time executor"))
    atomic_json(path, record)
    try:
        for key, trigger_key, runner in (("video_vae", "vae_trigger", run_video_vae),
                                          ("text_encoder", "encoder_trigger", run_encoder)):
            triggered = evidence.get(trigger_key) is True
            unavailable = key == "text_encoder" and triggered and not admission["auxiliary_encoder"]
            if not triggered or unavailable:
                record[key] = _skip(key, triggered, evidence, unavailable=unavailable)
            else:
                context.check_budget()
                record["active_component"] = key
                atomic_json(path, record)
                record[key] = runner(context)
                record[key]["triggered"] = True
                filename = "conditional-video-vae.json" if key == "video_vae" else "conditional-text-encoder.json"
                source = Path(context.output) / filename
                record[key]["component_evidence"] = dict(path=str(source), sha256=file_digest(source))
                if record[key].get("decoder_cleanup_passed") is False:
                    raise RuntimeError("conditional component cleanup unconfirmed; further work stopped")
            record["active_component"] = None
            atomic_json(path, record)
        components = [record[key] for key in ("video_vae", "text_encoder")]
        record.update(status="completed", independently_qualified=[key for key in ("video_vae", "text_encoder")
            if record[key].get("online_qualified") is True],
            decision="值得继续" if any(row.get("online_qualified") for row in components) else
                "未验证" if any(row.get("decision") == "未验证" or
                    row.get("status") in ("failed", "incomplete", "budget_stopped") for row in components) else
                "简单策略已足够" if all(row.get("decision") == "简单策略已足够" for row in components) else "当前配置无收益")
        atomic_json(path, record)
        return record
    except BaseException as error:
        record.update(status="incomplete", decision="未验证", reason=type(error).__name__ + ": " + str(error))
        atomic_json(path, record)
        raise
