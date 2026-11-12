"""New bounded integration ledger; no reuse of consumed exploratory test origins."""

import json
from collections import defaultdict
from pathlib import Path

from ..common import Blocked, atomic_json, digest, read_jsonl, sha256, write_jsonl
from ..transforms.probes import audio_hash
from .audio import read_audio
from .prepare import validate_splits


def check_pilot_budget(cfg):
    p = cfg["pilot"]
    runtime = cfg["runtime"]
    if not p.get("enabled", False):
        raise Blocked("Pilot needs explicit enabled=true budget config")
    if runtime["budget"] != "pilot" or not 21 <= p["max_optimizer_steps"] <= 2000:
        raise ValueError(
            "Explicit bounded pilot optimizer budget required; smoke remains <=20"
        )
    total = sum(p["origins_by_role"].values())
    if not 1 <= total <= p["max_origins"] <= 1024:
        raise ValueError("Pilot origin cap exceeded")
    if not 1 <= p["min_epochs"] <= p["max_epochs"] <= 200 or p["patience_epochs"] < 1:
        raise ValueError("Invalid convergence budget")
    if p["batch_size"] < 1 or len(set(p["seeds"])) != len(p["seeds"]):
        raise ValueError("Invalid batching/seeds")
    if "rawnet2" in [e["id"] for e in cfg["experts"]]:
        raise Blocked("Unknown-provenance RawNet2 is quarantined")
    if cfg["mode"] != "integration_pretrained":
        raise Blocked(
            "Pilot is not research_holdout: cross-speaker source map / generic SSL exposure uncertified"
        )
    return total


def prepare_pilot(cfg):
    check_pilot_budget(cfg)
    p = cfg["pilot"]
    out = Path(cfg["output"])
    if (out / "audit_manifest.jsonl").exists():
        raise ValueError("Pilot ledger exists; do not resplit a consumed experiment")
    audit = json.loads(Path(p["supervision_audit"]).read_text())
    selected = {e["id"] for e in cfg["experts"]}
    verified = [e for e in audit["experts"] if e["id"] in selected]
    if len(verified) != len(selected) or not all(
        e["provenance_pass"] for e in verified
    ):
        raise Blocked("Expert source audit has not passed")
    exposed_ids = set()
    exposed_speakers = set()
    for e in verified:
        for role in ("fit", "source_val"):
            exposed_ids.update(e["supervision"][role]["ids"])
            exposed_speakers.update(e["supervision"][role]["speakers"])
    root = Path(cfg["data"]["asv2019"])
    protocol = root / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
    old = read_jsonl(p["prior_development_ledger"])
    old_ids = {r["id"] for r in old}
    old_speakers = {r["speaker_id"] for r in old}
    data = []
    for line in protocol.read_text().splitlines():
        speaker, ident, _, attack, label = line.split()
        if speaker in exposed_speakers or ident in exposed_ids or ident in old_ids:
            continue
        data.append(
            {
                "id": ident,
                "origin_id": ident,
                "source_group": speaker,
                "speaker_id": speaker,
                "label": int(label == "spoof"),
                "attack": attack,
                "audio": str(root / f"ASVspoof2019_LA_dev/flac/{ident}.flac"),
                "condition": "no_new_processing",
                "source_overlap_status": "unknown_cross_speaker_mapping",
            }
        )
    all_speakers = {r["speaker_id"] for r in data}
    fresh = sorted(
        all_speakers - old_speakers, key=lambda s: digest([p["split_seed"], s])
    )
    labels_by_speaker = {
        s: {r["label"] for r in data if r["speaker_id"] == s} for s in all_speakers
    }
    mixed = {s for s, labels in labels_by_speaker.items() if labels == {0, 1}}
    fresh_mixed = [s for s in fresh if s in mixed]
    if not fresh_mixed:
        raise Blocked("Fresh test speaker pool has no spoof/bonafide mixed group")
    test_speakers = {fresh_mixed[0], *([s for s in fresh if s != fresh_mixed[0]][:2])}
    if len(test_speakers) < 3:
        raise Blocked(
            "Need >=3 new test speaker groups, excluding upstream selection and prior dev groups"
        )
    rest = sorted(
        all_speakers - test_speakers, key=lambda s: digest([p["split_seed"], s])
    )
    if len(rest) < 9:
        raise Blocked("Insufficient grouped dev pool")
    roles = {s: "test" for s in test_speakers}
    for role in ("fusion_val", "calibration"):
        eligible = [s for s in rest if s in mixed]
        if not eligible:
            raise Blocked("Each selection/calibration role needs both labels")
        group = [eligible[0]] + [
            s for s in rest if s != eligible[0] and s not in mixed
        ][:2]
        if len(group) < 3:
            group += [s for s in rest if s not in group][: 3 - len(group)]
        for s in group:
            roles[s] = role
            rest.remove(s)
    roles.update({s: "fusion_train" for s in rest})
    buckets = defaultdict(list)
    for r in data:
        r["role"] = roles[r["speaker_id"]]
        buckets[(r["role"], r["label"], r["speaker_id"])].append(r)
    for bucket in buckets.values():
        bucket.sort(key=lambda r: digest([p["split_seed"], r["id"]]))
    chosen = []
    for role, n in p["origins_by_role"].items():
        speakers = sorted(s for s, v in roles.items() if v == role)
        for i in range(n):
            label = i % 2
            available = [s for s in speakers if buckets[(role, label, s)]]
            if not available:
                raise Blocked("Insufficient label coverage within fixed group budget")
            speaker = available[(i // 2) % len(available)]
            chosen.append(buckets[(role, label, speaker)].pop())
    summary = validate_splits(chosen)
    seen_audio = {}
    for r in chosen:
        r["audio_file_sha256"] = sha256(r["audio"])
        r["waveform_sha256"] = audio_hash(read_audio(r["audio"]))
        if (
            r["waveform_sha256"] in seen_audio
            and seen_audio[r["waveform_sha256"]] != r["role"]
        ):
            raise ValueError("Duplicate waveform across roles")
        seen_audio[r["waveform_sha256"]] = r["role"]
    consumed = json.loads(Path(p["consumed_registry"]).read_text())
    if {r["origin_id"] for r in chosen} & set(consumed["origins"]):
        raise ValueError("Consumed external exploration origin reused")
    if {r["waveform_sha256"] for r in chosen} & set(
        consumed.get("waveform_sha256", [])
    ):
        raise ValueError("Consumed waveform reused")
    write_jsonl(out / "audit_manifest.jsonl", chosen)
    write_jsonl(
        out / "inference_manifest.jsonl",
        [
            {k: r[k] for k in ("id", "origin_id", "source_group", "audio")}
            for r in chosen
        ],
    )
    result = {
        "status": "prepared_new_integration_pilot",
        "origins": len(chosen),
        "roles": summary,
        "expert_ids": sorted(selected),
        "excluded_upstream_speakers": sorted(exposed_speakers),
        "test_speakers": sorted(test_speakers),
        "upstream_exact_ID_overlap": 0,
        "upstream_speaker_overlap": 0,
        "prior_development_test_speaker_overlap": 0,
        "speaker_label_counts": {
            s: {
                str(y): sum(r["speaker_id"] == s and r["label"] == y for r in chosen)
                for y in (0, 1)
            }
            for s in sorted(roles)
        },
        "test_spoof_speakers": len(
            {r["speaker_id"] for r in chosen if r["role"] == "test" and r["label"] == 1}
        ),
        "group_limitation": "official dev has bonafide-only speaker groups; fresh test has only one spoof speaker group, so no broad-generalization claim",
        "waveform_cross_role_duplicates": 0,
        "consumed_exploration_origin_overlap": 0,
        "protocol_sha256": sha256(protocol),
        "ledger_sha256": sha256(out / "audit_manifest.jsonl"),
        "research_holdout_pass": False,
        "scope": "integration pilot only; speaker-conservative grouping; cross-speaker source maps and generic SSL exposure remain unknown",
    }
    atomic_json(out / "data-audit.json", result)
    return result
