"""Deterministic historical T materialization, separate from diagnostic probes P."""

import copy
import os
import tempfile
from pathlib import Path

import soundfile as sf

from ..common import (
    Blocked,
    atomic_json,
    budget,
    digest,
    read_jsonl,
    write_jsonl,
)
from ..transforms.probes import apply_probe
from .audio import read_audio
from .prepare import validate_splits

OPERATIONS = {
    "identity": {"id": "identity", "operation": "identity"},
    "lowpass4500": {"id": "lowpass4500", "operation": "lowpass", "cutoff_hz": 4500},
    "white25": {"id": "white25", "operation": "additive_white_noise", "snr_db": 25},
    "MP3_48": {
        "id": "MP3_48",
        "operation": "ffmpeg_roundtrip",
        "encoder": "libmp3lame",
        "bitrate_kbps": 48,
    },
    "AAC_64kbps": {
        "id": "AAC_64kbps",
        "operation": "ffmpeg_roundtrip",
        "encoder": "aac",
        "bitrate_kbps": 64,
    },
    "Opus_32kbps": {
        "id": "Opus_32kbps",
        "operation": "ffmpeg_roundtrip",
        "encoder": "libopus",
        "bitrate_kbps": 32,
    },
}


def conditions_for_origin(row, p):
    if row["role"] == "fusion_train":
        choice = min(
            p["training_t_candidates"],
            key=lambda c: digest(
                [p["environment_assignment_seed"], row["origin_id"], c]
            ),
        )
        return ["identity", choice]
    return list(p["conditions_by_role"][row["role"]])


def materialize_versions(cfg, base_manifest):
    """Only a caller-audited origin manifest; v2 preregistration still checks sources.

    No expert is run here. Every failed request remains in the audit ledger, and
    prevents preregistration rather than disappearing or receiving random data.
    """
    p = cfg["pilot"]
    root = Path(cfg["output"])
    rows = read_jsonl(base_manifest)
    validate_splits(rows)
    budget(cfg["runtime"]["budget"], len({r["origin_id"] for r in rows}))
    if len({r["origin_id"] for r in rows}) != len(rows):
        raise ValueError("Base ledger must contain one row/origin")
    if root.exists() and any(
        (root / name).exists()
        for name in (
            "audit_manifest.jsonl",
            "experiment-start.json",
            "preregistration.json",
            "current_audio",
        )
    ):
        raise ValueError("Never overwrite materialized/frozen development inputs")
    counts = {
        role: sum(r["role"] == role for r in rows) for role in {r["role"] for r in rows}
    }
    if counts != p["origins_by_role"]:
        raise Blocked("Base origin budget differs from recipe")
    if any(c not in OPERATIONS for row in rows for c in conditions_for_origin(row, p)):
        raise ValueError("Unknown historical environment condition in recipe")
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "current_audio"
    destination.mkdir()
    audit = []
    total_bytes = 0
    for row in rows:
        for condition in conditions_for_origin(row, p):
            ident = "current-" + digest([row["origin_id"], condition])[:32]
            target = destination / (ident + ".wav")
            item = copy.deepcopy(row)
            item.update(
                id=ident,
                condition=condition,
                audio=str(target),
                historical_processing=[OPERATIONS[condition]],
                generation_status="pending",
                input_base_id=row["id"],
                T_truth_forward_input=False,
            )
            try:
                x = read_audio(row["audio"])
                y, meta = apply_probe(x, OPERATIONS[condition])
                # WAV FLOAT has a small header; quota before mutation, exact after.
                if total_bytes + y.nbytes + 4096 > p["max_output_bytes"]:
                    raise Blocked("Current audio output quota reached")
                fd, name = tempfile.mkstemp(dir=destination, suffix=".wav")
                os.close(fd)
                try:
                    sf.write(name, y, 16000, subtype="FLOAT")
                    os.replace(name, target)
                finally:
                    if os.path.exists(name):
                        os.unlink(name)
                total_bytes += target.stat().st_size
                item.update(
                    generation_status="valid",
                    processing=meta,
                    length_preserved=len(x) == len(y),
                )
            except Exception as err:
                item.update(generation_status="failed", generation_error=str(err))
            audit.append(item)
    write_jsonl(root / "audit_manifest.jsonl", audit)
    write_jsonl(
        root / "inference_manifest.jsonl",
        [
            {k: r[k] for k in ("id", "origin_id", "source_group", "audio")}
            for r in audit
        ],
    )
    result = {
        "status": "materialized_only_not_source_certified",
        "origins": len(rows),
        "requests": len(audit),
        "failed": sum(r["generation_status"] != "valid" for r in audit),
        "current_audio_bytes": total_bytes,
        "historical_T_and_diagnostic_P_separate": True,
        "expert_inference_performed": False,
        "new_confirmation_scores_read": False,
    }
    atomic_json(root / "version-generation.json", result)
    return result
