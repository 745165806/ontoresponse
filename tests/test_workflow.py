"""Synthetic end-to-end schema regression, never a real-data score source."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from ontoresponse.common import atomic_json, digest, write_jsonl
from ontoresponse.evaluation.workflow import calibrate, evaluate, freeze
from ontoresponse.evidence.cache import Cache
from ontoresponse.evidence.dataset import load_batch
from ontoresponse.ontology.registry import (
    OntologyRegistry,
    export_evidence,
    validate_evidence,
)
from ontoresponse.train.fit import fit, forward, restore


def fixture(tmp_path):
    OntologyRegistry()
    probes = yaml.safe_load(Path("configs/probes.yaml").read_text())["probes"]
    cfg = {
        "mode": "integration_pretrained",
        "runtime": {"budget": "smoke", "device": "cpu"},
        "output": str(tmp_path / "out"),
        "cache": str(tmp_path / "cache"),
        "crop": {
            "samples": 64600,
            "starts": [0],
            "padding": "repeat",
            "aggregation": "mean_hidden_mean_logits",
        },
        "experts": [{"id": f"unit-expert-{i}"} for i in range(3)],
    }
    cache = Cache(cfg["cache"])
    audit = []
    records = []
    for row in range(16):
        role = ["fusion_train", "fusion_val", "calibration", "test"][row // 4]
        ident = "unit-" + str(row)
        audit.append(
            {
                "id": ident,
                "origin_id": ident,
                "source_group": "g" + str(row),
                "speaker_id": "s" + str(row),
                "label": row % 2,
                "role": role,
                "condition": "aac",
                "source_overlap_status": "unknown",
                "audio": "synthetic_fixture_not_a_file",
            }
        )
        views = []
        for p in [
            {"id": "original", "family": "identity", "operation": "identity"}
        ] + probes:
            audio_spec = {
                "stage": "audio",
                "input_audio_sha256": digest(ident),
                "operation": p,
            }
            key = cache.key(audio_spec)
            cache.put(
                key,
                audio_spec,
                {"waveform": np.ones(160, dtype="float32") * 0.1},
                {"input_audio_sha256": digest(ident)},
            )
            views.append(
                {
                    "probe_id": p["id"],
                    "family": p["family"],
                    "status": "valid",
                    "artifact_key": key,
                }
            )
        experts = {}
        for i, e in enumerate(cfg["experts"]):
            identity = {
                "id": e["id"],
                "checkpoint_sha256": digest(["unit-checkpoint-fixture", i]),
                "crop": cfg["crop"],
                "score_direction": "higher_is_spoof",
            }
            measurements = []
            for v, view in enumerate(views):
                spec = {
                    "stage": "expert",
                    "expert": identity,
                    "artifact_key": view["artifact_key"],
                }
                key = cache.key(spec)
                cache.put(
                    key,
                    spec,
                    {
                        "embedding": np.arange(2 + i, dtype="float32")
                        + row * 0.1
                        + v * 0.01,
                        "margin": np.array([row * 0.3 + v * 0.1], dtype="float32"),
                    },
                    {},
                )
                measurements.append(
                    {"probe_id": view["probe_id"], "status": "valid", "cache_key": key}
                )
            experts[e["id"]] = {
                "status": "complete",
                "identity": identity,
                "measurements": measurements,
            }
        records.append(
            {
                "id": ident,
                "status": "scored",
                "quality": [0.1, 0.2, 0.0, 0.0],
                "views": views,
                "experts": experts,
            }
        )
    index = tmp_path / "index.jsonl"
    ledger = tmp_path / "audit.jsonl"
    write_jsonl(index, records)
    write_jsonl(ledger, audit)
    return cfg, index, ledger, audit, records


def test_train_resume_calibrate_freeze_evaluate(tmp_path):
    cfg, index, ledger, audit, records = fixture(tmp_path)
    r = fit(cfg, index, ledger, "response_mlp", 17, 2, width=16)
    checkpoint = Path(r["checkpoint"])
    cp, model, stats, reg = restore(checkpoint)
    assert r["save_load_exact"] and cp["stats"].fit_role == "fusion_train"
    continued = fit(
        cfg, index, ledger, "response_mlp", 17, 3, width=16, resume=checkpoint
    )
    assert continued["optimizer_steps"] == 3
    calibrate(cfg, checkpoint, index, ledger)
    test = tmp_path / "test.jsonl"
    write_jsonl(test, [r for r in audit if r["role"] == "test"])
    lock = tmp_path / "freeze.json"
    freeze(cfg, checkpoint, test, lock)
    result = evaluate(cfg, checkpoint, index, test, lock, tmp_path / "evaluation")
    assert result["N_requested"] == 4 and result["N_failed"] == 0
    payload = json.loads(lock.read_text())
    payload["test_manifest_sha256"] = "drift"
    atomic_json(lock, payload)
    with pytest.raises(ValueError):
        evaluate(cfg, checkpoint, index, test, lock)


def test_audit_mutation_does_not_change_tensor_inputs_and_stats(tmp_path):
    cfg, index, ledger, audit, records = fixture(tmp_path)
    before = load_batch(cfg, index, ledger, "test")
    for r in audit:
        if r["role"] == "test":
            r.update(
                label=1 - r["label"],
                attack="changed",
                environment_truth="changed",
                speaker_id="new-" + r["speaker_id"],
            )
    write_jsonl(ledger, audit)
    after = load_batch(cfg, index, ledger, "test")
    for a, b in zip(before.hidden, after.hidden):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(before.scores, after.scores, atol=0, rtol=0)
    # Fit cannot consume test labels/features; fitting roles alone determine normalizers.
    train = load_batch(cfg, index, ledger, "fusion_train")
    from reference.core import ExpertStats

    stats = ExpertStats.fit(*train.inputs(), role="fusion_train")
    snap = copy.deepcopy(stats)
    stats.transform([x + 1e6 for x in before.hidden], before.scores + 1e6, before.valid)
    assert all(torch.equal(a, b) for a, b in zip(snap.hidden_means, stats.hidden_means))


def test_missing_all_requests_still_return_review(tmp_path):
    cfg, index, ledger, audit, records = fixture(tmp_path)
    r = fit(cfg, index, ledger, "typed_response", 17, 1, width=16)
    cp, model, stats, reg = restore(r["checkpoint"])
    for row in records:
        if row["id"] == "unit-12":
            row["experts"] = {
                e["id"]: {"status": "blocked", "error": "synthetic missing fixture"}
                for e in cfg["experts"]
            }
            row["status"] = "review"
    write_jsonl(index, records)
    batch = load_batch(cfg, index, ledger, "test", cp["dims"])
    out = forward(model, batch, stats, reg)
    assert (
        len(out.logits) == 4
        and not out.observed[0]
        and torch.isnan(out.probabilities()[0])
    )


def test_identity_and_registry_parameter_drift_rejected(tmp_path):
    cfg, index, ledger, audit, records = fixture(tmp_path)
    validate_evidence(cfg, records, Cache(cfg["cache"]))
    records[0]["experts"]["unit-expert-0"]["identity"]["crop"]["starts"] = [1]
    with pytest.raises(ValueError):
        validate_evidence(cfg, records, Cache(cfg["cache"]))


def test_rdf_actual_observations_export_and_semantics(tmp_path):
    cfg, index, ledger, audit, records = fixture(tmp_path)
    path = tmp_path / "evidence.ttl"
    r = export_evidence(cfg, index, path)
    assert r["registry_hash"] == OntologyRegistry().digest
    text = path.read_text()
    assert (
        "prov:wasGeneratedBy" in text
        and "orf:hasBefore" in text
        and "orf:hasAfter" in text
    )
    assert "orf:same_probe_cross_expert" in text


def test_freeze_detects_audio_replacement_without_manifest_change(tmp_path):
    from ontoresponse.evaluation.workflow import check_lock

    cfg, index, ledger, audit, records = fixture(tmp_path)
    r = fit(cfg, index, ledger, "response_mlp", 17, 1, width=16)
    cp = Path(r["checkpoint"])
    calibrate(cfg, cp, index, ledger)
    content = tmp_path / "frozen-audio-fixture.bin"
    content.write_bytes(b"unit fixture content")
    rows = [row for row in audit if row["role"] == "test"]
    for row in rows:
        row["audio"] = str(content)
    manifest = tmp_path / "test.jsonl"
    write_jsonl(manifest, rows)
    lock = freeze(cfg, cp, manifest, tmp_path / "freeze.json")
    check_lock(cfg, cp, manifest, lock)
    content.write_bytes(b"replaced unit fixture content")
    with pytest.raises(ValueError, match="waveform content drift"):
        check_lock(cfg, cp, manifest, lock)


def test_frozen_training_and_calibration_cannot_be_overwritten(tmp_path):
    cfg, index, ledger, audit, records = fixture(tmp_path)
    result = fit(cfg, index, ledger, "response_mlp", 17, 1, width=16)
    cp = Path(result["checkpoint"])
    calibrate(cfg, cp, index, ledger)
    test = tmp_path / "test.jsonl"
    write_jsonl(test, [row for row in audit if row["role"] == "test"])
    freeze(cfg, cp, test)
    from ontoresponse.common import sha256

    before = sha256(cp)
    cal_before = sha256(cp.parent / "calibration.json")
    with pytest.raises(ValueError, match="Frozen run"):
        fit(cfg, index, ledger, "response_mlp", 17, 2, width=16, resume=cp)
    with pytest.raises(ValueError, match="Frozen calibration"):
        calibrate(cfg, cp, index, ledger)
    assert sha256(cp) == before and sha256(cp.parent / "calibration.json") == cal_before
