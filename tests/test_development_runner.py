"""Synthetic integration only; no synthetic expert replaces a real data run."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from test_workflow import fixture

from ontoresponse.common import Blocked, atomic_json, sha256, write_jsonl
from ontoresponse.data.connected import connected_groups
from ontoresponse.evidence.dataset import load_batch
from ontoresponse.ontology.registry import OntologyRegistry
from ontoresponse.preregistration import freeze_development, require_preregistration
from ontoresponse.train.development import (
    epoch_schedule,
    h1_decision,
    select_family_trials,
    validation_metrics,
)
from ontoresponse.train.pilot import fit_one, lock_and_evaluate


def batches(tmp_path):
    cfg, index, ledger, audit, records = fixture(tmp_path)
    train = load_batch(cfg, index, ledger, "fusion_train")
    val = load_batch(cfg, index, ledger, "fusion_val")

    # Three valid historical conditions per same origin; views remain views.
    def versions(b, conditions):
        ix = torch.arange(len(b.ids)).repeat(len(conditions))
        return replace(
            b,
            ids=[ident + "-" + c for c in conditions for ident in b.ids],
            hidden=[h[ix] for h in b.hidden],
            scores=b.scores[ix],
            valid=b.valid[ix],
            quality=b.quality[ix],
            labels=b.labels[ix],
            origins=b.origins * len(conditions),
            groups=b.groups * len(conditions),
            audit=[
                dict(r, condition=c, id=r["id"] + "-" + c)
                for c in conditions
                for r in b.audit
            ],
        )

    train = versions(train, ["identity", "noise"])
    val = versions(val, ["identity", "aac", "opus"])
    cfg["output"] = str(tmp_path / "experiment")
    Path(cfg["output"]).mkdir()
    write_jsonl(Path(cfg["output"]) / "audit_manifest.jsonl", train.audit + val.audit)
    base = yaml.safe_load(Path("configs/pilot-20261110.yaml").read_text())["pilot"]
    cfg["pilot"] = copy.deepcopy(base)
    cfg["pilot"].update(
        width=8,
        max_epochs=6,
        min_epochs=6,
        patience_epochs=20,
        max_optimizer_steps=24,
        batch_size=1,
        validation_eer_conditions=["aac", "opus"],
        require_convergence=True,
    )
    return cfg, train, val


def test_label_free_architecture_independent_epoch_schedule(tmp_path):
    _, train, val = batches(tmp_path)
    order = epoch_schedule(train, 17, 1)
    assert len(order) == len(set(train.origins)) == 4
    assert len({train.origins[i] for i in order}) == 4
    mutated = replace(
        train,
        labels=1 - train.labels,
        audit=[dict(r, label=1 - r["label"], attack="changed") for r in train.audit],
    )
    torch.manual_seed(999)
    torch.randn(1234)
    assert order.tolist() == epoch_schedule(mutated, 17, 1).tolist()
    train.valid[[0, 4]] = False
    with pytest.raises(Blocked, match="No valid"):
        epoch_schedule(train, 17, 1)


def test_macro_eer_differs_from_pooled_and_does_not_use_identity(tmp_path):
    _, _, val = batches(tmp_path)
    # Each condition perfect but incomparable offsets; pooled EER is misleading.
    scores = torch.tensor(
        [100.0, 101.0, 100.0, 101.0, -100.0, -99.0, -100.0, -99.0, 1.0, 0.0, 1.0, 0.0]
    )
    result = validation_metrics(val, scores, ["aac", "opus"])
    assert result["eer"] == 0.5 and result["per_condition"]["identity"]["eer"] == 0
    assert (
        result["per_condition"]["aac"]["eer"] == 0
        and result["per_condition"]["opus"]["eer"] == 1
    )
    changed = scores.clone()
    changed[:4] = -scores[:4]
    assert validation_metrics(val, changed, ["aac", "opus"])["eer"] == result["eer"]
    with pytest.raises(Blocked):
        validation_metrics(val, scores, ["absent"])
    with pytest.raises(ValueError):
        validation_metrics(
            replace(val, audit=[dict(r, role="test") for r in val.audit]),
            scores,
            ["aac"],
        )
    mask = torch.ones(12, dtype=torch.bool)
    mask[1] = False
    with pytest.raises(Blocked):
        validation_metrics(val, scores, ["aac"], mask)


@pytest.mark.parametrize("seed", [17, 29, 43])
@pytest.mark.parametrize(
    "model",
    [
        "score_lr",
        "score_mlp",
        "original_mlp",
        "tta_stats",
        "all_views_mlp",
        "response_mlp",
    ],
)
def test_six_baselines_multiversion_training_restore_and_condition_metrics(
    tmp_path, model, seed
):
    cfg, train, val = batches(tmp_path)
    result = fit_one(cfg, model, seed, train, val, OntologyRegistry())
    cp = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
    assert (
        result["steps"] == 24
        and result["save_load_exact"]
        and not result["convergence_claim"]
    )
    assert result["val_metrics"]["primary_conditions"] == ["aac", "opus"]
    assert all(
        h["selected_train_origins"] == 4 and h["train_measurement_count"] == 8
        for h in cp["history"]
    )
    assert all(
        np.isfinite(h["gradient_norm_max"]) and h["learning_rate"] > 0
        for h in cp["history"]
    )
    assert cp["stats"].fit_role == "fusion_train"


def results_for_gate():
    return [
        dict(
            model=model,
            seed=seed,
            trial_id="t1",
            val_eer=eer,
            val_bce=0.2,
            parameters=100,
            convergence_claim=True,
            stop_reason="validation_patience",
        )
        for model, eer in [
            ("original_mlp", 0.2),
            ("response_mlp", 0.1),
            ("all_views_mlp", 0.15),
        ]
        for seed in [17, 29, 43]
    ]


def test_unconverged_and_insufficient_never_support_h1_or_load_test(
    tmp_path, monkeypatch
):
    cfg, train, val = batches(tmp_path)
    p = cfg["pilot"]
    p["h1_models"] = ["original_mlp", "response_mlp", "all_views_mlp"]
    p["h1_gate"]["min_groups_per_class"] = 2
    # Two mixed-label clusters: no class-degenerate bootstrap draws.
    val = replace(
        val, groups=["mixed-" + str((i % 4) // 2) for i in range(len(val.ids))]
    )
    runs = results_for_gate()
    assert h1_decision(cfg, runs, val)["H2_allowed"]
    runs[0]["convergence_claim"] = False
    gate = h1_decision(cfg, runs, val)
    assert gate["status"] == "INCONCLUSIVE_CONVERGENCE" and not gate["H2_allowed"]
    p["h1_gate"]["min_groups_per_class"] = 5
    gate = h1_decision(cfg, runs, val)
    assert gate["status"] == "INSUFFICIENT_DISCRIMINATION" and not gate["H2_allowed"]
    p["require_h1_support_before_confirmation"] = True
    atomic_json(Path(cfg["output"]) / "H1-gate.json", gate)

    def forbid(*args, **kwargs):
        raise AssertionError("No calibration/val/test load allowed")

    monkeypatch.setattr("ontoresponse.train.pilot.load_batch", forbid)
    assert lock_and_evaluate(cfg, runs) == []
    assert (
        json.loads((Path(cfg["output"]) / "confirmation-not-run.json").read_text())[
            "test_loaded"
        ]
        is False
    )


def test_trial_selection_requires_common_complete_seed_budget():
    runs = results_for_gate()
    selected, _ = select_family_trials(runs, [17, 29, 43], ["original_mlp"])
    assert len(selected["original_mlp"]) == 3
    with pytest.raises(ValueError):
        select_family_trials(runs[:-1], [17, 29, 43], ["all_views_mlp"])
    with pytest.raises(ValueError):
        select_family_trials(runs + [runs[0]], [17, 29, 43], ["original_mlp"])


def prereg_fixture(tmp_path):
    root = tmp_path / "seal"
    root.mkdir()
    cfg = {
        "output": str(root),
        "experts": [{"id": "fixture", "checkpoint": str(root / "cp")}],
        "pilot": {
            "protocol_version": 2,
            "preregistration": str(root / "preregister.json"),
            "source_map_manifest": str(root / "mapping.json"),
            "official_map_archive": str(root / "official.fixture"),
            "consumption_ledger": str(root / "consumed.jsonl"),
            "supervision_audit": str(root / "supervision.json"),
            "require_convergence": True,
            "require_h1_support_before_confirmation": True,
            "seeds": [17, 29, 43],
            "h1_models": [
                "score_lr",
                "score_mlp",
                "original_mlp",
                "tta_stats",
                "all_views_mlp",
                "response_mlp",
            ],
            "candidate_grid": [{"width": 8, "learning_rate": 0.001}],
            "trials_per_family": 1,
            "validation_eer_conditions": ["aac", "opus"],
            "min_epochs": 40,
            "max_optimizer_steps": 2000,
            "conditions_by_role": {
                "fusion_train": ["identity", "noise"],
                "fusion_val": ["identity", "aac", "opus"],
                "calibration": ["identity"],
                "test": ["identity", "aac", "opus"],
            },
            "origins_by_role": {
                "fusion_train": 4,
                "fusion_val": 4,
                "calibration": 4,
                "test": 4,
            },
            "h1_gate": {"min_groups_per_class": 2},
        },
    }
    (root / "cp").write_text(
        "SYNTHETIC fixture checkpoint not used for expert inference"
    )
    (root / "official.fixture").write_text(
        "SYNTHETIC source mapping fixture, not certified real data"
    )
    universe = [
        {
            "id": f"o{i}",
            "origin_id": f"o{i}",
            "speaker_id": f"s{i}",
            "source_ids": [f"source{i}"],
        }
        for i in range(17)
    ]
    cc = connected_groups(universe, {r["id"]: r["source_ids"] for r in universe})
    audio = root / "wave.fixture"
    audio.write_bytes(b"SYNTHETIC waveform hash fixture; never used for real inference")
    rows = []
    for i in range(16):
        role = ["fusion_train", "fusion_val", "calibration", "test"][i // 4]
        for condition in cfg["pilot"]["conditions_by_role"][role]:
            rows.append(
                dict(
                    id=f"o{i}_{condition}",
                    origin_id=f"o{i}",
                    speaker_id=f"s{i}",
                    source_group=cc[f"o{i}"],
                    role=role,
                    label=i % 2,
                    condition=condition,
                    audio=str(audio),
                    audio_file_sha256=sha256(audio),
                )
            )
    write_jsonl(root / "audit_manifest.jsonl", rows)
    write_jsonl(
        root / "inference_manifest.jsonl",
        [{k: r[k] for k in ("id", "origin_id", "source_group", "audio")} for r in rows],
    )
    write_jsonl(
        root / "consumed.jsonl", [dict(id="o16", origin_id="o16", speaker_id="s16")]
    )
    atomic_json(root / "data-audit.json", {"source_mapping_complete": True})
    atomic_json(
        root / "mapping.json",
        {
            "mapping_complete": True,
            "official_archive_sha256": sha256(root / "official.fixture"),
            "recordings": universe,
        },
    )
    atomic_json(
        root / "supervision.json",
        {
            "experts": [
                {
                    "id": "fixture",
                    "provenance_pass": True,
                    "checkpoint_sha256": sha256(root / "cp"),
                    "supervision": {
                        "fit": {"ids": [], "speakers": []},
                        "source_val": {"ids": [], "speakers": []},
                    },
                }
            ]
        },
    )
    return cfg, root


def test_preregister_freezes_before_outputs_and_rejects_config_data_drift(tmp_path):
    cfg, root = prereg_fixture(tmp_path)
    lock = freeze_development(cfg)
    assert lock["data"]["source_mapping_complete"] and not lock["test_used"]
    require_preregistration(cfg)
    changed = copy.deepcopy(cfg)
    changed["pilot"]["validation_eer_conditions"] = ["identity"]
    with pytest.raises(Blocked, match="drift"):
        require_preregistration(changed)
    with pytest.raises(ValueError, match="exists"):
        freeze_development(cfg)
    (root / "audit_manifest.jsonl").write_text("changed")
    with pytest.raises(Blocked, match="drift"):
        require_preregistration(cfg)


def test_preregister_rejects_cross_speaker_consumed_source_component(tmp_path):
    cfg, root = prereg_fixture(tmp_path)
    path = root / "mapping.json"
    mapping = json.loads(path.read_text())
    mapping["recordings"][16]["source_ids"] = ["source0"]
    atomic_json(path, mapping)
    cc = connected_groups(
        mapping["recordings"], {r["id"]: r["source_ids"] for r in mapping["recordings"]}
    )
    rows = [
        json.loads(l) for l in (root / "audit_manifest.jsonl").read_text().splitlines()
    ]
    for row in rows:
        row["source_group"] = cc[row["origin_id"]]
    write_jsonl(root / "audit_manifest.jsonl", rows)
    with pytest.raises(Blocked, match="Consumed"):
        freeze_development(cfg)


def test_preregister_rejects_late_lock_and_smoke_budget(tmp_path):
    cfg, root = prereg_fixture(tmp_path)
    (root / "runs").mkdir()
    with pytest.raises(Blocked, match="precede"):
        freeze_development(cfg)
    cfg["pilot"]["max_optimizer_steps"] = 20
    with pytest.raises(Blocked, match="smoke"):
        freeze_development(cfg)


def test_preregister_rejects_waveform_changes_after_freeze(tmp_path):
    cfg, root = prereg_fixture(tmp_path)
    freeze_development(cfg)
    (root / "wave.fixture").write_bytes(b"changed waveform")
    with pytest.raises(Blocked, match="current_audio"):
        require_preregistration(cfg)


def test_materialized_T_rule_is_label_independent_and_failures_are_retained(tmp_path):
    import soundfile as sf

    from ontoresponse.common import read_jsonl
    from ontoresponse.data.development import (
        conditions_for_origin,
        materialize_versions,
    )

    wave = tmp_path / "base.wav"
    x = (0.1 * np.sin(np.arange(3200) * 0.1)).astype("float32")
    sf.write(wave, x, 16000, subtype="FLOAT")
    rows = [
        dict(
            id=f"base{i}",
            origin_id=f"o{i}",
            source_group=f"g{i}",
            speaker_id=f"s{i}",
            label=i % 2,
            role=role,
            audio=str(wave),
        )
        for i, role in enumerate(
            ["fusion_train", "fusion_train", "fusion_val", "fusion_val"]
        )
    ]
    base = tmp_path / "base.jsonl"
    write_jsonl(base, rows)
    cfg = {
        "output": str(tmp_path / "versions"),
        "runtime": {"budget": "smoke"},
        "pilot": {
            "origins_by_role": {"fusion_train": 2, "fusion_val": 2},
            "training_t_candidates": ["white25", "lowpass4500"],
            "environment_assignment_seed": 271828,
            "conditions_by_role": {"fusion_val": ["identity"]},
            "max_output_bytes": 2**20,
        },
    }
    assert conditions_for_origin(rows[0], cfg["pilot"]) == conditions_for_origin(
        dict(rows[0], label=1), cfg["pilot"]
    )
    result = materialize_versions(cfg, base)
    assert (
        result["requests"] == 6
        and result["failed"] == 0
        and not result["expert_inference_performed"]
    )
    audit = read_jsonl(Path(cfg["output"]) / "audit_manifest.jsonl")
    inference = read_jsonl(Path(cfg["output"]) / "inference_manifest.jsonl")
    assert all(
        set(r) == {"id", "origin_id", "source_group", "audio"} for r in inference
    )
    assert all(
        r["length_preserved"] and r["audio_file_sha256"] == sha256(r["audio"])
        for r in audit
    )
    assert all(len(sf.read(r["audio"])[0]) == len(x) for r in audit)
    with pytest.raises(ValueError, match="overwrite"):
        materialize_versions(cfg, base)
    bad = copy.deepcopy(cfg)
    bad["output"] = str(tmp_path / "failed")
    bad["pilot"]["max_output_bytes"] = 1
    failed = materialize_versions(bad, base)
    assert failed["requests"] == failed["failed"] == 6
    assert len(read_jsonl(Path(bad["output"]) / "inference_manifest.jsonl")) == 6


def test_unconverged_H2_stops_before_calibration_or_confirmation(tmp_path, monkeypatch):
    cfg, _, _ = batches(tmp_path)
    cfg["pilot"]["require_h1_support_before_confirmation"] = True
    atomic_json(Path(cfg["output"]) / "H1-gate.json", {"status": "SUPPORTED_IN_PILOT"})
    runs = [
        dict(model="typed_response", seed=17, parameters=100, convergence_claim=False)
    ]

    def forbid(*args, **kwargs):
        raise AssertionError("No data may be loaded after unconfirmed convergence")

    monkeypatch.setattr("ontoresponse.train.pilot.load_batch", forbid)
    assert lock_and_evaluate(cfg, runs) == []
    note = json.loads((Path(cfg["output"]) / "confirmation-not-run.json").read_text())
    assert note["reason"] == "INCONCLUSIVE_CONVERGENCE" and not note["test_loaded"]
