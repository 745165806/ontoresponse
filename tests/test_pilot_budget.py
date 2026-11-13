import copy
from pathlib import Path

import pytest
import torch
import yaml
from test_workflow import fixture

from ontoresponse.data.pilot import check_pilot_budget
from ontoresponse.evidence.dataset import load_batch
from ontoresponse.ontology.registry import OntologyRegistry
from ontoresponse.pilot_gate import preflight
from ontoresponse.train.pilot import fit_one, operating_point


def config():
    return yaml.safe_load(
        (Path(__file__).parents[1] / "configs/pilot-20261110.yaml").read_text()
    )


def test_explicit_pilot_cap_and_rawnet_quarantine():
    c = config()
    assert check_pilot_budget(c) == 128
    for key, value in [
        ("max_origins", 1025),
        ("max_optimizer_steps", 20),
        ("max_optimizer_steps", 2001),
    ]:
        bad = copy.deepcopy(c)
        bad["pilot"][key] = value
        with pytest.raises((ValueError, RuntimeError)):
            check_pilot_budget(bad)
    c["experts"].append({"id": "rawnet2"})
    with pytest.raises(RuntimeError, match="quarantined"):
        check_pilot_budget(c)


def test_missing_inputs_cannot_start_pilot(tmp_path):
    c = config()
    c["output"] = str(tmp_path)
    c["pilot"]["required_next_stage_document"] = str(tmp_path / "missing.md")
    result = preflight(c)
    assert result["status"] == "BLOCKED" and not result["research_holdout_pass"]
    assert any("missing" in s for s in result["blocked"])


def test_pilot_needs_inputs_and_budget_but_no_document_or_checksum_seal(tmp_path):
    from ontoresponse.common import write_jsonl

    cfg, index, ledger, rows, records = fixture(tmp_path)
    cfg["pilot"] = copy.deepcopy(config()["pilot"])
    cfg["runtime"]["budget"] = "pilot"
    cfg["pilot"]["origins_by_role"] = {
        role: 4 for role in ("fusion_train", "fusion_val", "calibration", "test")
    }
    checkpoint = tmp_path / "unit-preflight.pt"
    checkpoint.write_bytes(
        b"Synthetic existence fixture; never used for expert inference"
    )
    audio = tmp_path / "unit-preflight.wav"
    audio.write_bytes(b"Synthetic existence fixture; never used for expert inference")
    for expert in cfg["experts"]:
        expert.update(task_trained=True, checkpoint=str(checkpoint))
    for row in rows:
        row["audio"] = str(audio)
    root = Path(cfg["output"])
    write_jsonl(root / "audit_manifest.jsonl", rows)
    write_jsonl(
        root / "inference_manifest.jsonl",
        [{k: r[k] for k in ("id", "origin_id", "source_group", "audio")} for r in rows],
    )
    result = preflight(cfg)
    assert (
        result["status"] == "PASS_INTEGRATION_PILOT"
        and not result["research_holdout_pass"]
    )
    assert not (root / "preregistration.json").exists()
    assert not any("sha256" in key or "digest" in key for key in result)
    rows[-1]["source_group"] = rows[0]["source_group"]
    write_jsonl(root / "audit_manifest.jsonl", rows)
    assert any("leakage" in reason for reason in preflight(cfg)["blocked"])


def test_calibrated_logit_threshold_avoids_probability_saturation():
    point = operating_point([0, 0, 1, 1], [20.0, 30.0, 40.0, 50.0], "fusion_val", 0.1)
    assert (
        point["space"] == "calibrated_logit_float64"
        and point["threshold"] == 40
        and point["actual_dev_fpr"] == 0
    )
    with pytest.raises(ValueError):
        operating_point([0, 1], [1.0, 2.0], "test", 0.1)


def test_synthetic_pilot_curve_more_than_twenty_updates(tmp_path):
    # Only a numerical optimizer test; no synthetic scores enter extraction/real pilot.
    cfg, index, ledger, audit, records = fixture(tmp_path)
    cfg["pilot"] = copy.deepcopy(config()["pilot"])
    cfg["output"] = str(tmp_path / "pilot-unit")
    cfg["pilot"].update(
        max_epochs=8,
        min_epochs=8,
        patience_epochs=8,
        batch_size=1,
        max_optimizer_steps=32,
        width=16,
    )
    from ontoresponse.common import write_jsonl

    Path(cfg["output"]).mkdir(parents=True)
    write_jsonl(Path(cfg["output"]) / "audit_manifest.jsonl", audit)
    train = load_batch(cfg, index, ledger, "fusion_train")
    val = load_batch(cfg, index, ledger, "fusion_val")
    result = fit_one(cfg, "relation_tied", 17, train, val, OntologyRegistry())
    assert result["steps"] == 32 and result["epochs"] == 8 and result["save_load_exact"]
    cp = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
    assert cp["best_step"] == cp["history"][cp["best_epoch"] - 1]["optimizer_steps"]
    assert (Path(result["checkpoint"]).parent / "convergence.jsonl").is_file()
