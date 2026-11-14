import copy
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from test_workflow import fixture

from ontoresponse.evidence.dataset import load_batch
from ontoresponse.models.factory import build_model
from ontoresponse.ontology.registry import OntologyRegistry
from ontoresponse.train.fit import forward, restore
from ontoresponse.train.pilot import predict_batches
from reference.core import ExpertStats
from scripts.round1_batch import (
    check_evidence,
    pilot_rows,
    refit,
)
from scripts.round1_batch import (
    test_rows as evaluation_rows,
)


def test_pilot_preserves_two_complete_groups_and_cap():
    pool = [
        dict(id=f"{g}-{y}-{i}", source_group=g, label=y)
        for g in ("a", "b")
        for y in (0, 1)
        for i in range(8)
    ]
    chosen = pilot_rows(pool, 16, 17)
    assert len(chosen) == 16 and len({r["id"] for r in chosen}) == 16
    for group in ("a", "b"):
        assert sum(r["source_group"] == group and r["label"] == 0 for r in chosen) == 4
    with pytest.raises(ValueError):
        pilot_rows(pool, 1028, 17)


def test_progress_and_hidden_are_not_in_la_eval(tmp_path):
    path = tmp_path / "keys.txt"
    path.write_text(
        "s a none loc A07 spoof notrim eval\n"
        "s b none loc A07 spoof notrim progress\n"
        "s c none loc bonafide bonafide only_speech hidden\n"
    )
    rows = evaluation_rows(
        {"data": {"asv2021la": str(tmp_path), "asv2021la_keys": str(path)}},
        "asv2021_la",
    )
    assert [r["id"] for r in rows] == ["2021LA:a"]
    assert rows[0]["official_phase"] == "eval" and rows[0]["role"] == "test"


def test_partial_probe_failure_cannot_be_silently_packed():
    valid = dict(
        id="a",
        complete_all_experts=True,
        experts={"e": dict(measurements=[dict(status="valid")])},
    )
    check_evidence([valid])
    broken = copy.deepcopy(valid)
    broken["experts"]["e"]["measurements"][0]["status"] = "failed"
    with pytest.raises(RuntimeError, match="Incomplete"):
        check_evidence([broken])


def test_batched_prediction_and_fixed_epoch_refit(tmp_path):
    cfg, index, ledger, _, _ = fixture(tmp_path)
    train = load_batch(cfg, index, ledger, "fusion_train")
    val = load_batch(cfg, index, ledger, "fusion_val")
    reg = OntologyRegistry()
    stats = ExpertStats.fit(*train.inputs(), role="fusion_train")
    model = build_model(
        [h.shape[-1] for h in train.hidden], reg, "typed_response", 8
    ).eval()
    with torch.inference_mode():
        expected = forward(model, val, stats, reg)
    actual = predict_batches(model, val, stats, reg, 1)
    torch.testing.assert_close(actual.logits, expected.logits, atol=1e-6, rtol=1e-5)
    assert actual.observed.tolist() == expected.observed.tolist()
    cfg["output"] = str(tmp_path / "refit")
    cfg["pilot"] = dict(learning_rate=0.001, weight_decay=0.0001, batch_size=2)
    checkpoint = refit(cfg, "typed_response", 17, train, 8, 2, reg)
    cp, loaded, scaler, registry = restore(checkpoint)
    assert cp["epochs"] == 2 and cp["train_ids"] == train.ids and cp["val_ids"] == []
    assert not set(cp["train_ids"]) & set(val.ids)
    assert len(cp["history"]) == 2
    assert torch.isfinite(
        predict_batches(loaded, val, scaler, registry, 2).logits
    ).all()
    assert (
        json.loads(
            (Path(checkpoint).parent / "history.jsonl").read_text().splitlines()[-1]
        )["epoch"]
        == 2
    )


def test_parameterless_refit_selects_on_development_and_never_creates_optimizer(
    tmp_path,
):
    cfg, index, ledger, _, _ = fixture(tmp_path)
    train = load_batch(cfg, index, ledger, "fusion_train")
    scores = train.scores.clone()
    signal = train.labels * 2 - 1
    scores[:, 0, 0] = -signal
    scores[:, 1, 0] = signal
    scores[:, 2, 0] = 0
    groups = ["a", "a", "b", "b"]
    train = replace(
        train,
        scores=scores,
        groups=groups,
        audit=[dict(r, source_group=g) for r, g in zip(train.audit, groups)],
    )
    cfg["output"] = str(tmp_path / "single-refit")
    # A parameterless expert must work without optimizer settings or optimizer construction.
    with patch("torch.optim.AdamW", side_effect=AssertionError("No optimizer allowed")):
        checkpoint = refit(
            cfg, "best_single_expert", 17, train, 64, 1, OntologyRegistry()
        )
    cp, model, stats, reg = restore(checkpoint)
    assert int(model.chosen_expert) == 1
    assert cp["epochs"] == 0 and cp["history"] == []
    assert cp["single_expert_selection"]["test_used"] is False
    assert cp["single_expert_selection"]["chosen"]["mean_eer"] == 0
    assert torch.isfinite(predict_batches(model, train, stats, reg, 2).logits).all()
    # Re-entering a completed refit reuses the checkpoint.
    assert refit(cfg, "best_single_expert", 17, train, 64, 1, reg) == checkpoint
    forbidden = replace(train, audit=[dict(r, role="test") for r in train.audit])
    cfg["output"] = str(tmp_path / "forbidden-selection")
    with pytest.raises(ValueError, match="development evidence only"):
        refit(cfg, "best_single_expert", 17, forbidden, 64, 1, reg)
