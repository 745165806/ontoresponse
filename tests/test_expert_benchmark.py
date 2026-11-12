"""Full expert benchmark coverage and fixed-score semantics (unit only)."""

from pathlib import Path

import pytest

from scripts.expert_benchmark import Waveforms, job_spec, protocol_rows, summarize


def recipe(tmp_path):
    path = tmp_path / "protocol.txt"
    path.write_text(
        "s1 e1 - - bonafide\ns2 e2 - A07 spoof\ns3 e3 - - bonafide\ns4 e4 - A08 spoof\n"
    )
    return {
        "protocol": str(path),
        "budget": {"expected_requests": 4, "expected_bonafide": 2, "expected_spoof": 2},
    }


def test_full_protocol_rejects_count_or_duplicate_drift(tmp_path):
    cfg = recipe(tmp_path)
    assert len(protocol_rows(cfg)) == 4
    cfg["budget"]["expected_spoof"] = 3
    with pytest.raises(ValueError, match="counts"):
        protocol_rows(cfg)
    cfg = recipe(tmp_path)
    Path(cfg["protocol"]).write_text("s1 e1 - - bonafide\ns2 e1 - A07 spoof\n")
    with pytest.raises(ValueError, match="Duplicate"):
        protocol_rows(cfg)


def test_failed_request_kept_and_cannot_be_reported_as_complete(tmp_path, monkeypatch):
    cfg = recipe(tmp_path)
    rows = protocol_rows(cfg)
    predictions = [dict(r, status="failed") for r in rows]
    result = summarize(cfg, rows, predictions)
    assert result["N_requested"] == result["N_failed"] == 4
    assert result["N_scored"] == 0 and result["status"] == "INCOMPLETE"
    with pytest.raises(ValueError, match="complete ordered protocol"):
        summarize(cfg, rows, predictions[:-1])
    with pytest.raises(ValueError, match="complete ordered protocol"):
        summarize(cfg, rows, list(reversed(predictions)))


def test_native_and_margin_are_reported_separately_without_eval_selection(
    tmp_path, monkeypatch
):
    cfg = recipe(tmp_path)
    rows = protocol_rows(cfg)
    observed = []

    def audit_metric(cfg, labels, scores):
        observed.append((list(labels), list(scores)))
        return {"fixture_metric_call": len(observed)}

    monkeypatch.setattr("scripts.expert_benchmark.official_eer", audit_metric)
    # Native is perfectly ordered while margin is deliberately reversed.
    predictions = [
        dict(
            r,
            status="scored",
            native_bonafide_score=2 if r["label"] == 0 else -2,
            spoof_margin=3 if r["label"] == 0 else -3,
        )
        for r in rows
    ]
    result = summarize(cfg, rows, predictions)
    assert result["status"] == "COMPLETE"
    assert result["native_grouped_roc"]["eer"] == 0
    assert result["margin_grouped_roc"]["eer"] == 1
    assert not result["score_direction_chosen_on_eval"]
    assert observed[0][1] == [2, -2, 2, -2]
    assert observed[1][1] == [-3, 3, -3, 3]


def test_author_reference_overrides_do_not_mutate_existing_checkpoint():
    base = {
        "experts": [
            {"id": "aasist", "checkpoint": "existing.pt", "state_key": "model_state"}
        ]
    }
    ref = job_spec(
        base,
        {
            "id": "aasist_author_reference",
            "base_expert": "aasist",
            "checkpoint": "author.pth",
            "state_key": None,
        },
    )
    assert ref["checkpoint"] == "author.pth" and ref["state_key"] is None
    assert base["experts"][0]["checkpoint"] == "existing.pt"


def test_waveform_dataset_has_no_label_or_audit_input(tmp_path):
    data = Waveforms(["not_present"], tmp_path, {"samples": 64600})
    row = data[0]
    assert row["status"] == "failed" and row["id"] == "not_present"
    assert "label" not in row and "attack" not in row
