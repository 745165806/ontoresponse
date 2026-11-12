"""Unit/synthetic only: protocol and timeline invariants, no efficacy claims."""

import numpy as np
import pytest
import torch

from ontoresponse.data.audio import coverage, crop_starts, crops
from ontoresponse.data.connected import connected_groups
from ontoresponse.evaluation.stage_gate import discrimination_gate


@pytest.mark.parametrize("length", [160, 1000, 64600, 64601, 129200, 190000])
def test_cover_has_no_gaps_and_aligned_probe_coordinates(length):
    p = {
        "samples": 64600,
        "strategy": "cover",
        "padding": "repeat",
        "aggregation": "mean_hidden_mean_logits",
    }
    x = np.arange(length, dtype="float32") + 1
    a = crops(x, p)
    b = crops(x + 2, p)
    torch.testing.assert_close(b - a, torch.full_like(a, 2))
    c = coverage(length, p)
    assert c["coverage_fraction"] == 1 and c["unique_samples"] == length
    assert a.shape == (c["crop_count"], 64600)
    assert crop_starts(length, p) == crop_starts(length, p)


def test_union_coverage_handles_duplicate_overlapping_fixed_crops():
    p = {"samples": 200, "starts": [100, 0, 100]}
    assert coverage(500, p)["unique_samples"] == 300


def test_gate_floor_groups_degeneracy_and_test_guard():
    labels = [0, 1] * 6
    groups = [f"g{i}" for i in range(12)]
    gate = discrimination_gate(labels, groups, [0, 0, 0])
    assert gate["status"] == "INSUFFICIENT_DISCRIMINATION"
    assert gate["reasons"] == ["BASELINE_BOUND_PRECLUDES_PRESPECIFIED_IMPROVEMENT"]
    assert (
        discrimination_gate(labels, groups, [0.1, 0.1, 0.1])["status"]
        == "ELIGIBLE_FOR_H1"
    )
    gate = discrimination_gate([0, 1], ["a", "b"], [0.1], degenerate_fraction=0.5)
    assert set(gate["reasons"]) == {
        "TOO_FEW_INDEPENDENT_GROUPS_PER_CLASS",
        "DEGENERATE_GROUP_BOOTSTRAP",
    }
    with pytest.raises(ValueError):
        discrimination_gate(labels, groups, [0.1], role="test")
    with pytest.raises(ValueError):
        discrimination_gate(labels, groups, [float("nan")])


def test_connected_sources_join_speakers_transitively_and_unknown_blocks():
    rows = [
        {"id": x, "speaker_id": s}
        for x, s in [("a", "s1"), ("b", "s2"), ("c", "s2"), ("d", "s3")]
    ]
    links = {"a": ["u1"], "b": ["u1", "u2"], "c": ["u3"], "d": ["u4"]}
    groups = connected_groups(rows, links)
    assert groups["a"] == groups["b"] == groups["c"] and groups["d"] != groups["a"]
    assert groups == connected_groups(list(reversed(rows)), links)
    with pytest.raises(ValueError, match="incomplete"):
        connected_groups(rows, {"a": ["u1"]})
