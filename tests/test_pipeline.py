"""UNIT/SYNTHETIC only. These fixtures are never substituted for expert extraction."""

import copy
import inspect

import numpy as np
import pytest
import torch

from ontoresponse.common import Blocked, budget, write_jsonl
from ontoresponse.data.audio import crops
from ontoresponse.data.external import overlap_report
from ontoresponse.data.prepare import shard, validate_splits
from ontoresponse.evaluation.metrics import choose_threshold, metrics
from ontoresponse.evaluation.report import paired_report, summarize
from ontoresponse.evidence.cache import Cache
from ontoresponse.evidence.dataset import origin_weights
from ontoresponse.experts.base import AudioInput, FrozenExpert
from ontoresponse.models.baselines import BASELINES, OrdinaryFusion
from ontoresponse.models.factory import build_model, parameter_count
from ontoresponse.ontology.registry import OntologyRegistry
from ontoresponse.train.calibrate import Platt
from ontoresponse.transforms.environment import exposure_audit
from ontoresponse.transforms.probes import apply_probe, audio_hash

torch.set_num_threads(1)


def waveform():
    t = np.arange(32017) / 16000
    return (
        0.15 * np.sin(2 * np.pi * 310 * t) + 0.05 * np.sin(2 * np.pi * 6800 * t)
    ).astype("float32")


def args():
    torch.manual_seed(12)
    return (
        [torch.randn(4, 7, d) for d in (5, 7, 9)],
        torch.randn(4, 3, 7),
        torch.ones(4, 3, 7, dtype=torch.bool),
        OntologyRegistry().features,
        torch.zeros(4, 4),
    )


def test_budget_fail_closed():
    for mode, n, steps in [("smoke", 65, 1), ("pilot", 1025, 1), ("smoke", 5, 21)]:
        with pytest.raises(ValueError):
            budget(mode, n, steps)
    with pytest.raises(Blocked):
        budget("full", 1, 1)
    budget("smoke", 64, 20)
    budget("pilot", 1024, 20)


def test_label_free_contract_and_missing_weights():
    assert set(inspect.signature(AudioInput).parameters) == {"id", "waveform"}
    assert set(inspect.signature(FrozenExpert.predict_utterances).parameters) == {
        "self",
        "audio_batch_without_labels",
        "batch_size",
    }
    batch = inspect.signature(FrozenExpert.predict_utterances).parameters["batch_size"]
    assert batch.kind is inspect.Parameter.KEYWORD_ONLY and batch.default == 1
    with pytest.raises(Blocked):
        FrozenExpert({"task_trained": False}, {})
    with pytest.raises(Blocked):
        FrozenExpert(
            {
                "task_trained": True,
                "repository": ".",
                "checkpoint": "/absent-real-model",
            },
            {},
        )


def test_split_origin_speaker_source_leakage():
    a = {
        "id": "a",
        "origin_id": "a",
        "speaker_id": "s1",
        "source_group": "g1",
        "role": "fusion_train",
        "label": 0,
    }
    for key in ("origin_id", "speaker_id", "source_group"):
        b = {
            "id": "b",
            "origin_id": "b",
            "speaker_id": "s2",
            "source_group": "g2",
            "role": "test",
            "label": 1,
        }
        b[key] = a[key]
        with pytest.raises(ValueError):
            validate_splits([a, b])
    with pytest.raises(ValueError):
        validate_splits([a, a])


def test_stable_complete_shards():
    ids = [f"group-{i}" for i in range(101)]
    parts = [{x for x in ids if shard(x, 4) == i} for i in range(4)]
    assert set.union(*parts) == set(ids)
    assert sum(map(len, parts)) == len(ids)
    assert all(not (parts[i] & parts[j]) for i in range(4) for j in range(i))


def test_crop_full_timeline_and_repeat():
    x = waveform()
    policy = {
        "samples": 64600,
        "starts": [0, 100],
        "padding": "repeat",
        "aggregation": "mean_hidden_mean_logits",
    }
    result = crops(x, policy)
    assert result.shape == (2, 64600)
    np.testing.assert_array_equal(result[1, : len(x) - 100], x[100:])
    with pytest.raises(ValueError):
        crops(np.zeros(16000, dtype="float32"), policy)


@pytest.mark.parametrize("op", ["identity", "lossless"])
def test_identity_lossless_exact(op):
    x = waveform()
    y, meta = apply_probe(x, {"operation": op})
    np.testing.assert_array_equal(x, y)
    assert meta["input_samples"] == meta["output_samples"] == len(x)


def test_noise_shared_direction_and_physical_snr():
    x = waveform()
    spec = lambda snr: {"operation": "additive_white_noise", "snr_db": snr}
    a, ma = apply_probe(x, spec(30))
    b, mb = apply_probe(x, spec(20))
    again, _ = apply_probe(x, spec(30))
    np.testing.assert_array_equal(a, again)
    assert ma["seed"] == mb["seed"]
    np.testing.assert_allclose((a - x) * 10**0.5, b - x, atol=1e-7)
    assert abs(20 * np.log10(np.linalg.norm(x) / np.linalg.norm(a - x)) - 30) < 1e-4


def test_filter_attenuates_high_frequency_and_parameter_validation():
    x = waveform()
    y, _ = apply_probe(x, {"operation": "lowpass", "cutoff_hz": 3400, "order": 6})
    assert len(y) == len(x) and not np.array_equal(x, y)
    for spec in [
        {"operation": "lowpass", "cutoff_hz": 8000},
        {"operation": "additive_white_noise", "snr_db": -1},
    ]:
        with pytest.raises(ValueError):
            apply_probe(x, spec)
    with pytest.raises(Blocked):
        apply_probe(x, {"operation": "invented_codec"})


def test_cache_identity_corruption_and_numeric_dedup(tmp_path):
    c = Cache(tmp_path)
    spec = {
        "audio": audio_hash(waveform()),
        "expert": "a",
        "checkpoint": "real-fixture-placeholder-unit-only",
        "crop": [0],
    }
    key = c.key(spec)
    arrays = {"embedding": np.array([0.0, 1.0]), "margin": np.array([0.0])}
    c.put(key, spec, arrays, {"status": "valid"})
    got = c.get(key, spec)
    assert got[0]["margin"][0] == 0
    assert key != c.key({**spec, "expert": "b"})
    assert key != c.key({**spec, "checkpoint": "changed"})
    assert key != c.key({**spec, "crop": [1]})
    with pytest.raises(ValueError):
        c.get(key, {**spec, "expert": "b"})
    with pytest.raises(ValueError):
        c.put(
            key,
            spec,
            {"embedding": np.array([2.0, 3.0]), "margin": np.array([0.0])},
            {},
        )
    with pytest.raises(ValueError):
        c.put(c.key({"bad": 1}), {"bad": 1}, {"margin": [np.nan]}, {})
    (tmp_path / (key + ".npz")).write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        c.get(key)


@pytest.mark.parametrize("name", BASELINES)
def test_baseline_bce_masks_and_no_evidence(name):
    reg = OntologyRegistry()
    net = OrdinaryFusion([5, 7, 9], reg.families, 4, name, width=16, dropout=0).eval()
    h, s, v, p, q = args()
    v[:, 1, 2] = False
    missing_h = [x.clone() for x in h]
    missing_s = s.clone()
    missing_h[1][:, 2] = float("nan")
    missing_s[:, 1, 2] = float("nan")
    a = net(h, s, v, p, q)
    b = net(missing_h, missing_s, v, p, q)
    torch.testing.assert_close(a.logits, b.logits, atol=0, rtol=0)
    torch.nn.functional.binary_cross_entropy_with_logits(
        b.logits, torch.tensor([0.0, 1.0, 0.0, 1.0])
    ).backward()
    assert any(x.grad is not None and x.grad.abs().sum() > 0 for x in net.parameters())
    v[:, :, 0] = False
    out = net(h, s, v, p, q)
    assert not out.observed.any() and out.probabilities().isnan().all()


@pytest.mark.parametrize(
    "name",
    [
        "typed_response",
        "untyped_response",
        "set_attention",
        "wrong_family",
        "drop_anchor",
        "drop_same_probe",
        "drop_same_family",
        "no_response",
        "missing_probe",
        "single_expert_1",
    ],
)
def test_relation_models_real_computation_and_roundtrip(name):
    reg = OntologyRegistry()
    torch.manual_seed(1)
    net = build_model([5, 7, 9], reg, name, width=16).eval()
    h, s, v, p, q = args()
    a = net(h, s, v, p, q)
    torch.nn.functional.binary_cross_entropy_with_logits(
        a.logits, torch.tensor([0.0, 1.0, 0.0, 1.0])
    ).backward()
    assert parameter_count(net) > 0
    state = copy.deepcopy(net.state_dict())
    net2 = build_model([5, 7, 9], reg, name, width=16).eval()
    net2.load_state_dict(state)
    torch.testing.assert_close(a.logits, net2(h, s, v, p, q).logits, atol=0, rtol=0)


def test_wrong_members_preserve_degrees_and_change_output():
    reg = OntologyRegistry()
    a = reg.adjacency(3)
    b = reg.wrong_members(3, 17)
    assert not torch.equal(a, b)
    assert torch.equal(a.sum(-1), b.sum(-1))
    net = build_model([5, 7, 9], reg, "typed_response", width=16).eval()
    h, s, v, p, q = args()
    before = net(h, s, v, p, q).logits
    net.adjacency.copy_(b)
    after = net(h, s, v, p, q).logits
    assert not torch.equal(before, after)


def test_origin_versions_equal_total_weight():
    w = origin_weights(["a", "a", "b", "c", "c", "c"])
    assert w[:2].sum() == w[2] and w[3:].sum() == w[2]


def test_exposure_includes_probe_of_environment():
    ledger = exposure_audit(
        [[], ["lowpass"]],
        ["noise", "mp3"],
        [["lowpass", "noise"], ["aac"], ["noise", "lowpass"]],
        None,
    )
    assert [x["classification"] for x in ledger["candidates"]] == [
        "exposed_in_T_or_P_of_T",
        "fusion_unseen_operator",
        "fusion_unseen_ordered_composition",
    ]
    assert all(x["all_experts_unseen"] is None for x in ledger["candidates"])


def test_unknown_source_not_disjoint():
    r = overlap_report(
        [{"id": i} for i in ("a", "b", "c")], ["s1"], {"a": ["s1"], "b": ["s2"]}
    )
    assert r["counts"] == {"known_overlap": 1, "known_disjoint": 1, "unknown": 1}


def test_metrics_direction_ties_and_dev_threshold():
    y = [0, 0, 1, 1]
    assert metrics(y, [-2, -1, 1, 2])["eer"] == 0
    assert metrics(y, [2, 1, -1, -2])["eer"] == 1
    assert metrics(y, [0, 0, 0, 0])["eer"] == 0.5
    with pytest.raises(ValueError):
        choose_threshold(y, [0.1, 0.2, 0.8, 0.9], "test")
    r = choose_threshold(y, [0.1, 0.2, 0.8, 0.9], "fusion_val")
    assert r["actual_dev_fpr"] == 0 and r["low_fpr_estimate_supported"] is False


def test_calibration_isolated_and_monotonic():
    with pytest.raises(ValueError):
        Platt.fit([-2, 2], [0, 1], "test")
    params = Platt.fit([-3, -1, 1, 3], [0, 0, 1, 1], "calibration")
    assert (
        params["scale"] > 0 and np.diff(Platt.apply([-3, -1, 1, 3], params)).min() > 0
    )


def test_full_coverage_and_recomputation():
    rows = [
        {
            "id": str(i),
            "status": "scored",
            "label": y,
            "logit": s,
            "p": p,
            "threshold": 0.5,
            "condition": "aac",
            "source_group": "g" + str(i // 2),
        }
        for i, (y, s, p) in enumerate(
            [(0, -2, 0.1), (1, 2, 0.9), (0, -1, 0.2), (1, 1, 0.8)]
        )
    ]
    rows.append(
        {
            "id": "failed",
            "status": "review",
            "label": 1,
            "p": None,
            "logit": None,
            "threshold": 0.5,
            "condition": "aac",
            "source_group": "g3",
        }
    )
    report = summarize(rows)
    assert (
        report["N_requested"] == 5
        and report["N_scored"] == 4
        and report["N_failed"] == 1
    )
    assert report["unseen_macro_eer"] == 0 and report["failure_policy_bounds"][
        "tpr"
    ] == [2 / 3, 1.0]
    paired = paired_report(rows, copy.deepcopy(rows), 30)
    assert paired["N_common_scored"] == 4
    boot = paired["condition_bootstrap"]["aac"]
    assert boot["ci95"] == [0.0, 0.0] and boot["independent_groups"] == 2


def test_merge_preserves_every_request_and_rejects_loss_or_duplication(tmp_path):
    from ontoresponse.evidence.extract import merge

    manifest = tmp_path / "requests.jsonl"
    write_jsonl(manifest, [{"id": "a"}, {"id": "b"}])
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    destination = tmp_path / "merged.jsonl"
    write_jsonl(first, [{"id": "a", "status": "review"}])
    write_jsonl(second, [{"id": "b", "status": "scored"}])
    assert merge([first, second], manifest, destination)["requested"] == 2
    with pytest.raises(ValueError):
        merge([first], manifest, destination)
    with pytest.raises(ValueError):
        merge([first, first, second], manifest, destination)
