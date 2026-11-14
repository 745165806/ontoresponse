"""Synthetic contracts for batching only; real experiments use trained checkpoints."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ontoresponse.evidence.cache import Cache
from ontoresponse.evidence.extract import _batched_measurements
from ontoresponse.experts.base import AudioInput, FrozenExpert, Prediction


def test_batching_preserves_per_utterance_crop_aggregation_and_order():
    expert = FrozenExpert.__new__(FrozenExpert)
    expert.device = torch.device("cpu")
    expert.identity = {"id": "synthetic-contract-only"}
    expert.spec = dict(spoof_index=0, bonafide_index=1)
    expert.crop_policy = dict(
        samples=160,
        starts=[0, 20],
        padding="repeat",
        aggregation="mean_hidden_mean_logits",
    )
    calls = []

    def run(x):
        calls.append(len(x))
        h = torch.stack([x.mean(1), x[:, 0]], 1)
        return h, torch.stack([h[:, 0], -h[:, 0]], 1)

    expert._run = run
    items = [
        AudioInput(str(i), np.arange(n, dtype="float32") / n)
        for i, n in enumerate([165, 260, 400])
    ]
    expected = expert.predict_utterances(items)
    calls.clear()
    actual = expert.predict_utterances(items, batch_size=3)
    assert calls == [3, 3] and actual.ids == expected.ids
    torch.testing.assert_close(actual.embeddings, expected.embeddings, atol=0, rtol=0)
    torch.testing.assert_close(
        actual.spoof_margins, expected.spoof_margins, atol=0, rtol=0
    )
    with pytest.raises(ValueError):
        expert.predict_utterances(items, batch_size=0)
    with pytest.raises(TypeError):
        expert.predict_utterances([dict(label=1)], batch_size=4)


def test_batched_cache_correspondence_actual_batch_calls_and_resume(tmp_path):
    cache = Cache(tmp_path)
    records = []
    for i in range(3):
        views = []
        for p, offset in [("original", 0), ("probe", 10)]:
            spec = dict(stage="synthetic-audio", record=i, probe=p)
            key = cache.key(spec)
            cache.put(
                key, spec, {"waveform": np.full(160, i + offset, dtype="float32")}, {}
            )
            views.append(dict(probe_id=p, artifact_key=key, status="valid"))
        records.append(dict(id=str(i), views=views, experts={}))
    calls = []

    def predict(items, batch_size):
        calls.append((len(items), batch_size))
        values = torch.tensor([float(x.waveform[0]) for x in items])
        return Prediction(
            [x.id for x in items],
            values[:, None],
            values,
            torch.ones(len(items), dtype=torch.bool),
            {"utterance_measurements": [dict(id=x.id) for x in items]},
        )

    expert = SimpleNamespace(
        spec={"id": "unit"}, identity={"id": "unit"}, predict_utterances=predict
    )
    policy = dict(batch_size=4, cudnn_allow_tf32=True)
    cfg = dict(runtime={"cache_revision": 2})
    _batched_measurements(cfg, records, expert, cache, [{"id": "probe"}], policy)
    assert calls == [(4, 4), (2, 4)]
    for i, row in enumerate(records):
        for j, entry in enumerate(row["experts"]["unit"]["measurements"]):
            assert entry["status"] == "valid"
            arrays, meta = cache.get(entry["cache_key"])
            assert arrays["embedding"][0] == i + 10 * j
            assert meta["spec"]["artifact_key"] == row["views"][j]["artifact_key"]
            assert meta["spec"]["inference_policy"] == policy
    _batched_measurements(cfg, records, expert, cache, [{"id": "probe"}], policy)
    assert calls == [(4, 4), (2, 4)]
