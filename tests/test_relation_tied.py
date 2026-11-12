"""Synthetic numerical equivalence; not expert inference or efficacy tests."""

import copy

import pytest
import torch

from ontoresponse.models.factory import build_model, parameter_count
from ontoresponse.models.relation_tied import (
    copy_equal_typed_to_tied,
    copy_tied_to_typed,
)
from ontoresponse.ontology.registry import OntologyRegistry


@pytest.mark.parametrize("missing", [False, True])
def test_four_channel_forward_backward_exact_copy(missing):
    torch.manual_seed(192)
    torch.set_num_threads(1)
    reg = OntologyRegistry()
    typed = build_model([3, 5], reg, "typed_response", width=16).double().eval()
    tied = build_model([3, 5], reg, "relation_tied", width=16).double().eval()
    copy_tied_to_typed(tied, typed)
    assert typed.adjacency.shape == tied.adjacency.shape == (4, 14, 14)
    assert torch.equal(typed.adjacency, tied.adjacency)
    for layer in tied.layers:
        assert len(layer.maps) == 4 and all(m is layer.maps[0] for m in layer.maps)
        assert len({m.weight.data_ptr() for m in layer.maps}) == 1
    h = [torch.randn(3, 7, d, dtype=torch.float64) for d in [3, 5]]
    scores = torch.randn(3, 2, 7, dtype=torch.float64)
    valid = torch.ones_like(scores, dtype=torch.bool)
    if missing:
        valid[0, 0, 2:] = False
        valid[1, 1, 0] = False
        valid[2] = False
        h = [
            torch.where(valid[:, i, :, None], a, torch.full_like(a, float("nan")))
            for i, a in enumerate(h)
        ]
        scores = torch.where(valid, scores, torch.full_like(scores, float("nan")))
    args = (
        h,
        scores,
        valid,
        reg.features.double(),
        torch.zeros(3, 4, dtype=torch.float64),
    )
    a = typed(*args)
    b = tied(*args)
    torch.testing.assert_close(a.logits, b.logits, atol=1e-12, rtol=1e-12)
    a.logits.sum().backward()
    b.logits.sum().backward()
    for first, second in zip(typed.layers, tied.layers):
        torch.testing.assert_close(
            sum(m.weight.grad for m in first.maps),
            second.maps[0].weight.grad,
            atol=1e-11,
            rtol=1e-11,
        )
        torch.testing.assert_close(
            first.self_map.weight.grad,
            second.self_map.weight.grad,
            atol=1e-11,
            rtol=1e-11,
        )
    # Exactly 3 matrices per layer are removed; no self/head/adapter change.
    assert parameter_count(typed) - parameter_count(tied) == 2 * 3 * 16 * 16
    copy_equal_typed_to_tied(typed, tied)
    restored = build_model([3, 5], reg, "relation_tied", width=16).double().eval()
    restored.load_state_dict(copy.deepcopy(tied.state_dict()))
    torch.testing.assert_close(restored(*args).logits, b.logits, atol=0, rtol=0)


def test_unequal_typed_matrices_cannot_be_silently_copied():
    reg = OntologyRegistry()
    typed = build_model([3, 5], reg, "typed_response", width=16)
    tied = build_model([3, 5], reg, "relation_tied", width=16)
    with pytest.raises(ValueError, match="must be equal"):
        copy_equal_typed_to_tied(typed, tied)


def test_separate_normalization_is_not_union():
    from ontoresponse.models.relation_tied import RelationTiedLayer
    from reference.core import RelationalLayer

    tied = RelationTiedLayer(4, 4, 0).double().eval()
    union = RelationalLayer(4, 1, 0).double().eval()
    with torch.no_grad():
        tied.self_map.weight.zero_()
        tied.maps[0].weight.copy_(torch.eye(4))
        union.self_map.weight.copy_(tied.self_map.weight)
        union.maps[0].weight.copy_(tied.maps[0].weight)
    z = torch.tensor(
        [
            [
                [0.0, 1.0, 0.0, 2.0],
                [2.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float64,
    )
    mask = torch.ones(1, 4, dtype=torch.bool)
    adj = torch.zeros(4, 4, 4, dtype=torch.float64)
    adj[0, 0, 1] = 1
    adj[1, 0, 2:] = 1
    assert not torch.allclose(
        tied(z, mask, adj), union(z, mask, (adj.sum(0, keepdim=True) > 0).double())
    )


def test_same_seed_preserves_every_nonrelation_parameter():
    reg = OntologyRegistry()
    torch.manual_seed(83)
    typed = build_model([3, 5], reg, "typed_response", width=16)
    torch.manual_seed(83)
    tied = build_model([3, 5], reg, "relation_tied", width=16)
    for key, value in typed.state_dict().items():
        if ".maps." not in key or ".maps.0." in key:
            assert torch.equal(value, tied.state_dict()[key]), key
