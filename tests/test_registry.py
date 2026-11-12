from pathlib import Path

import torch
from rdflib import Graph, Namespace

from reference.core import RELATIONS, compile_relations
from reference.registry import export_topology_fixture, load_registry

ROOT = Path(__file__).resolve().parents[1]


def test_registry_consistency():
    reg = load_registry(ROOT)
    assert len(reg.probe_ids) == 6
    assert reg.probe_features.shape == (6, 4)
    assert len(reg.digest) == 64
    assert reg.digest == load_registry(ROOT).digest


def test_rdf_numeric_topology_match(tmp_path):
    reg = load_registry(ROOT)
    graph = export_topology_fixture(reg, 3, tmp_path / "fixture.ttl")
    OR = Namespace("https://example.org/ontoresponse#")
    EX = Namespace("https://example.org/synthetic-topology/")
    expected = compile_relations(3, reg.families)
    for r, name in enumerate(RELATIONS):
        triples = list(graph.triples((None, OR[name], None)))
        assert len(triples) == int(expected[r].sum())
        for target, source in torch.nonzero(expected[r]).tolist():
            assert (EX[f"node-{source}"], OR[name], EX[f"node-{target}"]) in graph


def test_ontology_and_shapes_syntax_parse_only():
    assert len(Graph().parse(ROOT / "ontology/ontology.ttl", format="turtle")) > 0
    assert len(Graph().parse(ROOT / "ontology/shapes.ttl", format="turtle")) > 0
