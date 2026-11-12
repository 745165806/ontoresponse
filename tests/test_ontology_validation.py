"""Positive and deliberately invalid measurement graphs; no detection claims."""

from pathlib import Path

import pytest
from pyshacl import validate
from rdflib import RDF, Graph, Literal, Namespace
from test_workflow import fixture

from ontoresponse.common import write_jsonl
from ontoresponse.ontology.registry import export_evidence
from ontoresponse.ontology.validate_graph import validate_numeric_graph
from ontoresponse.resources import resource_path

OR = Namespace("https://example.org/ontoresponse#")
PROV = Namespace("http://www.w3.org/ns/prov#")


def graph_fixture(tmp_path):
    cfg, index, ledger, audit, records = fixture(tmp_path)
    write_jsonl(index, records[:1])
    path = tmp_path / "positive.ttl"
    export_evidence(cfg, index, path)
    return Graph().parse(path)


def conforms(graph):
    return validate(
        graph, shacl_graph=str(resource_path("ontology/shapes.ttl")), inference="rdfs"
    )[0]


def test_positive_graph_and_exact_registry_edges(tmp_path):
    graph = graph_fixture(tmp_path)
    assert conforms(graph)
    result = validate_numeric_graph(graph, 3)
    assert result["recordings"] == 1 and result["edges"] == 90


@pytest.mark.parametrize(
    "mutation",
    ["expert", "root", "crop", "score", "probe_run", "missing_model", "same_probe"],
)
def test_negative_measurement_graphs_rejected(tmp_path, mutation):
    graph = graph_fixture(tmp_path)
    response = next(graph.subjects(RDF.type, OR.ResponseObservation))
    after = graph.value(response, OR.hasAfter)
    if mutation == "expert":
        graph.set((after, OR.forExpert, OR.UnknownExpert))
    elif mutation == "root":
        graph.set((after, OR.rootRecording, OR.UnrelatedRecording))
        graph.add((OR.UnrelatedRecording, RDF.type, OR.Recording))
    elif mutation == "crop":
        graph.set((after, OR.cropPolicy, Literal("different-crop")))
    elif mutation == "score":
        graph.set((after, OR.scoreDirection, Literal("higher_is_bonafide")))
    elif mutation == "missing_model":
        graph.remove((after, OR.forExpert, None))
    elif mutation == "probe_run":
        current = graph.value(response, OR.respondsTo)
        other = next(p for p in graph.subjects(RDF.type, OR.ProbeRun) if p != current)
        graph.set((response, OR.respondsTo, other))
    elif mutation == "same_probe":
        current = graph.value(response, OR.respondsTo)
        other = next(
            p
            for p in graph.subjects(RDF.type, OR.ResponseObservation)
            if graph.value(p, OR.respondsTo) != current
        )
        graph.add((response, OR.same_probe_cross_expert, other))
    assert not conforms(graph), mutation


@pytest.mark.parametrize("mutation", ["delete_edge", "rename_hash", "duplicate_index"])
def test_negative_topology_graphs_rejected(tmp_path, mutation):
    graph = graph_fixture(tmp_path)
    if mutation == "delete_edge":
        graph.remove(next(graph.triples((None, OR.same_probe_cross_expert, None))))
    elif mutation == "rename_hash":
        graph.set(
            (
                next(graph.subjects(OR.numericNodeIndex, None)),
                OR.registryHash,
                Literal("wrong"),
            )
        )
    else:
        pairs = list(graph.subject_objects(OR.numericNodeIndex))
        graph.set((pairs[1][0], OR.numericNodeIndex, pairs[0][1]))
    with pytest.raises(ValueError):
        validate_numeric_graph(graph, 3)


def test_packaged_measurement_resources_match_source():
    root = Path(__file__).resolve().parents[1]
    for name in [
        "configs/probes.yaml",
        "configs/ontology.yaml",
        "ontology/shapes.ttl",
        "ontology/ontology.ttl",
    ]:
        assert (root / name).read_bytes() == (
            root / "ontoresponse/_resources" / name
        ).read_bytes()
