"""Single configuration source for topology and a topology-only RDF fixture.

export_topology_fixture is NOT a full evidence/provenance exporter and does not
satisfy production SHACL shapes; it is intentionally marked as a structural test.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import yaml
import torch
from .core import RELATIONS, compile_relations

@dataclass(frozen=True)
class Registry:
    probe_ids: tuple[str, ...]
    families: tuple[str, ...]
    probe_features: torch.Tensor
    digest: str


def load_registry(root: str | Path | None = None) -> Registry:
    if root is None:
        from ontoresponse.resources import resource_path
        probe_path=resource_path('configs/probes.yaml');onto_path=resource_path('configs/ontology.yaml')
    else:
        root=Path(root);probe_path=root/'configs/probes.yaml';onto_path=root/'configs/ontology.yaml'
    probe_cfg = yaml.safe_load(probe_path.read_text())
    onto_cfg = yaml.safe_load(onto_path.read_text())
    if tuple(r["name"] for r in onto_cfg["relations"]) != RELATIONS:
        raise ValueError("Configured relations and implemented compiler differ")
    probes = probe_cfg["probes"]
    ids = tuple(p["id"] for p in probes)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate probe ID")
    family_names = ("bandwidth", "noise", "lossy_coding")
    features=[]
    for p in probes:
        family=p["family"]
        if family not in family_names:
            raise ValueError("Unsupported probe family in reference metadata encoder")
        onehot=[float(family==f) for f in family_names]
        if family=="bandwidth":
            value=p["cutoff_hz"]/(probe_cfg["sample_rate_hz"]/2)
        elif family=="noise":
            value=p["snr_db"]/50.0
        else:
            value=p["bitrate_kbps"]/128.0
        features.append(onehot+[value])
    digest=hashlib.sha256(json.dumps([probe_cfg,onto_cfg],sort_keys=True).encode()).hexdigest()
    return Registry(ids,tuple(p["family"] for p in probes),torch.tensor(features),digest)


def export_topology_fixture(reg: Registry, experts: int, destination: str | Path):
    from rdflib import Graph, Literal, Namespace, RDF, XSD
    OR=Namespace("https://example.org/ontoresponse#")
    EX=Namespace("https://example.org/synthetic-topology/")
    graph=Graph()
    graph.bind("orf",OR)
    graph.add((EX.fixture,RDF.type,OR.TopologyFixture))
    graph.add((EX.fixture,OR.registryHash,Literal(reg.digest)))
    k=len(reg.families)
    a=compile_relations(experts,reg.families)
    for idx in range(experts*(k+1)):
        node=EX[f"node-{idx}"]
        graph.add((node,OR.numericNodeIndex,Literal(idx,datatype=XSD.integer)))
        graph.add((node,RDF.type,OR.OriginalToken if idx<experts else OR.ResponseToken))
        i=idx if idx<experts else (idx-experts)//k
        graph.add((node,OR.expertIndex,Literal(i,datatype=XSD.integer)))
        if idx>=experts:
            graph.add((node,OR.probeId,Literal(reg.probe_ids[(idx-experts)%k])))
    for r,name in enumerate(RELATIONS):
        for target,source in torch.nonzero(a[r]).tolist():
            graph.add((EX[f"node-{source}"],OR[name],EX[f"node-{target}"]))
    path=Path(destination)
    path.parent.mkdir(parents=True,exist_ok=True)
    graph.serialize(path,format="turtle")
    return graph
