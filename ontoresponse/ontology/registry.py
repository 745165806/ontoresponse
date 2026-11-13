"""One registry drives both numerical topology and evidence provenance export."""

import json
from pathlib import Path

import numpy as np
import torch

from reference.core import RELATIONS, compile_relations
from reference.registry import load_registry

from ..common import Blocked, digest, read_jsonl, unique
from ..evidence.cache import Cache
from ..resources import resource_path


class OntologyRegistry:
    def __init__(self, root=None):
        self.reference = load_registry(root)
        self.probe_ids = self.reference.probe_ids
        self.families = self.reference.families
        self.features = self.reference.probe_features
        self.digest = self.reference.digest

    def adjacency(self, experts):
        return compile_relations(experts, self.families)

    def wrong_members(self, experts, seed):
        rng = np.random.default_rng(seed)
        base = self.adjacency(experts)
        for _ in range(100):
            order = rng.permutation(len(self.families))
            families = [self.families[i] for i in order]
            candidate = compile_relations(experts, families)
            if not torch.equal(candidate, base) and torch.equal(
                candidate.sum(-1), base.sum(-1)
            ):
                return candidate
        raise ValueError("Cannot construct degree-preserving changed family membership")


def validate_evidence(cfg, records, cache):
    unique(records)
    reg = OntologyRegistry()
    identities = {}
    import yaml

    specs = yaml.safe_load(resource_path("configs/probes.yaml").read_text())["probes"]
    for r in records:
        if not r.get("views"):
            continue
        if [v["probe_id"] for v in r["views"]] != ["original", *reg.probe_ids]:
            raise ValueError("Registry/probe mismatch")
        if [v["family"] for v in r["views"][1:]] != list(reg.families):
            raise ValueError("Incorrect semantic family metadata")
        for name, e in r["experts"].items():
            if e["status"] != "complete":
                continue
            ident = e["identity"]
            if ident["score_direction"] != "higher_is_spoof":
                raise ValueError("Unknown score semantics")
            if name in identities and identities[name] != ident:
                raise ValueError("Mixed expert versions")
            identities[name] = ident
            for v, m in zip(r["views"], e["measurements"], strict=True):
                if m["probe_id"] != v["probe_id"]:
                    raise ValueError("Mismatched probe order")
                if m["status"] != "valid":
                    continue
                _, meta = cache.get(m["cache_key"])
                if (
                    meta["spec"]["expert"] != ident
                    or meta["spec"]["artifact_key"] != v["artifact_key"]
                ):
                    raise ValueError(
                        "before/after model/crop/artifact provenance mismatch"
                    )
                _, audio_meta = cache.get(v["artifact_key"])
                if (
                    v["probe_id"] != "original"
                    and audio_meta["spec"]["operation"]
                    != specs[list(reg.probe_ids).index(v["probe_id"])]
                ):
                    raise ValueError(
                        "Cached operation parameters differ from frozen registry"
                    )
                original = cache.get(r["views"][0]["artifact_key"])[1]
                source = audio_meta["spec"].get(
                    "audio", audio_meta["extra"].get("input_audio_sha256")
                )
                before_source = original["spec"].get(
                    "audio", original["extra"].get("input_audio_sha256")
                )
                if source is None or source != before_source:
                    raise ValueError("Probe not derived from current recording")
    return reg


def export_evidence(cfg, index, destination, shacl=False):
    records = read_jsonl(index)
    cache = Cache(cfg["cache"])
    reg = validate_evidence(cfg, records, cache)
    triples = [
        "@prefix orf: <https://example.org/ontoresponse#> .",
        "@prefix ex: <https://example.org/ontoresponse/evidence/> .",
        "@prefix prov: <http://www.w3.org/ns/prov#> .",
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
    ]
    q = lambda x: json.dumps(str(x), ensure_ascii=False)
    eids = [e["id"] for e in cfg["experts"]]
    a = reg.adjacency(len(eids))
    k = len(reg.probe_ids)
    for name in eids:
        identity = next(
            (
                r["experts"][name]["identity"]
                for r in records
                if r["experts"].get(name, {}).get("status") == "complete"
            ),
            None,
        )
        triples.append(f"ex:model-{name} a orf:DetectorModel .")
        if identity and identity.get("checkpoint"):
            triples.append(
                f"ex:model-{name} orf:checkpointPath {q(identity['checkpoint']['path'])} ."
            )
    for rec in records:
        rid = digest(rec["id"])[:20]
        root = "ex:recording-" + rid
        triples.append(f"{root} a orf:Recording .")
        nodes = {}
        for i, name in enumerate(eids):
            e = rec["experts"].get(name, {})
            if e.get("status") != "complete":
                continue
            for v, m in enumerate(e["measurements"]):
                if m["status"] != "valid":
                    continue
                key = m["cache_key"]
                obs = "ex:observation-" + digest([rid, name, v, key])
                run = "ex:expert-run-" + digest([rid, name, v, key])
                input_recording = (
                    root if v == 0 else "ex:artifact-" + rec["views"][v]["artifact_key"]
                )
                if v > 0:
                    triples.append(
                        f"{input_recording} a orf:Recording ; prov:wasDerivedFrom {root} ."
                    )
                triples.append(
                    f"{run} a orf:ExpertRun ; orf:usesModel ex:model-{name} ; prov:used {input_recording} ."
                )
                triples.append(
                    f'{obs} a orf:ExpertObservation ; orf:rootRecording {root} ; orf:forExpert ex:model-{name} ; prov:wasGeneratedBy {run} ; orf:measurementId {q(key)} ; orf:status "valid" ; orf:scoreDirection "higher_is_spoof" ; orf:cropPolicy {q(json.dumps(e["identity"]["crop"], sort_keys=True))} .'
                )
                if v == 0:
                    nodes[i] = obs
                else:
                    before = e["measurements"][0]
                    if before["status"] != "valid":
                        continue
                    bid = "ex:observation-" + digest(
                        [rid, name, 0, before["cache_key"]]
                    )
                    artifact = rec["views"][v]["artifact_key"]
                    probe = reg.probe_ids[v - 1]
                    pr = "ex:probe-run-" + digest([rid, artifact])
                    op = "ex:operation-" + probe
                    _, am = cache.get(artifact)
                    triples.append(
                        f"{op} a orf:ProbeOperation ; orf:parameters {q(json.dumps(am['spec']['operation'], sort_keys=True))} ."
                    )
                    derived = "ex:artifact-" + artifact
                    triples.append(
                        f"{derived} a orf:Recording ; prov:wasDerivedFrom {root} ; prov:wasGeneratedBy {pr} ."
                    )
                    triples.append(
                        f"{pr} a orf:ProbeRun ; orf:usesProbe {op} ; prov:used {root} ."
                    )
                    response = "ex:response-" + digest([rid, name, v, key])
                    triples.append(
                        f"{response} a orf:ResponseObservation ; orf:hasBefore {bid} ; orf:hasAfter {obs} ; orf:respondsTo {pr} ."
                    )
                    nodes[len(eids) + i * k + v - 1] = response
        for idx, node in nodes.items():
            triples.append(f'{node} orf:numericNodeIndex "{idx}"^^xsd:integer .')
        for ri, relation in enumerate(RELATIONS):
            for target, source in torch.nonzero(a[ri]).tolist():
                if target in nodes and source in nodes:
                    triples.append(f"{nodes[source]} orf:{relation} {nodes[target]} .")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(triples) + "\n")
    result = {
        "status": "exported_python_validated",
        "probe_ids": list(reg.probe_ids),
        "requests": len(records),
        "rdf": str(destination),
        "shacl": "NOT_RUN",
    }
    if shacl:
        try:
            from pyshacl import validate
        except ImportError as err:
            raise Blocked(
                "pyshacl unavailable; Python validation and RDF export completed"
            ) from err
        conforms, graph, report = validate(
            str(destination),
            shacl_graph=str(resource_path("ontology/shapes.ttl")),
            inference="rdfs",
        )
        result.update(shacl="passed" if conforms else "failed", report=report)
        if not conforms:
            raise ValueError(report)
    return result
