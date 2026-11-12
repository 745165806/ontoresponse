"""The full recording-source universe is shared by split preparation and freezing."""

import json
from pathlib import Path

from ..common import Blocked, read_jsonl, sha256
from .connected import connected_groups


def coverage_protocols(cfg):
    paths = {}
    data = cfg.get("data", {})
    if data.get("asv2019"):
        root = Path(data["asv2019"]) / "ASVspoof2019_LA_cm_protocols"
        for split in ("train", "dev", "eval"):
            paths["2019_" + split] = (
                root
                / f"ASVspoof2019.LA.cm.{split}.{'trn' if split == 'train' else 'trl'}.txt"
            )
    if data.get("asv2021df_keys"):
        paths["2021_DF"] = Path(data["asv2021df_keys"]) / "DF/CM/trial_metadata.txt"
    return paths


def source_universe(cfg):
    p = cfg["pilot"]
    manifest = Path(p["source_map_manifest"])
    archive = Path(p["official_map_archive"])
    if not manifest.is_file() or not archive.is_file():
        raise Blocked(
            "Official source archive and complete normalized mapping are required"
        )
    mapping = json.loads(manifest.read_text())
    if mapping.get("mapping_complete") is not True or mapping.get(
        "official_archive_sha256"
    ) != sha256(archive):
        raise Blocked("Incomplete or unpinned official source mapping")
    universe = mapping["recordings"]
    components = connected_groups(
        universe, {r["id"]: r["source_ids"] for r in universe}
    )
    for name, protocol in coverage_protocols(cfg).items():
        if not protocol.is_file():
            raise Blocked("Missing source coverage protocol: " + name)
        expected = {
            line.split()[1]
            for line in protocol.read_text().splitlines()
            if line.strip()
        }
        missing = expected - components.keys()
        if missing:
            raise Blocked(
                f"Source universe omits {len(missing)} official {name} recording IDs"
            )
    return universe, components


def excluded_components(cfg, universe, components):
    consumed = read_jsonl(cfg["pilot"]["consumption_ledger"])
    known = {
        r["id"]: r
        for r in json.loads(Path(cfg["pilot"]["supervision_audit"]).read_text())[
            "experts"
        ]
    }
    origins = {r.get("origin_id", r["id"]) for r in consumed}
    speakers = {r["speaker_id"] for r in consumed if r.get("speaker_id")}
    for expert in cfg["experts"]:
        record = known.get(expert["id"], {})
        if not record.get("provenance_pass"):
            raise Blocked("Unknown expert task provenance")
        if record.get("checkpoint_sha256") != sha256(expert["checkpoint"]):
            raise Blocked("Expert checkpoint differs from supervision audit")
        for role in ("fit", "source_val"):
            origins.update(record["supervision"][role]["ids"])
            speakers.update(record["supervision"][role]["speakers"])
    excluded = {
        components[r["id"]]
        for r in universe
        if r["id"] in origins
        or r.get("origin_id", r["id"]) in origins
        or bool(({r["speaker_id"]} | set(r.get("linked_speaker_ids", []))) & speakers)
    }
    return excluded
