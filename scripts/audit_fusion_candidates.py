"""Metadata-only source/speaker audit for fusion dataset recommendations.

Reads the real official archive without extracting paths or inventing mappings.
No expert scores, audio predictions, data-role assignment or training is used.
"""

import csv
import hashlib
import io
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

from ontoresponse.common import atomic_json, sha256


def run(root=Path("reports/expert-benchmark-20261111")):
    archive_path = root / "ASVspoof2019_2021_VCTK_VCC_MetaInfo.tar.gz"
    protocol_root = Path("${DATA_ROOT}/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols")
    protocols = {}
    for split, suffix in [
        ("train", "train.trn"),
        ("dev", "dev.trl"),
        ("eval", "eval.trl"),
    ]:
        path = protocol_root / f"ASVspoof2019.LA.cm.{suffix}.txt"
        protocols[split] = {
            parts[1]: dict(speaker=parts[0], label=int(parts[-1] == "spoof"))
            for parts in map(str.split, path.read_text().splitlines())
        }
    with tarfile.open(archive_path) as archive:
        inventory = {}
        maps = {}
        for name in [
            "ASVspoof2019_LA_VCTK_MetaInfo.tsv",
            "ASVspoof2021_DF_VCTK_MetaInfo.tsv",
            "ASVspoof2021_DF_VCC_MetaInfo.tsv",
        ]:
            data = archive.extractfile(name).read()
            rows = list(csv.DictReader(io.StringIO(data.decode()), delimiter="\t"))
            ids = [r["ASVspoof_ID"] for r in rows]
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate official map ID: " + name)
            inventory[name] = {
                "rows": len(rows),
                "columns": list(rows[0]),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            maps[name] = {r["ASVspoof_ID"]: r for r in rows}
        readme = archive.extractfile("README.txt").read().decode()
        (root / "official-map-README.txt").write_text(readme)
    mapping = maps["ASVspoof2019_LA_VCTK_MetaInfo.tsv"]
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    processed = {}
    summaries = {}
    raw_sources = {}
    targets = {}
    for split, protocol in protocols.items():
        missing = sorted(set(protocol) - set(mapping))
        if missing:
            raise ValueError(f"{split}: missing {len(missing)} official IDs")
        rows = []
        sources = set()
        speakers = set()
        kinds = Counter()
        for ident, audit in protocol.items():
            m = mapping[ident]
            bona = m["VCTK_ID"]
            vc = m["VC_source_VCTK_ID"]
            target = m["TTS_VC_target_speaker"]
            if audit["label"] == 0:
                if bona == "-" or vc != "-" or target != "-":
                    raise ValueError("Bonafide metadata mismatch")
                speaker = bona.split("_")[0]
                source = bona
                kind = "bonafide"
            else:
                if bona != "-" or target == "-":
                    raise ValueError("Spoof metadata mismatch")
                speaker = target
                if vc != "-":
                    source = vc
                    kind = "voice_conversion"
                elif m["TTS_text"] != "-":
                    source = None
                    kind = "text_to_speech"
                else:
                    source = None
                    kind = "unresolved_spoof"
            find(speaker)
            speakers.add(speaker)
            kinds[kind] += 1
            if source:
                sources.add(source)
                union(speaker, source.split("_")[0])
            rows.append(
                dict(
                    id=ident,
                    label=audit["label"],
                    speaker=speaker,
                    source=source,
                    kind=kind,
                )
            )
        processed[split] = rows
        raw_sources[split] = sources
        targets[split] = speakers
        summaries[split] = {
            "n": len(rows),
            "labels": dict(Counter(r["label"] for r in rows)),
            "target_speakers": len(speakers),
            "mapped_acoustic_sources": len(sources),
            "generation_kinds": dict(kinds),
            "official_ID_coverage_complete": True,
        }
    component_rows = defaultdict(list)
    for split, rows in processed.items():
        for r in rows:
            component_rows[find(r["speaker"])].append((split, r["label"]))
    for split, rows in processed.items():
        summaries[split]["connected_components"] = len(
            {find(r["speaker"]) for r in rows}
        )
        summaries[split]["components_per_label"] = {
            str(y): len({find(r["speaker"]) for r in rows if r["label"] == y})
            for y in (0, 1)
        }
    pairs = {}
    for a, b in [("train", "dev"), ("train", "eval"), ("dev", "eval")]:
        pairs[a + "__" + b] = {
            "same_target_speakers": len(targets[a] & targets[b]),
            "same_acoustic_source_utterances": len(raw_sources[a] & raw_sources[b]),
            "same_source_target_connected_components": len(
                {find(r["speaker"]) for r in processed[a]}
                & {find(r["speaker"]) for r in processed[b]}
            ),
        }
    key = Path(
        "${DATA_ROOT}/asvspoof2019/LA/ASVspoof2021_DF_keys_full/DF/CM/trial_metadata.txt"
    )
    df_ids = {parts[1] for parts in map(str.split, key.read_text().splitlines())}
    df_maps = {
        **maps["ASVspoof2021_DF_VCTK_MetaInfo.tsv"],
        **maps["ASVspoof2021_DF_VCC_MetaInfo.tsv"],
    }
    report = {
        "status": "METADATA_AUDIT_ONLY",
        "archive": str(archive_path),
        "archive_sha256": sha256(archive_path),
        "source_url": "https://www.asvspoof.org/asvspoof2019/ASVspoof2019_2021_VCTK_VCC_MetaInfo.tar.gz",
        "inventory": inventory,
        "ASV2019": summaries,
        "ASV2019_cross_partition": pairs,
        "ASV2019_conservative_speaker_source_components": [
            {
                "public_component_id": name,
                "size": len(rows),
                "by_partition_label": dict(
                    Counter(split + ":" + str(label) for split, label in rows)
                ),
            }
            for name, rows in sorted(component_rows.items(), key=lambda r: -len(r[1]))
        ],
        "ASV2021_DF": {
            "protocol_ids": len(df_ids),
            "map_ids": len(df_maps),
            "missing_protocol_ids": len(df_ids - set(df_maps)),
            "unexpected_map_ids": len(set(df_maps) - df_ids),
        },
        "interpretation": "Components conservatively join target and VC-source speakers. TTS text is not fabricated into an acoustic-source ID. This audit does not assign roles, certify task-HPO independence, or authorize leaking source groups.",
        "expert_scores_used": False,
        "new_training_started": False,
    }
    atomic_json(root / "official-source-map-audit.json", report)
    print({k: report[k] for k in ("ASV2019", "ASV2019_cross_partition", "ASV2021_DF")})
    return report


if __name__ == "__main__":
    run()
