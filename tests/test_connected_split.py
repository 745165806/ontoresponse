"""Source allocation tests use explicit synthetic metadata/waves, not research data."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from test_development_runner import prereg_fixture

from ontoresponse.common import Blocked, write_jsonl
from ontoresponse.data.connected_split import allocate_connected
from ontoresponse.data.development import materialize_versions
from ontoresponse.data.prepare_connected import (
    audit_materialized,
    prepare_connected_base,
)
from ontoresponse.preregistration import freeze_development, require_preregistration


def metadata(groups=12, per_class=3):
    rows = []
    cc = {}
    for g in range(groups):
        for y in (0, 1):
            for j in range(per_class):
                ident = f"origin-{g}-{y}-{j}"
                rows.append(
                    dict(
                        id=ident,
                        origin_id=ident,
                        label=y,
                        speaker_id=f"s{g}",
                        audio="not_read",
                    )
                )
                cc[ident] = f"component-{g}"
    return rows, cc


def test_component_allocation_exact_roles_and_order_invariance():
    rows, cc = metadata()
    roles = {"fusion_train": 12, "fusion_val": 8, "calibration": 8, "test": 8}
    a, report = allocate_connected(rows, cc, set(), roles, minimum=2, seed=17)
    b, again = allocate_connected(
        list(reversed(rows)),
        cc,
        set(),
        dict(reversed(list(roles.items()))),
        minimum=2,
        seed=17,
    )
    assert (
        a == b
        and report["status"] == "ALLOCATED_METADATA_ONLY"
        and not report["test_scores_read"]
    )
    ownership = {}
    for row in a:
        assert ownership.setdefault(row["source_group"], row["role"]) == row["role"]
    assert (
        len(ownership) == 12
    )  # maximize independent group participation, not just the floor
    for role, n in roles.items():
        part = [r for r in a if r["role"] == role]
        assert len(part) == n and sum(r["label"] for r in part) == n // 2
        for y in (0, 1):
            assert len({r["source_group"] for r in part if r["label"] == y}) >= 2
    assert all(
        v["precision_status"] == "NOT_POWERED_FROM_METADATA_ONLY"
        for v in report["precision"].values()
    )
    # Arbitrary scores cannot influence allocation: the solver never receives them.
    scored = [dict(r, expert_score=1000 - r["label"]) for r in rows]
    c, _ = allocate_connected(scored, cc, set(), roles, minimum=2, seed=17)
    assert [r["id"] for r in a] == [r["id"] for r in c]


def test_class_component_shortage_is_not_repaired_with_file_randomization():
    rows, cc = metadata(groups=2, per_class=40)
    chosen, report = allocate_connected(
        rows, cc, set(), {"fusion_train": 16, "fusion_val": 8}, minimum=2
    )
    assert not chosen and report["status"] == "BLOCKED_INSUFFICIENT_COMPONENTS"
    assert report["eligible_origins_by_class"] == {"0": 80, "1": 80}
    assert "INSUFFICIENT_DISJOINT_COMPONENTS_CLASS_1" in report["reasons"]


def test_exclusion_removes_whole_component_and_its_all_classes():
    rows, cc = metadata()
    selected, report = allocate_connected(
        rows, cc, {"component-0"}, {"fusion_train": 8, "fusion_val": 8}, minimum=2
    )
    assert report["excluded_candidate_origins"] == 6
    assert all(r["source_group"] != "component-0" for r in selected)
    with pytest.raises(ValueError):
        allocate_connected(rows + [rows[0]], cc, set(), {"fusion_train": 8}, minimum=2)
    with pytest.raises(ValueError):
        allocate_connected(rows, {}, set(), {"fusion_train": 8}, minimum=2)


def test_connected_base_to_versions_to_audit_to_freeze_without_expert_scores(tmp_path):
    cfg, assets = prereg_fixture(tmp_path)
    root = tmp_path / "new"
    cfg["output"] = str(root)
    cfg["runtime"] = {"budget": "smoke"}
    p = cfg["pilot"]
    p["preregistration"] = str(root / "preregistration.json")
    p.update(
        training_t_candidates=["white25", "lowpass4500"],
        environment_assignment_seed=17,
        max_output_bytes=2**20,
    )
    p["conditions_by_role"] = {
        role: ["identity", "white25", "lowpass4500"] for role in p["origins_by_role"]
    }
    p["validation_eer_conditions"] = ["white25", "lowpass4500"]
    candidates = []
    for i in range(16):
        audio = assets / f"o{i}.wav"
        t = np.arange(1200, dtype="float32")
        wave = (0.1 * np.sin((0.03 + i * 0.011) * t)).astype("float32")
        sf.write(audio, wave, 16000, subtype="FLOAT")
        candidates.append(
            dict(
                id=f"o{i}",
                origin_id=f"o{i}",
                speaker_id=f"s{i}",
                label=i % 2,
                audio=str(audio),
            )
        )
    manifest = assets / "candidates.jsonl"
    write_jsonl(manifest, candidates)
    result = prepare_connected_base(cfg, manifest)
    assert result["status"] == "PREPARED_CONNECTED_BASE"
    with pytest.raises(ValueError):
        prepare_connected_base(cfg, manifest)
    result = materialize_versions(cfg, root / "base_manifest.jsonl")
    assert result["failed"] == 0 and result["requests"] == 44
    data = audit_materialized(cfg)
    assert (
        data["origins"] == 16
        and data["waveform_checks"] == 44
        and data["source_mapping_complete"]
    )
    freeze_development(cfg)
    require_preregistration(cfg)
    with pytest.raises(ValueError):
        audit_materialized(cfg)
    assert not (root / "extraction-0-of-1.jsonl").exists()


def test_mapping_unknown_prevents_even_metadata_split(tmp_path):
    cfg, assets = prereg_fixture(tmp_path)
    cfg["runtime"] = {"budget": "smoke"}
    cfg["output"] = str(tmp_path / "unknown")
    cfg["pilot"]["source_map_manifest"] = str(tmp_path / "absent")
    with pytest.raises(Blocked, match="Official source"):
        prepare_connected_base(cfg, assets / "not_read.jsonl")
    assert not Path(cfg["output"]).exists()


def test_complete_flag_cannot_omit_official_protocol_recordings(tmp_path):
    from ontoresponse.data.source_audit import source_universe

    cfg, assets = prereg_fixture(tmp_path)
    cfg["data"] = {"asv2019": str(assets / "data")}
    protocols = assets / "data/ASVspoof2019_LA_cm_protocols"
    protocols.mkdir(parents=True)
    for split in ("train", "dev", "eval"):
        suffix = "trn" if split == "train" else "trl"
        (protocols / f"ASVspoof2019.LA.cm.{split}.{suffix}.txt").write_text(
            "s0 o0 - - bonafide\n"
        )
    source_universe(cfg)
    (protocols / "ASVspoof2019.LA.cm.eval.trl.txt").write_text(
        "s0 missing_official_recording - - bonafide\n"
    )
    with pytest.raises(Blocked, match="omits 1 official"):
        source_universe(cfg)
