"""Synthetic privacy/export fixtures; no expert outputs are fabricated in production."""

import hashlib
import json
import math
import subprocess
import sys
import zipfile
from pathlib import Path

from ontoresponse.audit_export import PERSONAL, export_audit, verify_audit_zip


def test_shareable_zip_has_no_audio_weights_paths_and_recomputes(tmp_path):
    root = tmp_path / "source"
    run = root / "reports/implementation/runs/unit-seed17/external-evaluation"
    run.mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "ontoresponse").mkdir()
    original = Path(__file__).parents[1] / "scripts/recompute_predictions.py"
    (root / "scripts/recompute_predictions.py").write_text(original.read_text())
    # Assemble the synthetic path so publishing this test does not pre-redact
    # the fixture and accidentally weaken the privacy regression check.
    private_path = "/".join(("", "home", "testperson", "private", "data"))
    (root / "ontoresponse/unit.py").write_text(f"DATA={private_path!r}\n")
    (run / "original.wav").write_bytes(b"synthetic forbidden audio placeholder")
    model = run.parent / "model.pt"
    model.write_bytes(b"unit hash fixture only; not an expert")
    checkpoint = hashlib.sha256(model.read_bytes()).hexdigest()
    (run.parent / "freeze.json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": checkpoint,
                "calibration": {
                    "calibration": {"scale": 1.0, "bias": 0.0, "role": "calibration"}
                },
            }
        )
    )
    rows = []
    for i, (s, y) in enumerate([(-2, 0), (-1, 0), (1, 1), (2, 1)]):
        rows.append(
            {
                "id": str(i),
                "origin_id": str(i),
                "source_group": "g" + str(i),
                "label": y,
                "logit": s,
                "p": 1 / (1 + math.exp(-s)),
                "threshold": 0.5,
                "status": "scored",
                "condition": "original",
            }
        )
    (run / "per_utterance_predictions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    target = tmp_path / "share.zip"
    result = export_audit(root, target)
    assert result["prediction_rows"] == 4
    assert verify_audit_zip(target)["status"] == "verified"
    unpack = tmp_path / "unpack"
    unpack.mkdir()
    with zipfile.ZipFile(target) as z:
        assert not any(Path(name).suffix in (".pt", ".wav") for name in z.namelist())
        assert not any(PERSONAL.search(z.read(name).decode()) for name in z.namelist())
        z.extractall(unpack)
    result = subprocess.run(
        [sys.executable, "audit_recompute.py"],
        cwd=unpack,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    measured = json.loads((unpack / "metrics_recomputed_public.json").read_text())
    assert next(iter(measured.values()))["eer"] == 0

    # Publishing must omit oversized reports and raw prediction tables while
    # preserving the small, independently recomputable evidence above.
    limit = 1024 * 1024
    oversized = root / "reports/oversized.log"
    oversized.write_bytes(b"x" * (limit + 1))
    large_predictions = root / "reports/full/per_utterance_predictions.jsonl"
    large_predictions.parent.mkdir()
    large_predictions.write_bytes(b"x" * (limit + 1))
    bounded = tmp_path / "bounded.zip"
    assert export_audit(root, bounded, max_file_bytes=limit)["prediction_rows"] == 4
    with zipfile.ZipFile(bounded) as archive:
        manifest = json.loads(archive.read("PUBLIC_FILES.json"))
        assert all(set(row) == {"path", "bytes"} for row in manifest["files"])
        excluded = {row["path"] for row in manifest["intentional_exclusions"]}
        assert "reports/oversized.log" in excluded
        assert "reports/full/per_utterance_predictions.jsonl" in excluded
        assert "reports/oversized.log" not in archive.namelist()
        assert all(info.file_size <= limit for info in archive.infolist())
