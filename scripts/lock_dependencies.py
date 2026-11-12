"""Lock the actually tested Python environment and pre-existing pure Python layer."""

import base64
import csv
import hashlib
import importlib.metadata as md
import json
import sys
import tarfile
from pathlib import Path

from ontoresponse.common import atomic_json, sha256


def main():
    layer = Path("${EXTERNAL_PATH}")
    out = Path("reports/next-stage-20261110")
    packages = []
    errors = []
    for dist in sorted(
        md.distributions(path=[str(layer)]), key=lambda d: d.metadata["Name"].lower()
    ):
        metadata = dist.metadata
        checked = 0
        script_records = []
        for row in csv.reader(dist.read_text("RECORD").splitlines()):
            name, expected, size = row
            if not expected or name.endswith(".pyc"):
                continue
            # pip --target relocates console scripts; never resolve outside the audited tree.
            if name.startswith("../../bin/"):
                p = layer / "bin" / Path(name).name
                script_records.append(name)
            else:
                p = layer / name
            if not p.is_file():
                errors.append({"package": metadata["Name"], "missing": name})
                continue
            algorithm, value = expected.split("=", 1)
            raw = (
                base64.urlsafe_b64encode(hashlib.sha256(p.read_bytes()).digest())
                .decode()
                .rstrip("=")
            )
            if algorithm != "sha256" or raw != value:
                errors.append({"package": metadata["Name"], "hash_mismatch": name})
            checked += 1
        packages.append(
            {
                "name": metadata["Name"],
                "version": dist.version,
                "requires_python": metadata.get("Requires-Python"),
                "requires_dist": metadata.get_all("Requires-Dist") or [],
                "record_files_checked": checked,
                "relocated_script_records": script_records,
            }
        )
    names = {d.metadata["Name"].lower().replace("_", "-") for d in md.distributions()}
    effective = [md.distribution(name) for name in names]
    installed = sorted(
        ({"name": d.metadata["Name"], "version": d.version} for d in effective),
        key=lambda r: r["name"].lower(),
    )
    archive = out / "ontology-dependencies.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=3) as stream:
        for p in sorted(layer.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                stream.add(p, arcname=p.relative_to(layer), recursive=False)
    atomic_json(
        out / "dependency-lock.json",
        {
            "python": sys.executable,
            "python_version": sys.version,
            "layer_source": "pre-existing /tmp/ontology-validation-deps copied to isolated task directory; no network download succeeded",
            "layer_packages": packages,
            "record_errors": errors,
            "layer_archive": archive.name,
            "layer_archive_sha256": sha256(archive),
            "all_visible_distributions": installed,
            "scope": "exact tested environment snapshot and offline layer archive; not a claim of a clean upstream wheel resolution",
        },
    )
    (out / "requirements-tested.txt").write_text(
        "\n".join(f"{r['name']}=={r['version']}" for r in installed) + "\n"
    )
    print(
        json.dumps(
            {
                "packages": len(packages),
                "record_errors": errors,
                "layer_sha256": sha256(archive),
            }
        )
    )
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
