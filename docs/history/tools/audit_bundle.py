"""Self-contained export/check tool; stdlib only, runnable outside the repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

EXCLUDED = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    ".agents",
    ".codex",
    ".aws",
    "deliverables",
    ".next-deps",
    "build",
    "dist",
}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory(root, include_evidence=True):
    files = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if any(part in EXCLUDED or part.endswith(".egg-info") for part in rel.parts):
            continue
        if p.is_symlink():
            raise ValueError(
                "Symlink is not an independently archived file: " + str(rel)
            )
        if not p.is_file():
            continue
        if not include_evidence and rel.parts[0] in ("artifacts", "reports"):
            continue
        files.append(
            {"path": rel.as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)}
        )
    return files


def historical_manifests(root):
    results = []
    for name in ("MANIFEST.json", "MANIFEST.implementation.json"):
        p = root / name
        if not p.is_file():
            continue
        data = json.loads(p.read_text())
        entries = (
            data
            if isinstance(data, list)
            else [
                dict(path=k, sha256=v)
                for category in ("files", "result_files")
                for k, v in data.get(category, {}).items()
            ]
        )
        missing = []
        changed = []
        for entry in entries:
            source = root / entry["path"]
            if not source.is_file():
                missing.append(entry["path"])
            elif sha(source) != entry["sha256"]:
                changed.append(entry["path"])
        results.append(
            {
                "manifest": name,
                "entries": len(entries),
                "missing": missing,
                "changed": changed,
                "note": "Historical reference-bundle hashes are not silently rewritten to match this revision",
            }
        )
    return results


def export(root, destination):
    root = Path(root).resolve()
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError("Do not overwrite an exported snapshot")
    entries = inventory(root)
    manifest = {
        "schema": "ontoresponse-audit-bundle-1",
        "files": entries,
        "historical_manifest_audit": historical_manifests(root),
        "external_assets": "Large external expert weights and original corpora are not redistributed; their paths, existence and SHA256 are audited separately.",
        "evidence_policy": "Contains actual logs, per-request predictions, frozen fusion checkpoints and local evidence caches; STATUS is context only.",
    }
    import io

    with tarfile.open(destination, "w:gz", compresslevel=3) as archive:
        for entry in entries:
            archive.add(root / entry["path"], arcname=entry["path"], recursive=False)
        data = json.dumps(manifest, indent=2, sort_keys=True).encode()
        info = tarfile.TarInfo("BUNDLE_MANIFEST.json")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    result = verify(destination)
    result.update(
        archive_sha256=sha(destination), archive_bytes=destination.stat().st_size
    )
    return result


def verify(path):
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate tar members")
        for m in members:
            p = Path(m.name)
            if not m.isfile() or p.is_absolute() or ".." in p.parts:
                raise ValueError("Unsafe/non-file archive member")
        manifest = json.load(archive.extractfile("BUNDLE_MANIFEST.json"))
        expected = {r["path"] for r in manifest["files"]} | {"BUNDLE_MANIFEST.json"}
        if set(names) != expected:
            raise ValueError("Tar files differ from manifest")
        for entry in manifest["files"]:
            stream = archive.extractfile(entry["path"])
            h = hashlib.sha256()
            n = 0
            for data in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(data)
                n += len(data)
            if n != entry["bytes"] or h.hexdigest() != entry["sha256"]:
                raise ValueError("Hash/size mismatch: " + entry["path"])
    return {
        "status": "verified",
        "files_verified": len(manifest["files"]),
        "archive": str(path),
        "historical_manifest_audit": manifest["historical_manifest_audit"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["export", "verify"])
    parser.add_argument("path")
    parser.add_argument("--root", default=".")
    parser.add_argument("--report")
    args = parser.parse_args()
    result = (
        export(args.root, args.path) if args.command == "export" else verify(args.path)
    )
    if args.report:
        Path(args.report).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
