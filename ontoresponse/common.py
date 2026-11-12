from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


class Blocked(RuntimeError):
    """A real resource or a protocol gate is unavailable."""


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def code_hash():
    root = Path(__file__).resolve().parents[1]
    return digest(
        {
            str(p.relative_to(root)): sha256(p)
            for folder in ("ontoresponse", "reference")
            for p in sorted((root / folder).rglob("*.py"))
        }
    )


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
    try:
        with os.fdopen(fd, "w") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def unique(rows, key="id"):
    ids = [r[key] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate " + key)


def budget(mode, origins, steps=0):
    if mode not in ("smoke", "pilot"):
        raise Blocked(
            "Full runs need a separate explicit frozen budget; this CLI supports smoke/pilot"
        )
    limit = 64 if mode == "smoke" else 1024
    if not 0 < origins <= limit or not 0 <= steps <= 20:
        raise ValueError(
            f"{mode} budget exceeded: origins <= {limit}; optimizer steps <= 20"
        )
