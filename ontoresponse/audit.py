from __future__ import annotations

import importlib.metadata
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import psutil
import torch

from .common import code_hash, sha256


def command(args):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return {
            "command": args,
            "exit_code": r.returncode,
            "stdout": r.stdout[-12000:],
            "stderr": r.stderr[-3000:],
        }
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"command": args, "exit_code": None, "error": str(e)}


def doctor(cfg):
    resources = {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_count": torch.cuda.device_count(),
        "cpu_count": psutil.cpu_count(),
        "memory": dict(psutil.virtual_memory()._asdict()),
        "disk": dict(zip(("total", "used", "free"), shutil.disk_usage("."))),
        "nvidia_smi": command(["nvidia-smi"]),
        "ffmpeg": command(["ffmpeg", "-version"]),
        "encoders": command(["ffmpeg", "-hide_banner", "-encoders"]),
    }
    deps = {}
    for name in (
        "torch",
        "numpy",
        "scipy",
        "soundfile",
        "PyYAML",
        "pytest",
        "rdflib",
        "pyshacl",
        "fairseq",
        "scikit-learn",
    ):
        try:
            deps[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            deps[name] = None
    experts, blocked = [], []
    if not resources["cuda_available"]:
        blocked.append("GPU: CUDA unavailable in this session; CPU work continues")
    for e in cfg["experts"]:
        item = {
            "id": e["id"],
            "supervision": e.get("supervision"),
            "artifacts": {},
            "repository": e["repository"],
            "git": command(["git", "-C", e["repository"], "rev-parse", "HEAD"]),
        }
        for name in ("checkpoint", "frontend"):
            if e.get(name):
                p = Path(e[name])
                item["artifacts"][name] = {
                    "path": str(p),
                    "exists": p.is_file(),
                    "sha256": sha256(p) if p.is_file() else None,
                    "bytes": p.stat().st_size if p.is_file() else None,
                }
                if not p.is_file():
                    blocked.append(e["id"] + ": missing " + name)
        repo = Path(e["repository"])
        for name in ("LICENSE", "NOTICE", "README.md", e["module"]):
            p = repo / name
            if p.is_file():
                item.setdefault("source_files", {})[name] = sha256(p)
        item["license_text"] = (
            (repo / "LICENSE").read_text() if (repo / "LICENSE").is_file() else None
        )
        item["research_holdout_eligible"] = (
            e.get("supervision", {}).get("independent_holdout") is True
        )
        if not item["research_holdout_eligible"]:
            blocked.append(e["id"] + ": research_holdout supervision not certified")
        experts.append(item)
    data = {
        k: {"path": v, "exists": bool(v and Path(v).exists())}
        for k, v in cfg["data"].items()
    }
    return {
        "status": "PARTIAL" if blocked else "READY",
        "mode": cfg["mode"],
        "resources": resources,
        "dependencies": deps,
        "project_code_sha256": code_hash(),
        "project_git": command(["git", "rev-parse", "HEAD"]),
        "experts": experts,
        "data": data,
        "blocked": blocked,
        "external_source_mapping": None,
        "formal_protocol_frozen": False,
    }
