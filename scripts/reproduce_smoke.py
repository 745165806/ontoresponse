"""Reproduce the bounded integration smoke from user-owned local resources.

Run from the repository root: py310/bin/python -m scripts.reproduce_smoke
No network, full dataset extraction, or expert training is launched.
Use a fresh output/cache directory in a copied local config for a new experiment.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

from ontoresponse.common import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local.yaml")
    args = parser.parse_args()
    config = args.config
    cfg = yaml.safe_load(Path(config).read_text())
    root = Path(cfg["output"])
    root.mkdir(parents=True, exist_ok=True)
    commands = [
        ["ontoresponse", "doctor"],
        ["ontoresponse", "prepare-data", "--max-origins", "24"],
        ["ontoresponse", "check-experts"],
        ["ontoresponse", "extract", "--max-origins", "24"],
        ["ontoresponse", "export-ontology"],
        ["scripts.run_bounded"],
        ["ontoresponse", "prepare-external", "--max-origins", "8"],
        ["ontoresponse", "make-environments"],
        ["scripts.evaluate_bounded"],
    ]
    journal = []
    for number, command in enumerate(commands):
        cmd = [sys.executable, "-m", *command, "--config", config]
        log = root / f"reproduce-{number:02d}.log"
        started = time.time()
        with log.open("w") as stream:
            r = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
        journal.append(
            {
                "command": cmd,
                "exit_code": r.returncode,
                "log": str(log),
                "seconds": time.time() - started,
            }
        )
        atomic_json(root / "reproduce-journal.json", journal)
        print(command, r.returncode, flush=True)
        if r.returncode:
            raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
