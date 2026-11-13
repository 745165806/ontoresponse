"""Check source/role separation or optionally save a readable experiment snapshot."""

import argparse
import json
from pathlib import Path

import yaml

from ontoresponse.common import Blocked, atomic_json
from ontoresponse.data.pilot import check_pilot_budget
from ontoresponse.preregistration import (
    freeze_development,
    validate_development_data,
    validate_v2_recipe,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--report")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    try:
        validate_v2_recipe(cfg)
        if args.freeze:
            check_pilot_budget(cfg)
            result = freeze_development(cfg)
        else:
            result = {
                "status": "DEVELOPMENT_INPUTS_VALID",
                "pilot_enabled": cfg["pilot"]["enabled"],
                "data": validate_development_data(cfg),
                "test_used": False,
            }
        code = 0
    except (Blocked, ValueError, KeyError, OSError) as error:
        result = {
            "status": "BLOCKED",
            "reason": str(error),
            "freeze_written": False,
            "test_used": False,
        }
        code = 2
    if args.report:
        atomic_json(args.report, result)
    print(json.dumps(result, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
