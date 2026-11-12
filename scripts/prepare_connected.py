"""Source-connected preparation only; never invokes an expert or fusion training."""

import argparse
import json
from pathlib import Path

import yaml

from ontoresponse.common import Blocked, atomic_json
from ontoresponse.data.prepare_connected import (
    audit_materialized,
    prepare_connected_base,
)
from ontoresponse.preregistration import validate_v2_recipe


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidates")
    parser.add_argument("--audit-materialized", action="store_true")
    parser.add_argument("--report")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    try:
        validate_v2_recipe(cfg)
        if args.audit_materialized:
            result = audit_materialized(cfg)
            code = 0
        elif args.candidates:
            result = prepare_connected_base(cfg, args.candidates)
            code = 0 if result["status"] == "PREPARED_CONNECTED_BASE" else 2
        else:
            raise ValueError("Supply --candidates or --audit-materialized")
    except (Blocked, ValueError, KeyError, OSError) as err:
        result = {"status": "BLOCKED", "reason": str(err), "test_scores_read": False}
        code = 2
    if args.report:
        atomic_json(args.report, result)
    print(json.dumps(result, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
