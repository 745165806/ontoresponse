"""Build historical T versions from an explicitly supplied grouped origin ledger."""

import argparse
import json
from pathlib import Path

import yaml

from ontoresponse.data.development import materialize_versions
from ontoresponse.data.pilot import check_pilot_budget
from ontoresponse.preregistration import validate_v2_recipe


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-manifest", required=True)
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    validate_v2_recipe(cfg)
    check_pilot_budget(cfg)
    result = materialize_versions(cfg, args.base_manifest)
    print(json.dumps(result, indent=2))
    return 2 if result["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
