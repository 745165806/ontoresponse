"""Export bounded public source/evidence with private and temporary paths redacted."""

import argparse
import json
import re
import zipfile
from pathlib import Path

from ontoresponse import audit_export
from ontoresponse.common import atomic_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/public-export.zip")
    parser.add_argument("--max-file-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--report")
    args = parser.parse_args(argv)
    # Temporary directories may also contain the local username. This changes
    # text sanitization only; no training/inference code or frozen data is edited.
    audit_export.PERSONAL = re.compile(
        audit_export.PERSONAL.pattern
        + r"|/tmp/(?:pytest-of-[^/\s\"\']+|ontoresponse[^/\s\"\']*)(?:/[^\s\"\'<>;,)}\]]*)?"
    )
    destination = Path(args.output)
    result = audit_export.export_audit(
        ".", destination, max_file_bytes=args.max_file_bytes
    )
    with zipfile.ZipFile(destination) as archive:
        payload = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(payload["PUBLIC_FILES.json"])
    # A private tar inventory is historical evidence, not a promise that its
    # audio members were delivered in this shareable ZIP.
    for name in payload:
        if name.endswith("/external-assets.json"):
            data = json.loads(payload[name])
            for item in data.get("assets", []):
                item["included_in_bundle"] = False
                item["scope"] = (
                    "external/private asset; not included in this shareable ZIP"
                )
            payload[name] = json.dumps(data, indent=2).encode()
    for row in manifest["files"]:
        data = payload[row["path"]]
        row["bytes"] = len(data)
    manifest["policy"] = (
        "Shareable ZIP excludes all audio/weights/cache archives; historical private inventory statements have a separate scope."
    )
    payload["PUBLIC_FILES.json"] = json.dumps(manifest, indent=2).encode()
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for name, data in sorted(payload.items()):
            archive.writestr(name, data)
    result = audit_export.verify_audit_zip(destination)
    result.update(
        zip_bytes=destination.stat().st_size,
    )
    atomic_json(
        args.report or destination.parent / "public-export-verification.json", result
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
