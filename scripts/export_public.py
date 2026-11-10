"""Export the CODEX A7 shareable bundle with extra temporary-path redaction."""
from pathlib import Path
import re,json,zipfile,hashlib
from ontoresponse import audit_export
from ontoresponse.common import atomic_json

def main():
    # Temporary directories may also contain the local username. This changes
    # text sanitization only; no training/inference code or frozen data is edited.
    audit_export.PERSONAL=re.compile(audit_export.PERSONAL.pattern+r'|/tmp/(?:pytest-of-[^/\s\"\']+|ontoresponse[^/\s\"\']*)(?:/[^\s\"\'<>;,)}\]]*)?')
    destination=Path('reports/codex-next-stage-20261110/audit_bundle.zip')
    result=audit_export.export_audit('.',destination)
    with zipfile.ZipFile(destination) as archive:payload={name:archive.read(name) for name in archive.namelist()}
    manifest=json.loads(payload['AUDIT_MANIFEST.json'])
    # A private tar inventory is historical evidence, not a promise that its
    # audio members were delivered in this shareable ZIP.
    for name in payload:
        if name.endswith('/external-assets.json'):
            data=json.loads(payload[name])
            for item in data.get('assets',[]):
                item['included_in_bundle']=False
                item['scope']='external/private asset; not included in this shareable ZIP'
            payload[name]=json.dumps(data,indent=2).encode()
    for row in manifest['files']:
        data=payload[row['path']];row['bytes']=len(data);row['sha256']=hashlib.sha256(data).hexdigest()
    manifest['policy']='Shareable ZIP excludes all audio/weights/cache archives; historical private inventory statements have a separate scope.'
    payload['AUDIT_MANIFEST.json']=json.dumps(manifest,indent=2).encode()
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,data in sorted(payload.items()):archive.writestr(name,data)
    result=audit_export.verify_audit_zip(destination)
    result.update(zip_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),zip_bytes=destination.stat().st_size)
    atomic_json('reports/codex-next-stage-20261110/public-export-verification.json',result)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
