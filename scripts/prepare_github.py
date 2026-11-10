"""Prepare only the sanitized shareable source/evidence snapshot for GitHub."""
from pathlib import Path
import argparse,json,subprocess,zipfile,hashlib
from ontoresponse.audit_export import verify_audit_zip,FORBIDDEN
from ontoresponse.common import atomic_json,sha256


def run(args,cwd):
    result=subprocess.run(args,cwd=cwd,capture_output=True,text=True,check=True)
    return result.stdout.strip()


def main():
    p=argparse.ArgumentParser();p.add_argument('--zip',default='reports/codex-next-stage-20261110/audit_bundle.zip');p.add_argument('--email',default='745165806@qq.com');a=p.parse_args()
    archive=Path(a.zip).resolve();verify_audit_zip(archive)
    stage=Path('${EXTERNAL_PATH}')
    if stage.exists():raise ValueError('Publishing stage exists; do not overwrite a reviewed commit')
    stage.mkdir()
    with zipfile.ZipFile(archive) as z:z.extractall(stage)  # names already verified, no path traversal/symlinks
    (stage/'.gitignore').write_text('artifacts/\ndeliverables/\n__pycache__/\n.pytest_cache/\n*.pt\n*.pth\n*.npz\n*.npy\n*.wav\n*.flac\n*.mp3\n*.m4a\n*.ogg\n*.bundle\n.env\n*.pem\n*.key\n')
    for path in stage.rglob('*'):
        if path.is_file() and path.suffix in FORBIDDEN:raise ValueError('Forbidden published asset '+str(path))
    run(['git','init','-b','main'],stage);run(['git','config','user.name','745165806'],stage);run(['git','config','user.email',a.email],stage)
    run(['git','add','--all'],stage);run(['git','commit','-m','OntoResponse v0.5: audited staged integration pilot and portable evidence'],stage)
    head=run(['git','rev-parse','HEAD'],stage)
    bundle=Path('deliverables/ontoresponse-github-20261110.bundle').resolve();bundle.parent.mkdir(exist_ok=True)
    run(['git','bundle','create',str(bundle),'--all'],stage);verification=run(['git','bundle','verify',str(bundle)],stage)
    atomic_json('reports/codex-next-stage-20261110/github-publish-ready.json',{'local_stage':str(stage),'commit':head,'commit_email':a.email,
      'source_zip_sha256':sha256(archive),'bundle':str(bundle),'bundle_sha256':sha256(bundle),'bundle_verification':verification,
      'branch':'main','audio_weights_credentials_included':False,'uploaded':False,
      'remote_blocker':'No authenticated GitHub connector/CLI and shell HTTPS/DNS connection unavailable; email is not a remote repository URL',
      'expected_remote_visibility':'private unless user specifies otherwise'})
    print('sanitized commit',head,'ready; upload is not claimed')
if __name__=='__main__':main()
