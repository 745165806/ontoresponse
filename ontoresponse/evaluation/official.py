"""Use audited upstream tools; never label a small subset as official full evaluation."""
from pathlib import Path
import subprocess
import sys
from ..common import Blocked,read_jsonl,unique,sha256,atomic_json

def asv2021df_full(predictions,keys,script,destination):
    rows=read_jsonl(predictions);unique(rows)
    metadata=Path(keys)/'CM/trial_metadata.txt'
    if not metadata.is_file() or not Path(script).is_file():raise Blocked('Missing official DF evaluator/keys')
    expected={line.split()[1] for line in metadata.read_text().splitlines() if line.strip()}
    if {r['id'] for r in rows}!=expected:raise Blocked('Official DF full scoring requires all key IDs, no subset/missing IDs')
    if any(r['status']!='scored' for r in rows):raise Blocked('Official metric has no missing-request semantics; report coverage separately')
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    scores=destination/'official-scores.txt'
    # The official evaluator expects higher bonafide; invert the declared spoof margin by definition, never test label fitting.
    scores.write_text(''.join(f"{r['id']} {-r['logit']:.12g}\n" for r in rows))
    result=subprocess.run([sys.executable,str(Path(script).resolve()),str(scores.resolve()),str(Path(keys).resolve()),'eval'],capture_output=True,text=True,timeout=300)
    report={'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr,
            'evaluator_sha256':sha256(script),'metadata_sha256':sha256(metadata),'requests':len(rows)}
    atomic_json(destination/'official-result.json',report)
    return report

def asv5_track1(*args,**kwargs):
    raise Blocked('ASVspoof5 Track1 official audio/protocol/tool version must be audited and pinned before evaluation; no automatic download')
