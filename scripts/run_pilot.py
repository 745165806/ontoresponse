"""Run only a reviewed and audited explicit new-directory pilot recipe."""
from pathlib import Path
import argparse,json
import yaml,torch
from ontoresponse.pilot_gate import require_pass,preflight
from ontoresponse.evidence.extract import extract
from ontoresponse.train.pilot import run_training,lock_and_evaluate
from ontoresponse.common import atomic_json,sha256

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/pilot-20261110.yaml');parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text());torch.set_num_threads(cfg['runtime']['threads'])
    if args.preflight_only:
        result=preflight(cfg);print(json.dumps(result,indent=2));return 2 if result['blocked'] else 0
    require_pass(cfg);root=Path(cfg['output'])
    if (root/'experiment-start.json').exists():raise ValueError('Existing experiment must not be restarted/overwritten with new choices')
    atomic_json(root/'experiment-start.json',{'config':cfg,'config_sha256':sha256(args.config),'preflight_sha256':sha256(root/'preflight.json')})
    extraction=extract(cfg,root/'inference_manifest.jsonl',cfg['pilot']['max_origins'])
    atomic_json(root/'extraction-summary.json',extraction)
    from ontoresponse.common import read_jsonl,Blocked
    records=read_jsonl(extraction['index'])
    if any('CACHE_QUOTA' in str(row) for row in records):raise Blocked('Cache quota reached; no training performed')
    if sum(p.stat().st_size for p in root.rglob('*') if p.is_file())>cfg['pilot']['max_output_bytes']:raise Blocked('Output budget reached before training')
    results=run_training(cfg)
    # Save curves before any final scoring, independent of whether loss has plateaued.
    import os
    os.environ.setdefault('MPLCONFIGDIR','${EXTERNAL_PATH}')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ontoresponse.common import read_jsonl
    for run in results:
        folder=Path(run['checkpoint']).parent;hist=read_jsonl(folder/'convergence.jsonl');fig,axis=plt.subplots(figsize=(6,4))
        axis.plot([r['optimizer_steps'] for r in hist],[r['train_eval_bce'] for r in hist],label='fusion_train BCE')
        axis.plot([r['optimizer_steps'] for r in hist],[r['val_bce'] for r in hist],label='fusion_val BCE')
        axis.set(xlabel='Optimizer steps',ylabel='Source-weighted BCE',title=f"{run['model']} seed {run['seed']}");axis.legend();fig.tight_layout();fig.savefig(folder/'convergence.png',dpi=160);plt.close(fig)
    reports=lock_and_evaluate(cfg,results)
    atomic_json(root/'completion.json',{'status':'completed_integration_pilot','models':len(results),'test_reports':len(reports),'research_holdout':False})
    return 0
if __name__=='__main__':raise SystemExit(main())
