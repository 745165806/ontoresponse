"""Freeze before external extraction/evaluation. No test-driven selection."""
from pathlib import Path
import copy
import argparse
import json
import yaml
import torch
from ontoresponse.common import atomic_json,read_jsonl
from ontoresponse.evaluation.workflow import calibrate,freeze,evaluate
from ontoresponse.evaluation.report import paired_report
from ontoresponse.evaluation.metrics import paired_macro_bootstrap
from ontoresponse.evidence.extract import extract

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/local.yaml');args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text());torch.set_num_threads(cfg['runtime']['threads'])
    root=Path(cfg['output']);selection=json.loads((root/'selection.json').read_text());test=root/'external/conditions-audit.jsonl'
    variants=[(selection['best_nonontology']['model'],selection['best_nonontology']['capacity']),('typed_response','fixed')]
    checkpoints=[]
    # Train/val/calibration roles only; test ledger is frozen without reading model outputs.
    for model,capacity in variants:
        for seed in (17,29,43):
            cp=root/'runs'/f'{model}-{capacity}-seed{seed}'/'model.pt'
            calibrate(cfg,cp,root/'extraction-0-of-1.jsonl',root/'audit_manifest.jsonl')
            freeze(cfg,cp,test)
            checkpoints.append((model,seed,cp))
            print('calibrated and frozen',model,seed,flush=True)
    external=copy.deepcopy(cfg);external['output']=str(root/'external')
    (root/'external/config.yaml').write_text(yaml.safe_dump(external,sort_keys=False))
    result=extract(external,root/'external/conditions-inference.jsonl',8)
    atomic_json(root/'external/extraction-summary.json',result)
    reports=[];predictions={}
    for model,seed,cp in checkpoints:
        destination=cp.parent/'external-evaluation'
        report=evaluate(external,cp,result['index'],test,cp.parent/'freeze.json',destination)
        predictions[(model,seed)]=read_jsonl(destination/'per_utterance_predictions.jsonl')
        reports.append({'model':model,'seed':seed,'report':report})
        print('evaluated',model,seed,report['N_scored'],report['unseen_macro_eer'],flush=True)
    comparisons=[]
    for seed in (17,29,43):
        a=predictions[('typed_response',seed)];b=predictions[(variants[0][0],seed)]
        common=[(x,y) for x,y in zip(a,b,strict=True) if x['status']==y['status']=='scored']
        if any(x['id']!=y['id'] for x,y in common):raise ValueError('Unpaired IDs')
        paired=paired_report(a,b,replicates=2000)
        if common:
            macro=paired_macro_bootstrap([x['label'] for x,y in common],[x['logit'] for x,y in common],
                    [y['logit'] for x,y in common],[x['source_group'] for x,y in common],[x['condition'] for x,y in common],replicates=2000)
        else:macro=None
        comparisons.append({'seed':seed,'paired':paired,'primary_macro':macro})
    atomic_json(root/'external-results.json',{'status':'integration_smoke_only','selection':selection,'reports':reports,'paired_comparisons':comparisons,
        'limitations':['8 origins only; this is a software/feasibility smoke, not formal effect estimation',
                       'source overlap unknown; no independence claim','public/local expert dev-selection overlaps fusion roles',
                       'AAC/Opus unseen by fusion training only; all-expert pretraining exposure unknown'],
        'primary_metric':'AAC/Opus macro EER, conditions fixed before final prediction',
        'seed_aggregation':'each seed reported separately; bootstrap samples source/speaker groups; seeds are not independent recordings'})
if __name__=='__main__':main()
