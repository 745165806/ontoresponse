"""Prespecified bounded engineering comparison. Never evaluates final test."""
from pathlib import Path
import argparse
import json
import yaml
import torch
from ontoresponse.common import atomic_json
from ontoresponse.models.baselines import BASELINES
from ontoresponse.train.fit import fit

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/local.yaml');p.add_argument('--steps',type=int,default=20)
    args=p.parse_args();cfg=yaml.safe_load(Path(args.config).read_text());torch.set_num_threads(cfg['runtime']['threads'])
    root=Path(cfg['output']);index=root/'extraction-0-of-1.jsonl';ledger=root/'audit_manifest.jsonl'
    results=[]
    def run(name,seed,capacity='fixed',width=64):
        result=fit(cfg,index,ledger,name,seed,args.steps,width=width,capacity=capacity)
        results.append(result);print(name,seed,capacity,result['parameters'],result['val_metrics']['eer'],flush=True)
        atomic_json(root/'comparison.json',{'status':'real_data_integration_smoke','results':results,
            'test_used':False,'warning':'8 training / 8 validation requests; expert checkpoint selection overlaps dev; no research efficacy claim'})
    # Stage P3 always finishes first.
    for name in BASELINES:
        for seed in (17,29,43):run(name,seed)
    atomic_json(root/'p3-completed.json',{'models':list(BASELINES),'seeds':[17,29,43],'steps':args.steps,'status':'trained_real_cache_smoke'})
    # P4/P5 same-cache models, followed by count-matched and stronger ordinary baselines.
    for name in ('set_attention','untyped_response','typed_response','wrong_family','drop_anchor','drop_same_probe','drop_same_family','no_response','missing_probe'):
        for seed in (17,29,43):run(name,seed)
    for name in ('single_expert_0','single_expert_1','single_expert_2'):
        for seed in (17,29,43):run(name,seed)
    for name in ('response_mlp','all_views_mlp','set_attention','untyped_response'):
        for seed in (17,29,43):run(name,seed,'matched')
    for name in ('response_mlp','all_views_mlp','set_attention'):
        for seed in (17,29,43):run(name,seed,'stronger',128)
    # Best non-ontology architecture chosen on mean fusion_val EER, tie BCE; seed fixed 17 for demo calibration.
    baseline_names=set(BASELINES)|{'set_attention'}
    keys={(r['model'],r['capacity']) for r in results if r['model'] in baseline_names}
    ranked=[]
    for key in keys:
        family=[r for r in results if (r['model'],r['capacity'])==key]
        ranked.append((sum(r['val_metrics']['eer'] for r in family)/len(family),sum(r['val_metrics']['nll'] for r in family)/len(family),key))
    winner=min(ranked)[2]
    selection={'role':'fusion_val','criterion':'mean EER across 3 seeds, tie mean NLL; fixed seed17 deployment demo',
               'best_nonontology':{'model':winner[0],'capacity':winner[1]},'candidate_test_access':False,
               'checkpoint':next(r['checkpoint'] for r in results if (r['model'],r['capacity'])==winner and r['seed']==17)}
    atomic_json(root/'selection.json',selection)
if __name__=='__main__':main()
