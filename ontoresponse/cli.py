from __future__ import annotations
import argparse
from pathlib import Path
import torch
import yaml
from .common import Blocked, atomic_json

def main(argv=None):
    p=argparse.ArgumentParser(description='OntoResponse bounded real-expert pipeline')
    p.add_argument('command',choices=['doctor','audit','audit-export','prepare-data','prepare-external','make-environments','export-ontology','check-experts','extract','merge','train','ablate','calibrate','freeze','evaluate','infer','report'])
    p.add_argument('--config',default='configs/local.yaml');p.add_argument('--max-origins',type=int,default=24)
    p.add_argument('--source-root');p.add_argument('--manifest');p.add_argument('--index');p.add_argument('--output');p.add_argument('--model',default='response_mlp')
    p.add_argument('--steps',type=int,default=20);p.add_argument('--seed',type=int,default=17)
    p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shard-count',type=int,default=1)
    p.add_argument('--shacl',action='store_true');p.add_argument('--indices',nargs='+');p.add_argument('--audio');p.add_argument('--checkpoint');p.add_argument('--lock');p.add_argument('--role',default='test')
    args=p.parse_args(argv)
    cfg=yaml.safe_load(Path(args.config).read_text());torch.set_num_threads(cfg['runtime'].get('threads',4))
    root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    try:
        if args.command=='audit-export':
            from .audit_export import export_audit
            result=export_audit(args.source_root or Path(__file__).resolve().parents[1],args.output or root/'audit_bundle.zip')
        elif args.command in ('doctor','audit'):
            from .audit import doctor
            result=doctor(cfg);atomic_json(args.output or root/'audit.json',result)
        elif args.command=='prepare-data':
            from .data.prepare import prepare
            result=prepare(cfg,args.max_origins)
        elif args.command=='prepare-external':
            from .data.external import prepare_inthewild
            result=prepare_inthewild(cfg,args.max_origins)
        elif args.command=='make-environments':
            from .data.environments import make_external_conditions
            result=make_external_conditions(cfg)
        elif args.command=='export-ontology':
            from .ontology.registry import export_evidence
            result=export_evidence(cfg,args.index or root/'extraction-0-of-1.jsonl',args.output or root/'evidence.ttl',shacl=args.shacl)
        elif args.command=='check-experts':
            from .validation import check_experts
            result=check_experts(cfg,args.manifest or root/'inference_manifest.jsonl')
            atomic_json(args.output or root/'real-checkpoints.json',result)
        elif args.command=='extract':
            from .evidence.extract import extract
            result=extract(cfg,args.manifest or root/'inference_manifest.jsonl',args.max_origins,args.shard_index,args.shard_count)
        elif args.command=='merge':
            from .evidence.extract import merge
            result=merge(args.indices,args.manifest or root/'inference_manifest.jsonl',args.output or root/'extraction.jsonl')
        elif args.command in ('train','ablate'):
            from .train.fit import fit,ablate
            fn=fit if args.command=='train' else ablate
            result=fn(cfg,args.index or root/'extraction-0-of-1.jsonl',args.manifest or root/'audit_manifest.jsonl',args.model,args.seed,args.steps,args.output)
        elif args.command in ('calibrate','freeze','evaluate','infer','report'):
            from .evaluation.workflow import execute
            result=execute(args,cfg)
        print(__import__('json').dumps(result,indent=2,allow_nan=False))
        if result.get('status')=='BLOCKED':return 2
        return 0
    except (Blocked,ValueError,FileNotFoundError) as err:
        result={'status':'BLOCKED','command':args.command,'error':str(err)}
        atomic_json(root/(args.command+'-blocked.json'),result)
        print(__import__('json').dumps(result,indent=2));return 2
