"""Independent stdlib audit of frozen per-request predictions. Never fits anything."""
from __future__ import annotations
import argparse,hashlib,json,math
from collections import defaultdict
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def sigmoid(v):return 1/(1+math.exp(-v)) if v>=0 else math.exp(v)/(1+math.exp(v))

def decision(r):
    space=r.get('threshold_space','calibrated_probability')
    value=r['calibrated_logit'] if space.startswith('calibrated_logit') else r['logit'] if space=='raw_logit' else r['p']
    return value>=r['threshold']

def independent_metrics(records):
    scored=[r for r in records if r['status']=='scored'];y=[r['label'] for r in scored]
    if set(y)!={0,1}:raise ValueError('Both classes required')
    if any(r['label'] not in (0,1) or not all(isinstance(r[k],(int,float)) and math.isfinite(r[k]) for k in ('logit','p','threshold')) or not 0<=r['p']<=1 for r in scored):raise ValueError('Nonfinite/invalid scored row')
    n0=y.count(0);n1=y.count(1);threshold_groups=defaultdict(list)
    for r in scored:threshold_groups[r['logit']].append(r['label'])
    fpr=0.;fnr=1.;eer=None;fp=tp=0
    for score,labels in sorted(threshold_groups.items(),reverse=True):
        old_fpr,old_fnr=fpr,fnr;fp+=labels.count(0);tp+=labels.count(1);fpr=fp/n0;fnr=1-tp/n1
        if old_fpr-old_fnr<=0<=fpr-fnr:
            alpha=-(old_fpr-old_fnr)/((fpr-fnr)-(old_fpr-old_fnr));eer=old_fpr+alpha*(fpr-old_fpr);break
    eps=2.220446049250313e-16
    nll=-sum(r['label']*math.log(min(1-eps,max(eps,r['p'])))+(1-r['label'])*math.log(1-min(1-eps,max(eps,r['p']))) for r in scored)/len(scored)
    brier=sum((r['p']-r['label'])**2 for r in scored)/len(scored)
    fp=sum(decision(r) for r in scored if r['label']==0);tp=sum(decision(r) for r in scored if r['label']==1)
    bona=[r['logit'] for r in scored if r['label']==0];spoof=[r['logit'] for r in scored if r['label']==1]
    auroc=sum((a>b)+.5*(a==b) for a in spoof for b in bona)/(n0*n1)
    tp=fp=0;previous_recall=0.;auprc=0.
    for score,labels in sorted(threshold_groups.items(),reverse=True):
        tp+=labels.count(1);fp+=labels.count(0);recall=tp/n1
        auprc+=(recall-previous_recall)*tp/(tp+fp);previous_recall=recall
    fp=sum(decision(r) for r in scored if r['label']==0);tp=sum(decision(r) for r in scored if r['label']==1)
    bins=defaultdict(list)
    for r in scored:bins[min(9,int(r['p']*10))].append(r)
    reliability=[{'bin':i,'n':len(v),'mean_p':sum(r['p'] for r in v)/len(v),'spoof_fraction':sum(r['label'] for r in v)/len(v)} for i,v in sorted(bins.items())]
    ece=sum(v['n']*abs(v['mean_p']-v['spoof_fraction']) for v in reliability)/len(scored)
    return {'n_requested':len(records),'n_scored':len(scored),'n_failed':len(records)-len(scored),
            'eer':eer,'eer_convention':'linear ROC crossing after grouping tied scores','actual_fpr':fp/n0,'actual_tpr':tp/n1,
            'auroc':auroc,'spoof_auprc':auprc,'coverage':len(scored)/len(records),'nll':nll,'brier':brier,'ece_10_equal_width':ece,'reliability_bins':reliability}

def audit(root,out):
    root=Path(root);ledger={r['id']:r for r in rows(root/'audit_manifest.jsonl')};external={r['id']:r for r in rows(root/'external/conditions-audit.jsonl')}
    results=[]
    for path in sorted(root.glob('runs/*/external-evaluation/per_utterance_predictions.jsonl')):
        records=rows(path);run=path.parent.parent;cal=read(run/'calibration.json');lock=read(run/'freeze.json')
        if len({r['id'] for r in records})!=len(records) or {r['id'] for r in records}!=set(lock['test_ids']):raise ValueError('Prediction request coverage mismatch')
        if sha(run/'model.pt')!=lock['checkpoint_sha256'] or cal!=lock['calibration']:raise ValueError('Model/calibration freeze drift')
        if sha(root/'audit_manifest.jsonl')!=cal['manifest_sha256'] or sha(root/'extraction-0-of-1.jsonl')!=cal['index_sha256']:raise ValueError('Role ledger/cache lineage drift')
        if set(cal['calibration_ids'])!={i for i,r in ledger.items() if r['role']=='calibration'}:raise ValueError('Wrong calibration IDs')
        if cal['calibration']['role']!='calibration' or cal['threshold']['role']!='fusion_val':raise ValueError('Wrong calibration/threshold role')
        if set(lock['test_ids'])&set(ledger):raise ValueError('External ID leakage')
        scale=cal['calibration']['scale'];bias=cal['calibration']['bias'];threshold=cal['threshold']['threshold']
        if scale<=0 or not 0<=threshold<=math.nextafter(1.,math.inf):raise ValueError('Invalid direction/threshold space')
        equivalent_logit_threshold=(math.log(threshold/(1-threshold))-bias)/scale if 0<threshold<1 else None
        max_error=0.;decisions_match=True
        for r in records:
            if (r['label'],r['origin_id'],r['source_group'],r['condition'])!=tuple(external[r['id']][k] for k in ('label','origin_id','source_group','condition')):raise ValueError('Audit annotation mismatch')
            if r['checkpoint_sha256']!=lock['checkpoint_sha256'] or r['threshold']!=threshold:raise ValueError('Per-row freeze mismatch')
            if r['status']!='scored':
                if r['p'] is not None:raise ValueError('Missing request carries probability')
                continue
            expected=sigmoid(r['logit']*scale+bias);max_error=max(max_error,abs(r['p']-expected))
            if equivalent_logit_threshold is not None:decisions_match &= (r['p']>=threshold)==(r['logit']>=equivalent_logit_threshold)
        if max_error>3e-7 or not decisions_match:raise ValueError('Probability transform or threshold-space mismatch')
        grouped=defaultdict(list)
        for r in records:grouped[r['condition']].append(r)
        by_condition={k:independent_metrics(v) for k,v in grouped.items()}
        old_report=read(path.parent/'report.json');differences=[]
        for k,m in by_condition.items():
            for metric in ('eer','actual_fpr','actual_tpr','nll','brier','auroc','spoof_auprc'):
                error=abs(m[metric]-old_report['conditions'][k][metric])
                if error>1e-10:differences.append({'condition':k,'metric':metric,'difference':error})
        results.append({'run':run.name,'prediction_path':str(path),'prediction_sha256':sha(path),'conditions':by_condition,
            'differences_from_old_report':differences,'max_probability_error_vs_float64_platt':max_error,
            'threshold_space':'calibrated_probability','equivalent_raw_logit_threshold':equivalent_logit_threshold,
            'threshold_decision_equivalence':decisions_match if equivalent_logit_threshold is not None else None,
            'saturation_warning':('threshold at probability boundary; float32 saturated p=1 is not a finite raw-logit threshold; preserve saved probability policy' if equivalent_logit_threshold is None else None),
            'probabilities_exactly_one':sum(r.get('p')==1 for r in records),'threshold_role':'fusion_val','calibration_role':'calibration',
            'score_direction':'higher logit/p means spoof; never inverted from observed EER','legacy_probability_rounding':'float32 Platt output tolerated within 3e-7',
            'research_holdout':False,'data_status':'consumed_exploration'})
    if not results:raise ValueError('No per-request predictions found; summary is not evidence')
    result={'status':'independently_recomputed','implementation':'stdlib only; no ontoresponse metric/selection/calibration functions imported',
            'runs':results,'fitting_performed':False,'warning':'Old expert selection overlaps dev calibration/validation; existing freezes retained'}
    Path(out).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='reports/implementation');p.add_argument('--out',required=True);a=p.parse_args()
    result=audit(a.root,a.out);print(json.dumps({'status':result['status'],'runs':len(result['runs']),'differences':sum(len(r['differences_from_old_report']) for r in result['runs'])}))
