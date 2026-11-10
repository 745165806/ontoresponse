"""Explicit pilot optimizer budget, grouped ledgers and convergence history.

Never called by smoke fit; never overwrites an existing experiment. Test tensors
are only loaded for scoring after model/calibration/operating-point locks exist.
"""
from pathlib import Path
import copy,json,math
from dataclasses import replace
import numpy as np
import torch
from reference.core import ExpertStats,source_weighted_bce
from ..common import Blocked,atomic_json,write_jsonl,read_jsonl,sha256,digest,code_hash
from ..data.pilot import check_pilot_budget
from ..evidence.dataset import load_batch,origin_weights
from ..models.factory import build_model,parameter_count
from ..ontology.registry import OntologyRegistry,validate_evidence
from ..evidence.cache import Cache
from ..evaluation.metrics import metrics
from ..train.calibrate import Platt
from .fit import set_seed,forward,atomic_torch


def subset(batch,indices):
    ix=indices.tolist()
    return replace(batch,ids=[batch.ids[i] for i in ix],hidden=[h[indices] for h in batch.hidden],scores=batch.scores[indices],
                   valid=batch.valid[indices],quality=batch.quality[indices],labels=batch.labels[indices],
                   origins=[batch.origins[i] for i in ix],groups=[batch.groups[i] for i in ix],audit=[batch.audit[i] for i in ix])


def operating_point(labels,calibrated_logits,role,target_fpr):
    if role!='fusion_val' or not 0<target_fpr<1:raise ValueError('Development-only operating point')
    y=np.asarray(labels);z=np.asarray(calibrated_logits,dtype=np.float64)
    if set(y)!={0,1} or not np.isfinite(z).all():raise ValueError('Invalid threshold data')
    candidates=np.unique(np.r_[z,np.nextafter(z.max(),np.inf)])
    threshold=min(v for v in candidates if (z[y==0]>=v).mean()<=target_fpr)
    return {'space':'calibrated_logit_float64','comparison':'>=','threshold':float(threshold),'role':role,
            'target_dev_fpr':target_fpr,'actual_dev_fpr':float((z[y==0]>=threshold).mean()),'n_dev_bonafide':int((y==0).sum()),
            'warning':'pilot empirical operating point; small development counts limit FPR precision'}


def fit_one(cfg,name,seed,train,val,reg):
    p=cfg['pilot'];out=Path(cfg['output'])/'runs'/f'{name}-seed{seed}'
    if out.exists():raise ValueError('Do not overwrite pilot runs')
    out.mkdir(parents=True);set_seed(seed);dims=[h.shape[-1] for h in train.hidden]
    model=build_model(dims,reg,name,p['width'],seed);model.expert_identities=train.identities
    stats=ExpertStats.fit(*train.inputs(),role='fusion_train')
    optimizer=None if name=='best_single_expert' else torch.optim.AdamW(model.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'])
    if len(set(train.origins))!=len(train.origins):raise ValueError('This pilot recipe expects one training version/origin')
    history=[];best=None;steps=0;stale=0;stop='max_epochs'
    if name=='best_single_expert':
        candidates=[]
        for i in range(len(dims)):
            mask=val.valid[:,i,0]
            measured=metrics(val.labels[mask].numpy(),val.scores[:,i,0][mask].numpy())
            candidates.append((measured['eer'],i))
        model.chosen_expert.fill_(min(candidates)[1])
    epochs=1 if optimizer is None else p['max_epochs']
    for epoch in range(1,epochs+1):
        model.train();order=torch.randperm(len(train.ids));loss_sum=0.;batches=0
        for start in ([] if optimizer is None else range(0,len(order),p['batch_size'])):
            if steps>=p['max_optimizer_steps']:break
            batch=subset(train,order[start:start+p['batch_size']]);optimizer.zero_grad(set_to_none=True)
            result=forward(model,batch,stats,reg);loss=source_weighted_bce(result,batch.labels,origin_weights(batch.origins))
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);optimizer.step()
            steps+=1;loss_sum+=float(loss.detach());batches+=1
        model.eval()
        with torch.inference_mode():
            a=forward(model,train,stats,reg);b=forward(model,val,stats,reg)
            train_loss=float(source_weighted_bce(a,train.labels,origin_weights(train.origins)))
            val_loss=float(source_weighted_bce(b,val.labels,origin_weights(val.origins)))
            val_metric=metrics(val.labels.numpy(),b.logits.numpy())
        improved=best is None or val_loss<best['val_bce']-1e-6
        if improved:
            best={'epoch':epoch,'val_bce':val_loss,'val_eer':val_metric['eer'],'state':copy.deepcopy(model.state_dict())};stale=0
        else:stale+=1
        history.append({'epoch':epoch,'optimizer_steps':steps,'optimization_bce':loss_sum/max(1,batches),
                        'train_eval_bce':train_loss,'val_bce':val_loss,'val_eer':val_metric['eer'],'best_epoch':best['epoch']})
        write_jsonl(out/'convergence.jsonl',history)
        if epoch>=p['min_epochs'] and stale>=p['patience_epochs']:stop='validation_patience';break
        if steps>=p['max_optimizer_steps']:stop='max_optimizer_steps';break
    checkpoint={'schema':'ontoresponse-fusion-1','mode':'integration_pretrained','claim_scope':'integration_pilot_only',
        'model_name':name,'dims':dims,'width':p['width'],'seed':seed,'model_state':best['state'],'stats':stats,'registry_hash':reg.digest,
        'expert_specs':cfg['experts'],'expert_identities':train.identities,'crop':cfg['crop'],'last_model_state':model.state_dict(),
        'optimizer_state':optimizer.state_dict() if optimizer is not None else None,'rng_state':torch.get_rng_state(),'last_step':steps,'history':history,
        'train_ids':train.ids,'val_ids':val.ids,'best_epoch':best['epoch'],'best_step':history[best['epoch']-1]['optimizer_steps'],'selection_rule':'min fusion_val BCE within fixed convergence budget',
        'code_sha256':code_hash(),'pilot_budget':p,'ledger_sha256':sha256(Path(cfg['output'])/'audit_manifest.jsonl')}
    atomic_torch(out/'model.pt',checkpoint)
    model.load_state_dict(best['state']);model.eval()
    from .fit import restore
    _,restored,loaded_stats,loaded_reg=restore(out/'model.pt')
    with torch.inference_mode():torch.testing.assert_close(forward(model,val,stats,reg).logits,forward(restored,val,loaded_stats,loaded_reg).logits,atol=0,rtol=0)
    result={'model':name,'seed':seed,'parameters':parameter_count(model),'epochs':len(history),'steps':steps,'best_epoch':best['epoch'],
            'val_bce':best['val_bce'],'val_eer':best['val_eer'],'stop_reason':stop,'convergence_claim':stop=='validation_patience',
            'checkpoint':str(out/'model.pt'),'checkpoint_sha256':sha256(out/'model.pt'),'save_load_exact':True}
    atomic_json(out/'result.json',result);return result


def run_training(cfg):
    from ..pilot_gate import require_pass
    require_pass(cfg)
    check_pilot_budget(cfg);root=Path(cfg['output']);manifest=root/'audit_manifest.jsonl';index=root/'extraction-0-of-1.jsonl'
    reg=validate_evidence(cfg,read_jsonl(index),Cache(cfg['cache']))
    train=load_batch(cfg,index,manifest,'fusion_train');val=load_batch(cfg,index,manifest,'fusion_val',[h.shape[-1] for h in train.hidden])
    results=[]
    from ..evidence.diagnostics import scaler_diagnostics,crop_coverage
    stats=ExpertStats.fit(*train.inputs(),role='fusion_train')
    atomic_json(root/'scaler-diagnostics.json',scaler_diagnostics(train,stats,[e['id'] for e in cfg['experts']],'fusion_train'))
    atomic_json(root/'crop-coverage.json',crop_coverage(read_jsonl(manifest),cfg['crop'],[e['id'] for e in cfg['experts']]))
    def run_names(names):
        for name in names:
            for seed in cfg['pilot']['seeds']:
                results.append(fit_one(cfg,name,seed,train,val,reg));atomic_json(root/'training-results.json',results)
                print(name,seed,results[-1]['steps'],results[-1]['stop_reason'],flush=True)
    run_names(cfg['pilot']['h1_models'])
    def average(name,key):return float(np.mean([r[key] for r in results if r['model']==name]))
    chosen_response=min(('response_mlp','all_views_mlp'),key=lambda n:(average(n,'val_eer'),average(n,'val_bce'),n))
    improved=sum(next(r for r in results if r['model']==chosen_response and r['seed']==seed)['val_eer'] < next(r for r in results if r['model']=='original_mlp' and r['seed']==seed)['val_eer']-1e-9 for seed in cfg['pilot']['seeds'])
    supported=(average('original_mlp','val_eer')-average(chosen_response,'val_eer')>=cfg['pilot']['h1_gate']['min_absolute_val_eer_improvement'] and improved>=2 and average(chosen_response,'val_bce')<=average('original_mlp','val_bce'))
    gate={'status':'SUPPORTED_IN_PILOT' if supported else 'NOT_SUPPORTED_IN_PILOT','selected_response':chosen_response,
          'original_mean_val_eer':average('original_mlp','val_eer'),'response_mean_val_eer':average(chosen_response,'val_eer'),
          'improved_seeds':improved,'definition':cfg['pilot']['h1_gate'],'test_used':False,
          'interpretation':'Exploratory validation gate only, not statistical proof; no H2 escalation when H1 is unsupported'}
    atomic_json(root/'H1-gate.json',gate)
    if supported:run_names(cfg['pilot']['h2_models']+cfg['pilot']['confirmation_models'])
    else:atomic_json(root/'H2-not-run.json',{'reason':'H1 unsupported under preregistered validation gate','not_run':cfg['pilot']['h2_models']+cfg['pilot']['confirmation_models']})
    candidates={r['model'] for r in results if r['model'] in ('response_mlp','all_views_mlp','set_attention','full_attention')}
    chosen=min(candidates,key=lambda name:(np.mean([r['val_eer'] for r in results if r['model']==name]),np.mean([r['val_bce'] for r in results if r['model']==name]),name))
    selection={'role':'fusion_val','selected_nonontology':chosen,'rule':'3-seed mean validation EER, tie mean BCE','results_sha256':sha256(root/'training-results.json')}
    atomic_json(root/'selection.json',selection)
    return results


def lock_and_evaluate(cfg,results):
    root=Path(cfg['output']);manifest=root/'audit_manifest.jsonl';index=root/'extraction-0-of-1.jsonl';p=cfg['pilot']
    cal=load_batch(cfg,index,manifest,'calibration');val=load_batch(cfg,index,manifest,'fusion_val')
    # Complete every lock before reading any test tensor or selecting using test scores.
    locks=[]
    from .fit import restore
    for run in results:
        cp,model,stats,reg=restore(run['checkpoint'])
        with torch.inference_mode():c=forward(model,cal,stats,reg);v=forward(model,val,stats,reg)
        if not c.observed.all() or not v.observed.all():raise Blocked('Calibration/validation coverage failure')
        params=Platt.fit(c.logits.numpy(),cal.labels.numpy(),'calibration')
        z=v.logits.double().numpy()*params['scale']+params['bias'];point=operating_point(val.labels.numpy(),z,'fusion_val',p['target_dev_fpr'])
        lock={'status':'frozen_integration_pilot','model_sha256':sha256(run['checkpoint']),'calibration':params,'operating_point':point,
              'ledger_sha256':sha256(manifest),'index_sha256':sha256(index),'config':cfg,'registry_hash':reg.digest,
              'code_sha256':code_hash(),'data_audit_sha256':sha256(root/'data-audit.json'),'selection_sha256':sha256(root/'selection.json'),
              'test_requests':[r['id'] for r in read_jsonl(manifest) if r['role']=='test'],'research_holdout':False}
        path=Path(run['checkpoint']).parent/'freeze.json'
        if path.exists():raise ValueError('Pilot lock already exists')
        atomic_json(path,lock);locks.append((run,lock))
    test=load_batch(cfg,index,manifest,'test');reports=[]
    for run,lock in locks:
        _,model,stats,reg=restore(run['checkpoint'])
        with torch.inference_mode():out=forward(model,test,stats,reg)
        params=lock['calibration'];z=out.logits.double().numpy()*params['scale']+params['bias'];prob=torch.tensor(z).sigmoid().numpy()
        point=lock['operating_point'];records=[]
        for i,ident in enumerate(test.ids):
            ok=bool(out.observed[i]);records.append({'id':ident,'origin_id':test.origins[i],'source_group':test.groups[i],
                'status':'scored' if ok else 'review','label':int(test.labels[i]),'condition':'no_new_processing',
                'logit':float(out.logits[i]) if ok else None,'calibrated_logit':float(z[i]) if ok else None,'p':float(prob[i]) if ok else None,
                'raw_logit':float(out.logits[i]) if ok else None,'p_calibrated':float(prob[i]) if ok else None,
                'prediction':int(z[i]>=point['threshold']) if ok else None,'speaker_group':test.audit[i]['speaker_id'],
                'calibrator_id':digest(params),'model_id':run['checkpoint_sha256'],
                'threshold_space':point['space'],'threshold':point['threshold'],'model_sha256':lock['model_sha256']})
        write_jsonl(Path(run['checkpoint']).parent/'per_utterance_predictions.jsonl',records)
        valid=np.array([r['status']=='scored' for r in records]);y=test.labels.numpy()[valid]
        measured=metrics(y,out.logits.numpy()[valid],prob[valid]) if set(y)=={0.,1.} else {'status':'undefined'}
        decisions=z[valid]>=point['threshold']
        measured.update(N_requested=len(records),N_scored=int(valid.sum()),N_failed=int((~valid).sum()),
                        actual_fpr=float(decisions[y==0].mean()) if (y==0).any() else None,
                        actual_tpr=float(decisions[y==1].mean()) if (y==1).any() else None,threshold_space=point['space'])
        report={'model':run['model'],'seed':run['seed'],'metrics':measured,'scope':'integration_pilot_only'}
        atomic_json(Path(run['checkpoint']).parent/'test-report.json',report);reports.append(report)
    atomic_json(root/'test-reports.json',reports)
    return reports
