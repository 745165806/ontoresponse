"""Training-only numerical diagnostics. These never change a fitted transform."""
from pathlib import Path
import numpy as np
import torch
from reference.core import ExpertStats
from ..common import atomic_json


def distribution(values):
    x=np.asarray(values,dtype=float).reshape(-1);x=x[np.isfinite(x)]
    return {'n':len(x),'quantiles':np.quantile(x,[0,.01,.5,.99,1]).tolist() if len(x) else None}


def scaler_diagnostics(batch,stats,expert_ids,role):
    if role!='fusion_train':raise ValueError('Numerical scale decisions/diagnostics in this recipe use fusion_train only')
    h,s=stats.transform(*batch.inputs());records=[]
    for i,name in enumerate(expert_ids):
        original=batch.hidden[i][batch.valid[:,i,0],0]
        raw_std=original.std(0,unbiased=False).numpy()
        response=h[i][:,1:]-h[i][:,:1];rmask=batch.valid[:,i,1:]&batch.valid[:,i,:1]
        records.append({'expert':name,'hidden_std':distribution(raw_std),'std_floor':1e-5,
            'hidden_floor_fraction':float((raw_std<=1e-5).mean()),
            'score_raw_std':float(batch.scores[:,i,0][batch.valid[:,i,0]].std(unbiased=False)),
            'score_floor_hit':bool(stats.score_scales[i]<=1e-5),
            'valid_z_abs':distribution(h[i][batch.valid[:,i]].abs().numpy()),
            'original_norm':distribution(h[i][:,0][batch.valid[:,i,0]].norm(dim=-1).numpy()),
            'response_norm':distribution(response[rmask].norm(dim=-1).numpy()),
            'extreme_abs_z_gt_20_fraction':float((h[i][batch.valid[:,i]].abs()>20).float().mean()),
            'fit_role':'fusion_train','transform_changed':False})
    return {'role':role,'records':records,'policy':'Diagnostic only; no adjustment using validation or test'}


def crop_coverage(rows,crop_policy,expert_ids):
    records=[]
    import soundfile as sf
    for row in rows:
        info=sf.info(row['audio']);length=info.frames;observed=set()
        for start in crop_policy['starts']:
            observed.update(range(start,min(length,start+crop_policy['samples'])))
        records.append({'id':row['id'],'origin_id':row['origin_id'],'recording_seconds':length/info.samplerate,
            'crop_count':len(crop_policy['starts']),'model_input_seconds_per_expert':crop_policy['samples']*len(crop_policy['starts'])/info.samplerate,
            'unique_audio_seconds_observed':len(observed)/info.samplerate,'coverage_fraction':len(observed)/length,
            'repeated_short_input':length<crop_policy['samples'],'experts':expert_ids})
    return records
