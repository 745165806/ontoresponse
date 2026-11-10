"""Compare conventions on labelled numerical fixtures; never change real scores."""
from pathlib import Path
import importlib.util,json
import numpy as np
from scripts.recompute_predictions import independent_metrics,sigmoid
from ontoresponse.common import sha256,atomic_json

path=Path('${EXPERT_ROOT}/SSL_Anti-spoofing/eval_metrics_DF.py')
spec=importlib.util.spec_from_file_location('upstream_metric',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
rows=[]
for name,scores in [('perfect',[-2,-1,1,2]),('reverse',[2,1,-1,-2]),('constant',[0,0,0,0]),('tied',[0,1,1,2])]:
    y=np.array([0,0,1,1]);s=np.array(scores,dtype=float)
    table=[{'status':'scored','label':int(a),'logit':float(b),'p':sigmoid(float(b)),'threshold':.5} for a,b in zip(y,s)]
    ours=independent_metrics(table)
    # Fixed semantic conversion: official higher means bonafide.
    official=module.compute_eer(-s[y==0],-s[y==1])[0]
    rows.append({'fixture':name,'grouped_interpolated_EER':ours['eer'],'upstream_nearest_DET_EER':float(official)})
atomic_json('reports/codex-next-stage-20261110/metric-conventions.json',{'fixtures':rows,'level':'unit/synthetic only',
 'official_source':'https://github.com/asvspoof-challenge/2021/blob/main/eval-package/eval_metrics.py',
 'local_vendored_source_sha256':sha256(path),'local_vendored_copy':'SSL_Anti-spoofing eval_metrics_DF.py',
 'difference':'Own report groups ties and linearly crosses ROC; upstream chooses nearest DET point and stable per-trial tie ordering. Constant-score result can differ.',
 'official_full_ASV2021_evaluation':'NOT_RUN; source/coverage/budget gates still apply','no_score_direction_search':True})
print(rows)
