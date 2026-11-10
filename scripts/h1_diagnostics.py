"""Post-selection validation diagnostics only; no extra fitting or test tuning."""
from pathlib import Path
import numpy as np,yaml,json
from ontoresponse.evidence.dataset import load_batch
from ontoresponse.evaluation.metrics import metrics
from ontoresponse.common import atomic_json
cfg=yaml.safe_load(Path('configs/pilot-20261110.yaml').read_text());root=Path(cfg['output'])
batch=load_batch(cfg,root/'extraction-0-of-1.jsonl',root/'audit_manifest.jsonl','fusion_val')
experts=[];errors=[]
for i,e in enumerate(cfg['experts']):
    valid=batch.valid[:,i,0].numpy();s=batch.scores[:,i,0].numpy();y=batch.labels.numpy()
    predicted=s>=0;errors.append(predicted!=y)
    experts.append({'expert':e['id'],'val_metrics_raw_margin':metrics(y[valid],s[valid]),
        'native_margin_zero_threshold_errors':int((predicted[valid]!=y[valid]).sum()),'threshold_source':'fixed native class mapping, not tuned on test'})
atomic_json('reports/codex-next-stage-20261110/H1-diagnostics.json',{'role':'fusion_val','experts':experts,
    'original_margin_Pearson':float(np.corrcoef(batch.scores[:,:,0].numpy().T)[0,1]),
    'both_experts_wrong':int(np.logical_and(*errors).sum()),'either_expert_wrong':int(np.logical_or(*errors).sum()),
    'H1_gate':json.loads((root/'H1-gate.json').read_text()),
    'interpretation':'Validation EER is at ceiling already. This pilot cannot establish incremental observation benefit; it does not establish that response fusion is useless.',
    'no_hyperparameter_or_threshold_change':True,'cross_expert_raw_embedding_comparison':False})
print('validation-only expert complementarity recorded')
