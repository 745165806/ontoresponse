import gc
from pathlib import Path
import torch
from .common import read_jsonl
from .data.audio import read_audio
from .experts.base import FrozenExpert,AudioInput
from .transforms.probes import apply_probe

def check_experts(cfg,manifest):
    rows=read_jsonl(manifest)[:2]
    results=[]
    for spec in cfg['experts']:
        expert=None
        try:
            expert=FrozenExpert(spec,cfg['crop'],cfg['runtime']['device'])
            controls=[]
            for r in rows:
                x=read_audio(r['audio'])
                original=expert.predict_utterances([AudioInput(r['id'],x)])
                for op in ('identity','lossless'):
                    y,meta=apply_probe(x,{'operation':op})
                    after=expert.predict_utterances([AudioInput(r['id'],y)])
                    torch.testing.assert_close(original.embeddings,after.embeddings,atol=1e-6,rtol=1e-5)
                    torch.testing.assert_close(original.spoof_margins,after.spoof_margins,atol=1e-6,rtol=1e-5)
                    controls.append({'id':r['id'],'operation':op,'max_hidden_delta':float((original.embeddings-after.embeddings).abs().max()),'max_margin_delta':float((original.spoof_margins-after.spoof_margins).abs().max())})
            consistency=expert.check_upstream(AudioInput(rows[0]['id'],read_audio(rows[0]['audio'])))
            results.append({'expert':spec['id'],'status':'passed','upstream':consistency,'controls':controls})
        except Exception as err:
            results.append({'expert':spec['id'],'status':'BLOCKED','error':str(err)})
        finally:
            if expert:expert.close()
            del expert;gc.collect()
        print('checked '+spec['id']+': '+results[-1]['status'],flush=True)
    return {'status':'passed' if all(r['status']=='passed' for r in results) else 'BLOCKED',
            'level':'real_checkpoint_and_real_audio_CPU' if cfg['runtime']['device']=='cpu' else 'real_checkpoint_and_real_audio_GPU',
            'results':results}
