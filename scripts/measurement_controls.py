from pathlib import Path
import json,gc
import numpy as np
import torch,yaml
from ontoresponse.common import read_jsonl,atomic_json
from ontoresponse.transforms.probes import apply_probe,codec_roundtrip
from ontoresponse.data.audio import read_audio,crops
from ontoresponse.experts.base import FrozenExpert,AudioInput

ROOT=Path('reports/codex-next-stage-20261110')

def main():
    ROOT.mkdir(exist_ok=True);cfg=yaml.safe_load(Path('configs/pilot-20261110.yaml').read_text());torch.set_num_threads(4)
    requests=read_jsonl('reports/pilot-20261110/inference_manifest.jsonl')[:2];audio=[read_audio(r['audio']) for r in requests];results=[]
    for spec in cfg['experts']:
        model=FrozenExpert(spec,cfg['crop']);one=model.predict_utterances([AudioInput('one',audio[0])]);two=model.predict_utterances([AudioInput('label=spoof,filename_marker',audio[0])])
        torch.testing.assert_close(one.embeddings,two.embeddings,atol=0,rtol=0);torch.testing.assert_close(one.spoof_margins,two.spoof_margins,atol=0,rtol=0)
        x=torch.cat([crops(a,cfg['crop']) for a in audio])
        together=model._run(x);separate=[model._run(x[i:i+1]) for i in range(len(x))]
        hidden=torch.cat([r[0] for r in separate]);logits=torch.cat([r[1] for r in separate])
        torch.testing.assert_close(together[0],hidden,atol=1e-4,rtol=1e-4);torch.testing.assert_close(together[1],logits,atol=1e-4,rtol=1e-4)
        model.model.to('cpu').eval();again=model.predict_utterances([AudioInput('repeat',audio[0])]);torch.testing.assert_close(one.embeddings,again.embeddings,atol=0,rtol=0)
        results.append({'expert':spec['id'],'status':'passed','max_batch_hidden_abs_error':float((together[0]-hidden).abs().max()),
             'max_batch_logit_abs_error':float((together[1]-logits).abs().max()),'atol':1e-4,'rtol':1e-4,'label_or_filename_marker_invariant':True,
             'repeat_exact':True,'CPU_to_CPU_eval_preserved':not any(m.training for m in model.model.modules()),'GPU_migration':'BLOCKED: no device nodes'})
        model.close();del model;gc.collect();print(spec['id'],'batch/repeat/ID controls passed',flush=True)
    codec=[]
    for n in (8000,32017,72021,98765):
        t=np.arange(n)/16000;wave=(.15*np.sin(2*np.pi*440*t)+.03*np.sin(2*np.pi*2700*t)).astype('float32')
        noise_a,ma=apply_probe(wave,{'operation':'additive_white_noise','snr_db':20});noise_b,mb=apply_probe(wave,{'operation':'additive_white_noise','snr_db':20})
        assert np.array_equal(noise_a,noise_b) and ma['seed']==mb['seed']
        for encoder,rate in [('libmp3lame',32),('libmp3lame',64),('aac',64),('libopus',32)]:
            try:
                y,meta=codec_roundtrip(wave,encoder,rate);assert len(y)==n and np.isfinite(y).all();codec.append({'samples':n,'encoder':encoder,'bitrate':rate,'status':'passed','metadata':meta})
            except Exception as e:codec.append({'samples':n,'encoder':encoder,'bitrate':rate,'status':'BLOCKED','error':str(e)})
    atomic_json(ROOT/'measurement-controls.json',{'real_checkpoint':results,'codec_signal_controls':codec,'synthetic_controls_not_detection_scores':True})
    print('codec statuses',[r['status'] for r in codec])
if __name__=='__main__':main()
