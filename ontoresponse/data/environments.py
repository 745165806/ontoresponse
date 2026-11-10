"""Construct controlled external current inputs; keep historical truth in audit only."""
from pathlib import Path
import copy
import soundfile as sf
from ..common import read_jsonl,write_jsonl,atomic_json,digest
from .audio import read_audio
from ..transforms.probes import codec_roundtrip
from ..transforms.environment import exposure_audit

def make_external_conditions(cfg):
    root=Path(cfg['output'])/'external';originals=read_jsonl(root/'audit_manifest.jsonl')
    traces=[('no_new_processing',None),('aac',('aac',64)),('opus',('libopus',32))]
    audit=[];inference=[]
    for r in originals:
        for condition,codec in traces:
            item=copy.deepcopy(r);ident='request-'+digest([r['id'],condition])[:24]
            item.update(id=ident,condition=condition,historical_processing=[] if codec is None else [codec[0]])
            destination=root/'current_audio'/(ident+'.wav');destination.parent.mkdir(parents=True,exist_ok=True)
            try:
                x=read_audio(r['audio'])
                if codec is None:y=x;processing={'operation':'identity'}
                else:y,processing=codec_roundtrip(x,*codec)
                sf.write(destination,y,16000,subtype='FLOAT')
                item.update(audio=str(destination),generation_status='valid',processing=processing)
            except Exception as err:item.update(audio=str(destination),generation_status='failed',generation_error=str(err))
            audit.append(item);inference.append({k:item[k] for k in ('id','origin_id','source_group','audio')})
    write_jsonl(root/'conditions-audit.jsonl',audit);write_jsonl(root/'conditions-inference.jsonl',inference)
    report={'origins':len(originals),'requests':len(audit),'generated':sum(r['generation_status']=='valid' for r in audit),
            'exposure':exposure_audit([[]],['lowpass','additive_white_noise','mp3'],[['aac'],['opus']],None),
            'scope':'fusion-unseen codec operators; expert exposure unknown; engineering smoke only'}
    atomic_json(root/'conditions.json',report);return report
