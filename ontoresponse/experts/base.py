"""Real task-trained experts only. No label/audit fields are accepted."""
from __future__ import annotations
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch
import torch
import yaml
from ..common import Blocked, sha256, digest
from ..data.audio import crops

@dataclass(frozen=True)
class AudioInput:
    id: str
    waveform: object

@dataclass
class Prediction:
    ids: list[str]
    embeddings: torch.Tensor
    spoof_margins: torch.Tensor
    valid: torch.Tensor
    metadata: dict

def load_module(path):
    if not path.is_file(): raise Blocked('Missing upstream source '+str(path))
    spec=importlib.util.spec_from_file_location('ontoresponse_upstream_'+sha256(path)[:16], path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

class FrozenExpert:
    def __init__(self, spec, crop_policy, device='cpu'):
        if spec.get('task_trained') is not True: raise Blocked('A task-trained checkpoint is required')
        self.spec, self.crop_policy, self.device = spec, crop_policy, torch.device(device)
        self.patch_info = {}
        repo=Path(spec['repository']); cp=Path(spec['checkpoint'])
        if not cp.is_file(): raise Blocked('Missing task checkpoint '+str(cp))
        if {spec['spoof_index'],spec['bonafide_index']} != {0,1}: raise ValueError('Explicit class mapping required')
        source=repo/spec['module']; kind=spec['kind']
        if kind == 'ssl_aasist':
            if not spec.get('frontend') or not Path(spec['frontend']).is_file(): raise Blocked('Missing SSL frontend')
            import fairseq
            original_merge=fairseq.tasks.merge_with_parent
            removed=[]
            def merge(dc, cfg, remove_missing=False):
                if type(dc).__name__=='AudioPretrainingConfig':
                    from omegaconf import open_dict
                    with open_dict(cfg):
                        for key in sorted(set(cfg.keys())-set(dc.__dataclass_fields__)):
                            removed.append('task.'+key);del cfg[key]
                return original_merge(dc,cfg,remove_missing=remove_missing)
            text=source.read_text(); old="cp_path = 'xlsr2_300m.pt'"
            if text.count(old)!=1: raise Blocked('SSL frontend path patch context changed')
            text=text.replace(old,'cp_path = '+repr(str(Path(spec['frontend']).resolve())))
            namespace={'__file__':str(source),'__name__':'ontoresponse_ssl_upstream'}
            original_load=torch.load
            def trusted_load(*args,**kw):
                kw.setdefault('weights_only',False)
                return original_load(*args,**kw)
            with patch.object(fairseq.tasks,'merge_with_parent',merge), patch.object(torch,'load',trusted_load):
                exec(compile(text,str(source),'exec'),namespace)
                self.model=namespace['Model'](None,self.device)
            self.patch_info={'frontend_path_only':True,'removed_task_fields':removed,
                             'fairseq_version':fairseq.__version__,
                             'eval_device_fix':'move all parameters before eval; reject dtype/device mismatch at inference'}
        else:
            module=load_module(source)
            config=yaml.safe_load((repo/spec['config']).read_text())
            self.model=(module.Model(config['model_config']) if kind=='aasist' else module.RawNet2(config['model']))
        # Trusted, user-owned checkpoints; never load untrusted remote pickle.
        state=torch.load(cp,map_location='cpu',weights_only=False)
        self.training_record={}
        if spec.get('state_key'):
            self.training_record={k:state.get(k) for k in ('epoch','global_step','class_index_map','architecture','fit','source_val','training','model_patch','task_weight_origin')}
            mapping=state.get('class_index_map')
            if mapping and mapping != {'spoof':spec['spoof_index'],'bonafide':spec['bonafide_index']}:
                raise Blocked('Checkpoint class mapping contradicts configured mapping')
            if not state.get('global_step',0): raise Blocked('No evidence of task training in project checkpoint')
            state=state[spec['state_key']]
        self.model.load_state_dict(state,strict=True)
        del state
        self.model.to(device=self.device,dtype=torch.float32).eval().requires_grad_(False)
        self.identity={'id':spec['id'],'checkpoint_sha256':sha256(cp),'source_sha256':sha256(source),
                       'frontend_sha256':sha256(spec['frontend']) if spec.get('frontend') else None,
                       'crop':crop_policy,'spoof_index':spec['spoof_index'],'bonafide_index':spec['bonafide_index'],
                       'score_direction':'higher_is_spoof','patch':self.patch_info}
        self._hidden=None
        self._hook=None
        if kind=='ssl_aasist':
            def capture(module, args): self._hidden=args[0].detach().clone()
            self._hook=self.model.out_layer.register_forward_pre_hook(capture)
        elif kind=='rawnet2':
            def capture(module,args,output): self._hidden=output.detach().clone()
            self._hook=self.model.fc1_gru.register_forward_hook(capture)
        self.kind=kind

    def close(self):
        if self._hook: self._hook.remove();self._hook=None

    @torch.inference_mode()
    def _run(self,x):
        if x.dtype!=torch.float32 or x.device!=self.device: raise ValueError('Expert device/dtype must remain fixed')
        if any(m.training for m in self.model.modules()): raise RuntimeError('Expert unexpectedly in train mode')
        self._hidden=None
        raw=self.model(x,Freq_aug=False) if self.kind=='aasist' else self.model(x)
        if self.kind=='aasist': hidden,logits=raw
        else: hidden,logits=self._hidden,raw
        if hidden is None or logits.shape[-1]!=2: raise RuntimeError('Upstream output contract mismatch')
        if any(m.training for m in self.model.modules()): raise RuntimeError('Upstream inference changed eval mode')
        return hidden,logits

    @torch.inference_mode()
    def predict_utterances(self, audio_batch_without_labels):
        ids=[];hs=[];margins=[]
        for item in audio_batch_without_labels:
            if type(item) is not AudioInput: raise TypeError('Only AudioInput(id, waveform); no labels or audit metadata')
            h,l=self._run(crops(item.waveform,self.crop_policy).to(self.device))
            ids.append(item.id);hs.append(h.mean(0).cpu())
            margins.append((l[:,self.spec['spoof_index']]-l[:,self.spec['bonafide_index']]).mean().cpu())
        if not ids or len(ids)!=len(set(ids)): raise ValueError('Empty/duplicate input IDs')
        h=torch.stack(hs);s=torch.stack(margins)
        if not torch.isfinite(h).all() or not torch.isfinite(s).all(): raise RuntimeError('Nonfinite expert observation')
        return Prediction(ids,h,s,torch.ones(len(ids),dtype=torch.bool),dict(self.identity,hidden_dim=h.shape[-1]))

    @torch.inference_mode()
    def check_upstream(self, item, atol=1e-6, rtol=1e-5):
        x=crops(item.waveform,self.crop_policy).to(self.device)
        h,l=self._run(x)
        h=h.clone();l=l.clone()
        # Remove the observing hook and invoke the actual upstream module directly.
        self.close()
        captured=[]
        if self.kind=='ssl_aasist': handle=self.model.out_layer.register_forward_pre_hook(lambda m,args:captured.append(args[0].detach().clone()))
        elif self.kind=='rawnet2': handle=self.model.fc1_gru.register_forward_hook(lambda m,args,out:captured.append(out.detach().clone()))
        else: handle=None
        direct=self.model(x,Freq_aug=False) if self.kind=='aasist' else self.model(x)
        direct_h,direct_l=direct if self.kind=='aasist' else (captured[0],direct)
        if handle: handle.remove()
        torch.testing.assert_close(l,direct_l,atol=atol,rtol=rtol)
        torch.testing.assert_close(h,direct_h,atol=atol,rtol=rtol)
        if self.kind=='ssl_aasist':
            self._hook=self.model.out_layer.register_forward_pre_hook(lambda m,args:setattr(self,'_hidden',args[0].detach().clone()))
        elif self.kind=='rawnet2':
            self._hook=self.model.fc1_gru.register_forward_hook(lambda m,args,out:setattr(self,'_hidden',out.detach().clone()))
        return {'id':item.id,'status':'passed','hidden_dim':h.shape[-1],
                'max_logit_abs_error':float((l-direct_l).abs().max()),
                'max_hidden_abs_error':float((h-direct_h).abs().max()),'atol':atol,'rtol':rtol,
                'logits':l.cpu().tolist(),'metadata':self.identity,'training_record':self.training_record,
                'limitation':'same loaded upstream implementation with hook removed; not an independent reimplementation'}
