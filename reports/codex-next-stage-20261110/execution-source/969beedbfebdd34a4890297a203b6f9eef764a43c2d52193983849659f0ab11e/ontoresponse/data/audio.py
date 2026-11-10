import numpy as np
import soundfile as sf
import torch

def read_audio(path):
    x,sr = sf.read(path, dtype='float32', always_2d=True)
    if sr != 16000: raise ValueError('Expected 16000Hz; resampling must be explicit and common to all classes')
    x = x.mean(1)
    validate_audio(x)
    return x

def validate_audio(x):
    x=np.asarray(x)
    if x.ndim!=1 or len(x)<160 or not np.isfinite(x).all(): raise ValueError('Invalid or short waveform')
    if np.sqrt(np.mean(x.astype(np.float64)**2))<1e-7: raise ValueError('Silent waveform')
    return x

def crops(x, policy):
    validate_audio(x)
    n=policy['samples']
    if n<160 or policy.get('padding')!='repeat' or policy.get('aggregation')!='mean_hidden_mean_logits':
        raise ValueError('Unsupported crop/aggregation policy')
    out=[]
    for start in policy['starts']:
        if not isinstance(start,int) or start<0 or start>=len(x): raise ValueError('Illegal crop start')
        # Repeat the full aligned waveform, not a separately padded suffix.
        wave=np.tile(x, (start+n+len(x)-1)//len(x))
        out.append(wave[start:start+n])
    if not out: raise ValueError('No crops')
    return torch.from_numpy(np.stack(out).astype('float32'))

def quality(x):
    return [float(np.log1p(len(x)/16000)), float(20*np.log10(max(np.sqrt(np.mean(x.astype('float64')**2)),1e-10)))/100,
            float(np.mean(np.abs(x)>=.999)),float(np.mean(np.abs(x)<1e-4))]
