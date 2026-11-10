"""Apply complete-waveform probes before any expert crops."""
from __future__ import annotations
import hashlib
import io
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import scipy
from scipy.signal import butter, sosfiltfilt
import soundfile as sf
from ..common import Blocked, digest
from ..data.audio import validate_audio

def audio_hash(x):
    return hashlib.sha256(np.asarray(x,dtype='<f4').tobytes()).hexdigest()

def codec_roundtrip(x, encoder, bitrate=None, _timing_control=False):
    formats={'libmp3lame':('mp3','mp3'),'aac':('m4a','ipod'),'libopus':('ogg','ogg')}
    if encoder not in formats: raise Blocked('Unsupported codec '+encoder)
    suffix,fmt=formats[encoder]
    with tempfile.TemporaryDirectory(prefix='ontoresponse-codec-') as tmp:
        raw=Path(tmp)/'input.wav';encoded=Path(tmp)/('encoded.'+suffix);decoded=Path(tmp)/'decoded.wav'
        sf.write(raw,x,16000,subtype='FLOAT')
        args=['ffmpeg','-nostdin','-v','error','-y','-i',str(raw),'-c:a',encoder]
        if bitrate: args+=['-b:a',str(bitrate)+'k']
        args+=['-f',fmt,str(encoded)]
        decode=['ffmpeg','-nostdin','-v','error','-y','-i',str(encoded),'-ar','16000','-ac','1','-c:a','pcm_f32le',str(decoded)]
        for cmd in (args,decode):
            try: p=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
            except OSError as e: raise Blocked('Codec binary missing') from e
            if p.returncode: raise Blocked('Real codec failed: '+p.stderr[-1500:])
        y,sr=sf.read(decoded,dtype='float32')
        if sr!=16000: raise ValueError('Codec changed sample rate')
        # MP3 gapless container metadata makes ffmpeg remove encoder delay.
        # Other codecs with non-exact alignment remain explicitly blocked.
        alignment={'decoded_samples_before_alignment':len(y),'tail_trim_samples':0,'timing_control':None}
        if _timing_control:
            return y,{}
        if len(y)!=len(x):
            # Encode a label-free timing fixture at this exact length/rate/bitrate.
            # Only a verified constant nonnegative delay can be removed; no guessed padding.
            control=np.zeros(len(x),dtype='float32')
            positions=[len(x)//4,len(x)//2,3*len(x)//4]
            for pos,amp in zip(positions,[.3,.5,.7]):control[pos]=amp
            timing,_=codec_roundtrip(control,encoder,bitrate,_timing_control=True)
            lags=[]
            for pos in positions:
                radius=min(2048,max(16,len(x)//16))  # disjoint windows also on short signals
                start=max(0,pos-radius);end=min(len(timing),pos+radius+1)
                lags.append(int(start+np.argmax(np.abs(timing[start:end]))-pos))
            if len(set(lags))!=1 or lags[0]<0 or len(y)-lags[0]<len(x):
                raise Blocked(f'Unverified codec alignment: input={len(x)}, decoded={len(y)}, timing_lags={lags}')
            lag=lags[0];alignment.update(timing_control={'impulse_positions':positions,'measured_lags':lags},tail_trim_samples=len(y)-lag-len(x))
            y=y[lag:lag+len(x)];alignment['manual_shift_samples']=lag
        clean=lambda cmd:[Path(a).name if a.startswith(tmp) else a for a in cmd]
        return y,{'encoder':encoder,'bitrate_kbps':bitrate,'encoded_bytes':encoded.stat().st_size,
                  'encode_command':clean(args),'decode_command':clean(decode),
                  'alignment':'ffmpeg_gapless_metadata_plus_verified_timing_control_if_length_differs','manual_shift_samples':alignment.get('manual_shift_samples',0),**alignment}

def apply_probe(x, spec):
    validate_audio(x)
    op=spec['operation']; meta={'operation':spec,'input_samples':len(x),'input_audio_sha256':audio_hash(x),
                              'sample_rate':16000,'scipy_version':scipy.__version__,'soundfile_version':sf.__version__}
    if op=='identity': y=x.copy()
    elif op=='lossless':
        buf=io.BytesIO();sf.write(buf,x,16000,format='WAV',subtype='FLOAT');buf.seek(0)
        y,sr=sf.read(buf,dtype='float32');meta['serialization']='WAV_FLOAT'
    elif op=='lowpass':
        cutoff=spec['cutoff_hz'];order=spec.get('order',6)
        if not 0<cutoff<8000 or not isinstance(order,int) or not 1<=order<=12: raise ValueError('Illegal lowpass')
        y=sosfiltfilt(butter(order,cutoff,fs=16000,output='sos'),x).astype('float32')
        meta['alignment']='zero_phase_sosfiltfilt'
    elif op=='additive_white_noise':
        snr=float(spec['snr_db'])
        if not 0<=snr<=80: raise ValueError('Illegal SNR')
        seed=int(digest([audio_hash(x),'noise_shared_direction_v1'])[:16],16)
        noise=np.random.Generator(np.random.PCG64(seed)).standard_normal(len(x))
        noise/=np.sqrt(np.mean(noise**2))
        rms=np.sqrt(np.mean(x.astype('float64')**2))
        y=(x+noise*rms*10**(-snr/20)).astype('float32');meta.update(seed=seed,shared_direction='audio_hash_noise_v1')
    elif op=='ffmpeg_roundtrip':
        rate=spec.get('bitrate_kbps')
        if not isinstance(rate,int) or rate<=0: raise ValueError('Illegal codec bitrate')
        y,extra=codec_roundtrip(x,spec['encoder'],rate);meta.update(extra)
    else: raise Blocked('Unimplemented real operation '+op)
    validate_audio(y)
    if len(y)!=len(x): raise ValueError('Probe must preserve aligned timeline')
    meta.update(output_samples=len(y),output_audio_sha256=audio_hash(y),
                clipping_fraction=float(np.mean(np.abs(y)>=.999)),status='valid')
    return y,meta
