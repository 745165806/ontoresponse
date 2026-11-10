"""Portable, sanitized audit ZIP. No audio, weights, array caches or credentials."""
from pathlib import Path
from collections import defaultdict
import hashlib,json,re,zipfile
from .common import sha256,read_jsonl,atomic_json,digest

TEXT={'.py','.yaml','.yml','.ttl','.md','.toml','.txt','.log','.xml','.json','.jsonl','.csv','.ini'}
FORBIDDEN={'.wav','.flac','.mp3','.m4a','.ogg','.pt','.pth','.npz','.npy','.whl','.gz','.zip','.bundle'}
PERSONAL=re.compile(r'/(?:home|media)/[A-Za-z0-9_.-]+(?:/[^\s\"\'<>;,)}\]]*)?')
CREDENTIAL=re.compile(r'ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16}|-----BEGIN [^-]*PRIVATE KEY-----')


def pseudo(value,kind):return ('group' if kind=='group' else 'record')+'-'+hashlib.sha256(('OntoResponse-audit-v1:'+str(value)).encode()).hexdigest()[:20]


def export_audit(root,destination):
    root=Path(root).resolve();destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():raise ValueError('Do not overwrite an exported audit ZIP')
    ids={};prediction_tables=[];calibrators={};models={};missing=[]
    # Explicit exported per-request schema; no model or audio is needed to recompute.
    for path in sorted((root/'reports').glob('**/per_utterance_predictions.jsonl')):
        if 'inference' in path.parts:continue
        run=path.parent.parent if path.parent.name in ('external-evaluation','evaluation') else path.parent
        freeze=run/'freeze.json';model_path=run/'model.pt'
        lock=json.loads(freeze.read_text()) if freeze.is_file() else {}
        cal=lock.get('calibration',{});params=cal.get('calibration',cal)
        if not {'scale','bias'}<=set(params):
            missing.append({'path':str(path.relative_to(root)),'reason':'missing frozen Platt parameters; cannot claim complete calibrator lineage'});continue
        cal_id=digest(params);calibrators[cal_id]={'a':params['scale'],'b':params['bias'],'fit_role':'calibration','source':'existing frozen calibration; never refitted during export'}
        model_id=lock.get('checkpoint_sha256',lock.get('model_sha256'))
        if not model_id:missing.append({'path':str(freeze.relative_to(root)),'reason':'missing model hash'});continue
        if model_path.is_file() and sha256(model_path)!=model_id:raise ValueError('Frozen model hash drift')
        models[model_id]={'method':run.name,'artifact_included':False,'reason':'weights excluded by audit export policy','sha256':model_id}
        rows=read_jsonl(path)
        external_path=root/'reports/implementation/external/conditions-audit.jsonl'
        external={r['id']:r for r in read_jsonl(external_path)} if external_path.exists() else {}
        for row in rows:
            for key,kind in [('id','request'),('origin_id','origin'),('source_group','group'),('speaker_group','group')]:
                if row.get(key):ids[str(row[key])]=pseudo(row[key],kind)
            speaker=row.get('speaker_group',external.get(row['id'],{}).get('speaker_id',row['source_group']))
            ids[str(speaker)]=pseudo(speaker,'group')
            ok=row['status']=='scored';space=row.get('threshold_space','calibrated_probability_float32')
            raw=row.get('raw_logit',row.get('logit'));prob=row.get('p_calibrated',row.get('p'));z=row.get('calibrated_logit')
            prediction=(int(z>=row['threshold']) if space.startswith('calibrated_logit') else int(prob>=row['threshold'])) if ok else None
            prediction_tables.append({'id':pseudo(row['id'],'request'),'raw_logit':raw,'p_calibrated':prob,
                'calibrated_logit':z,'threshold':row['threshold'],'threshold_space':space,'prediction':prediction,'label':row.get('label'),
                'origin_id':pseudo(row['origin_id'],'origin'),'source_group':pseudo(row['source_group'],'group'),'speaker_group':pseudo(speaker,'group'),
                'condition':row.get('condition','no_new_processing'),'method':run.name,'seed':int(re.search(r'seed(\d+)',run.name).group(1)),
                'status':row['status'],'calibrator_id':cal_id,'model_id':model_id,'role':'consumed_exploration' if 'implementation' in path.parts else 'confirmation_test'})
    if not prediction_tables:raise ValueError('No actual per-request predictions to export')
    replacements=[(str(root),'${PROJECT_ROOT}'),(str(root.parent),'${EXPERT_ROOT}'),('${DATA_ROOT}','${DATA_ROOT}'),('${USER_HOME}','${USER_HOME}')]
    def sanitize(text):
        for source,target in replacements:text=text.replace(source,target)
        text=PERSONAL.sub('${EXTERNAL_PATH}',text)
        text=CREDENTIAL.sub('<REDACTED_CREDENTIAL>',text)
        for source,target in sorted(ids.items(),key=lambda x:-len(x[0])):
            # Never replace short/numeric IDs inside Python literals, hashes, or counts.
            # Authoritative normalized prediction rows are pseudonymized structurally.
            if len(source)<5 or source.isdigit():continue
            text=re.sub(r'(?<![A-Za-z0-9_-])'+re.escape(source)+r'(?![A-Za-z0-9_-])',target,text)
        return text
    files={};original_hashes={};excluded=[]
    roots=['ontoresponse','reference','scripts','tests','ontology','configs','docs']
    candidates=[p for folder in roots for p in (root/folder).rglob('*') if p.is_file()]
    candidates += [p for p in root.iterdir() if p.is_file() and p.suffix in TEXT]
    for p in (root/'reports').rglob('*'):
        if not p.is_file():continue
        relative=p.relative_to(root)
        if 'training-code' in relative.parts or 'current_audio' in relative.parts or 'code-snapshots' in relative.parts:continue
        if p.suffix in TEXT:candidates.append(p)
        elif p.suffix in FORBIDDEN:excluded.append({'path':str(relative),'reason':'audio/weights/cache/archive excluded; this is not an unavailable delivered member'})
    for p in sorted(set(candidates)):
        relative=p.relative_to(root)
        if '__pycache__' in relative.parts or p.suffix not in TEXT:continue
        try:text=p.read_text()
        except UnicodeDecodeError:missing.append({'path':str(relative),'reason':'not a UTF-8 audit text file'});continue
        # Avoid duplicating the private legacy schema; normalized predictions below are authoritative.
        if p.name=='per_utterance_predictions.jsonl':continue
        name=str(relative);data=sanitize(text).encode();files[name]=data;original_hashes[name]=sha256(p)
    files['audit/predictions.jsonl']=(''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in prediction_tables)).encode()
    files['audit/calibrators.json']=json.dumps(calibrators,indent=2).encode();files['audit/models.json']=json.dumps(models,indent=2).encode()
    public_script='''from collections import defaultdict\nimport json\nfrom pathlib import Path\nfrom scripts.recompute_predictions import independent_metrics\nrows=[json.loads(line) for line in Path('audit/predictions.jsonl').read_text().splitlines()]\ngroups=defaultdict(list)\nfor r in rows:\n r=dict(r,logit=r['raw_logit'],p=r['p_calibrated'])\n groups[(r['method'],r['seed'],r['condition'])].append(r)\nreport={'|'.join(map(str,k)):independent_metrics(v) for k,v in groups.items()}\nPath('metrics_recomputed_public.json').write_text(json.dumps(report,indent=2)+'\\n')\nprint('recomputed',len(report),'condition/method/seed tables; no models or audio read')\n'''
    files['audit_recompute.py']=public_script.encode()
    files['AUDIT_README.txt']=('Extract ZIP then run: python3 audit_recompute.py\nNo audio/model/dependency installation is required for metric recomputation.\n'
        'This is a sanitized source/evidence mirror. Configure your own data/expert paths before running extraction.\n'
        'Tokens ${PROJECT_ROOT}, ${EXPERT_ROOT}, ${DATA_ROOT}, ${USER_HOME}, ${EXTERNAL_PATH} replace private paths.\n'
        'Stable pseudonyms preserve pairing/groups; no claim of irreversible anonymity for public dataset IDs.\n'
        'Original/large task checkpoints, audio, waveform/embedding caches and legacy private tar archives are deliberately not delivered.\n').encode()
    for name,data in files.items():
        if PERSONAL.search(data.decode()) or CREDENTIAL.search(data.decode()):raise ValueError('Privacy scan failed for '+name)
        if Path(name).suffix in FORBIDDEN:raise ValueError('Forbidden archive member '+name)
    manifest={'schema':'ontoresponse-shareable-audit-1','files':[{'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
        'private_source_sha256':original_hashes.get(name)} for name,data in sorted(files.items())],
        'missing':missing,'intentional_exclusions':excluded,'prediction_rows':len(prediction_tables),'weights_and_audio_included':False,
        'source_hash_scope':'SHA256 is of delivered sanitized bytes; optional private_source_sha256 identifies local pre-redaction bytes'}
    files['AUDIT_MANIFEST.json']=json.dumps(manifest,indent=2).encode()
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,data in sorted(files.items()):archive.writestr(name,data)
    result=verify_audit_zip(destination);result.update(zip_sha256=sha256(destination),zip_bytes=destination.stat().st_size)
    return result


def verify_audit_zip(path):
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate ZIP member')
        manifest=json.loads(archive.read('AUDIT_MANIFEST.json'))
        expected={r['path'] for r in manifest['files']}|{'AUDIT_MANIFEST.json'}
        if set(names)!=expected:raise ValueError('ZIP members differ from manifest')
        for row in manifest['files']:
            p=Path(row['path']);data=archive.read(row['path'])
            if p.is_absolute() or '..' in p.parts or p.suffix in FORBIDDEN:raise ValueError('Unsafe/forbidden member')
            if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('ZIP content mismatch')
            if PERSONAL.search(data.decode()) or CREDENTIAL.search(data.decode()):raise ValueError('Private path/credential-like content')
    return {'status':'verified','members':len(names),'missing':manifest['missing'],'prediction_rows':manifest['prediction_rows'],'audio_or_weights_included':False}
