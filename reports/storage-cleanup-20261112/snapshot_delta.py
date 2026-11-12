"""Content-preserving private snapshot deduplication; no extractall or silent overwrite.

The restored snapshot has the original file bytes and BUNDLE_MANIFEST. Its gzip
container hash is not claimed to match the retired original container.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def sha(path):
    result=hashlib.sha256()
    with open(path,'rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):result.update(part)
    return result.hexdigest()


def valid_name(name):
    p=Path(name)
    if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe snapshot member')


def add_bytes(archive,name,data):
    valid_name(name)
    info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644
    archive.addfile(info,io.BytesIO(data))


def create(base_path,old_path,output):
    if Path(output).exists():raise FileExistsError(output)
    with tarfile.open(base_path,'r:gz') as base,tarfile.open(old_path,'r:gz') as old:
        base_index=json.load(base.extractfile('BUNDLE_MANIFEST.json'))
        original_manifest=old.extractfile('BUNDLE_MANIFEST.json').read()
        original=json.loads(original_manifest)
        lookup={r['path']:r for r in base_index['files']}
        rows=[]
        for row in original['files']:
            valid_name(row['path'])
            same=lookup.get(row['path'])
            reuse=same is not None and same['sha256']==row['sha256'] and same['bytes']==row['bytes']
            rows.append(dict(row,location='base' if reuse else 'delta'))
        index={'schema':'ontoresponse-snapshot-delta-1','base_name':Path(base_path).name,
               'base_sha256':sha(base_path),'retired_name':Path(old_path).name,
               'retired_sha256':sha(old_path),'retired_bytes':Path(old_path).stat().st_size,
               'files':rows,'original_archive_byte_identity_preserved':False,
               'original_file_content_identity_preserved':True}
        with tarfile.open(output,'w:gz') as delta:
            for row in rows:
                if row['location']!='delta':continue
                data=old.extractfile(row['path']).read()
                if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:
                    raise ValueError('Original member hash mismatch')
                add_bytes(delta,'payload/'+row['path'],data)
            add_bytes(delta,'BUNDLE_MANIFEST.json',original_manifest)
            add_bytes(delta,'DELTA_MANIFEST.json',json.dumps(index,indent=2).encode())
    return index


def restore(base_path,delta_path,output):
    if Path(output).exists():raise FileExistsError(output)
    with tarfile.open(delta_path,'r:gz') as delta,tarfile.open(base_path,'r:gz') as base:
        index=json.load(delta.extractfile('DELTA_MANIFEST.json'))
        if sha(base_path)!=index['base_sha256']:raise ValueError('Retained base archive hash drift')
        with tarfile.open(output,'w:gz',compresslevel=3) as restored:
            for row in index['files']:
                valid_name(row['path'])
                source=base if row['location']=='base' else delta
                name=row['path'] if row['location']=='base' else 'payload/'+row['path']
                data=source.extractfile(name).read()
                if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:
                    raise ValueError('Reconstruction member hash mismatch: '+row['path'])
                add_bytes(restored,row['path'],data)
            add_bytes(restored,'BUNDLE_MANIFEST.json',delta.extractfile('BUNDLE_MANIFEST.json').read())
    return {'status':'restored_original_file_contents','members':len(index['files']),
            'sha256':sha(output),'byte_identity_to_original_archive_not_claimed':True}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['create','restore'])
    p.add_argument('--base',required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    result=create(a.base,a.source,a.output) if a.command=='create' else restore(a.base,a.source,a.output)
    print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))
