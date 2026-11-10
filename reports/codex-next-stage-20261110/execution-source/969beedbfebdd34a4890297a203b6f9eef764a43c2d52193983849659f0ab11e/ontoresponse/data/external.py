"""Bounded external requests; unknown source mapping stays unknown."""
import csv
from pathlib import Path
from ..common import Blocked,digest,sha256,atomic_json,write_jsonl,budget

def prepare_inthewild(cfg,max_origins=8):
    budget(cfg['runtime']['budget'],max_origins)
    root=Path(cfg['data']['inthewild']);meta=root/'meta.csv'
    if not meta.is_file():raise Blocked('Missing In-the-Wild metadata')
    rows=[]
    with meta.open() as f:
        for r in csv.DictReader(f):
            if r['label'] not in ('bona-fide','bonafide','spoof'):raise ValueError('Unknown external label')
            rows.append({'id':'itw-'+r['file'],'origin_id':'itw-'+r['file'],'audio':str(root/r['file']),
                         'speaker_id':'itw:'+r['speaker'],'source_group':'itw:'+r['speaker'],
                         'label':int(r['label']=='spoof'),'role':'test','condition':'no_new_processing',
                         'dataset':'inthewild','source_overlap_status':'unknown'})
    selected=[]
    groups={y:sorted([r for r in rows if r['label']==y],key=lambda r:digest(['external-prespecified17',r['id']])) for y in (0,1)}
    for i in range(max_origins):selected.append(groups[i%2][i//2])
    dest=Path(cfg['output'])/'external'
    write_jsonl(dest/'audit_manifest.jsonl',selected)
    write_jsonl(dest/'inference_manifest.jsonl',[{k:r[k] for k in ('id','origin_id','source_group','audio')} for r in selected])
    result={'status':'integration_only','dataset':'inthewild','metadata_sha256':sha256(meta),'n_requested':len(selected),
            'source_mapping_complete':False,'limitation':'metadata version pinned, selected audio hashes pinned during extraction; original download/release version unknown; speaker grouping only'}
    atomic_json(dest/'data-audit.json',result)
    return result

def overlap_report(rows,known_training_sources,source_map):
    counts={'known_overlap':0,'known_disjoint':0,'unknown':0};statuses={}
    for r in rows:
        sources=source_map.get(r['id'])
        status='unknown' if not sources else 'known_overlap' if set(sources)&set(known_training_sources) else 'known_disjoint'
        counts[status]+=1;statuses[r['id']]=status
    return {'counts':counts,'per_request':statuses,'unknown_is_not_independent':True}
