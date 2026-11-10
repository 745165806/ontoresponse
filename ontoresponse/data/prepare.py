"""Audit metadata never enters the expert tensor interface."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
from ..common import Blocked, unique, write_jsonl, digest, atomic_json, budget

ROLES = {'expert_train','expert_val','fusion_train','fusion_val','calibration','threshold','test'}

def validate_splits(rows):
    unique(rows)
    ownership = {}
    for r in rows:
        if r['role'] not in ROLES: raise ValueError('Unknown split role')
        if r.get('label') not in (0,1): raise ValueError('Unknown binary label')
        # Union constraints are checked by shared ownership, independent of row order.
        for key in ('origin_id','source_group','speaker_id'):
            if not r.get(key): raise ValueError('Missing grouping '+key)
            marker = (key, r[key])
            if marker in ownership and ownership[marker] != r['role']:
                raise ValueError('Cross-role leakage: '+str(marker))
            ownership[marker] = r['role']
    # Explicit source maps can connect different speakers via source_group.
    return {role: {'n': len(part), 'labels': dict(Counter(r['label'] for r in part)),
                   'speakers': len({r['speaker_id'] for r in part}),
                   'groups': len({r['source_group'] for r in part})}
            for role in sorted({r['role'] for r in rows})
            for part in [[r for r in rows if r['role'] == role]]}

def prepare(cfg, max_origins=24):
    budget(cfg['runtime']['budget'], max_origins)
    if cfg['mode'] != 'integration_pretrained':
        raise Blocked('research_holdout needs audited source mapping and experts trained on disjoint groups')
    root = Path(cfg['data']['asv2019'])
    protocol = root/'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
    if not protocol.is_file(): raise Blocked('Missing ASV2019 dev protocol')
    records = []
    for line in protocol.read_text().splitlines():
        speaker, ident, _, attack, label = line.split()
        records.append({'id':ident,'origin_id':ident,'speaker_id':speaker,'source_group':speaker,
                        'label':int(label=='spoof'),'attack':attack,
                        'audio': str(root/f'ASVspoof2019_LA_dev/flac/{ident}.flac')})
    speakers = sorted({r['speaker_id'] for r in records},key=lambda x:hashlib.sha256(('split17:'+x).encode()).hexdigest())
    n = len(speakers)
    if n < 6: raise Blocked('Too few speakers for three independent roles')
    a,b = max(2,n//2), max(2,n//4)
    roles = {s: 'fusion_train' if i<a else 'fusion_val' if i<a+b else 'calibration' for i,s in enumerate(speakers)}
    out = []
    # Bounded, balanced engineering subset, fixed before any expert outputs.
    buckets = defaultdict(list)
    for r in records:
        r['role'] = roles[r['speaker_id']]
        buckets[(r['role'],r['label'])].append(r)
    for group in buckets.values(): group.sort(key=lambda r: digest(r['id']))
    order = [(r,y) for r in ('fusion_train','fusion_val','calibration') for y in (0,1)]
    for i in range(max_origins):
        key = order[i%len(order)]
        if not buckets[key]: raise Blocked('Insufficient class coverage')
        out.append(buckets[key].pop(0))
    summary = validate_splits(out)
    if any(set(v['labels']) != {0,1} or v['groups']<2 for v in summary.values()):
        raise Blocked('Subset must cover both classes and >=2 groups in each role; increase bounded origins')
    destination = Path(cfg['output'])
    write_jsonl(destination/'audit_manifest.jsonl',out)
    write_jsonl(destination/'inference_manifest.jsonl',[{k:r[k] for k in ('id','origin_id','source_group','audio')} for r in out])
    report = {'status':'integration_only','summary':summary,'protocol_sha256': __import__('ontoresponse.common',fromlist=['sha256']).sha256(protocol),
              'ledger_hash':digest(out),'source_mapping_complete':False,
              'limitation':'speaker grouping only; cross-speaker source mapping unknown; dev overlaps upstream checkpoint selection; no final test consumed',
              'missing_audio':[r['id'] for r in out if not Path(r['audio']).is_file()]}
    atomic_json(destination/'splits.json',report)
    return report

def shard(origin_group, count):
    if count<1: raise ValueError('Invalid shard count')
    return int(hashlib.sha256(str(origin_group).encode()).hexdigest(),16)%count
