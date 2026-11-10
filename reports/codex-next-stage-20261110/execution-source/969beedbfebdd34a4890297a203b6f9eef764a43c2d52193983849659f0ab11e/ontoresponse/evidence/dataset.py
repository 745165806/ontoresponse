from dataclasses import dataclass
import numpy as np
import torch
from ..common import read_jsonl, unique, digest
from ..data.prepare import validate_splits
from .cache import Cache

@dataclass
class EvidenceBatch:
    ids: list
    hidden: list
    scores: torch.Tensor
    valid: torch.Tensor
    quality: torch.Tensor
    labels: torch.Tensor
    origins: list
    groups: list
    audit: list
    identities: dict
    def inputs(self): return self.hidden,self.scores,self.valid

def load_batch(cfg,index,manifest,role,expert_dims=None):
    # Filter the audit ledger before touching features; test cannot affect fitting.
    ledger=read_jsonl(manifest)
    if role == 'inference':
        unique(ledger);rows=ledger
        if any(set(r)!={'id','origin_id','source_group','audio'} for r in rows):raise ValueError('Unexpected inference fields')
    else:
        validate_splits(ledger)
        rows=[r for r in ledger if r['role']==role]
    if not rows: raise ValueError('No requests for '+role)
    records=read_jsonl(index);unique(records)
    lookup={r['id']:r for r in records}
    cache=Cache(cfg['cache']);experts=[e['id'] for e in cfg['experts']]
    dims={}
    # Derive dimensions from actual cached observations, never random heads or guessed hidden.
    for r in rows:
        if r['id'] not in lookup: raise ValueError('Missing request ID in extraction index')
        for name,entry in lookup[r['id']]['experts'].items():
            if name not in experts:continue
            if entry['status']!='complete':continue
            for v in entry['measurements']:
                if v['status']=='valid':
                    arrays,meta=cache.get(v['cache_key'])
                    d=len(arrays['embedding'])
                    if name in dims and dims[name]!=d:raise ValueError('Expert dimension drift')
                    dims[name]=d
    if expert_dims is not None:
        if len(expert_dims)!=len(experts):raise ValueError('Known expert dimensions mismatch')
        for name,d in zip(experts,expert_dims):
            if name in dims and dims[name]!=d:raise ValueError('Checkpoint embedding dimension drift')
            dims[name]=d
    if set(dims)!=set(experts): raise ValueError('BLOCKED: no real observations for experts '+str(set(experts)-set(dims)))
    from ..ontology.registry import OntologyRegistry
    reg=OntologyRegistry();count=len(rows);views=len(reg.probe_ids)+1
    hidden=[torch.full((count,views,dims[e]),float('nan')) for e in experts]
    scores=torch.full((count,len(experts),views),float('nan'));valid=torch.zeros_like(scores,dtype=torch.bool)
    quality=torch.zeros(count,4)
    identities={};actual_identities={};probe_order=None
    for b,r in enumerate(rows):
        rec=lookup[r['id']]
        if 'quality' in rec: quality[b]=torch.tensor(rec['quality'])
        order=[v['probe_id'] for v in rec.get('views',[])]
        if order:
            if probe_order is not None and order!=probe_order:raise ValueError('Probe order drift')
            if order!=['original',*reg.probe_ids]:raise ValueError('Cache uses a different probe registry')
            probe_order=order
        for i,e in enumerate(experts):
            entry=rec['experts'].get(e,{})
            if entry.get('status')!='complete': continue
            ident=digest(entry['identity'])
            if e in identities and identities[e]!=ident:raise ValueError('Mixed checkpoint/crop/score identity')
            identities[e]=ident;actual_identities[e]=entry['identity']
            for v,measurement in enumerate(entry['measurements']):
                if measurement['status']!='valid':continue
                arrays,meta=cache.get(measurement['cache_key'])
                if meta['spec']['artifact_key']!=rec['views'][v]['artifact_key']:raise ValueError('Cross-expert artifact mismatch')
                if meta['spec']['expert']!=entry['identity']:raise ValueError('Measurement provenance mismatch')
                hidden[i][b,v]=torch.from_numpy(arrays['embedding'])
                scores[b,i,v]=float(arrays['margin'][0]);valid[b,i,v]=True
    return EvidenceBatch([r['id'] for r in rows],hidden,scores,valid,quality,
                         torch.tensor([r.get('label',float('nan')) for r in rows],dtype=torch.float32),
                         [r['origin_id'] for r in rows],[r['source_group'] for r in rows],rows,actual_identities)

def origin_weights(origins):
    from collections import Counter
    counts=Counter(origins)
    return torch.tensor([1/counts[o] for o in origins],dtype=torch.float32)
