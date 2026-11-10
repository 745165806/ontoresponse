"""One valid environment version per origin per epoch, independent of labels."""
from collections import defaultdict
from ..common import digest,Blocked

def origin_uniform_epoch(origin_ids,version_ids,valid,seed,epoch):
    if not(len(origin_ids)==len(version_ids)==len(valid)):raise ValueError('Sampler lengths differ')
    groups=defaultdict(list)
    for i,(origin,version,ok) in enumerate(zip(origin_ids,version_ids,valid)):
        groups[origin]
        if ok:groups[origin].append((version,i))
    selected=[]
    for origin,rows in sorted(groups.items()):
        if not rows:raise Blocked('No valid environment version for origin '+str(origin))
        # Digest includes stable measurement IDs, never class/attack/domain labels.
        selected.append(min(rows,key=lambda pair:digest([seed,epoch,origin,pair[0]]))[1])
    return selected
