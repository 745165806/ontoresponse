from collections import defaultdict
import json
from pathlib import Path
from scripts.recompute_predictions import independent_metrics
rows=[json.loads(line) for line in Path('audit/predictions.jsonl').read_text().splitlines()]
groups=defaultdict(list)
for r in rows:
 r=dict(r,logit=r['raw_logit'],p=r['p_calibrated'])
 groups[(r['method'],r['seed'],r['condition'])].append(r)
report={'|'.join(map(str,k)):independent_metrics(v) for k,v in groups.items()}
Path('metrics_recomputed_public.json').write_text(json.dumps(report,indent=2)+'\n')
print('recomputed',len(report),'condition/method/seed tables; no models or audio read')
