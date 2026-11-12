"""Verify wheel imports/resources and existing validation checkpoint outside the repo."""
import json
from pathlib import Path

import torch
import yaml
import ontoresponse
from ontoresponse.common import atomic_json, code_hash, digest, sha256
from ontoresponse.evidence.dataset import load_batch
from ontoresponse.resources import resource_path
from ontoresponse.train.fit import forward, restore

repo = Path('${PROJECT_ROOT}')
installed = Path('${EXTERNAL_PATH}')
assert Path.cwd() == Path('/tmp')
assert Path(ontoresponse.__file__).is_relative_to(installed)
assert resource_path('configs/probes.yaml').is_relative_to(installed)
source_hash = digest({str(p.relative_to(repo)):sha256(p) for folder in ('ontoresponse','reference') for p in sorted((repo/folder).rglob('*.py'))})
assert source_hash == code_hash()
for name in ('configs/probes.yaml','configs/ontology.yaml','ontology/shapes.ttl','ontology/ontology.ttl'):
    assert resource_path(name).read_bytes() == (repo/name).read_bytes()

torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
cfg = yaml.safe_load((repo/'configs/pilot-20261110.yaml').read_text())
cfg['cache'] = str(repo/cfg['cache'])
checkpoint, model, stats, reg = restore(repo/'reports/pilot-20261110/runs/original_mlp-seed17/model.pt')
batch = load_batch(cfg, repo/'reports/pilot-20261110/extraction-0-of-1.jsonl',
    repo/'reports/pilot-20261110/audit_manifest.jsonl','fusion_val',checkpoint['dims'])
expected = torch.load(repo/'reports/cleanup-20261112/fusion-before.pt',weights_only=False)
with torch.inference_mode():
    actual = forward(model,batch,stats,reg)
    torch.testing.assert_close(actual.logits,expected['logits'],atol=0,rtol=0)
result = dict(status='passed', package=ontoresponse.__file__, source_code_sha256=code_hash(),
    installed_source_hash_equal=True, packaged_resources_equal=True, validation_requests=len(batch.ids),
    before_after_logits_exact=True, new_training=False, new_confirmation=False)
atomic_json(repo/'reports/cleanup-20261112/installed-verification.json',result)
print(json.dumps(result,indent=2))
