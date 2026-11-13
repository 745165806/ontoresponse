"""Consumed real-data GPU checks; 16 fusion origins, 2 steps, no new confirmation."""
import gc
import json
from pathlib import Path

import torch
import yaml

from ontoresponse.common import atomic_json, read_jsonl, write_jsonl
from ontoresponse.data.audio import crops, read_audio
from ontoresponse.evidence.cache import Cache
from ontoresponse.evidence.dataset import load_batch
from ontoresponse.evidence.extract import extract
from ontoresponse.experts.base import FrozenExpert
from ontoresponse.ontology.registry import validate_evidence
from ontoresponse.train.fit import fit, forward, restore

root = Path.cwd()
out = root / 'reports/simplify-experiments-20261113'
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
assert torch.cuda.is_available(), 'Real CUDA is required'
device = torch.device('cuda:0')
result = {'level':'real_checkpoint_real_data_GPU', 'gpu':torch.cuda.get_device_name(),
          'physical_gpu':0, 'new_confirmation':False, 'full_training':False,
          'unit_or_synthetic':False}

cfg = yaml.safe_load((root/'configs/pilot-20261110.yaml').read_text())
old = root/'reports/pilot-20261110/runs/original_mlp-seed17/model.pt'
cp, model, stats, reg = restore(old)
ledger = root/'reports/pilot-20261110/audit_manifest.jsonl'
index = root/'reports/pilot-20261110/extraction-0-of-1.jsonl'
batch = load_batch(cfg, index, ledger, 'fusion_val', cp['dims'])
baseline = torch.load(root/'reports/cleanup-20261112/fusion-before.pt', weights_only=False)
assert baseline['ids'] == batch.ids
with torch.inference_mode():
    cpu = forward(model, batch, stats, reg)
    torch.testing.assert_close(cpu.logits, baseline['logits'], rtol=0, atol=0)
    model.to(device)
    gpu = forward(model, batch, stats, reg)
    torch.testing.assert_close(gpu.logits.cpu(), baseline['logits'], rtol=1e-4, atol=1e-4)
result['old_fusion'] = {'requests':len(batch.ids), 'CPU_exact':True,
                       'GPU_max_error':float((gpu.logits.cpu()-baseline['logits']).abs().max())}
del model, batch, cpu, gpu

selected = []
rows = read_jsonl(ledger)
for role in ('fusion_train','fusion_val'):
    for label in (0,1):
        part = [r for r in rows if r['role']==role and r['label']==label][:4]
        assert len(part)==4
        selected.extend(part)
small_ledger = out/'training-input.jsonl'
write_jsonl(small_ledger, selected)
cfg['output'] = str(out/'fusion-smoke')
cfg['runtime'].update(device=str(device), budget='smoke')
trained = fit(cfg, index, small_ledger, 'typed_response', 17, 2, width=16)
assert trained['device']=='cuda:0' and trained['optimizer_steps']==2
saved = torch.load(trained['checkpoint'], map_location='cpu', weights_only=False)
assert not any('sha256' in key or key=='registry_hash' for key in saved)
result['fusion_smoke'] = trained

cfg = yaml.safe_load((root/'configs/local.yaml').read_text())
recipe = yaml.safe_load((root/'configs/expert-benchmark-20261111.yaml').read_text())
result['experts'] = []
for spec in cfg['experts']:
    with (root/'reports/expert-benchmark-20261111'/spec['id']/'per_utterance_predictions.jsonl').open() as stream:
        consumed = [json.loads(next(stream)) for _ in range(8)]
    waves = [read_audio(Path(recipe['audio_root'])/(r['id']+'.flac')) for r in consumed]
    expert = FrozenExpert(spec, cfg['crop'], device)
    assert 'checkpoint_sha256' not in expert.identity
    x = torch.cat([crops(w, cfg['crop']) for w in waves]).to(device)
    with torch.inference_mode():
        _, logits = expert._run(x)
    expected = torch.tensor([r['raw_outputs'] for r in consumed])
    torch.testing.assert_close(logits.cpu(), expected, rtol=1e-4, atol=1e-4)
    result['experts'].append({'expert':spec['id'], 'requests':8,
                              'max_output_error':float((logits.cpu()-expected).abs().max()),
                              'status':'passed'})
    expert.close()
    del expert, x, logits
    gc.collect()
    torch.cuda.empty_cache()
    print('expert passed:', spec['id'], flush=True)

# Run the new cache/observation path on one already consumed eval recording.
ident = consumed[0]['id']
cfg['output'] = str(out/'extract-smoke')
cfg['cache'] = str(root/'artifacts/simplify-experiments-20261113')
cfg['runtime'].update(device=str(device), budget='smoke')
manifest = out/'extract-input.jsonl'
write_jsonl(manifest, [{'id':ident,'origin_id':ident,'source_group':'consumed-'+ident,
                        'audio':str(Path(recipe['audio_root'])/(ident+'.flac'))}])
extracted = extract(cfg, manifest, max_origins=1)
assert extracted['scored']==1 and extracted['complete_all_experts']==1
records = read_jsonl(extracted['index'])
validate_evidence(cfg, records, Cache(cfg['cache']))
assert 'code_sha256' not in records[0]
result['extract_smoke'] = extracted
result['GPU_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
result['status']='passed'
atomic_json(out/'gpu-results.json', result)
print(json.dumps(result, indent=2), flush=True)
