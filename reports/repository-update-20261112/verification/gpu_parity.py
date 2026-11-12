"""Read-only parity against consumed frozen evidence; no fitting or new test selection."""
import gc
import json
from pathlib import Path

import torch
import yaml

from ontoresponse.common import atomic_json, code_hash, sha256
from ontoresponse.data.audio import crops, read_audio
from ontoresponse.evidence.dataset import load_batch
from ontoresponse.experts.base import FrozenExpert
from ontoresponse.train.fit import forward, restore
import ontoresponse
assert Path(ontoresponse.__file__).is_relative_to('${TEMP_ROOT}')

root = Path('${PROJECT_ROOT}')
outdir = root / 'reports/repository-update-20261112'
torch.set_num_threads(2)
torch.manual_seed(17)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert torch.cuda.is_available(), 'GPU validation must not be counted as a CPU pass'
device = torch.device('cuda:0')

cfg = yaml.safe_load((root/'configs/pilot-20261110.yaml').read_text())
checkpoint, model, stats, reg = restore(root/'reports/pilot-20261110/runs/original_mlp-seed17/model.pt')
batch = load_batch(cfg, root/'reports/pilot-20261110/extraction-0-of-1.jsonl',
                   root/'reports/pilot-20261110/audit_manifest.jsonl', 'fusion_val', checkpoint['dims'])
before = torch.load(root/'reports/cleanup-20261112/fusion-before.pt', weights_only=False)
assert before['ids'] == batch.ids
with torch.inference_mode():
    after = forward(model, batch, stats, reg)
    torch.testing.assert_close(before['logits'], after.logits, atol=0, rtol=0)
    assert torch.equal(before['observed'], after.observed)
    model.to(device)
    batch.hidden = [x.to(device) for x in batch.hidden]
    batch.scores = batch.scores.to(device)
    batch.valid = batch.valid.to(device)
    batch.quality = batch.quality.to(device)
    reg.features = reg.features.to(device)
    gpu = forward(model, batch, stats, reg)
    torch.testing.assert_close(before['logits'], gpu.logits.cpu(), atol=1e-4, rtol=1e-4)
fusion = dict(requests=len(batch.ids), CPU_before_after_exact=True,
              GPU_max_logit_error=float((before['logits']-gpu.logits.cpu()).abs().max()),
              checkpoint_sha256=sha256(root/'reports/pilot-20261110/runs/original_mlp-seed17/model.pt'))
print('fusion checkpoint parity passed', fusion, flush=True)
del model, batch, stats, reg
gc.collect()
torch.cuda.empty_cache()

cfg = yaml.safe_load((root/'configs/local.yaml').read_text())
audio_root = Path('${DATA_ROOT}/asvspoof2019/LA/ASVspoof2019_LA_eval/flac')
experts = []
for spec in cfg['experts']:
    source = root/'reports/expert-benchmark-20261111'/spec['id']/'per_utterance_predictions.jsonl'
    rows = []
    with source.open() as stream:
        for _ in range(8):
            rows.append(json.loads(next(stream)))
    inputs = torch.cat([crops(read_audio(audio_root/(r['id']+'.flac')), cfg['crop']) for r in rows])
    expert = FrozenExpert(spec, cfg['crop'], device)
    with torch.inference_mode():
        hidden, actual = expert._run(inputs.to(device))
        expected = torch.tensor([r['raw_outputs'] for r in rows], device=device)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    result = dict(expert=spec['id'], requests=len(rows), status='passed',
                  max_output_error=float((actual-expected).abs().max()),
                  checkpoint_sha256=expert.identity['checkpoint_sha256'])
    experts.append(result)
    print('real expert GPU parity', result, flush=True)
    expert.close()
    del expert, inputs, hidden, actual, expected
    gc.collect()
    torch.cuda.empty_cache()

atomic_json(outdir/'gpu-parity.json', dict(status='passed', level='real_checkpoint_consumed_real_data_GPU',
    source_code_sha256=code_hash(), gpu=torch.cuda.get_device_name(), physical_gpu=0,
    fusion=fusion, experts=experts, atol=1e-4, rtol=1e-4, new_training=False, new_confirmation=False))
