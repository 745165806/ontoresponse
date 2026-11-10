# 当前CLI（从实际 --help 生成）

生成日期2026-11-10。TASK_FOR_CODEX的仅reference描述是历史规格。

```text
usage: __main__.py [-h] [--config CONFIG] [--max-origins MAX_ORIGINS]
                   [--source-root SOURCE_ROOT] [--manifest MANIFEST]
                   [--index INDEX] [--output OUTPUT] [--model MODEL]
                   [--steps STEPS] [--seed SEED] [--shard-index SHARD_INDEX]
                   [--shard-count SHARD_COUNT] [--shacl]
                   [--indices INDICES [INDICES ...]] [--audio AUDIO]
                   [--checkpoint CHECKPOINT] [--lock LOCK] [--role ROLE]
                   {doctor,audit,audit-export,prepare-data,prepare-external,make-environments,export-ontology,check-experts,extract,merge,train,ablate,calibrate,freeze,evaluate,infer,report}

OntoResponse bounded real-expert pipeline

positional arguments:
  {doctor,audit,audit-export,prepare-data,prepare-external,make-environments,export-ontology,check-experts,extract,merge,train,ablate,calibrate,freeze,evaluate,infer,report}

options:
  -h, --help            show this help message and exit
  --config CONFIG
  --max-origins MAX_ORIGINS
  --source-root SOURCE_ROOT
  --manifest MANIFEST
  --index INDEX
  --output OUTPUT
  --model MODEL
  --steps STEPS
  --seed SEED
  --shard-index SHARD_INDEX
  --shard-count SHARD_COUNT
  --shacl
  --indices INDICES [INDICES ...]
  --audio AUDIO
  --checkpoint CHECKPOINT
  --lock LOCK
  --role ROLE
```

真实新pilot入口为 `python -m scripts.run_pilot --config configs/pilot-20261110.yaml`；只有显式预算enabled及预检通过才启动。`full_attention`为完整自注意力TransformerEncoder对照，未声称完整Set Transformer。`untyped_union`保留旧union计算，`relation_tied`单独保留四通道归一化。
