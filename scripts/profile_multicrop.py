"""Real-checkpoint measurement controls on already-consumed training audio only."""

import subprocess
import sys
from pathlib import Path

import torch
import yaml

from ontoresponse.common import atomic_json, read_jsonl
from ontoresponse.data.audio import coverage, crops, read_audio
from ontoresponse.experts.base import AudioInput, FrozenExpert
from ontoresponse.transforms.probes import apply_probe


def main():
    cfg = yaml.safe_load(Path("configs/pilot-20261110.yaml").read_text())
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(17)
    root = Path("reports/discrimination-20261110")
    resource = {
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": torch.cuda.device_count(),
        "device": "cpu",
        "threads": 4,
        "deterministic_algorithms": True,
        "seed": 17,
    }
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    resource["nvidia_smi"] = {
        "exit_code": smi.returncode,
        "stdout": smi.stdout,
        "stderr": smi.stderr,
    }
    resource["GPU_status"] = "AVAILABLE" if resource["cuda_available"] else "BLOCKED"
    atomic_json(root / "runtime.json", resource)
    rows = [
        r
        for r in read_jsonl("reports/pilot-20261110/audit_manifest.jsonl")
        if r["role"] == "fusion_train"
    ]
    import soundfile as sf

    # Structural duration extremes from consumed TRAIN only; no outputs drive selection.
    rows = sorted(rows, key=lambda r: sf.info(r["audio"]).frames)
    examples = [rows[0], rows[-1]]
    results = []
    for spec in cfg["experts"]:
        for policy_name, policy in [
            ("single", cfg["crop"]),
            (
                "cover",
                dict(
                    samples=64600,
                    strategy="cover",
                    padding="repeat",
                    aggregation="mean_hidden_mean_logits",
                    batch_size=2,
                ),
            ),
        ]:
            expert = FrozenExpert(spec, policy, "cpu")
            try:
                for row in examples:
                    x = read_audio(row["audio"])
                    item = AudioInput(row["id"], x)
                    first = expert.predict_utterances([item])
                    repeat = expert.predict_utterances([item])
                    torch.testing.assert_close(
                        first.embeddings, repeat.embeddings, atol=0, rtol=0
                    )
                    torch.testing.assert_close(
                        first.spoof_margins, repeat.spoof_margins, atol=0, rtol=0
                    )
                    controls = []
                    for operation in ("identity", "lossless", "lowpass"):
                        op = {"id": operation, "operation": operation}
                        if operation == "lowpass":
                            op["cutoff_hz"] = 3400
                        y, meta = apply_probe(x, op)
                        other = expert.predict_utterances([AudioInput(row["id"], y)])
                        assert (
                            coverage(len(x), policy)["starts"]
                            == coverage(len(y), policy)["starts"]
                        )
                        if operation in ("identity", "lossless"):
                            torch.testing.assert_close(
                                first.embeddings, other.embeddings, atol=1e-6, rtol=1e-5
                            )
                            torch.testing.assert_close(
                                first.spoof_margins,
                                other.spoof_margins,
                                atol=1e-6,
                                rtol=1e-5,
                            )
                        controls.append(
                            {
                                "operation": operation,
                                "same_coordinates": True,
                                "input_samples": len(x),
                                "output_samples": len(y),
                                "hidden_delta_max": float(
                                    (first.embeddings - other.embeddings).abs().max()
                                ),
                                "margin_delta": float(
                                    (first.spoof_margins - other.spoof_margins)
                                    .abs()
                                    .max()
                                ),
                                "timing": other.metadata["utterance_measurements"],
                            }
                        )
                    # Check streamed crop aggregation against full upstream crop batch.
                    with torch.inference_mode():
                        h, l = expert._run(crops(x, policy))
                    torch.testing.assert_close(
                        first.embeddings[0], h.mean(0), atol=1e-4, rtol=1e-4
                    )
                    torch.testing.assert_close(
                        first.spoof_margins[0],
                        (
                            l[:, spec["spoof_index"]] - l[:, spec["bonafide_index"]]
                        ).mean(),
                        atol=1e-4,
                        rtol=1e-4,
                    )
                    results.append(
                        {
                            "expert": spec["id"],
                            "policy": policy_name,
                            "id": row["id"],
                            "role": "consumed_fusion_train",
                            "checkpoint_sha256": expert.identity["checkpoint_sha256"],
                            "hidden_dim": first.embeddings.shape[1],
                            "single_utterance_embedding_and_margin": True,
                            "repeat_exact": True,
                            "upstream_full_batch_aggregation_close": True,
                            "profile": first.metadata["utterance_measurements"],
                            "controls": controls,
                            "status": "passed",
                        }
                    )
                    atomic_json(
                        root / "real-multicrop.json",
                        {
                            "level": "real_checkpoint_and_real_audio_CPU",
                            "rows": results,
                            "GPU_tested": False,
                        },
                    )
                    print(spec["id"], policy_name, row["id"], "passed", flush=True)
            finally:
                expert.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
