"""Synthetic wiring check. No scientific or real audio performance is measured."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from .core import EvidenceFusion, ExpertStats, source_weighted_bce
from .registry import load_registry, export_topology_fixture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/synthetic_smoke.json")
    args = parser.parse_args()
    torch.manual_seed(1729)
    torch.set_num_threads(1)
    b, e, k = 24, 3, 6
    dims = [8, 10, 12]  # deliberately synthetic, not asserted real model dimensions
    labels = torch.arange(b).remainder(2).float()
    sign = labels * 2 - 1
    hidden = [torch.randn(b, k+1, d) for d in dims]
    for h in hidden:
        h[:, :, 0] += sign[:, None] * 1.5
    scores = torch.randn(b, e, k+1) + sign[:, None, None] * 0.4
    valid = torch.ones_like(scores, dtype=torch.bool)
    registry = load_registry(Path(__file__).resolve().parents[1])
    families = registry.families
    probe_meta = registry.probe_features
    quality = torch.randn(b, 4)
    stats = ExpertStats.fit(hidden, scores, valid, role="fusion_train")
    hh, ss = stats.transform(hidden, scores, valid)
    results = {}
    for mode in ["flat", "untyped", "set_attention", "typed"]:
        torch.manual_seed(37)
        model = EvidenceFusion(dims, families, 4, width=16, dropout=0, mode=mode)
        opt = torch.optim.Adam(model.parameters(), lr=0.003)
        initial = float(source_weighted_bce(model(hh, ss, valid, probe_meta, quality), labels).detach())
        for _ in range(20):
            loss = source_weighted_bce(model(hh, ss, valid, probe_meta, quality), labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            out = model(hh, ss, valid, probe_meta, quality)
            final = float(source_weighted_bce(out, labels))
        results[mode] = {"parameters": sum(p.numel() for p in model.parameters()),
                         "initial_training_bce": initial, "final_training_bce": final,
                         "finite": bool(torch.isfinite(out.probabilities()).all())}
    report = {"data_kind": "SYNTHETIC_ONLY", "real_audio_used": False,
              "real_checkpoint_used": False, "scientific_performance_claim": False,
              "torch": torch.__version__, "batch": b, "steps": 20, "results": results,
              "warning": "Toy in-sample loss is only a wiring check; not a fair performance ranking."}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    report["registry_hash"] = registry.digest
    export_topology_fixture(registry, e, path.parent / "synthetic_topology.ttl")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
