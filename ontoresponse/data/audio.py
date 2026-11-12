import numpy as np
import soundfile as sf
import torch


def read_audio(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != 16000:
        raise ValueError(
            "Expected 16000Hz; resampling must be explicit and common to all classes"
        )
    x = x.mean(1)
    validate_audio(x)
    return x


def validate_audio(x):
    x = np.asarray(x)
    if x.ndim != 1 or len(x) < 160 or not np.isfinite(x).all():
        raise ValueError("Invalid or short waveform")
    if np.sqrt(np.mean(x.astype(np.float64) ** 2)) < 1e-7:
        raise ValueError("Silent waveform")
    return x


def crop_starts(length, policy):
    """Resolve coordinates once on the current input timeline, without labels."""
    n = policy["samples"]
    if length < 160 or n < 160:
        raise ValueError("Invalid waveform/crop length")
    if policy.get("strategy", "fixed") == "cover":
        # Nonoverlapping windows plus a final right-aligned window; no gaps.
        starts = list(range(0, max(1, length - n + 1), n))
        if length > n and starts[-1] != length - n:
            starts.append(length - n)
        return starts
    if policy.get("strategy", "fixed") != "fixed":
        raise ValueError("Unknown crop strategy")
    starts = policy["starts"]
    if not starts or any(type(s) is not int or not 0 <= s < length for s in starts):
        raise ValueError("Illegal crop starts")
    return list(starts)


def coverage(length, policy):
    starts = crop_starts(length, policy)
    end = total = 0
    for start in sorted(starts):
        stop = min(length, start + policy["samples"])
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return {
        "starts": starts,
        "crop_count": len(starts),
        "coverage_fraction": total / length,
        "unique_samples": total,
        "model_input_samples": len(starts) * policy["samples"],
    }


def crops(x, policy):
    validate_audio(x)
    n = policy["samples"]
    if (
        n < 160
        or policy.get("padding") != "repeat"
        or policy.get("aggregation") != "mean_hidden_mean_logits"
    ):
        raise ValueError("Unsupported crop/aggregation policy")
    out = []
    for start in crop_starts(len(x), policy):
        if not isinstance(start, int) or start < 0 or start >= len(x):
            raise ValueError("Illegal crop start")
        # Repeat the full aligned waveform, not a separately padded suffix.
        wave = np.tile(x, (start + n + len(x) - 1) // len(x))
        out.append(wave[start : start + n])
    if not out:
        raise ValueError("No crops")
    return torch.from_numpy(np.stack(out).astype("float32"))


def quality(x):
    return [
        float(np.log1p(len(x) / 16000)),
        float(20 * np.log10(max(np.sqrt(np.mean(x.astype("float64") ** 2)), 1e-10)))
        / 100,
        float(np.mean(np.abs(x) >= 0.999)),
        float(np.mean(np.abs(x) < 1e-4)),
    ]
