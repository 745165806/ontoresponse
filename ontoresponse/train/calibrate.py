import numpy as np
import torch


class Platt:
    @staticmethod
    def fit(logits, labels, role, steps=20):
        if role != "calibration":
            raise ValueError("Platt fitting is calibration-only")
        if not 1 <= steps <= 20:
            raise ValueError("Calibration smoke budget <=20 steps")
        s = torch.as_tensor(logits, dtype=torch.float64)
        y = torch.as_tensor(labels, dtype=torch.float64)
        if (
            s.shape != y.shape
            or not torch.isfinite(s).all()
            or set(y.tolist()) != {0.0, 1.0}
        ):
            raise ValueError("Invalid calibration data")
        # Positive scale cannot learn to reverse score direction on held-out labels.
        log_scale = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
        bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
        optimizer = torch.optim.Adam([log_scale, bias], lr=0.05)
        for _ in range(steps):
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                s * log_scale.exp() + bias, y
            )
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                log_scale.clamp_(-8, 8)
                bias.clamp_(-30, 30)
        return {
            "scale": float(log_scale.exp().detach()),
            "bias": float(bias.detach()),
            "role": role,
            "steps": steps,
        }

    @staticmethod
    def apply(logits, params):
        if params["scale"] <= 0:
            raise ValueError("Calibration must preserve direction")
        z = np.asarray(logits) * params["scale"] + params["bias"]
        from scipy.special import expit

        return expit(z)
