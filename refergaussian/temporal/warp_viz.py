from __future__ import annotations

import json
import os
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .base import BaseTemporalWarp


def sample_warp_artifacts(
    warp: BaseTemporalWarp, num_samples: int = 256, device: Optional[torch.device] = None
) -> Dict[str, object]:
    device = device or warp.device
    t = torch.linspace(0.0, 1.0, num_samples, device=device).unsqueeze(-1)
    tau = warp(t).detach().squeeze(-1).cpu().tolist()
    density = warp.sample_density(num_samples, device=device).detach().squeeze(-1).cpu().tolist()
    summary = warp.summary(num_samples, device=device)
    return {
        "t": [float(i) / max(num_samples - 1, 1) for i in range(num_samples)],
        "tau": tau,
        "density": density,
        "summary": summary,
    }


def save_warp_artifacts(
    warp: BaseTemporalWarp, output_dir: str, num_samples: int = 256, device: Optional[torch.device] = None
) -> Dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)
    artifacts = sample_warp_artifacts(warp, num_samples=num_samples, device=device)

    with open(os.path.join(output_dir, "warp_curve.json"), "w", encoding="utf-8") as handle:
        json.dump({"t": artifacts["t"], "tau": artifacts["tau"]}, handle, indent=2)
    with open(os.path.join(output_dir, "warp_density.json"), "w", encoding="utf-8") as handle:
        json.dump({"t": artifacts["t"], "density": artifacts["density"]}, handle, indent=2)
    with open(os.path.join(output_dir, "warp_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(artifacts["summary"], handle, indent=2)

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(artifacts["t"], artifacts["tau"], color="#0b7285", linewidth=2, label="tau(t)")
    ax1.set_xlabel("t")
    ax1.set_ylabel("tau(t)", color="#0b7285")
    ax1.tick_params(axis="y", labelcolor="#0b7285")

    ax2 = ax1.twinx()
    ax2.plot(artifacts["t"], artifacts["density"], color="#c92a2a", linewidth=1.5, linestyle="--", label="density")
    ax2.set_ylabel("density / local slope", color="#c92a2a")
    ax2.tick_params(axis="y", labelcolor="#c92a2a")

    ax1.set_title("Temporal Warp")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "time_warp.png"), dpi=160)
    plt.close(fig)
    return artifacts

