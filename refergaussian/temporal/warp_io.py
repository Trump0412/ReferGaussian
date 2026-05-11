from __future__ import annotations

import os
import re
from typing import Optional

import torch

from .modules import DensityIntegralWarp, IdentityWarp, MonotonicMLPWarp, ContextualMetricWarp
from .warp_viz import save_warp_artifacts


def build_temporal_warp(args, device: str = "cuda"):
    warp_enabled = bool(getattr(args, "warp_enabled", False))
    warp_type = getattr(args, "temporal_warp_type", "identity")
    hidden_dim = int(getattr(args, "warp_hidden_dim", 32))
    num_layers = int(getattr(args, "warp_num_layers", 2))
    num_bins = int(getattr(args, "warp_num_bins", 128))

    if not warp_enabled or warp_type == "identity":
        model = IdentityWarp()
    elif warp_type == "mlp":
        model = MonotonicMLPWarp(hidden_dim=hidden_dim, num_layers=num_layers)
    elif warp_type == "density":
        model = DensityIntegralWarp(hidden_dim=hidden_dim, num_layers=num_layers, num_bins=num_bins)
    elif warp_type in {"stellar", "stellar_metric", "local"}:
        model = ContextualMetricWarp(hidden_dim=hidden_dim, num_layers=num_layers)
    else:
        raise ValueError(f"Unsupported temporal warp type: {warp_type}")
    return model.to(device)


def build_temporal_warp_optimizer(warp, lr: float):
    parameters = [p for p in warp.parameters() if p.requires_grad]
    if not parameters:
        return None
    return torch.optim.Adam(parameters, lr=lr, eps=1.0e-15)


def attach_temporal_warp(gaussians, warp) -> None:
    gaussians.temporal_warp = warp


def temporal_root(model_path: str) -> str:
    return os.path.join(model_path, "temporal_warp")


def _iteration_dir(model_path: str, iteration: int) -> str:
    return os.path.join(temporal_root(model_path), f"iteration_{iteration}")


def _checkpoint_dir(model_path: str) -> str:
    return os.path.join(temporal_root(model_path), "checkpoints")


def _latest_iteration(model_path: str) -> Optional[int]:
    root = temporal_root(model_path)
    if not os.path.isdir(root):
        return None
    iterations = []
    for name in os.listdir(root):
        if name.startswith("iteration_"):
            try:
                iterations.append(int(name.split("_", 1)[1]))
            except ValueError:
                continue
    if not iterations:
        return None
    return max(iterations)


def save_temporal_warp(warp, model_path: str, iteration: int, optimizer=None, num_samples: int = 256):
    root = temporal_root(model_path)
    latest_dir = os.path.join(root, "latest")
    iter_dir = _iteration_dir(model_path, iteration)
    os.makedirs(latest_dir, exist_ok=True)
    os.makedirs(iter_dir, exist_ok=True)

    payload = {"state_dict": warp.state_dict(), "iteration": iteration}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()

    for output_dir in (latest_dir, iter_dir):
        torch.save(payload, os.path.join(output_dir, "warp.pth"))
        save_warp_artifacts(warp, output_dir, num_samples=num_samples)


def load_temporal_warp(warp, model_path: str, iteration: int = -1, optimizer=None) -> bool:
    if iteration == -1:
        iteration = _latest_iteration(model_path)
    if iteration is None:
        candidate = os.path.join(temporal_root(model_path), "latest", "warp.pth")
    else:
        candidate = os.path.join(_iteration_dir(model_path, iteration), "warp.pth")
        if not os.path.exists(candidate):
            candidate = os.path.join(temporal_root(model_path), "latest", "warp.pth")
    if not os.path.exists(candidate):
        return False

    payload = torch.load(candidate, map_location=warp.device)
    warp.load_state_dict(payload["state_dict"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return True


def save_temporal_warp_checkpoint(warp, model_path: str, stage: str, iteration: int, optimizer=None):
    checkpoint_dir = _checkpoint_dir(model_path)
    os.makedirs(checkpoint_dir, exist_ok=True)
    payload = {"state_dict": warp.state_dict(), "iteration": iteration, "stage": stage}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, os.path.join(checkpoint_dir, f"chkpnt_{stage}_{iteration}.pth"))


def load_temporal_warp_checkpoint(warp, checkpoint_path: str, optimizer=None) -> bool:
    if checkpoint_path is None:
        return False
    match = re.search(r"chkpnt_(coarse|fine)_(\d+)\.pth$", checkpoint_path)
    if match is None:
        return False
    stage, iteration = match.group(1), match.group(2)
    candidate = os.path.join(os.path.dirname(checkpoint_path), "temporal_warp", "checkpoints", f"chkpnt_{stage}_{iteration}.pth")
    if not os.path.exists(candidate):
        return False
    payload = torch.load(candidate, map_location=warp.device)
    warp.load_state_dict(payload["state_dict"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return True
