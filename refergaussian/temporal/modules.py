from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseTemporalWarp


class IdentityWarp(BaseTemporalWarp):
    def __init__(self):
        super().__init__(enabled=False)

    def _warp_impl(self, t: torch.Tensor, context: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        del context
        return t


class PositiveLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight_raw = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.softplus(self.weight_raw)
        return F.linear(x, weight, self.bias)


class MonotonicMLPWarp(BaseTemporalWarp):
    def __init__(self, hidden_dim: int = 32, num_layers: int = 2):
        super().__init__(enabled=True)
        num_layers = max(num_layers, 1)
        layers: List[nn.Module] = []
        in_features = 1
        for _ in range(num_layers):
            layers.append(PositiveLinear(in_features, hidden_dim))
            in_features = hidden_dim
        layers.append(PositiveLinear(in_features, 1))
        self.layers = nn.ModuleList(layers)

    def raw(self, t: torch.Tensor) -> torch.Tensor:
        hidden = t
        for idx, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if idx != len(self.layers) - 1:
                hidden = F.softplus(hidden)
        return hidden

    def _warp_impl(self, t: torch.Tensor, context: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        del context
        zero = torch.zeros((1, 1), device=t.device, dtype=t.dtype)
        one = torch.ones((1, 1), device=t.device, dtype=t.dtype)
        f0 = self.raw(zero)
        f1 = self.raw(one)
        denom = (f1 - f0).clamp_min(1.0e-6)
        return (self.raw(t) - f0) / denom


class DensityIntegralWarp(BaseTemporalWarp):
    def __init__(self, hidden_dim: int = 32, num_layers: int = 2, num_bins: int = 128):
        super().__init__(enabled=True)
        num_layers = max(num_layers, 1)
        self.num_bins = max(num_bins, 16)
        layers: List[nn.Module] = []
        in_features = 1
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.SiLU())
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, 1))
        self.mlp = nn.Sequential(*layers)

    def density(self, t: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.mlp(t)) + 1.0e-6

    def _warp_impl(self, t: torch.Tensor, context: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        del context
        grid = torch.linspace(0.0, 1.0, self.num_bins, device=t.device, dtype=t.dtype).unsqueeze(-1)
        density = self.density(grid)
        delta = 1.0 / (self.num_bins - 1)
        trap = 0.5 * (density[1:] + density[:-1]) * delta
        cdf = torch.cat([torch.zeros((1, 1), device=t.device, dtype=t.dtype), trap.cumsum(dim=0)], dim=0)
        cdf = cdf / cdf[-1].clamp_min(1.0e-6)

        x = t.squeeze(-1).clamp(0.0, 1.0)
        pos = x * (self.num_bins - 1)
        left = torch.floor(pos).long().clamp(0, self.num_bins - 2)
        right = left + 1
        frac = (pos - left.float()).unsqueeze(-1)
        return cdf[left] * (1.0 - frac) + cdf[right] * frac

    def sample_density(self, num_samples: int = 256, device: Optional[torch.device] = None) -> torch.Tensor:
        device = device or self.device
        t = torch.linspace(0.0, 1.0, num_samples, device=device).unsqueeze(-1)
        return self.density(t)

    def _smoothness_signal(
        self,
        t: torch.Tensor,
        tau: torch.Tensor,
        context: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        del tau
        del context
        return self.density(t)


class ContextualMetricWarp(BaseTemporalWarp):
    def __init__(self, hidden_dim: int = 32, num_layers: int = 2, max_context_samples: int = 256):
        super().__init__(enabled=True)
        num_layers = max(num_layers, 1)
        self.context_dim = 9
        self.max_shift = 0.35
        self.min_scale = 0.05
        self.max_context_samples = max(max_context_samples, 16)

        layers: List[nn.Module] = []
        in_features = self.context_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.SiLU())
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, 2))
        self.context_net = nn.Sequential(*layers)

    def _canonical_context(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        features = torch.zeros((batch_size, self.context_dim), device=device, dtype=dtype)
        features[:, 3] = 0.5
        features[:, 4] = 1.0
        return {
            "features": features,
            "time_anchor": torch.full((batch_size, 1), 0.5, device=device, dtype=dtype),
            "time_scale": torch.ones((batch_size, 1), device=device, dtype=dtype),
        }

    def _prepare_context(
        self, t: torch.Tensor, context: Optional[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        batch_size = t.shape[0]
        if context is None:
            return self._canonical_context(batch_size, t.device, t.dtype)

        prepared: Dict[str, torch.Tensor] = {}
        for key in ("features", "time_anchor", "time_scale"):
            value = context.get(key)
            if value is None:
                continue
            if value.ndim == 1:
                value = value.unsqueeze(-1)
            value = value.to(device=t.device, dtype=t.dtype)
            if value.shape[0] == 1 and batch_size > 1:
                value = value.expand(batch_size, -1)
            elif value.shape[0] != batch_size:
                raise ValueError(f"Context batch mismatch for {key}: got {value.shape[0]}, expected {batch_size}")
            prepared[key] = value

        if "features" not in prepared:
            return self._canonical_context(batch_size, t.device, t.dtype)
        feature_dim = prepared["features"].shape[-1]
        if feature_dim > self.context_dim:
            prepared["features"] = prepared["features"][:, : self.context_dim]
        elif feature_dim < self.context_dim:
            pad = torch.zeros((batch_size, self.context_dim - feature_dim), device=t.device, dtype=t.dtype)
            prepared["features"] = torch.cat([prepared["features"], pad], dim=-1)
        if "time_anchor" not in prepared:
            prepared["time_anchor"] = torch.full((batch_size, 1), 0.5, device=t.device, dtype=t.dtype)
        if "time_scale" not in prepared:
            prepared["time_scale"] = torch.ones((batch_size, 1), device=t.device, dtype=t.dtype)
        return prepared

    def _warp_impl(self, t: torch.Tensor, context: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        prepared = self._prepare_context(t, context)
        offsets = self.context_net(prepared["features"])
        anchor = prepared["time_anchor"].clamp(0.0, 1.0)
        scale = prepared["time_scale"].clamp_min(self.min_scale)

        shift = self.max_shift * torch.tanh((anchor - 0.5) + offsets[:, :1])
        scale = scale * torch.exp(0.35 * torch.tanh(offsets[:, 1:2]))
        scale = scale.clamp(self.min_scale, 4.0)

        raw = torch.sigmoid((t - shift) / scale)
        raw0 = torch.sigmoid((-shift) / scale)
        raw1 = torch.sigmoid((1.0 - shift) / scale)
        return (raw - raw0) / (raw1 - raw0).clamp_min(1.0e-6)

    def _limit_context(
        self, context: Optional[Dict[str, torch.Tensor]]
    ) -> Optional[Dict[str, torch.Tensor]]:
        if context is None or "features" not in context:
            return context
        num_points = context["features"].shape[0]
        if num_points <= self.max_context_samples:
            return context
        indices = torch.linspace(
            0,
            num_points - 1,
            self.max_context_samples,
            device=context["features"].device,
        ).round().long()
        limited: Dict[str, torch.Tensor] = {}
        for key, value in context.items():
            limited[key] = value.index_select(0, indices)
        return limited

    def _batched_warp(
        self, num_samples: int, device: torch.device, context: Optional[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        context = self._limit_context(context)
        if context is None:
            t = torch.linspace(0.0, 1.0, num_samples, device=device).unsqueeze(-1)
            return self.forward(t).unsqueeze(0)

        sample_count = context["features"].shape[0]
        t = torch.linspace(0.0, 1.0, num_samples, device=device).view(1, num_samples, 1).expand(sample_count, -1, -1)
        flat_t = t.reshape(-1, 1)
        flat_context = {
            key: value.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, value.shape[-1])
            for key, value in context.items()
        }
        tau = self.forward(flat_t, context=flat_context)
        return tau.view(sample_count, num_samples, 1)

    def sample_density(
        self, num_samples: int = 256, device: Optional[torch.device] = None, context: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        device = device or self.device
        tau = self._batched_warp(num_samples, device, context=context)
        if num_samples < 2:
            return torch.ones((tau.shape[0], 1), device=device)
        delta = 1.0 / (num_samples - 1)
        slopes = (tau[:, 1:] - tau[:, :-1]) / delta
        last = slopes[:, -1:].clone()
        return torch.cat([slopes, last], dim=1).mean(dim=0)

    def regularization_terms(
        self,
        num_samples: int = 128,
        device: Optional[torch.device] = None,
        context: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        device = device or self.device
        tau = self._batched_warp(num_samples, device, context=context)
        if num_samples < 2:
            zero = torch.zeros((), device=device)
            return {"mono": zero, "smooth": zero, "budget": zero}
        delta = 1.0 / max(num_samples - 1, 1)
        slopes = (tau[:, 1:] - tau[:, :-1]) / delta
        mono = torch.relu(self.min_slope - slopes).mean()
        if num_samples >= 3:
            smooth = (tau[:, 2:] - 2 * tau[:, 1:-1] + tau[:, :-2]).abs().mean()
        else:
            smooth = (tau[:, 1:] - tau[:, :-1]).abs().mean()
        budget_low = torch.relu(self.min_budget_slope - slopes)
        budget_high = torch.relu(slopes - self.max_budget_slope)
        budget = (budget_low + budget_high).mean()
        return {"mono": mono, "smooth": smooth, "budget": budget}

    def summary(
        self,
        num_samples: int = 256,
        device: Optional[torch.device] = None,
        context: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        device = device or self.device
        tau = self._batched_warp(num_samples, device, context=context).detach()
        if num_samples < 2:
            return {"tau_min": 0.0, "tau_max": 1.0, "non_uniformity": 0.0}
        delta = 1.0 / (num_samples - 1)
        slopes = ((tau[:, 1:] - tau[:, :-1]) / delta).detach()
        slope_mean = slopes.mean().item()
        slope_std = slopes.std(unbiased=False).item()
        non_uniformity = 0.0 if slope_mean == 0 else slope_std / max(abs(slope_mean), 1.0e-6)
        return {
            "tau_min": tau.min().item(),
            "tau_max": tau.max().item(),
            "slope_min": slopes.min().item(),
            "slope_max": slopes.max().item(),
            "non_uniformity": non_uniformity,
        }
