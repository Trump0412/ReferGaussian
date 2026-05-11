from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
from torch import nn


class BaseTemporalWarp(nn.Module, ABC):
    def __init__(self, enabled: bool = True):
        super().__init__()
        self.enabled = enabled
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)
        self.min_slope = 1.0e-4
        self.min_budget_slope = 0.2
        self.max_budget_slope = 5.0

    def forward(self, t: torch.Tensor, context: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        t = self._ensure_tensor(t)
        if not self.enabled:
            return t
        warped = self._warp_impl(t.clamp(0.0, 1.0), context=context)
        return warped.clamp(0.0, 1.0)

    @abstractmethod
    def _warp_impl(
        self, t: torch.Tensor, context: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        raise NotImplementedError

    def sample_density(self, num_samples: int = 256, device: Optional[torch.device] = None) -> torch.Tensor:
        device = device or self.device
        t = torch.linspace(0.0, 1.0, num_samples, device=device).unsqueeze(-1)
        tau = self.forward(t)
        if num_samples < 2:
            return torch.ones_like(tau)
        delta = 1.0 / (num_samples - 1)
        slopes = (tau[1:] - tau[:-1]) / delta
        last = slopes[-1:].clone()
        return torch.cat([slopes, last], dim=0)

    def regularization_terms(
        self,
        num_samples: int = 128,
        device: Optional[torch.device] = None,
        context: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        device = device or self.device
        t = torch.linspace(0.0, 1.0, num_samples, device=device).unsqueeze(-1)
        tau = self.forward(t, context=context)
        delta = 1.0 / max(num_samples - 1, 1)
        if num_samples < 2:
            zero = torch.zeros((), device=device)
            return {"mono": zero, "smooth": zero, "budget": zero}

        slopes = (tau[1:] - tau[:-1]) / delta
        mono = torch.relu(self.min_slope - slopes).mean()

        smooth_signal = self._smoothness_signal(t, tau, context=context)
        if smooth_signal.shape[0] >= 3:
            smooth = (smooth_signal[2:] - 2 * smooth_signal[1:-1] + smooth_signal[:-2]).abs().mean()
        elif smooth_signal.shape[0] >= 2:
            smooth = (smooth_signal[1:] - smooth_signal[:-1]).abs().mean()
        else:
            smooth = torch.zeros((), device=device)

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
        t = torch.linspace(0.0, 1.0, num_samples, device=device).unsqueeze(-1)
        tau = self.forward(t, context=context).detach().squeeze(-1)
        if num_samples < 2:
            return {"tau_min": 0.0, "tau_max": 1.0, "non_uniformity": 0.0}
        delta = 1.0 / (num_samples - 1)
        slopes = ((tau[1:] - tau[:-1]) / delta).detach()
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

    def _smoothness_signal(
        self,
        t: torch.Tensor,
        tau: torch.Tensor,
        context: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        del t
        del context
        return tau

    def _ensure_tensor(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.reshape(1, 1)
        elif t.ndim == 1:
            t = t.unsqueeze(-1)
        return t.to(self.device)

    @property
    def device(self) -> torch.device:
        return self._device_anchor.device
