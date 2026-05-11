from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class DenseResnet18(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = torch.nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
        )
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.stem(image)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return F.normalize(x, dim=1)


@lru_cache(maxsize=1)
def get_dense_resnet18() -> DenseResnet18:
    return DenseResnet18()


def _imagenet_normalize(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=image.dtype, device=image.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=image.dtype, device=image.device).view(1, 3, 1, 1)
    return (image - mean) / std


def feature_map_from_image(image_rgb: np.ndarray, device: str = "cuda") -> torch.Tensor:
    model = get_dense_resnet18().to(device)
    tensor = torch.from_numpy(np.asarray(image_rgb, dtype=np.float32)).to(device)
    if tensor.ndim != 3 or tensor.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got shape {tuple(tensor.shape)}")
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    tensor = _imagenet_normalize(tensor)
    with torch.no_grad():
        feat = model(tensor)
    return feat.squeeze(0)


def _mask_to_feature_resolution(mask: np.ndarray, feat_hw: Tuple[int, int], device: str) -> torch.Tensor:
    mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.float32)).to(device)[None, None]
    mask_tensor = F.interpolate(mask_tensor, size=feat_hw, mode="nearest")
    return mask_tensor[0, 0] > 0.5


def prototypes_from_masks(
    feature_map: torch.Tensor,
    positive_mask: np.ndarray,
    negative_mask: np.ndarray,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    device = feature_map.device
    feat_h, feat_w = int(feature_map.shape[1]), int(feature_map.shape[2])
    pos = _mask_to_feature_resolution(positive_mask, (feat_h, feat_w), device=device)
    neg = _mask_to_feature_resolution(negative_mask, (feat_h, feat_w), device=device)
    flat = feature_map.view(feature_map.shape[0], -1).transpose(0, 1)
    pos_flat = pos.reshape(-1)
    neg_flat = neg.reshape(-1)
    pos_proto = F.normalize(flat[pos_flat].mean(dim=0), dim=0) if torch.any(pos_flat) else None
    neg_proto = F.normalize(flat[neg_flat].mean(dim=0), dim=0) if torch.any(neg_flat) else None
    return pos_proto, neg_proto


def sample_feature_map(
    feature_map: torch.Tensor,
    pixels_xy: np.ndarray,
    image_h: int,
    image_w: int,
) -> torch.Tensor:
    device = feature_map.device
    pixels = np.asarray(pixels_xy, dtype=np.float32)
    if pixels.size == 0:
        return torch.empty((0, feature_map.shape[0]), device=device, dtype=feature_map.dtype)
    grid_x = np.clip((pixels[:, 0] / max(float(image_w - 1), 1.0)) * 2.0 - 1.0, -1.0, 1.0)
    grid_y = np.clip((pixels[:, 1] / max(float(image_h - 1), 1.0)) * 2.0 - 1.0, -1.0, 1.0)
    grid = torch.from_numpy(np.stack([grid_x, grid_y], axis=1)).to(device=device, dtype=feature_map.dtype)
    grid = grid.view(1, -1, 1, 2)
    feat = feature_map.unsqueeze(0)
    sampled = F.grid_sample(feat, grid, mode="bilinear", align_corners=True)
    sampled = sampled.squeeze(0).squeeze(-1).transpose(0, 1)
    return F.normalize(sampled, dim=1)


def prototype_similarity(
    sampled_features: torch.Tensor,
    positive_proto: torch.Tensor | None,
    negative_proto: torch.Tensor | None,
) -> tuple[np.ndarray, np.ndarray]:
    if sampled_features.numel() == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    if positive_proto is None:
        pos = torch.ones((sampled_features.shape[0],), device=sampled_features.device, dtype=sampled_features.dtype)
    else:
        pos = ((sampled_features * positive_proto[None, :]).sum(dim=1) + 1.0) * 0.5
    if negative_proto is None:
        neg = torch.zeros((sampled_features.shape[0],), device=sampled_features.device, dtype=sampled_features.dtype)
    else:
        neg = ((sampled_features * negative_proto[None, :]).sum(dim=1) + 1.0) * 0.5
    return pos.detach().cpu().numpy().astype(np.float32), neg.detach().cpu().numpy().astype(np.float32)
