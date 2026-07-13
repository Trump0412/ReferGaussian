import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from pytorch_msssim import ms_ssim

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = REPO_ROOT / "external" / "4DGaussians"
for candidate in (REPO_ROOT, EXTERNAL_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from lpipsPyTorch import LPIPS
from utils.image_utils import psnr
from utils.loss_utils import ssim


def load_image(path: Path, device: torch.device) -> torch.Tensor:
    image = Image.open(path)
    return tf.to_tensor(image).unsqueeze(0)[:, :3, :, :].to(device, non_blocking=True)


def list_png_filenames(path: Path) -> list[str]:
    return sorted(name for name in os.listdir(path) if name.endswith(".png"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--method", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--with-lpips", action="store_true")
    parser.add_argument("--out-name", default="full_metrics.json")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.run_dir)
    split_dir = run_dir / args.split
    method = args.method or sorted(path.name for path in split_dir.iterdir() if path.is_dir())[-1]
    method_dir = split_dir / method
    renders_dir = method_dir / "renders"
    gt_dir = method_dir / "gt"

    filenames = list_png_filenames(renders_dir)
    psnrs = []
    ssims = []
    ms_ssims = []
    lpips_vgg = []
    lpips_vgg_metric = LPIPS(net_type="vgg").to(device) if args.with_lpips else None

    for idx, name in enumerate(filenames, start=1):
        render = load_image(renders_dir / name, device)
        gt = load_image(gt_dir / name, device)
        with torch.no_grad():
            psnrs.append(float(psnr(render, gt).mean().item()))
            ssims.append(float(ssim(render, gt).mean().item()))
            ms_ssims.append(float(ms_ssim(render, gt, data_range=1.0, size_average=True).item()))
            if lpips_vgg_metric is not None:
                lpips_vgg.append(float(lpips_vgg_metric(render, gt).mean().item()))
        if idx % 50 == 0 or idx == len(filenames):
            print(f"[{idx}/{len(filenames)}] {method}")

    payload = {
        "method": method,
        "PSNR": float(np.mean(psnrs)) if psnrs else None,
        "SSIM": float(np.mean(ssims)) if ssims else None,
        "MS-SSIM": float(np.mean(ms_ssims)) if ms_ssims else None,
        "LPIPS-vgg": float(np.mean(lpips_vgg)) if lpips_vgg else None,
        "sample_count": len(filenames),
        "sample_total": len(filenames),
        "frame_names": filenames,
        "metric_mode": "full",
    }

    out_path = run_dir / args.out_name
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(out_path)


if __name__ == "__main__":
    main()
