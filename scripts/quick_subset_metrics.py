import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as tf

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = REPO_ROOT / "external" / "4DGaussians"
for candidate in (REPO_ROOT, EXTERNAL_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from lpipsPyTorch import LPIPS
from utils.image_utils import psnr
from utils.loss_utils import ssim


def load_image(path: Path) -> torch.Tensor:
    image = Image.open(path)
    return tf.to_tensor(image).unsqueeze(0)[:, :3, :, :].cuda(non_blocking=True)


def select_filenames(renders_dir: Path, max_frames: int) -> list[str]:
    names = sorted(name for name in os.listdir(renders_dir) if name.endswith(".png"))
    if len(names) <= max_frames:
        return names
    indices = np.linspace(0, len(names) - 1, max_frames, dtype=np.int32)
    return [names[index] for index in indices.tolist()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--method", default="")
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--with-lpips", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    split_dir = run_dir / args.split
    method = args.method or sorted(path.name for path in split_dir.iterdir() if path.is_dir())[-1]
    method_dir = split_dir / method
    renders_dir = method_dir / "renders"
    gt_dir = method_dir / "gt"

    filenames = select_filenames(renders_dir, args.max_frames)
    total_frames = len([name for name in os.listdir(renders_dir) if name.endswith(".png")])

    psnrs = []
    ssims = []
    lpips_vgg = []
    lpips_vgg_metric = LPIPS(net_type="vgg").cuda() if args.with_lpips else None

    for name in filenames:
        render = load_image(renders_dir / name)
        gt = load_image(gt_dir / name)
        with torch.no_grad():
            psnrs.append(float(psnr(render, gt).mean().item()))
            ssims.append(float(ssim(render, gt).mean().item()))
            if lpips_vgg_metric is not None:
                lpips_vgg.append(float(lpips_vgg_metric(render, gt).mean().item()))

    payload = {
        "method": method,
        "PSNR": float(np.mean(psnrs)) if psnrs else None,
        "SSIM": float(np.mean(ssims)) if ssims else None,
        "LPIPS-vgg": float(np.mean(lpips_vgg)) if lpips_vgg else None,
        "sample_count": len(filenames),
        "sample_total": total_frames,
        "frame_names": filenames,
        "metric_mode": "subset",
    }

    out_path = run_dir / "quick_metrics.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(out_path)


if __name__ == "__main__":
    main()
