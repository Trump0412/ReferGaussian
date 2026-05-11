#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _find_method_dir(run_dir: Path, split: str, method: str) -> Path:
    split_dir = run_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split dir: {split_dir}")
    if method:
        method_dir = split_dir / method
        if not method_dir.is_dir():
            raise FileNotFoundError(f"Missing method dir: {method_dir}")
        return method_dir
    candidates = sorted(path for path in split_dir.iterdir() if path.is_dir() and path.name.startswith("ours_"))
    if not candidates:
        raise FileNotFoundError(f"No ours_* method dir under {split_dir}")
    return candidates[-1]


def _load_triptych_images(
    baseline_run: Path,
    ours_run: Path,
    split: str,
    frame_name: str,
    baseline_method: str,
    ours_method: str,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    baseline_method_dir = _find_method_dir(baseline_run, split, baseline_method)
    ours_method_dir = _find_method_dir(ours_run, split, ours_method)
    baseline_render = Image.open(baseline_method_dir / "renders" / frame_name).convert("RGB")
    ours_render = Image.open(ours_method_dir / "renders" / frame_name).convert("RGB")
    gt_path = baseline_method_dir / "gt" / frame_name
    if not gt_path.exists():
        gt_path = ours_method_dir / "gt" / frame_name
    gt_image = Image.open(gt_path).convert("RGB")
    return baseline_render, ours_render, gt_image


def _draw_label(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, text: str, font: ImageFont.ImageFont) -> None:
    draw.rectangle((x, y, x + w, y + h), fill=(30, 30, 30))
    draw.text((x + 8, y + 6), text, fill=(240, 240, 240), font=font)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export baseline/ours/GT triptych screenshots or posters.")
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--ours-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--frame-name", action="append", dest="frame_names")
    parser.add_argument("--baseline-method", default="")
    parser.add_argument("--ours-method", default="")
    parser.add_argument("--baseline-label", default="4DGS")
    parser.add_argument("--ours-label", default="ReferGaussian")
    parser.add_argument("--gt-label", default="GT")
    parser.add_argument("--title", default="")
    parser.add_argument("--padding", type=int, default=20)
    args = parser.parse_args()

    if not args.frame_names:
        raise ValueError("At least one --frame-name is required")

    baseline_run = Path(args.baseline_run)
    ours_run = Path(args.ours_run)
    font = ImageFont.load_default()
    triptychs = [
        _load_triptych_images(
            baseline_run=baseline_run,
            ours_run=ours_run,
            split=args.split,
            frame_name=frame_name,
            baseline_method=args.baseline_method,
            ours_method=args.ours_method,
        )
        for frame_name in args.frame_names
    ]

    sample_w, sample_h = triptychs[0][0].size
    columns = 3
    rows = len(triptychs)
    label_h = 26
    row_title_h = 22
    title_h = 30 if args.title else 0
    canvas_w = columns * sample_w + (columns + 1) * args.padding
    canvas_h = title_h + rows * (sample_h + label_h + row_title_h + args.padding) + args.padding
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)

    if args.title:
        draw.text((args.padding, 8), args.title, fill=(235, 235, 235), font=font)

    y_cursor = title_h + args.padding
    labels = [args.baseline_label, args.ours_label, args.gt_label]
    for frame_name, (baseline_img, ours_img, gt_img) in zip(args.frame_names, triptychs):
        draw.text((args.padding, y_cursor), frame_name, fill=(220, 220, 220), font=font)
        y = y_cursor + row_title_h
        for idx, image in enumerate((baseline_img, ours_img, gt_img)):
            x = args.padding + idx * (sample_w + args.padding)
            canvas.paste(image, (x, y))
            _draw_label(draw, x, y + sample_h, sample_w, label_h, labels[idx], font)
        y_cursor = y + sample_h + label_h + args.padding

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
