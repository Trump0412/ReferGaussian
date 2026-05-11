#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SH_C0 = 0.28209479177387814


def _read_ply_vertex(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        props: list[tuple[str, str]] = []
        count: int | None = None
        fmt: str | None = None
        in_vertex = False
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"{path} ended before end_header")
            text = line.decode("ascii", errors="ignore").strip()
            if text.startswith("format "):
                fmt = text.split()[1]
            elif text.startswith("element "):
                tokens = text.split()
                in_vertex = len(tokens) == 3 and tokens[1] == "vertex"
                if in_vertex:
                    count = int(tokens[2])
            elif in_vertex and text.startswith("property "):
                tokens = text.split()
                if len(tokens) == 3:
                    props.append((tokens[2], tokens[1]))
            elif text == "end_header":
                break
        if fmt != "binary_little_endian":
            raise ValueError(f"Unsupported PLY format for {path}: {fmt}")
        if count is None:
            raise ValueError(f"PLY vertex count missing in {path}")
        type_map = {
            "char": "i1",
            "uchar": "u1",
            "short": "<i2",
            "ushort": "<u2",
            "int": "<i4",
            "uint": "<u4",
            "float": "<f4",
            "double": "<f8",
        }
        dtype = np.dtype([(name, type_map[kind]) for name, kind in props])
        return np.fromfile(handle, dtype=dtype, count=count)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _prepare_points(path: Path, max_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = _read_ply_vertex(path)
    names = set(vertices.dtype.names or [])
    xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)
    rgb = np.clip(
        np.stack([vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]], axis=1).astype(np.float32)
        * SH_C0
        + 0.5,
        0.0,
        1.0,
    )
    opacity = _sigmoid(vertices["opacity"].astype(np.float32)) if "opacity" in names else np.ones(len(xyz), dtype=np.float32)
    if {"scale_0", "scale_1", "scale_2"}.issubset(names):
        scale = np.exp(
            np.stack([vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]], axis=1).astype(np.float32)
        )
        point_size = np.cbrt(np.maximum(scale.prod(axis=1), 1e-12))
    else:
        point_size = np.ones(len(xyz), dtype=np.float32)

    finite_mask = np.isfinite(xyz).all(axis=1) & np.isfinite(rgb).all(axis=1) & np.isfinite(opacity) & np.isfinite(point_size)
    xyz = xyz[finite_mask]
    rgb = rgb[finite_mask]
    opacity = opacity[finite_mask]
    point_size = point_size[finite_mask]

    keep_mask = opacity >= np.quantile(opacity, 0.45)
    xyz = xyz[keep_mask]
    rgb = rgb[keep_mask]
    opacity = opacity[keep_mask]
    point_size = point_size[keep_mask]

    center = np.average(xyz, axis=0, weights=np.maximum(opacity, 1e-4))
    centered = xyz - center
    radius = np.linalg.norm(centered, axis=1)
    radius_limit = np.quantile(radius, 0.985)
    keep_mask = radius <= radius_limit
    xyz = xyz[keep_mask]
    rgb = rgb[keep_mask]
    opacity = opacity[keep_mask]
    point_size = point_size[keep_mask]

    score = opacity * (0.4 + 0.6 * np.clip(point_size / np.quantile(point_size, 0.95), 0.0, 1.0))
    if len(xyz) > max_points:
        indices = np.argsort(score)[-max_points:]
        xyz = xyz[indices]
        rgb = rgb[indices]
        opacity = opacity[indices]
        point_size = point_size[indices]

    center = np.average(xyz, axis=0, weights=np.maximum(opacity, 1e-4))
    xyz = xyz - center
    spread = np.quantile(np.abs(xyz), 0.985, axis=0)
    scale = float(max(spread.max(), 1e-4))
    xyz = xyz / scale

    rgb = np.clip(np.power(rgb, 0.9), 0.0, 1.0)
    alpha = np.clip(0.18 + 0.82 * np.sqrt(opacity), 0.0, 1.0)
    sizes = np.clip(6.0 + 52.0 * np.sqrt(point_size / np.quantile(point_size, 0.97)), 6.0, 70.0)
    colors = np.concatenate([rgb, alpha[:, None]], axis=1)
    return xyz, colors, sizes


def _style_axis(ax: plt.Axes, title: str) -> None:
    bg = "#0d1117"
    ax.set_facecolor(bg)
    ax.set_title(title, color="#e6edf3", fontsize=12, pad=10)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect((1.0, 1.0, 1.0))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.set_facecolor((0.05, 0.07, 0.10, 1.0))
            axis.pane.set_edgecolor((0.16, 0.19, 0.23, 1.0))
            axis.line.set_color((0.2, 0.24, 0.28, 1.0))
        except Exception:
            pass


def _render_panel(
    xyz: np.ndarray,
    colors: np.ndarray,
    sizes: np.ndarray,
    title: str,
    output_path: Path,
    views: list[tuple[str, float, float]],
) -> None:
    fig = plt.figure(figsize=(13.5, 4.8), facecolor="#0d1117")
    for idx, (label, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(1, len(views), idx, projection="3d")
        ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            c=colors,
            s=sizes,
            marker="o",
            linewidths=0.0,
            depthshade=False,
        )
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(-1.0, 1.0)
        _style_axis(ax, label)
    fig.suptitle(title, color="#f0f6fc", fontsize=18, y=0.98)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.86, wspace=0.04)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def _render_single_view(
    xyz: np.ndarray,
    colors: np.ndarray,
    sizes: np.ndarray,
    title: str,
    elev: float,
    azim: float,
) -> Image.Image:
    fig = plt.figure(figsize=(6.6, 6.6), dpi=180, facecolor="#0d1117")
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=colors,
        s=sizes,
        marker="o",
        linewidths=0.0,
        depthshade=False,
    )
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    _style_axis(ax, title)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92)
    fig.canvas.draw()
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3], mode="RGB")
    plt.close(fig)
    return image


def _build_pair(left_path: Path, right_path: Path, output_path: Path) -> None:
    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")
    gap = 18
    canvas = Image.new("RGB", (left.width + right.width + gap, max(left.height, right.height)), "#0d1117")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gap, 0))
    canvas.save(output_path)


def _build_rotation_gif(
    baseline_xyz: np.ndarray,
    baseline_colors: np.ndarray,
    baseline_sizes: np.ndarray,
    ours_xyz: np.ndarray,
    ours_colors: np.ndarray,
    ours_sizes: np.ndarray,
    baseline_label: str,
    ours_label: str,
    gif_title: str,
    output_path: Path,
    frame_count: int,
    duration_ms: int,
    elev: float,
    azim_start: float,
    azim_span: float,
) -> None:
    font = ImageFont.load_default()
    frames: list[Image.Image] = []
    padding = 14
    gap = 18
    title_h = 24 if gif_title else 0
    label_h = 24
    azimuths = np.linspace(azim_start, azim_start + azim_span, num=max(frame_count, 1), endpoint=False)

    for azim in azimuths:
        left = _render_single_view(baseline_xyz, baseline_colors, baseline_sizes, baseline_label, elev=elev, azim=float(azim))
        right = _render_single_view(ours_xyz, ours_colors, ours_sizes, ours_label, elev=elev, azim=float(azim))
        width = left.width + right.width + gap + 2 * padding
        height = max(left.height, right.height) + title_h + label_h + 2 * padding
        canvas = Image.new("RGB", (width, height), "#0d1117")
        draw = ImageDraw.Draw(canvas)
        if gif_title:
            draw.text((padding, 6), gif_title, fill=(235, 235, 235), font=font)
        y = padding + title_h
        x_left = padding
        x_right = padding + left.width + gap
        canvas.paste(left, (x_left, y))
        canvas.paste(right, (x_right, y))
        draw.rectangle((x_left, y + left.height, x_left + left.width, y + left.height + label_h), fill=(30, 30, 30))
        draw.rectangle((x_right, y + right.height, x_right + right.width, y + right.height + label_h), fill=(30, 30, 30))
        draw.text((x_left + 6, y + left.height + 5), baseline_label, fill=(240, 240, 240), font=font)
        draw.text((x_right + 6, y + right.height + 5), ours_label, fill=(240, 240, 240), font=font)
        frames.append(canvas)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=max(duration_ms, 20),
        loop=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render report-friendly Gaussian point cloud panels.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--ours", type=Path, required=True)
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=45000)
    parser.add_argument("--baseline-title", default="4DGS baseline")
    parser.add_argument("--ours-title", default="ReferGaussian")
    parser.add_argument("--gif-path", type=Path, default=None)
    parser.add_argument("--gif-title", default="")
    parser.add_argument("--gif-frames", type=int, default=36)
    parser.add_argument("--gif-duration-ms", type=int, default=90)
    parser.add_argument("--gif-elevation", type=float, default=18.0)
    parser.add_argument("--gif-azim-start", type=float, default=35.0)
    parser.add_argument("--gif-azim-span", type=float, default=360.0)
    args = parser.parse_args()

    views = [
        ("view 1", 18.0, 35.0),
        ("view 2", 18.0, 92.0),
        ("view 3", 18.0, 145.0),
    ]

    baseline_xyz, baseline_colors, baseline_sizes = _prepare_points(args.baseline, args.max_points)
    ours_xyz, ours_colors, ours_sizes = _prepare_points(args.ours, args.max_points)

    baseline_out = args.out_dir / "pointcloud_baseline_4dgs.png"
    ours_out = args.out_dir / "pointcloud_ours_worldtube.png"
    pair_out = args.out_dir / "pointcloud_pair.png"

    _render_panel(
        baseline_xyz,
        baseline_colors,
        baseline_sizes,
        f"{args.scene_name} {args.baseline_title}",
        baseline_out,
        views,
    )
    _render_panel(
        ours_xyz,
        ours_colors,
        ours_sizes,
        f"{args.scene_name} {args.ours_title}",
        ours_out,
        views,
    )
    _build_pair(baseline_out, ours_out, pair_out)

    if args.gif_path is not None:
        gif_title = args.gif_title or f"{args.scene_name} Gaussian Point Cloud Comparison"
        _build_rotation_gif(
            baseline_xyz=baseline_xyz,
            baseline_colors=baseline_colors,
            baseline_sizes=baseline_sizes,
            ours_xyz=ours_xyz,
            ours_colors=ours_colors,
            ours_sizes=ours_sizes,
            baseline_label=args.baseline_title,
            ours_label=args.ours_title,
            gif_title=gif_title,
            output_path=args.gif_path,
            frame_count=args.gif_frames,
            duration_ms=args.gif_duration_ms,
            elev=args.gif_elevation,
            azim_start=args.gif_azim_start,
            azim_span=args.gif_azim_span,
        )


if __name__ == "__main__":
    main()
