import argparse
from pathlib import Path

import imageio.v3 as iio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData

matplotlib.use("Agg")


def _latest_iteration_dir(run_dir: Path) -> Path:
    point_cloud_root = run_dir / "point_cloud"
    candidates = sorted(path for path in point_cloud_root.glob("iteration_*") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No point cloud iterations found under {point_cloud_root}")
    return candidates[-1]


def _load_colors(ply_path: Path) -> np.ndarray:
    ply = PlyData.read(str(ply_path))
    vertex = ply["vertex"]
    color_fields = ["red", "green", "blue"]
    if all(field in vertex.data.dtype.names for field in color_fields):
        colors = np.stack([vertex[field] for field in color_fields], axis=1).astype(np.float32) / 255.0
        return colors
    count = len(vertex["x"])
    return np.full((count, 3), 0.85, dtype=np.float32)


def _select_indices(payload: dict[str, np.ndarray], max_points: int, seed: int) -> np.ndarray:
    num_points = int(payload["xyz"].shape[0])
    if num_points <= max_points:
        return np.arange(num_points, dtype=np.int64)
    score = np.asarray(payload.get("occupancy_mass", np.ones((num_points,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    motion = np.asarray(payload.get("motion_score", np.zeros((num_points,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    priority = score + 0.25 * motion
    top_k = max(int(max_points * 0.7), min(max_points, num_points // 4))
    top_indices = np.argsort(priority)[-top_k:]
    remaining = np.setdiff1d(np.arange(num_points, dtype=np.int64), top_indices, assume_unique=False)
    rng = np.random.default_rng(seed)
    extra_count = max_points - top_indices.shape[0]
    if extra_count > 0 and remaining.size > 0:
        extra = rng.choice(remaining, size=min(extra_count, remaining.size), replace=False)
        indices = np.concatenate([top_indices, extra])
    else:
        indices = top_indices
    return np.sort(indices.astype(np.int64))


def _aligned_point_count(payload: dict[str, np.ndarray], colors: np.ndarray) -> int:
    candidate_counts: list[int] = [int(colors.shape[0])]
    for key in ("trajectories", "gate", "xyz", "occupancy_mass", "motion_score"):
        value = payload.get(key)
        if value is None:
            continue
        array = np.asarray(value)
        if array.ndim <= 0:
            continue
        candidate_counts.append(int(array.shape[0]))
    if not candidate_counts:
        raise ValueError("Unable to infer point count from payload.")
    count = min(candidate_counts)
    if count <= 0:
        raise ValueError("Aligned point count must be positive.")
    return count


def _set_equal_axes(ax, xyz: np.ndarray) -> None:
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    radius = max(radius, 1.0e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def export_pointcloud_video(
    run_dir: Path,
    output_path: Path,
    max_points: int,
    frame_stride: int,
    seed: int,
    fps: int,
    azimuth_step: float,
    elevation: float,
    point_size: float,
) -> Path:
    trajectory_path = run_dir / "entitybank" / "trajectory_samples.npz"
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Missing trajectory_samples.npz under {run_dir / 'entitybank'}")
    payload_npz = np.load(trajectory_path)
    payload = {key: payload_npz[key] for key in payload_npz.files}

    latest_iteration = _latest_iteration_dir(run_dir)
    ply_path = latest_iteration / "point_cloud.ply"
    colors = _load_colors(ply_path)

    aligned_count = _aligned_point_count(payload, colors)
    colors = colors[:aligned_count]
    payload = {
        key: (np.asarray(value)[:aligned_count] if np.asarray(value).ndim > 0 else np.asarray(value))
        for key, value in payload.items()
    }

    indices = _select_indices(payload, max_points=max_points, seed=seed)
    trajectories = np.asarray(payload["trajectories"], dtype=np.float32)[indices]
    gates = np.asarray(payload["gate"], dtype=np.float32)[indices].reshape(trajectories.shape[0], trajectories.shape[1])
    colors = colors[indices]
    time_values = np.asarray(payload["time_values"], dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    sampled_frame_ids = list(range(0, trajectories.shape[1], max(int(frame_stride), 1)))
    if sampled_frame_ids[-1] != trajectories.shape[1] - 1:
        sampled_frame_ids.append(trajectories.shape[1] - 1)

    for local_index, frame_id in enumerate(sampled_frame_ids):
        xyz = trajectories[:, frame_id, :]
        gate = np.clip(gates[:, frame_id], 0.0, 1.0)
        visible = gate >= 0.05
        if not np.any(visible):
            visible = gate >= 0.0
        frame_xyz = xyz[visible]
        frame_colors = colors[visible] * (0.35 + 0.65 * gate[visible, None])
        if frame_xyz.shape[0] == 0:
            frame_xyz = xyz
            frame_colors = colors

        fig = plt.figure(figsize=(7.5, 7.5), dpi=140)
        ax = fig.add_subplot(111, projection="3d")
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
        ax.scatter(
            frame_xyz[:, 0],
            frame_xyz[:, 1],
            frame_xyz[:, 2],
            s=float(point_size),
            c=frame_colors,
            alpha=0.95,
            linewidths=0.0,
            depthshade=False,
        )
        _set_equal_axes(ax, frame_xyz)
        ax.view_init(elev=float(elevation), azim=float(azimuth_step) * float(local_index))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.grid(False)
        ax.set_title(
            f"ReferGaussian point cloud  t={float(time_values[frame_id]):.3f}  points={int(frame_xyz.shape[0])}",
            color="white",
            pad=18,
        )
        fig.tight_layout(pad=0.2)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3]
        frames.append(frame)
        plt.close(fig)

    iio.imwrite(output_path, frames, fps=int(fps))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--max-points", type=int, default=30000)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--azimuth-step", type=float, default=4.0)
    parser.add_argument("--elevation", type=float, default=20.0)
    parser.add_argument("--point-size", type=float, default=0.8)
    args = parser.parse_args()

    output_path = export_pointcloud_video(
        run_dir=Path(args.run_dir),
        output_path=Path(args.output_path),
        max_points=int(args.max_points),
        frame_stride=int(args.frame_stride),
        seed=int(args.seed),
        fps=int(args.fps),
        azimuth_step=float(args.azimuth_step),
        elevation=float(args.elevation),
        point_size=float(args.point_size),
    )
    print(output_path)


if __name__ == "__main__":
    main()
