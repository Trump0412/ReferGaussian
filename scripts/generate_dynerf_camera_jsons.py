#!/usr/bin/env python3
"""Create camera metadata required by the DyNeRF referring pipeline.

Neural 3D Video / DyNeRF scenes provide ``poses_bounds.npy`` in LLFF format,
while the referring pipeline also expects one Nerfies-style ``camera/*.json``
file per image.  The supported monocular release layout uses a fixed ``cam00``
camera, so every frame receives the same calibrated camera record.

Example:
    python scripts/generate_dynerf_camera_jsons.py \
        --dataset-dir data/dynerf/coffee_martini
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import numpy as np


def load_llff_poses(dataset_dir: Path) -> tuple[np.ndarray, float, float, float]:
    """Load poses_bounds.npy and return (c2w_all, H, W, focal) for all cameras."""
    poses_arr = np.load(dataset_dir / "poses_bounds.npy")
    poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
    H, W, focal = poses[0, :, -1]
    return poses, float(H), float(W), float(focal)


def llff_c2w_to_nerfies_camera(
    c2w_raw: np.ndarray,
    H: float,
    W: float,
    focal: float,
    actual_H: int,
    actual_W: int,
) -> dict:
    """
    Convert LLFF c2w matrix to nerfies/HyperNeRF Camera JSON format.

    LLFF convention (raw, before neural_3D rearrangement):
      c2w[:, 0] = right  (x+ in camera, in world)
      c2w[:, 1] = down   (y+ in image space, in world)
      c2w[:, 2] = forward (z+ into the scene, in world)  -- actually -z in OpenGL but +z in OpenCV
      c2w[:, 3] = camera center in world

    nerfies Camera convention (HyperNeRF):
      orientation = R_w2c  (world-to-camera, so rows are camera axes expressed as world-to-camera)
      position = C  (camera center in world)
      focal_length: in pixels (matching actual_W)
      principal_point: [cx, cy] in pixels
      image_size: [W, H] actual image size used by pipeline
    """
    # Camera-to-world rotation: columns are [right, down, forward] in world space
    R_c2w = c2w_raw[:3, :3].astype(float)  # (3, 3)
    C = c2w_raw[:3, 3].astype(float)       # (3,) camera center in world

    # World-to-camera rotation (orientation)
    # orientation[i, :] = i-th camera axis expressed as world-to-camera row
    # = R_c2w.T since R_c2w is orthogonal
    R_w2c = R_c2w.T  # (3, 3)

    # Scale focal length if actual image size differs from poses metadata size
    # poses H, W correspond to the full-res images
    scale_x = actual_W / max(W, 1.0)
    scale_y = actual_H / max(H, 1.0)
    focal_scaled_x = focal * scale_x
    # Principal point: assume image center
    cx = actual_W / 2.0
    cy = actual_H / 2.0

    camera_json = {
        "orientation": R_w2c.tolist(),
        "position": C.tolist(),
        "focal_length": float(focal_scaled_x),
        "principal_point": [float(cx), float(cy)],
        "skew": 0.0,
        "pixel_aspect_ratio": 1.0,
        "radial_distortion": [0.0, 0.0, 0.0],
        "tangential_distortion": [0.0, 0.0],
        "image_size": [int(actual_W), int(actual_H)],
    }
    return camera_json


def get_actual_image_size(dataset_dir: Path, cam_name: str = "cam00") -> tuple[int, int]:
    """Get actual image size from cam00/images/ directory."""
    cam_dir = dataset_dir / cam_name / "images"
    imgs = sorted(cam_dir.glob("*.png"))
    if not imgs:
        raise FileNotFoundError(f"No PNG images found under {cam_dir}")
    from PIL import Image as PILImage
    with PILImage.open(imgs[0]) as img:
        W, H = img.size
    return int(W), int(H)


def get_frame_ids(dataset_dir: Path, cam_name: str = "cam00") -> list[str]:
    """Get list of frame image_ids from cam00/images/."""
    cam_dir = dataset_dir / cam_name / "images"
    imgs = sorted(cam_dir.glob("*.png"))
    return [img.stem for img in imgs]


def generate_camera_jsons(
    dataset_dir: Path,
    cam_index: int = 0,
    dry_run: bool = False,
) -> None:
    """Generate camera/XXXX.json files for a dynerf dataset."""
    dataset_dir = Path(dataset_dir)

    if not (dataset_dir / "poses_bounds.npy").exists():
        raise FileNotFoundError(f"poses_bounds.npy not found in {dataset_dir}")

    # Find preferred camera
    cam_candidates = sorted([d.name for d in dataset_dir.glob("cam*") if d.is_dir()])
    if not cam_candidates:
        raise FileNotFoundError(f"No cam* directories found in {dataset_dir}")
    # Prefer cam00
    cam_name = "cam00" if "cam00" in cam_candidates else cam_candidates[0]
    print(f"Using camera: {cam_name}")

    # Load poses
    poses, H_meta, W_meta, focal_meta = load_llff_poses(dataset_dir)
    N_cams = poses.shape[0]
    print(f"N cameras in poses_bounds: {N_cams}")
    print(f"Metadata: H={H_meta:.0f}, W={W_meta:.0f}, focal={focal_meta:.2f}")

    if cam_index >= N_cams:
        print(f"[warn] cam_index={cam_index} >= N_cams={N_cams}, using 0")
        cam_index = 0

    # Get the c2w for the selected camera
    c2w_raw = poses[cam_index, :, :4]  # (3, 4)
    print(f"c2w_raw for cam_index={cam_index}:\n{c2w_raw}")

    # Get actual image size from disk
    actual_W, actual_H = get_actual_image_size(dataset_dir, cam_name)
    print(f"Actual image size: {actual_W}x{actual_H}")

    # Build camera JSON
    camera_json = llff_c2w_to_nerfies_camera(
        c2w_raw, H_meta, W_meta, focal_meta, actual_H, actual_W
    )

    # Get all frame IDs
    frame_ids = get_frame_ids(dataset_dir, cam_name)
    print(f"N frames: {len(frame_ids)}")

    # Output directory
    camera_dir = dataset_dir / "camera"
    if not dry_run:
        camera_dir.mkdir(exist_ok=True)

    # Write one JSON per frame (all identical since camera doesn't move for monocular video)
    for frame_id in frame_ids:
        out_path = camera_dir / f"{frame_id}.json"
        if not dry_run:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(camera_json, f, indent=2)

    if not dry_run:
        print(f"Generated {len(frame_ids)} camera JSON files in {camera_dir}")
        # Verify
        sample = camera_dir / f"{frame_ids[0]}.json"
        loaded = json.loads(sample.read_text())
        print(f"Sample {sample.name}:")
        print(f"  focal_length: {loaded['focal_length']:.2f}")
        print(f"  principal_point: {loaded['principal_point']}")
        print(f"  image_size: {loaded['image_size']}")
    else:
        print(f"[dry-run] Would generate {len(frame_ids)} camera JSONs in {camera_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate dynerf camera JSONs for query pipeline")
    parser.add_argument("--dataset-dir", required=True, help="Path to dynerf dataset directory")
    parser.add_argument("--cam-index", type=int, default=0, help="Camera index in poses_bounds.npy to use (default: 0=cam00)")
    parser.add_argument("--dry-run", action="store_true", help="Don't write files, just print what would be done")
    args = parser.parse_args()

    generate_camera_jsons(
        dataset_dir=Path(args.dataset_dir),
        cam_index=args.cam_index,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
