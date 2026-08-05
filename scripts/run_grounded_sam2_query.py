import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "external" / "Grounded-SAM-2"):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from refergaussian.semantics.grounded_sam2_backend import run_grounded_sam2_query


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--query-plan-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--grounding-model-id", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--sam2-model-id", default="facebook/sam2-hiera-large")
    parser.add_argument("--grounding-model-revision", default=None)
    parser.add_argument("--sam2-model-revision", default=None)
    parser.add_argument("--detector-frame-stride", type=int, default=12)
    parser.add_argument("--max-detector-frames", type=int, default=12)
    parser.add_argument("--detection-top-k", type=int, default=3)
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--prompt-type", choices=["point", "box", "mask"], default="point")
    parser.add_argument("--num-point-prompts", type=int, default=16)
    parser.add_argument("--track-window-radius", type=int, default=120)
    parser.add_argument("--frame-subsample-stride", type=int, default=10)
    parser.add_argument("--num-anchor-seeds", type=int, default=3)
    parser.add_argument("--inference-seed", type=int, default=0)
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load pinned Grounded-SAM2 weights from the local cache only.",
    )
    args = parser.parse_args()

    out_dir = run_grounded_sam2_query(
        dataset_dir=Path(args.dataset_dir),
        query_plan_path=Path(args.query_plan_path),
        output_dir=Path(args.output_dir),
        grounding_model_id=args.grounding_model_id,
        sam2_model_id=args.sam2_model_id,
        grounding_model_revision=args.grounding_model_revision,
        sam2_model_revision=args.sam2_model_revision,
        detector_frame_stride=args.detector_frame_stride,
        max_detector_frames=args.max_detector_frames,
        detection_top_k=args.detection_top_k,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        prompt_type=args.prompt_type,
        num_point_prompts=args.num_point_prompts,
        track_window_radius=args.track_window_radius,
        frame_subsample_stride=args.frame_subsample_stride,
        num_anchor_seeds=args.num_anchor_seeds,
        inference_seed=args.inference_seed,
        local_files_only=args.local_files_only,
    )
    print(out_dir)


if __name__ == "__main__":
    main()
