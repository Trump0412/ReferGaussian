import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = PROJECT_ROOT / "external" / "4DGaussians"
for candidate in (PROJECT_ROOT, EXTERNAL_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from refergaussian.semantics import detect_query_proposals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-id", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    output_dir = detect_query_proposals(
        dataset_dir=Path(args.dataset_dir),
        query=args.query,
        output_dir=None if args.output_dir is None else Path(args.output_dir),
        model_id=args.model_id,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        top_k=args.top_k,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=args.device,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
