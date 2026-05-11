import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = PROJECT_ROOT / "external" / "4DGaussians"
for candidate in (PROJECT_ROOT, EXTERNAL_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from refergaussian.semantics.qwen_query_planner import plan_query_entities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--qwen-model", default=None)
    parser.add_argument("--frame-subsample-stride", type=int, default=10)
    parser.add_argument("--num-sampled-frames", type=int, default=9)
    parser.add_argument("--num-boundary-frames", type=int, default=15)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    payload = plan_query_entities(
        query=args.query,
        dataset_dir=Path(args.dataset_dir),
        output_path=Path(args.output_path),
        qwen_model=args.qwen_model,
        frame_subsample_stride=args.frame_subsample_stride,
        num_sampled_frames=args.num_sampled_frames,
        num_boundary_frames=args.num_boundary_frames,
        strict=bool(args.strict),
    )
    print(payload["query"])
    print(args.output_path)


if __name__ == "__main__":
    main()
