import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = REPO_ROOT / "external" / "4DGaussians"
for candidate in (REPO_ROOT, EXTERNAL_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from refergaussian.semantics import render_hypernerf_query_video
from refergaussian.semantics.trase_bridge import transfer_trase_semantics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-run-dir", required=True)
    parser.add_argument("--source-model-dir", required=True)
    parser.add_argument("--query-name", default=None)
    parser.add_argument("--min-match-score", type=float, default=0.35)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--render-native-query", action="store_true")
    parser.add_argument("--native-output-dir", default=None)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    out_dir = transfer_trase_semantics(
        target_run_dir=args.target_run_dir,
        source_model_dir=args.source_model_dir,
        query_name=args.query_name,
        min_match_score=args.min_match_score,
    )
    print(out_dir)

    if args.render_native_query:
        if args.query_name is None:
            raise ValueError("--render-native-query requires --query-name")
        if args.dataset_dir is None:
            raise ValueError("--render-native-query requires --dataset-dir")
        selection_path = (
            Path(args.target_run_dir)
            / "entitybank"
            / "trase_bridge"
            / "queries"
            / args.query_name
            / "selected_transferred.json"
        )
        native_output_dir = (
            None
            if args.native_output_dir is None
            else Path(args.native_output_dir)
        )
        native_dir = render_hypernerf_query_video(
            run_dir=args.target_run_dir,
            dataset_dir=args.dataset_dir,
            selection_path=selection_path,
            output_dir=native_output_dir,
            fps=args.fps,
        )
        print(native_dir)


if __name__ == "__main__":
    main()
