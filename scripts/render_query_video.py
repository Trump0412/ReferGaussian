import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from refergaussian.semantics import render_hypernerf_query_video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--selection-path", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--background-mode", choices=["render", "source"], default="render")
    args = parser.parse_args()

    output_dir = render_hypernerf_query_video(
        run_dir=Path(args.run_dir),
        dataset_dir=Path(args.dataset_dir),
        selection_path=Path(args.selection_path),
        output_dir=None if args.output_dir is None else Path(args.output_dir),
        fps=args.fps,
        stride=args.stride,
        background_mode=args.background_mode,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
