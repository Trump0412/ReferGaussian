import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = PROJECT_ROOT / "external" / "4DGaussians"
for candidate in (PROJECT_ROOT, EXTERNAL_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from refergaussian.semantics.qwen_pair_query import refine_qwen_query_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--query-dir", required=True)
    parser.add_argument("--qwen-model", default=None)
    parser.add_argument("--top-pairs", type=int, default=12)
    args = parser.parse_args()

    output_path = refine_qwen_query_pairs(
        run_dir=args.run_dir,
        query_dir=args.query_dir,
        qwen_model=args.qwen_model,
        top_pairs=args.top_pairs,
    )
    print(output_path)


if __name__ == "__main__":
    main()
