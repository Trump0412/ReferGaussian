import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = REPO_ROOT / "external" / "4DGaussians"
for candidate in (REPO_ROOT, EXTERNAL_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from refergaussian.semantics import score_native_query


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--query-name", default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--semantic-source", choices=["native", "qwen"], default="native")
    parser.add_argument("--qwen-model", default=None)
    args = parser.parse_args()

    out_dir = score_native_query(
        run_dir=args.run_dir,
        query=args.query,
        query_name=args.query_name,
        top_k=args.top_k,
        semantic_source=args.semantic_source,
        qwen_model=args.qwen_model,
    )
    print(out_dir)


if __name__ == "__main__":
    main()
