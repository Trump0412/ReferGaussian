import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = REPO_ROOT / "external" / "4DGaussians"
for candidate in (REPO_ROOT, EXTERNAL_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from refergaussian.semantics import export_qwen_semantic_assignments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--qwen-model", default=None)
    parser.add_argument("--max-entities", type=int, default=None)
    parser.add_argument("--query", default=None)
    parser.add_argument("--shortlist-k", type=int, default=24)
    args = parser.parse_args()

    out_path = export_qwen_semantic_assignments(
        run_dir=args.run_dir,
        qwen_model=args.qwen_model,
        max_entities=args.max_entities,
        query=args.query,
        shortlist_k=args.shortlist_k,
    )
    print(out_path)


if __name__ == "__main__":
    main()
