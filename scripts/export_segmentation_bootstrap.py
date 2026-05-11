import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = REPO_ROOT / "external" / "4DGaussians"
for candidate in (REPO_ROOT, EXTERNAL_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from refergaussian.semantics import export_segmentation_bootstrap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    out_path = export_segmentation_bootstrap(args.run_dir, top_k=args.top_k)
    print(out_path)


if __name__ == "__main__":
    main()
