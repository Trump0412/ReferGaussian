import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = REPO_ROOT / "external" / "4DGaussians"
for candidate in (REPO_ROOT, EXTERNAL_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from refergaussian.entitybank import export_entitybank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--sample-ratio", type=float, default=0.02)
    parser.add_argument("--min-cluster-size", type=int, default=10)
    parser.add_argument("--min-gaussians-per-entity", type=int, default=32)
    parser.add_argument("--max-entities", type=int, default=30)
    parser.add_argument("--proposal-dir", default=None)
    parser.add_argument("--proposal-strict", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    out_dir = export_entitybank(
        Path(args.run_dir),
        num_frames=args.num_frames,
        sample_ratio=args.sample_ratio,
        min_cluster_size=args.min_cluster_size,
        min_gaussians_per_entity=args.min_gaussians_per_entity,
        max_entities=args.max_entities,
        proposal_dir=args.proposal_dir,
        proposal_strict=bool(args.proposal_strict),
        output_dir=args.output_dir,
    )
    print(out_dir)


if __name__ == "__main__":
    main()
