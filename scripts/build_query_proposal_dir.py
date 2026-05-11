import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = PROJECT_ROOT / "external" / "4DGaussians"
for candidate in (PROJECT_ROOT, EXTERNAL_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from refergaussian.semantics.query_proposal_bridge import build_query_proposal_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--tracks-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-track-frames", type=int, default=16)
    parser.add_argument("--proposal-keep-ratio", type=float, default=0.03)
    parser.add_argument("--min-gaussians", type=int, default=256)
    parser.add_argument("--max-gaussians", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--opacity-power", type=float, default=0.0)
    parser.add_argument("--cluster-mode", choices=["support_only", "worldtube_consistency"], default="support_only")
    parser.add_argument("--seed-ratio", type=float, default=0.05)
    parser.add_argument("--expansion-factor", type=float, default=4.0)
    args = parser.parse_args()

    out_dir = build_query_proposal_dir(
        run_dir=Path(args.run_dir),
        dataset_dir=Path(args.dataset_dir),
        tracks_path=Path(args.tracks_path),
        output_dir=Path(args.output_dir),
        max_track_frames=args.max_track_frames,
        proposal_keep_ratio=args.proposal_keep_ratio,
        min_gaussians=args.min_gaussians,
        max_gaussians=args.max_gaussians,
        chunk_size=args.chunk_size,
        opacity_power=args.opacity_power,
        cluster_mode=args.cluster_mode,
        seed_ratio=args.seed_ratio,
        expansion_factor=args.expansion_factor,
    )
    print(out_dir)


if __name__ == "__main__":
    main()
