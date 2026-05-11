import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = PROJECT_ROOT / "external" / "4DGaussians"
for candidate in (PROJECT_ROOT, EXTERNAL_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from refergaussian.semantics.joint_embedding_cluster import build_joint_query_proposal_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--tracks-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-track-frames", type=int, default=16)
    parser.add_argument("--proposal-keep-ratio", type=float, default=0.10)
    parser.add_argument("--min-gaussians", type=int, default=2048)
    parser.add_argument("--max-gaussians", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1.0e-2)
    args = parser.parse_args()

    out_dir = build_joint_query_proposal_dir(
        run_dir=Path(args.run_dir),
        dataset_dir=Path(args.dataset_dir),
        tracks_path=Path(args.tracks_path),
        output_dir=Path(args.output_dir),
        max_track_frames=int(args.max_track_frames),
        proposal_keep_ratio=float(args.proposal_keep_ratio),
        min_gaussians=int(args.min_gaussians),
        max_gaussians=int(args.max_gaussians),
        chunk_size=int(args.chunk_size),
        embed_dim=int(args.embed_dim),
        num_steps=int(args.num_steps),
        lr=float(args.lr),
    )
    print(out_dir)


if __name__ == "__main__":
    main()
