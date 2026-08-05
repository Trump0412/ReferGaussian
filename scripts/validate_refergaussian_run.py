#!/usr/bin/env python3
"""Reject query evaluation on a run not trained by the released method."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from refergaussian.run_identity import (
    validate_query_ready_baseline_4dgs_run,
    validate_refergaussian_run,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="ReferGaussian training output directory.")
    parser.add_argument(
        "--allow-baseline-4dgs",
        action="store_true",
        help="Accept a query-ready phase=baseline checkpoint under an explicit protocol.",
    )
    args = parser.parse_args()

    errors = validate_refergaussian_run(args.run_dir)
    if errors and args.allow_baseline_4dgs:
        baseline_errors = validate_query_ready_baseline_4dgs_run(args.run_dir)
        if not baseline_errors:
            print(f"[ok] baseline 4DGS run identity verified: {args.run_dir}")
            return 0
        errors = baseline_errors
    if errors:
        print(f"[error] Refusing incompatible query run: {args.run_dir}", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 2
    print(f"[ok] ReferGaussian run identity verified: {args.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
