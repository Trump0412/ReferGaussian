#!/usr/bin/env python3
"""Reject query evaluation on a run not trained by the released method."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from refergaussian.run_identity import validate_refergaussian_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="ReferGaussian training output directory.")
    args = parser.parse_args()

    errors = validate_refergaussian_run(args.run_dir)
    if errors:
        print(f"[error] Refusing non-ReferGaussian run: {args.run_dir}", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 2
    print(f"[ok] ReferGaussian run identity verified: {args.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
