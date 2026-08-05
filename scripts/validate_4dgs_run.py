#!/usr/bin/env python3
"""Validate a frozen upstream 4DGS input before referring inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from refergaussian.run_identity import validate_query_ready_4dgs_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Upstream 4DGS output directory.")
    args = parser.parse_args()

    errors = validate_query_ready_4dgs_run(args.run_dir)
    if errors:
        print(f"[error] Refusing incompatible 4DGS input: {args.run_dir}", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 2
    print(f"[ok] query-ready upstream 4DGS input verified: {args.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
