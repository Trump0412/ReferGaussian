"""Check whether a query can use deterministic declared-instance selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from refergaussian.semantics.instance_contract import declared_multi_instance_group


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-plan-path", required=True)
    parser.add_argument("--tracks-path", required=True)
    parser.add_argument("--entitybank-path", required=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Print 1 when a declared group exists or 0 otherwise, always exiting successfully.",
    )
    args = parser.parse_args()

    group = declared_multi_instance_group(
        _read_json(Path(args.query_plan_path)),
        _read_json(Path(args.tracks_path)),
        _read_json(Path(args.entitybank_path)),
    )
    if group is None:
        if args.probe:
            print("0")
            return
        raise SystemExit(1)
    if args.probe:
        print("1")
        return
    if not args.quiet:
        print(json.dumps(group, sort_keys=True))


if __name__ == "__main__":
    main()
