#!/usr/bin/env python3
"""Verify that Grounded-SAM2 imports from this checkout, not another project."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grounded-sam2-root", required=True)
    args = parser.parse_args()

    root = Path(args.grounded_sam2_root).resolve()
    package_init = root / "sam2" / "__init__.py"
    if not package_init.is_file():
        parser.error(f"Grounded-SAM2 checkout is incomplete: {package_init}")

    sys.path.insert(0, str(root))
    import sam2

    package_file = Path(sam2.__file__).resolve()
    if not _inside(package_file, root):
        parser.error(
            "sam2 resolved outside the requested Grounded-SAM2 checkout: "
            f"{package_file}. Re-run scripts/setup_grounded_sam2.sh."
        )

    extension_spec = importlib.util.find_spec("sam2._C")
    extension_origin = None if extension_spec is None else extension_spec.origin
    if extension_origin and not _inside(Path(extension_origin), root):
        parser.error(
            "sam2._C resolved outside the requested Grounded-SAM2 checkout: "
            f"{extension_origin}. Re-run scripts/setup_grounded_sam2.sh."
        )

    print(f"[ok] sam2 package: {package_file}")
    print(f"[ok] sam2 extension: {extension_origin or 'not built'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
