#!/usr/bin/env python3
"""Train and test-render a standard upstream 4DGS scene for ReferGaussian."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "external" / "4DGaussians"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from refergaussian.run_identity import validate_query_ready_4dgs_run


def _default_data_root() -> Path:
    return Path(os.environ.get("GS_DATA_ROOT", ROOT / "data")).expanduser().resolve()


def _default_run_root() -> Path:
    return Path(os.environ.get("GS_RUN_ROOT", ROOT / "runs")).expanduser().resolve()


def _config_path(dataset: str, scene: str, override: str | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    scene_name = Path(scene).name
    if dataset == "hypernerf":
        scene_name = {"chickchicken": "chicken", "slice-banana": "banana"}.get(
            scene_name, scene_name
        )
    candidate = UPSTREAM / "arguments" / dataset / f"{scene_name}.py"
    if candidate.is_file():
        return candidate
    return UPSTREAM / "arguments" / dataset / "default.py"


def build_commands(args: argparse.Namespace) -> tuple[Path, Path, list[list[str]]]:
    source = (
        Path(args.source).expanduser().resolve()
        if args.source
        else args.data_root / args.dataset / args.scene
    )
    scene_name = Path(args.scene).name
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else args.run_root / "baseline_4dgs" / args.dataset / scene_name
    )
    config = _config_path(args.dataset, args.scene, args.config)
    train = [
        sys.executable,
        str(UPSTREAM / "train.py"),
        "-s",
        str(source),
        "-m",
        str(output),
        "--expname",
        f"{args.dataset}/{scene_name}",
        "--configs",
        str(config),
        "--port",
        str(args.port),
    ]
    render = [
        sys.executable,
        str(UPSTREAM / "render.py"),
        "--model_path",
        str(output),
        "--configs",
        str(config),
        "--skip_train",
    ]
    commands = [render] if args.render_only else [train, render]
    return source, output, commands


def _print_command(command: list[str]) -> None:
    print("+ " + shlex.join(command), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("hypernerf", "dynerf"), required=True)
    parser.add_argument(
        "--scene",
        required=True,
        help="Scene path below the dataset root, for example misc/americano or coffee_martini.",
    )
    parser.add_argument("--source", default=None, help="Optional explicit scene directory.")
    parser.add_argument("--output", default=None, help="Optional explicit 4DGS output directory.")
    parser.add_argument("--config", default=None, help="Optional upstream 4DGS config file.")
    parser.add_argument("--data-root", type=Path, default=_default_data_root())
    parser.add_argument("--run-root", type=Path, default=_default_run_root())
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Skip training and generate the test renders required by query inference.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    args = parser.parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.run_root = args.run_root.expanduser().resolve()

    source, output, commands = build_commands(args)
    if not (UPSTREAM / "train.py").is_file():
        parser.error("missing external/4DGaussians; run: bash scripts/setup.sh")
    config = Path(commands[0][commands[0].index("--configs") + 1])
    if not config.is_file():
        parser.error(f"4DGS config does not exist: {config}")
    if not args.dry_run and not source.is_dir():
        parser.error(f"scene directory does not exist: {source}")

    if not args.render_only and not args.dry_run and output.exists():
        errors = validate_query_ready_4dgs_run(output)
        if not errors:
            print(f"[ready] {output}")
            return 0
        parser.error(
            f"output already exists but is not query-ready: {output}. "
            "Resume upstream training explicitly or use --render-only for a completed model."
        )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    pythonpath = [str(ROOT), str(UPSTREAM)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    for command in commands:
        _print_command(command)
        if not args.dry_run:
            subprocess.run(command, cwd=UPSTREAM, env=env, check=True)

    if args.dry_run:
        print(f"4DGS output: {output}")
        return 0

    errors = validate_query_ready_4dgs_run(output)
    if errors:
        raise SystemExit("4DGS run is not query-ready:\n- " + "\n- ".join(errors))
    print(f"[ready] ReferGaussian 4DGS input: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
