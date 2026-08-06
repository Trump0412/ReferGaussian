#!/usr/bin/env python3
"""Run ReferGaussian and report Acc/vIoU on one released benchmark."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PUBLIC_PROFILE = "public_time_boundary_gated_v5_numeric"
PUBLIC_PROTOCOL = "release_public4_extension"
R4D_PROFILE = "r4d_renderer_consistent"
R4D_PROTOCOL = "release_r4d_dense89_renderer_consistent"
PUBLIC_SCENES = {
    "americano": ("misc", "hypernerf/misc/americano"),
    "chickchicken": ("interp", "hypernerf/interp/chickchicken"),
    "espresso": ("misc", "hypernerf/misc/espresso"),
    "split-cookie": ("misc", "hypernerf/misc/split-cookie"),
}


@dataclass(frozen=True)
class BenchmarkPlan:
    benchmark: str
    output_dir: Path
    result_json: Path
    result_md: Path
    commands: tuple[tuple[str, ...], ...]


def _python(script: str, *args: object) -> tuple[str, ...]:
    return (sys.executable, str(SCRIPTS / script), *(str(value) for value in args))


def _public_plan(args: argparse.Namespace, output: Path) -> BenchmarkPlan:
    annotation_root = (
        Path(args.annotation_root).expanduser().resolve()
        if args.annotation_root
        else args.data_root / "benchmarks" / "4dlangsplat" / "HyperNeRF-Annotation"
    )
    protocol = output / "protocol.json"
    manifest = output / "manifest.jsonl"
    query_root = output / "query_root"
    commands: list[tuple[str, ...]] = [
        _python(
            "build_4dlangsplat_query_protocol.py",
            "--annotation-root",
            annotation_root,
            "--output-json",
            protocol,
        ),
        _python(
            "build_public_query_manifest.py",
            "--output",
            manifest,
            "--output-root",
            query_root,
            "--protocol-json",
            protocol,
            "--protocol-id",
            PUBLIC_PROTOCOL,
            "--annotation-root",
            annotation_root,
            "--profile",
            PUBLIC_PROFILE,
            "--run-root",
            args.run_root,
            "--data-root",
            args.data_root,
            "--run-namespace",
            args.run_namespace,
            "--gpus",
            *args.gpus,
        ),
        _python(
            "preflight_query_batch.py",
            "--manifest",
            manifest,
            "--protocol-id",
            PUBLIC_PROTOCOL,
            "--profile",
            PUBLIC_PROFILE,
            "--strict-release",
            "--gpu",
            *args.gpus,
            "--require-visible-gpu",
            "--create-output-root",
        ),
    ]
    run_command = list(
        _python(
            "run_query_batch.py",
            "--manifest",
            manifest,
            "--protocol-id",
            PUBLIC_PROTOCOL,
            "--profile",
            PUBLIC_PROFILE,
            "--gpu",
            *args.gpus,
            "--strict-release",
            "--timeout",
            args.timeout,
        )
    )
    if args.force:
        run_command.append("--force-rerun")
    commands.append(tuple(run_command))

    scene_results: list[Path] = []
    for scene, (_group, dataset_relative) in PUBLIC_SCENES.items():
        scene_result = output / f"{scene}_eval.json"
        scene_results.append(scene_result)
        commands.append(
            _python(
                "evaluate_public_query_protocol.py",
                "--protocol-json",
                protocol,
                "--query-manifest",
                manifest,
                "--annotation-dir",
                annotation_root / scene,
                "--dataset-dir",
                args.data_root / dataset_relative,
                "--query-root",
                query_root,
                "--scene",
                scene,
                "--category",
                "temporal_state_reference",
                "--require-complete",
                "--output-json",
                scene_result,
                "--output-md",
                output / f"{scene}_eval.md",
            )
        )

    result_json = output / "official_eval.json"
    result_md = output / "official_eval.md"
    commands.append(
        _python(
            "aggregate_public_query_evaluations.py",
            "--inputs",
            *scene_results,
            "--expected-queries",
            9,
            "--require-complete",
            "--output-json",
            result_json,
            "--output-md",
            result_md,
        )
    )
    return BenchmarkPlan("4dlangsplat", output, result_json, result_md, tuple(commands))


def _r4d_plan(args: argparse.Namespace, output: Path) -> BenchmarkPlan:
    benchmark_root = (
        Path(args.benchmark_root).expanduser().resolve()
        if args.benchmark_root
        else args.data_root / "benchmarks" / "r4d_bench_qa"
    )
    benchmark = benchmark_root / "benchmark_all_queries.json"
    query_metadata = benchmark_root / "evaluation" / "R4D-Bench_queries.json"
    manifest = output / "manifest.jsonl"
    query_root = output / "query_root"
    camera_root = output / "source_camera_views"
    commands: list[tuple[str, ...]] = [
        _python(
            "build_r4d_query_manifest.py",
            "--benchmark",
            benchmark,
            "--query-metadata",
            query_metadata,
            "--protocol-id",
            R4D_PROTOCOL,
            "--profile",
            R4D_PROFILE,
            "--output",
            manifest,
            "--output-root",
            query_root,
            "--run-root",
            args.run_root,
            "--data-root",
            args.data_root,
            "--run-namespace",
            args.run_namespace,
            "--gpus",
            *args.gpus,
        ),
        _python(
            "preflight_query_batch.py",
            "--manifest",
            manifest,
            "--protocol-id",
            R4D_PROTOCOL,
            "--profile",
            R4D_PROFILE,
            "--strict-release",
            "--gpu",
            *args.gpus,
            "--require-visible-gpu",
            "--create-output-root",
        ),
    ]
    run_command = list(
        _python(
            "run_query_batch.py",
            "--manifest",
            manifest,
            "--protocol-id",
            R4D_PROTOCOL,
            "--profile",
            R4D_PROFILE,
            "--gpu",
            *args.gpus,
            "--strict-release",
            "--timeout",
            args.timeout,
        )
    )
    if args.force:
        run_command.append("--force-rerun")
    commands.append(tuple(run_command))

    render_command = list(
        _python(
            "rerender_query_outputs.py",
            "--manifest",
            manifest,
            "--benchmark",
            benchmark,
            "--output-root",
            camera_root,
            "--profile",
            R4D_PROFILE,
            "--gpu",
            args.gpus[0],
            "--require-complete",
        )
    )
    if args.force:
        render_command.append("--force")
    commands.append(tuple(render_command))

    result_json = camera_root / "official_eval.json"
    result_md = camera_root / "official_eval.md"
    commands.append(
        _python(
            "evaluate_ours_benchmark.py",
            "--benchmark",
            benchmark,
            "--query-root-map",
            camera_root / "query_root_map.json",
            "--dataset-dir-map",
            camera_root / "dataset_dir_map.json",
            "--query-manifest",
            manifest,
            "--output-json",
            result_json,
            "--output-md",
            result_md,
            "--require-complete",
        )
    )
    return BenchmarkPlan("r4d-bench", output, result_json, result_md, tuple(commands))


def build_plan(args: argparse.Namespace) -> BenchmarkPlan:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (ROOT / "reports" / f"{args.benchmark}_{stamp}").resolve()
    )
    return _public_plan(args, output) if args.benchmark == "4dlangsplat" else _r4d_plan(args, output)


def _format_percent(value: object) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _print_result(plan: BenchmarkPlan) -> None:
    payload = json.loads(plan.result_json.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    tiou = summary.get("temporal_tIoU", summary.get("tIoU"))
    print("\n=== ReferGaussian result ===")
    print(f"Benchmark: {plan.benchmark}")
    print(f"Acc:  {_format_percent(summary.get('Acc'))}")
    print(f"vIoU: {_format_percent(summary.get('vIoU'))}")
    if tiou is not None:
        print(f"tIoU: {_format_percent(tiou)}")
    print(f"Report: {plan.result_md}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=("4dlangsplat", "r4d-bench"))
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("GS_DATA_ROOT", ROOT / "data")))
    parser.add_argument("--run-root", type=Path, default=Path(os.environ.get("GS_RUN_ROOT", ROOT / "runs")))
    parser.add_argument("--run-namespace", default="baseline_4dgs")
    parser.add_argument("--annotation-root", default=None, help="Override 4DLangSplat annotations.")
    parser.add_argument("--benchmark-root", default=None, help="Override R4D-Bench-QA files.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--timeout", type=int, default=3600, help="Per-query timeout in seconds.")
    parser.add_argument("--force", action="store_true", help="Replace existing query outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Print the exact pipeline without running it.")
    args = parser.parse_args()
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
        parser.error("--gpus requires one or more unique non-negative ids")
    args.data_root = args.data_root.expanduser().resolve()
    args.run_root = args.run_root.expanduser().resolve()
    plan = build_plan(args)

    if not args.dry_run and plan.output_dir.exists() and any(plan.output_dir.iterdir()):
        parser.error(f"output directory is not empty: {plan.output_dir}")

    for command in plan.commands:
        print("+ " + shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)
    if args.dry_run:
        print(f"Result: {plan.result_md}")
        return 0
    _print_result(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
