#!/usr/bin/env python3
"""
Multi-GPU batch query runner.

Reads a JSONL manifest where each line describes a query to run on a specific GPU.
Each GPU processes its assigned queries serially; GPUs run in parallel.

Manifest fields per line (JSON):
    query_id      : str   — unique identifier for this query
    query         : str   — natural-language query text
    run_dir       : str   — path to the 3D scene run directory
    dataset_dir   : str   — path to the dataset directory
    output_root   : str   — directory where query outputs and traces are written
    gpu           : int   — which GPU to use (0 or 1)
    annotation_dir: str   — (optional) path to annotation directory for diagnostics
"""

from __future__ import annotations

import argparse
import json
import os
import selectors
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SCRIPT = REPO_ROOT / "scripts" / "run_query_specific_worldtube_pipeline.sh"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run queries from a JSONL manifest on one or more GPUs in parallel."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to JSONL manifest file (one JSON object per line).",
    )
    parser.add_argument(
        "--profile",
        default="boundary_shape_v2",
        help="Query eval profile name (default: boundary_shape_v2).",
    )
    parser.add_argument(
        "--gpu",
        nargs="+",
        type=int,
        default=[0, 1],
        help="GPU indices to use (default: 0 1).",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Set QUERY_FORCE_RERUN=1 so the pipeline ignores cached results.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Per-query timeout in seconds (default: 3600, i.e. 1 hour).",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Keep the batch process exit code at 0 even if some queries fail; failures are still logged.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

def load_manifest(path: str) -> list[dict]:
    """Load and validate a JSONL manifest file."""
    items: list[dict] = []
    required = {"query_id", "query", "run_dir", "dataset_dir", "output_root", "gpu"}

    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"[warn] skipping manifest line {lineno}: invalid JSON ({exc})",
                    file=sys.stderr,
                )
                continue

            missing = required - obj.keys()
            if missing:
                print(
                    f"[warn] skipping manifest line {lineno}: missing keys {sorted(missing)}",
                    file=sys.stderr,
                )
                continue

            items.append(obj)

    return items


def group_by_gpu(items: list[dict], gpus: list[int]) -> dict[int, list[dict]]:
    """Partition manifest items by their assigned GPU."""
    groups: dict[int, list[dict]] = {g: [] for g in gpus}
    for item in items:
        gpu = item["gpu"]
        if gpu not in groups:
            print(
                f"[warn] query '{item['query_id']}' assigned to GPU {gpu} "
                f"which is not in the active GPU list; skipping",
                file=sys.stderr,
            )
            continue
        groups[gpu].append(item)
    return groups


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    """Return a UTC timestamp string for logging."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _ensure_dir(path: str) -> None:
    """Create a directory (and parents) if it does not exist."""
    os.makedirs(path, exist_ok=True)


def _write_json_atomic(path: str, payload: object) -> None:
    """Atomically write JSON to *path* via a temp file + rename."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def _append_jsonl(path: str, record: dict) -> None:
    """Append a single JSON record as a line to a JSONL file."""
    _ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


# ---------------------------------------------------------------------------
# Single-query execution
# ---------------------------------------------------------------------------

def run_one_query(
    item: dict,
    *,
    profile: str,
    force_rerun: bool,
    timeout: int,
) -> dict:
    """Execute one query via the pipeline shell script.

    Returns an efficiency-trace dictionary.
    """
    query_id: str = item["query_id"]
    query_text: str = item["query"]
    run_dir: str = item["run_dir"]
    dataset_dir: str = item["dataset_dir"]
    output_root: str = item["output_root"]
    gpu: int = item["gpu"]
    annotation_dir: str = item.get("annotation_dir", "")

    query_output_root = os.path.join(output_root, query_id)
    mllm_trace_path = os.path.join(query_output_root, "mllm_trace.jsonl")
    log_dir = os.path.join(output_root, "logs")
    log_file = os.path.join(log_dir, f"gpu{gpu}_{query_id}.log")
    item_profile = str(item.get("profile") or profile)
    item_env = item.get("env") if isinstance(item.get("env"), dict) else {}

    _ensure_dir(log_dir)
    _ensure_dir(query_output_root)

    # Build environment
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["QUERY_EVAL_PROFILE"] = item_profile
    if force_rerun:
        env["QUERY_FORCE_RERUN"] = "1"
    env["QUERY_OUTPUT_ROOT_OVERRIDE"] = query_output_root
    env["QUERY_GPU_LOCK_FILE"] = f"/tmp/refergaussian_query_gpu{gpu}.lock"
    env["MLLM_TRACE_PATH"] = mllm_trace_path
    if annotation_dir:
        env["QUERY_ANNOTATION_DIR"] = annotation_dir
    for key, value in item_env.items():
        if key:
            env[str(key)] = str(value)

    cmd = [
        "bash",
        str(PIPELINE_SCRIPT),
        run_dir,
        dataset_dir,
        query_text,
        query_id,
    ]

    # Log preamble
    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write(f"\n{'=' * 60}\n")
        lf.write(f"[{_utc_now()}] START  query_id={query_id}  gpu={gpu}\n")
        lf.write(f"[{_utc_now()}] CMD   {' '.join(cmd)}\n")
        lf.write(f"[{_utc_now()}] CWD   {REPO_ROOT}\n")
        lf.write(f"[{_utc_now()}] ENV   CUDA_VISIBLE_DEVICES={gpu}  "
                 f"QUERY_EVAL_PROFILE={item_profile}  force_rerun={force_rerun}\n")
        if item_env:
            lf.write(f"[{_utc_now()}] ENV_OVERRIDES {json.dumps(item_env, ensure_ascii=False)}\n")
        lf.write(f"{'=' * 60}\n\n")
        lf.flush()

    start = time.monotonic()
    started_at_utc = _utc_now()
    exit_code: int = -1
    error_msg: str = ""

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        deadline = start + float(timeout)

        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write("--- PIPELINE OUTPUT ---\n")
            lf.flush()
            if proc.stdout is not None:
                selector = selectors.DefaultSelector()
                selector.register(proc.stdout, selectors.EVENT_READ)
                while True:
                    if time.monotonic() > deadline:
                        try:
                            os.killpg(proc.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            try:
                                os.killpg(proc.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            proc.wait(timeout=10)
                        raise subprocess.TimeoutExpired(cmd, timeout)
                    ready = selector.select(timeout=0.5)
                    for key, _mask in ready:
                        line = key.fileobj.readline()
                        if line:
                            lf.write(line)
                            lf.flush()
                    if proc.poll() is not None:
                        while True:
                            line = proc.stdout.readline()
                            if not line:
                                break
                            lf.write(line)
                        break
                selector.close()
            exit_code = int(proc.wait())
            lf.write(f"\n[{_utc_now()}] EXIT  query_id={query_id}  "
                     f"exit_code={exit_code}  elapsed={time.monotonic() - start:.1f}s\n")

    except subprocess.TimeoutExpired:
        exit_code = -2
        error_msg = f"timeout after {timeout}s"
        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write(f"\n[{_utc_now()}] TIMEOUT  query_id={query_id}  "
                     f"limit={timeout}s\n")
        print(f"[{_utc_now()}] GPU{gpu} TIMEOUT {query_id} after {timeout}s",
              file=sys.stderr)

    except FileNotFoundError:
        exit_code = -3
        error_msg = f"pipeline script not found: {PIPELINE_SCRIPT}"
        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write(f"\n[{_utc_now()}] ERROR  query_id={query_id}: {error_msg}\n")

    except Exception as exc:
        exit_code = -4
        error_msg = str(exc)
        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write(f"\n[{_utc_now()}] EXCEPTION  query_id={query_id}: {error_msg}\n")

    elapsed = round(time.monotonic() - start, 3)

    trace: dict = {
        "query_id": query_id,
        "gpu": gpu,
        "returncode": exit_code,
        "exit_code": exit_code,
        "elapsed_seconds": elapsed,
        "error": error_msg,
        "run_dir": run_dir,
        "dataset_dir": dataset_dir,
        "output_root": query_output_root,
        "log_path": log_file,
        "profile": item_profile,
        "force_rerun": bool(force_rerun),
        "env_overrides": item_env,
        "started_at_utc": started_at_utc,
        "finished_at_utc": _utc_now(),
    }
    return trace


# ---------------------------------------------------------------------------
# GPU worker thread
# ---------------------------------------------------------------------------

def _gpu_worker(
    gpu: int,
    items: list[dict],
    *,
    profile: str,
    force_rerun: bool,
    timeout: int,
    results: list[dict],
    lock: threading.Lock,
) -> None:
    """Process every query assigned to *gpu* serially."""
    for idx, item in enumerate(items, 1):
        query_id = item["query_id"]
        print(f"[{_utc_now()}] GPU{gpu} [{idx}/{len(items)}] starting {query_id}")

        trace = run_one_query(
            item,
            profile=profile,
            force_rerun=force_rerun,
            timeout=timeout,
        )

        with lock:
            results.append(trace)
            # Write efficiency trace into the manifest-level output_root
            _append_jsonl(
                os.path.join(item["output_root"], "efficiency_trace.jsonl"),
                trace,
            )

        status = "OK" if trace["exit_code"] == 0 else f"FAIL(rc={trace['exit_code']})"
        print(
            f"[{_utc_now()}] GPU{gpu} [{idx}/{len(items)}] finished {query_id}  "
            f"{status}  {trace['elapsed_seconds']:.1f}s"
            + (f"  error: {trace['error']}" if trace["error"] else "")
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    # Pre-flight: does the pipeline script exist?
    if not PIPELINE_SCRIPT.exists():
        print(f"ERROR: pipeline script not found: {PIPELINE_SCRIPT}", file=sys.stderr)
        return 1

    # Load manifest
    items = load_manifest(args.manifest)
    if not items:
        print("ERROR: no valid items in manifest", file=sys.stderr)
        return 1

    # Partition by GPU
    groups = group_by_gpu(items, args.gpu)

    print(f"Loaded {len(items)} queries from {args.manifest}")
    print(f"GPUs: {args.gpu}")
    print(f"Profile: {args.profile}")
    print(f"Force rerun: {args.force_rerun}")
    print(f"Per-query timeout: {args.timeout}s")
    for gpu, gpu_items in groups.items():
        print(f"  GPU {gpu}: {len(gpu_items)} queries")
    print()

    results: list[dict] = []
    lock = threading.Lock()
    started_at_utc = _utc_now()
    t0 = time.monotonic()

    # Launch one thread per GPU
    threads: list[threading.Thread] = []
    for gpu, gpu_items in groups.items():
        if not gpu_items:
            continue
        t = threading.Thread(
            target=_gpu_worker,
            args=(gpu, gpu_items),
            kwargs={
                "profile": args.profile,
                "force_rerun": args.force_rerun,
                "timeout": args.timeout,
                "results": results,
                "lock": lock,
            },
            daemon=False,
        )
        t.start()
        threads.append(t)

    # Wait for all workers to finish
    for t in threads:
        t.join()

    total_elapsed = round(time.monotonic() - t0, 1)

    # ---- Summary ----
    succeeded = sum(1 for r in results if r["exit_code"] == 0)
    failed = len(results) - succeeded

    print()
    print("=" * 60)
    print("BATCH COMPLETE")
    print(f"  Total queries : {len(results)}")
    print(f"  Succeeded     : {succeeded}")
    print(f"  Failed        : {failed}")
    print(f"  Wall time     : {total_elapsed:.1f}s")
    print("=" * 60)

    # Per-query table
    if results:
        print()
        print(f"{'Status':>6s}  {'Query ID':<40s}  {'GPU':>3s}  {'Time (s)':>10s}  Error")
        print("-" * 90)
        for r in sorted(results, key=lambda x: x["query_id"]):
            status = "OK" if r["exit_code"] == 0 else f"RC={r['exit_code']}"
            print(
                f"{status:>6s}  {r['query_id']:<40s}  {r['gpu']:>3d}  "
                f"{r['elapsed_seconds']:>10.1f}  {r['error'] or ''}"
            )

    # Write a batch-summary JSON per unique manifest-level output_root
    unique_roots: set[str] = {item["output_root"] for item in items}
    for root in unique_roots:
        _ensure_dir(root)
        summary = {
            "total_queries": len(results),
            "succeeded": succeeded,
            "failed": failed,
            "total_elapsed_seconds": total_elapsed,
            "gpus": args.gpu,
            "profile": args.profile,
            "force_rerun": args.force_rerun,
            "started_at_utc": started_at_utc,
            "finished_at_utc": _utc_now(),
            "allow_failures": bool(args.allow_failures),
            "results": sorted(results, key=lambda row: str(row.get("query_id", ""))),
        }
        _write_json_atomic(os.path.join(root, "batch_summary.json"), summary)

    return 0 if failed == 0 or args.allow_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
