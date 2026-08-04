#!/usr/bin/env python3
"""Backward-compatible entry point for the renamed multi-GPU batch runner."""

from run_query_batch import main


if __name__ == "__main__":
    raise SystemExit(main())
