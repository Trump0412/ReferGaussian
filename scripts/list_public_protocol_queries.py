#!/usr/bin/env python3
"""List deterministic protocol query ids and text for a selected category."""

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("protocol_json")
parser.add_argument("--category", default="temporal_state_reference")
args = parser.parse_args()

protocol_path = Path(args.protocol_json)
payload = json.loads(protocol_path.read_text(encoding="utf-8"))
for item in payload.get("queries", []):
    if args.category and str(item.get("category", "")) != args.category:
        continue
    slug = str(item.get("query_slug", "")).strip()
    query = str(item.get("query", "")).strip()
    if slug and query:
        print(f"{slug}\t{query}")
