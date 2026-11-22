"""Shared trace format and JSON/CSV I/O. Uses only the standard library."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path


SCHEMA_VERSION = 1
TRACE_SCOPE = "prefill_then_new_position_per_decode_forward"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_line(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def read_jsonl(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def read_trace(path):
    records = read_jsonl(path)
    meta = next(records, {})
    if (meta.get("schema_version"), meta.get("trace_scope")) != (SCHEMA_VERSION, TRACE_SCOPE):
        raise ValueError(f"Unsupported trace format in {path}")
    requests = list(records)
    if not requests or len(requests) != meta["request_count"]:
        raise ValueError(f"Incomplete trace in {path}")
    return meta, requests


def write_csv(path, columns_or_rows, rows=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows is None:
        rows = list(columns_or_rows)
        columns = list(rows[0].keys()) if rows else []
    else:
        columns = list(columns_or_rows)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        writer.writerows(rows)
