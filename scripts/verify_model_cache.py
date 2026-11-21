"""Verify cached checkpoint blobs and safetensors structure without loading a model."""

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loop_moe.collect import download_model
from loop_moe.trace import read_json


def verify_file(path):
    blob = path.resolve()
    expected = blob.name
    size = path.stat().st_size
    if len(expected) == 64:
        algorithm, digest = "sha256", hashlib.sha256()
    elif len(expected) == 40:
        algorithm, digest = "git-sha1", hashlib.sha1(f"blob {size}\0".encode())
    else:
        raise ValueError(f"Not a content-addressed Hugging Face cache blob: {path}")
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"Checksum mismatch: {path}")
    result = {"filename": path.name, "bytes": size, "algorithm": algorithm, "digest": expected}
    if path.suffix == ".safetensors":
        with path.open("rb") as stream:
            header_length = struct.unpack("<Q", stream.read(8))[0]
            if not 0 < header_length <= min(size - 8, 64 * 1024 * 1024):
                raise ValueError(f"Invalid safetensors header length: {path}")
            header = json.loads(stream.read(header_length))
        tensors = {key: value for key, value in header.items() if key != "__metadata__"}
        data_bytes = size - 8 - header_length
        for key, tensor in tensors.items():
            start, stop = tensor["data_offsets"]
            if not 0 <= start <= stop <= data_bytes:
                raise ValueError(f"Invalid tensor offsets: {path}:{key}")
        result.update(tensor_count=len(tensors), dtypes=sorted({t["dtype"] for t in tensors.values()}))
    elif path.suffix == ".json":
        read_json(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/expanded.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    report = []
    for model in config["models"]:
        checkpoint = download_model(config["checkpoints"][model], True)
        files = [verify_file(path) for path in sorted(checkpoint.rglob("*")) if path.is_file()]
        weight_files = {row["filename"] for row in files if row["filename"].endswith(".safetensors")}
        if not weight_files:
            raise ValueError(f"No weights: {model}")
        index = checkpoint / "model.safetensors.index.json"
        if index.exists() and set(read_json(index)["weight_map"].values()) != weight_files:
            raise ValueError(f"Missing or unexpected weight shards: {model}")
        row = {"model": model, "revision": config["checkpoints"][model]["revision"],
               "checkpoint": str(checkpoint), "files": files,
               "weight_bytes": sum(f["bytes"] for f in files if f["filename"] in weight_files)}
        report.append(row)
        print(f"Verified {model}: {len(weight_files)} weight files, {row['weight_bytes']:,} bytes", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
