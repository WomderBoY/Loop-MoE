"""Build a fixed, balanced routing workload from public prompt datasets."""

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen


SOURCES = {
    "dolly": {
        "revision": "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
        "url": "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/"
               "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a/databricks-dolly-15k.jsonl",
        "homepage": "https://huggingface.co/datasets/databricks/databricks-dolly-15k",
        "license": "CC-BY-SA-3.0",
    },
    "gsm8k": {
        "revision": "3101c7d5072418e28b9008a6636bde82a006892c",
        "url": "https://raw.githubusercontent.com/openai/grade-school-math/"
               "3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl",
        "homepage": "https://github.com/openai/grade-school-math",
        "license": "MIT",
    },
    "mbpp": {
        "revision": "e2c3f9b85ff9804147541060bf35d68bf14f2258",
        "url": "https://raw.githubusercontent.com/google-research/google-research/"
               "e2c3f9b85ff9804147541060bf35d68bf14f2258/mbpp/sanitized-mbpp.json",
        "homepage": "https://github.com/google-research/google-research/tree/master/mbpp",
        "license": "CC-BY-4.0",
    },
}
SEED = 20261119


def download(item, cache):
    name, source = item
    path = cache / f"{name}.json"
    if not path.exists():
        with urlopen(source["url"], timeout=120) as response:
            path.write_bytes(response.read())
    data = path.read_bytes()
    records = json.loads(data) if name == "mbpp" else [json.loads(line) for line in data.splitlines()]
    return name, records, dict(source, sha256=hashlib.sha256(data).hexdigest())


def sample(records, count, namespace):
    records = sorted(records, key=lambda row: hashlib.sha256(
        f"{SEED}:{namespace}:{row[0]}".encode()).digest())
    # Deduplicate by source text before sampling.
    unique = {}
    for source_id, text in records:
        if 20 <= len(text) <= 1400:
            unique.setdefault(text.strip(), source_id)
    selected = [(source_id, text) for text, source_id in list(unique.items())[:count]]
    if len(selected) != count:
        raise ValueError(f"Not enough eligible prompts for {namespace}")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("configs/prompts_expanded.jsonl"))
    args = parser.parse_args()
    if args.out.exists() or args.out.with_suffix(".sources.json").exists():
        parser.error("Output already exists; use a new path to preserve the fixed workload")
    args.cache.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda item: download(item, args.cache), SOURCES.items()))
    datasets = {name: records for name, records, _ in results}
    sources = {name: info for name, _, info in results}
    groups = {"text": [], "math": [], "code": []}
    for category in ("general_qa", "open_qa", "creative_writing", "brainstorming"):
        candidates = [(str(i), row["instruction"]) for i, row in enumerate(datasets["dolly"])
                      if row["category"] == category and not row.get("context", "").strip()]
        groups["text"].extend(("dolly", source_id, text + "\n\nAnswer:")
                              for source_id, text in sample(candidates, 16, category))
    groups["math"] = [("gsm8k", source_id, text + "\nShow your work.\nSolution:")
                      for source_id, text in sample(
                          [(str(i), row["question"]) for i, row in enumerate(datasets["gsm8k"])],
                          64, "math")]
    groups["code"] = [("mbpp", source_id, text + "\n\nSolution in Python:\n")
                      for source_id, text in sample(
                          [(str(row["task_id"]), row["prompt"]) for row in datasets["mbpp"]
                           if 11 <= row["task_id"] <= 510], 64, "code")]
    prompts = []
    # Interleave categories so each 24-request shard contains 8 of each.
    for i in range(64):
        for category, group in groups.items():
            source, source_id, text = group[i]
            prompts.append({"id": f"expanded-{category}-{i + 1:03d}", "category": category,
                            "text": text, "source": source, "source_id": source_id})
    if len({row["text"] for row in prompts}) != len(prompts):
        raise ValueError("Duplicate prompt text across categories")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prompts),
                        encoding="utf-8")
    manifest = {
        "seed": SEED, "request_count": len(prompts), "category_counts": dict(Counter(
            row["category"] for row in prompts)), "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "sources": sources,
        "selection": "SHA256(seed:namespace:source_id) order; unique 20..1400 character source texts; "
                     "Dolly no-context 16 each general_qa/open_qa/creative_writing/brainstorming; "
                     "GSM8K test 64; sanitized MBPP test task IDs 11..510, 64. Answers/tests excluded.",
        "purpose": "Expert-routing workload only; not a quality benchmark. Dolly subset with answer suffix "
                   "is distributed under CC-BY-SA-3.0; preserve source attribution and share-alike terms.",
    }
    args.out.with_suffix(".sources.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)
                                                    + "\n", encoding="utf-8")
    print(json.dumps({"path": str(args.out), **{key: manifest[key] for key in
                     ("request_count", "category_counts", "sha256")}}, indent=2))


if __name__ == "__main__":
    main()
