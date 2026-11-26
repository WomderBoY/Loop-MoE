"""Dataset/model asset downloading and deterministic benchmark builders."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


DATASET_SPECS: dict[str, tuple[str, list[str]]] = {
    "2wiki": (
        "framolfese/2WikiMultihopQA",
        ["data/validation-00000-of-00001.parquet"],
    ),
    "wikitext": (
        "Salesforce/wikitext",
        [
            "wikitext-2-raw-v1/train-00000-of-00001.parquet",
            "wikitext-2-raw-v1/validation-00000-of-00001.parquet",
            "wikitext-2-raw-v1/test-00000-of-00001.parquet",
        ],
    ),
    "humaneval": (
        "openai/openai_humaneval",
        ["openai_humaneval/test-00000-of-00001.parquet"],
    ),
}

DYNAMIC_DATASET_REPOS: dict[str, tuple[str, list[str]]] = {
    "gsm8k": ("openai/gsm8k", ["main/train", "main/test"]),
    "math500": ("HuggingFaceH4/MATH-500", ["test"]),
    "arc_challenge": ("allenai/ai2_arc", ["ARC-Challenge/test", "ARC-Challenge/validation"]),
    "arc_easy": ("allenai/ai2_arc", ["ARC-Easy/test"]),
    "openbookqa": ("allenai/openbookqa", ["main/test"]),
}

MODEL_REPOS: dict[str, tuple[str, str]] = {
    "ouro14b": ("ByteDance/Ouro-1.4B", "ByteDance__Ouro-1.4B"),
    "ouro14b_thinking": ("ByteDance/Ouro-1.4B-Thinking", "ByteDance__Ouro-1.4B-Thinking"),
    "ouro26b_thinking": ("ByteDance/Ouro-2.6B-Thinking", "ByteDance__Ouro-2.6B-Thinking"),
    "ifm_ouro": ("IFM/LoopedLM-P1-moe-ouro-500b", "IFM__LoopedLM-P1-moe-ouro-500b"),
    "ifm_huginn": ("IFM/LoopedLM-P1-moe-huginn-500b", "IFM__LoopedLM-P1-moe-huginn-500b"),
    "inkling2_mini": ("thinkingmachines/Inkling-2-Mini", "thinkingmachines__Inkling-2-Mini"),
}

# Models with verified high-speed ModelScope Alibaba Cloud CDN mirrors for large weight shards
MODELSCOPE_LARGE_SHARDS: dict[str, list[tuple[str, int]]] = {
    "inkling2_mini": [
        ("model-00001-of-00001.safetensors", 19526019168),
        ("mtp.safetensors", 635801416),
    ],
}


def _probe_remote_range_size(url: str, timeout: int = 15) -> int:
    """Probe exact remote file size via HTTP Range: bytes=0-0 (works through 302 CDN redirects)."""
    import urllib.request

    req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "loop-moe-downloader/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        cr = resp.headers.get("Content-Range", "")
        m = re.search(r"/(\d+)\s*$", cr)
        if m:
            return int(m.group(1))
        cl = resp.headers.get("Content-Length")
        if cl and resp.status == 200:
            return int(cl)
    raise RuntimeError(f"Could not determine remote file size via Range probe for {url}")


def fast_range_download_file(
    urls: list[str],
    dest: Path,
    *,
    num_workers: int = 16,
    chunk_size_bytes: int = 16 * 1024 * 1024,
    expected_size: int | None = None,
    log_interval_sec: float = 5.0,
) -> Path:
    """High-speed resumable multi-connection HTTP Range downloader (os.pwrite, zero RAM blowup).

    Tries each URL in `urls` in order (e.g., ModelScope Alibaba Cloud CDN first, then hf-mirror).
    Logs live throughput (MB/s) and ETA every `log_interval_sec` seconds.
    """
    import concurrent.futures
    import threading
    import time
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and expected_size is not None and dest.stat().st_size == expected_size:
        print(f"[download] Fast shard already cached: {dest.name} ({expected_size / (1024**3):.2f} GiB)", flush=True)
        return dest

    chosen_url: str | None = None
    total_size: int = 0
    last_err: Exception | None = None
    for u in urls:
        try:
            sz = _probe_remote_range_size(u)
            if expected_size is not None and sz != expected_size:
                print(f"[download] Warning: {u} size {sz} != expected {expected_size}; trying next mirror.", flush=True)
                continue
            chosen_url = u
            total_size = sz
            break
        except Exception as exc:
            last_err = exc
            print(f"[download] Probe failed for {u}: {exc}", flush=True)

    if chosen_url is None:
        raise RuntimeError(f"All mirrors failed Range probe for {dest.name}: {last_err}")

    if dest.is_file() and dest.stat().st_size == total_size:
        print(f"[download] Fast shard already cached: {dest.name} ({total_size / (1024**3):.2f} GiB)", flush=True)
        return dest

    part_path = dest.with_name(dest.name + ".part")
    state_path = dest.with_name(dest.name + ".chunks.json")

    chunks: list[tuple[int, int, int]] = []
    cid = 0
    for start in range(0, total_size, chunk_size_bytes):
        end = min(total_size - 1, start + chunk_size_bytes - 1)
        chunks.append((cid, start, end))
        cid += 1

    done_ids: set[int] = set()
    if part_path.is_file() and state_path.is_file():
        try:
            st = json.loads(state_path.read_text(encoding="utf-8"))
            if st.get("total_size") == total_size and st.get("chunk_size") == chunk_size_bytes:
                done_ids = set(int(i) for i in st.get("done", []))
        except Exception:
            done_ids = set()

    # Pre-create / open sparse part file for parallel os.pwrite
    fd = os.open(str(part_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        os.ftruncate(fd, total_size)
        pending = [c for c in chunks if c[0] not in done_ids]
        already_bytes = sum((e - s + 1) for i, s, e in chunks if i in done_ids)
        print(
            f"[download] Fast {num_workers}-thread Range download: {dest.name} "
            f"({total_size / (1024**3):.2f} GiB, {len(done_ids)}/{len(chunks)} chunks cached) <- {chosen_url}",
            flush=True,
        )

        lock = threading.Lock()
        bytes_new = 0
        t_start = time.monotonic()
        t_last = t_start
        stop_flag = threading.Event()

        def save_state() -> None:
            tmp_st = state_path.with_name(state_path.name + ".tmp")
            tmp_st.write_text(
                json.dumps({"total_size": total_size, "chunk_size": chunk_size_bytes, "done": sorted(done_ids)}),
                encoding="utf-8",
            )
            os.replace(tmp_st, state_path)

        def download_chunk(item: tuple[int, int, int]) -> None:
            nonlocal bytes_new, t_last
            chunk_id, s_off, e_off = item
            want = e_off - s_off + 1
            for attempt in range(6):
                if stop_flag.is_set():
                    return
                try:
                    req = urllib.request.Request(
                        chosen_url,
                        headers={"Range": f"bytes={s_off}-{e_off}", "User-Agent": "loop-moe-downloader/1.0"},
                    )
                    with urllib.request.urlopen(req, timeout=45) as resp:
                        buf = bytearray()
                        while len(buf) < want:
                            blk = resp.read(min(1024 * 1024, want - len(buf)))
                            if not blk:
                                break
                            buf.extend(blk)
                    if len(buf) != want:
                        raise IOError(f"Short read chunk {chunk_id}: got {len(buf)}/{want}")
                    # Positional write directly to offset (thread-safe POSIX pwrite)
                    written = 0
                    mv = memoryview(buf)
                    while written < want:
                        n_w = os.pwrite(fd, mv[written:], s_off + written)
                        if n_w <= 0:
                            raise IOError(f"pwrite returned {n_w}")
                        written += n_w
                    with lock:
                        done_ids.add(chunk_id)
                        bytes_new += want
                        now = time.monotonic()
                        if now - t_last >= log_interval_sec or len(done_ids) == len(chunks):
                            t_last = now
                            cur_tot = already_bytes + bytes_new
                            pct = 100.0 * cur_tot / max(total_size, 1)
                            rate_mb = (bytes_new / (1024.0 * 1024.0)) / max(now - t_start, 1e-3)
                            rem_s = ((total_size - cur_tot) / (1024.0 * 1024.0)) / max(rate_mb, 1e-3)
                            print(
                                f"[download] {dest.name}: {cur_tot / (1024**3):.2f}/{total_size / (1024**3):.2f} GiB "
                                f"({pct:5.1f}%) | {rate_mb:5.1f} MB/s | ETA {int(rem_s // 60)}m{int(rem_s % 60):02d}s",
                                flush=True,
                            )
                            save_state()
                    return
                except Exception as exc:
                    if attempt == 5:
                        stop_flag.set()
                        raise RuntimeError(f"Chunk {chunk_id} ({s_off}-{e_off}) failed after 6 retries: {exc}") from exc
                    time.sleep(min(2.0 * (attempt + 1), 10.0))

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as pool:
            futs = [pool.submit(download_chunk, item) for item in pending]
            for fut in concurrent.futures.as_completed(futs):
                fut.result()
        save_state()
        os.fsync(fd)
    finally:
        os.close(fd)

    if part_path.stat().st_size != total_size:
        raise RuntimeError(f"Size mismatch after download: {part_path.stat().st_size} != {total_size}")
    os.replace(part_path, dest)
    if state_path.exists():
        state_path.unlink()
    dt = max(time.monotonic() - t_start, 1e-3)
    print(
        f"[download] Completed {dest.name} ({total_size / (1024**3):.2f} GiB) in {dt:.1f}s "
        f"(avg {(bytes_new / (1024**2)) / dt:.1f} MB/s)",
        flush=True,
    )
    return dest


def _model_shards_complete(target: Path) -> bool:
    """Verify config.json AND every shard listed in model.safetensors.index.json (if present)."""
    if not (target / "config.json").is_file():
        return False
    idx_file = target / "model.safetensors.index.json"
    if idx_file.is_file():
        try:
            wm = json.loads(idx_file.read_text(encoding="utf-8")).get("weight_map", {})
            req_shards = sorted(set(wm.values()))
            if req_shards and all((target / s).is_file() and (target / s).stat().st_size > 0 for s in req_shards):
                return True
            return False
        except Exception:
            return False
    return any(p.stat().st_size > 0 for p in target.glob("*.safetensors"))


def _download_dynamic_dataset(ds_key: str, ds_root: Path) -> str:
    """Download parquet/jsonl files for a dataset by querying HfApi file listing."""
    from huggingface_hub import HfApi, hf_hub_download

    repo_id, patterns = DYNAMIC_DATASET_REPOS[ds_key]
    ds_target = ds_root / ds_key
    ds_target.mkdir(parents=True, exist_ok=True)

    existing = list(ds_target.rglob("*.parquet")) + list(ds_target.rglob("*.jsonl"))
    if existing and all(p.stat().st_size > 0 for p in existing):
        print(f"[download] Dataset {ds_key} already cached ({len(existing)} files).")
        return str(ds_target)

    api = HfApi()
    all_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    chosen: list[str] = []
    for f in all_files:
        if not (f.endswith(".parquet") or f.endswith(".jsonl") or f.endswith(".json")):
            continue
        if any(pat in f for pat in patterns):
            chosen.append(f)
    if not chosen:
        # Fallback: try refs/convert/parquet
        try:
            pq_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision="refs/convert/parquet")
            for f in pq_files:
                if f.endswith(".parquet"):
                    chosen.append(f)
            revision = "refs/convert/parquet"
        except Exception as exc:
            raise RuntimeError(f"Could not locate data files for {repo_id}: {all_files[:20]}") from exc
    else:
        revision = "main"

    for rel_file in chosen:
        print(f"[download] Downloading {repo_id}@{revision}:{rel_file} -> {ds_target}")
        hf_hub_download(
            repo_id=repo_id,
            filename=rel_file,
            repo_type="dataset",
            revision=revision,
            local_dir=str(ds_target),
        )
    return str(ds_target)


def download_requested_assets(
    assets: list[str],
    model_dir: str | Path,
    dataset_dir: str | Path,
) -> dict[str, str]:
    """Download/verify models and parquet/jsonl datasets via HF mirror (idempotent)."""
    from huggingface_hub import hf_hub_download, snapshot_download

    model_root = Path(model_dir)
    ds_root = Path(dataset_dir)
    model_root.mkdir(parents=True, exist_ok=True)
    ds_root.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, str] = {}

    for asset in assets:
        asset = asset.strip()
        if not asset:
            continue
        if asset in MODEL_REPOS:
            repo_id, local_name = MODEL_REPOS[asset]
            target = model_root / local_name
            if _model_shards_complete(target):
                print(f"[download] Model {repo_id} already cached at {target}")
            elif asset in MODELSCOPE_LARGE_SHARDS:
                # 1) Fetch small config/tokenizer files quickly via snapshot_download (skip huge safetensors!)
                target.mkdir(parents=True, exist_ok=True)
                print(f"[download] Fetching configs/tokenizer for {repo_id} -> {target}", flush=True)
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(target),
                    ignore_patterns=["*.png", "*.jpg", "*.gif", "*.md", "*.safetensors"],
                )
                # 2) Download large safetensors shards via 16-connection ModelScope Alibaba CDN (fallback: hf-mirror)
                hf_ep = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
                workers = int(os.environ.get("DOWNLOAD_WORKERS", "16"))
                for fname, exp_sz in MODELSCOPE_LARGE_SHARDS[asset]:
                    urls = [
                        f"https://modelscope.cn/models/{repo_id}/resolve/master/{fname}",
                        f"{hf_ep}/{repo_id}/resolve/main/{fname}",
                    ]
                    fast_range_download_file(urls, target / fname, num_workers=workers, expected_size=exp_sz)
            else:
                print(f"[download] Downloading model {repo_id} -> {target} (endpoint={os.environ.get('HF_ENDPOINT', 'default')})")
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(target),
                    ignore_patterns=["*.png", "*.jpg", "*.gif", "*.md"],
                )
            resolved[asset] = str(target)
        elif asset == "datasets":
            for ds_key, (repo_id, files) in DATASET_SPECS.items():
                ds_target = ds_root / ds_key
                ds_target.mkdir(parents=True, exist_ok=True)
                for rel_file in files:
                    out_file = ds_target / Path(rel_file).name
                    if out_file.is_file() and out_file.stat().st_size > 0:
                        print(f"[download] Dataset {ds_key}/{out_file.name} already cached.")
                        continue
                    print(f"[download] Downloading {repo_id}:{rel_file} -> {ds_target}")
                    cached = hf_hub_download(
                        repo_id=repo_id,
                        filename=rel_file,
                        repo_type="dataset",
                        local_dir=str(ds_target),
                    )
                    src = Path(cached)
                    if src != out_file and not out_file.exists():
                        out_file.symlink_to(src)
                resolved[ds_key] = str(ds_target)
        elif asset in DYNAMIC_DATASET_REPOS:
            resolved[asset] = _download_dynamic_dataset(asset, ds_root)
        else:
            raise ValueError(f"Unknown asset key: {asset!r}")
    return resolved


# ---------------------------------------------------------------------------
# Math & Multiple-Choice Benchmark Loaders (GSM8K, MATH-500, ARC-Challenge)
# ---------------------------------------------------------------------------

def extract_gsm8k_gold(answer_text: str) -> str:
    m = re.search(r"####\s*([^\n]+)", answer_text)
    raw = m.group(1) if m else answer_text.strip().splitlines()[-1]
    return normalize_math_answer(raw)


def extract_boxed(text: str) -> str | None:
    """Extract the last \\boxed{...} content with balanced brace handling."""
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    pos = start
    while pos < len(text) and depth > 0:
        ch = text[pos]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        pos += 1
    if depth == 0:
        return text[start : pos - 1].strip()
    return text[start:].strip()


def normalize_math_answer(ans: str) -> str:
    s = str(ans).strip()
    s = s.replace(",", "").replace("$", "").replace("\\!", "").replace("\\ ", " ")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\left|\\right", "", s)
    s = s.strip().rstrip(".")
    # Canonicalize simple numeric strings (e.g., "42.0" -> "42")
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", s):
        try:
            f = float(s)
            if abs(f - round(f)) < 1e-9:
                return str(int(round(f)))
            return f"{f:.6g}"
        except ValueError:
            pass
    return re.sub(r"\s+", "", s)


def extract_predicted_math_answer(text: str, *, mode: str = "gsm8k") -> str:
    """Extract final predicted math answer from model completion."""
    boxed = extract_boxed(text)
    if boxed is not None:
        return normalize_math_answer(boxed)
    m = re.search(r"####\s*([-+]?[\d,]+(?:\.\d+)?)", text)
    if m:
        return normalize_math_answer(m.group(1))
    # Search after </think> if present
    tail = text.split("</think>")[-1] if "</think>" in text else text
    boxed_tail = extract_boxed(tail)
    if boxed_tail is not None:
        return normalize_math_answer(boxed_tail)
    if mode == "gsm8k":
        nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", tail)
        if nums:
            return normalize_math_answer(nums[-1])
    else:
        # Look for "answer is ..." or last line
        m_ans = re.search(r"(?:answer|result)\s*(?:is|=|:)\s*([^\n.]+)", tail, flags=re.IGNORECASE)
        if m_ans:
            return normalize_math_answer(m_ans.group(1))
        nums = re.findall(r"[-+]?\d[\d,]*(?:/\d+|\.\d+)?", tail)
        if nums:
            return normalize_math_answer(nums[-1])
    return ""


def load_gsm8k_split(
    dataset_dir: str | Path,
    split: str = "test",
    *,
    n_samples: int = 100,
    seed: int = 20261124,
) -> list[dict[str, Any]]:
    p_dir = Path(dataset_dir) / "gsm8k"
    cands = sorted(p_dir.rglob(f"*{split}*.parquet"))
    if not cands:
        raise FileNotFoundError(f"GSM8K {split} parquet not found under {p_dir}")
    rows = _read_parquet_rows(cands[0])
    rng = random.Random(seed)
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)
    out: list[dict[str, Any]] = []
    for i in idxs[:n_samples]:
        r = rows[i]
        out.append(
            {
                "id": f"gsm8k_{split}_{i}",
                "question": str(r["question"]).strip(),
                "solution": str(r["answer"]).strip(),
                "gold": extract_gsm8k_gold(str(r["answer"])),
            }
        )
    return out


def load_math500_samples(
    dataset_dir: str | Path,
    *,
    n_samples: int = 100,
    max_level: int = 5,
    seed: int = 20261124,
) -> list[dict[str, Any]]:
    p_dir = Path(dataset_dir) / "math500"
    pq_cands = sorted(p_dir.rglob("*.parquet"))
    jl_cands = sorted(p_dir.rglob("*.jsonl"))
    rows: list[dict[str, Any]] = []
    if pq_cands:
        rows = _read_parquet_rows(pq_cands[0])
    elif jl_cands:
        with open(jl_cands[0], "r", encoding="utf-8") as fp:
            for line in fp:
                if line.strip():
                    rows.append(json.loads(line))
    else:
        raise FileNotFoundError(f"MATH-500 data not found under {p_dir}")

    filtered = [r for r in rows if int(r.get("level", 3)) <= max_level]
    rng = random.Random(seed)
    idxs = list(range(len(filtered)))
    rng.shuffle(idxs)
    out: list[dict[str, Any]] = []
    for i in idxs[:n_samples]:
        r = filtered[i]
        out.append(
            {
                "id": str(r.get("unique_id", f"math500_{i}")),
                "question": str(r["problem"]).strip(),
                "solution": str(r.get("solution", "")).strip(),
                "gold": normalize_math_answer(str(r.get("answer", ""))),
                "level": int(r.get("level", 0)),
                "subject": str(r.get("subject", "unknown")),
            }
        )
    return out


def _load_mcq_parquet(
    p_dir: Path,
    *,
    prefix: str,
    n_samples: int,
    seed: int,
    question_key: str = "question",
) -> list[dict[str, Any]]:
    cands = sorted(p_dir.rglob("*test*.parquet"))
    if not cands:
        cands = sorted(p_dir.rglob("*.parquet"))
    if not cands:
        raise FileNotFoundError(f"{prefix} parquet not found under {p_dir}")
    rows = _read_parquet_rows(cands[0])
    rng = random.Random(seed)
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)
    out: list[dict[str, Any]] = []
    for i in idxs:
        r = rows[i]
        choices = r.get("choices", {})
        labels = list(choices.get("label", []))
        texts = list(choices.get("text", []))
        ans_key = str(r.get("answerKey", "")).strip()
        if ans_key not in labels or len(labels) < 2:
            continue
        q_val = r.get(question_key, r.get("question_stem", ""))
        out.append(
            {
                "id": str(r.get("id", f"{prefix}_{i}")),
                "question": str(q_val).strip(),
                "choice_labels": labels,
                "choice_texts": texts,
                "gold_idx": labels.index(ans_key),
            }
        )
        if n_samples > 0 and len(out) >= n_samples:
            break
    return out


def load_arc_challenge_samples(
    dataset_dir: str | Path,
    *,
    n_samples: int = 200,
    seed: int = 20261124,
) -> list[dict[str, Any]]:
    return _load_mcq_parquet(
        Path(dataset_dir) / "arc_challenge",
        prefix="arc_c",
        n_samples=n_samples,
        seed=seed,
        question_key="question",
    )


def load_arc_easy_samples(
    dataset_dir: str | Path,
    *,
    n_samples: int = 600,
    seed: int = 20261124,
) -> list[dict[str, Any]]:
    return _load_mcq_parquet(
        Path(dataset_dir) / "arc_easy",
        prefix="arc_e",
        n_samples=n_samples,
        seed=seed,
        question_key="question",
    )


def load_openbookqa_samples(
    dataset_dir: str | Path,
    *,
    n_samples: int = 500,
    seed: int = 20261124,
) -> list[dict[str, Any]]:
    return _load_mcq_parquet(
        Path(dataset_dir) / "openbookqa",
        prefix="obqa",
        n_samples=n_samples,
        seed=seed,
        question_key="question_stem",
    )


# ---------------------------------------------------------------------------
# Probe 1A: Controllable Paired Two-Hop QA (question switch + position swap)
# ---------------------------------------------------------------------------

_FIRST_NAMES = [
    "Clara", "Martin", "Nadia", "Owen", "Tessa", "Victor", "Hannah", "Julian",
    "Mina", "Arthur", "Sonia", "Felix", "Elena", "Darius", "Iris", "Simon",
    "Linnea", "Caleb", "Freya", "Gideon", "Maren", "Tobias", "Astrid", "Nolan",
    "Sylvie", "Hector", "Adele", "Roman", "Vera", "Bram", "Nora", "Lucian",
]
_LAST_NAMES = [
    "Vance", "Lindholm", "Kovacs", "Mercer", "Sato", "Delgado", "Brenner", "Okafor",
    "Halvorsen", "Moreau", "Takahashi", "Keller", "Reyes", "Novak", "Whitaker", "Solis",
    "Alvarez", "Bergstrom", "Choudhury", "Ferreira", "Gallagher", "Hoffman", "Ibrahim", "Jensen",
]
_PROJECTS = [
    "Cobalt", "Meridian", "Halcyon", "Vesper", "Solstice", "Aegis", "Boreal", "Cinder",
    "Delta", "Equinox", "Falcon", "Granite", "Helios", "Indigo", "Juniper", "Kestrel",
    "Lantern", "Monarch", "Nimbus", "Opal", "Polaris", "Quartz", "Redwood", "Saffron",
    "Tempest", "Umbra", "Veridian", "Warden", "Zephyr", "Alder", "Basalt", "Cypress",
]
_CITIES = [
    "Bergen", "Graz", "Tampere", "Ghent", "Porto", "Salzburg", "Utrecht", "Brno",
    "Aarhus", "Lille", "Verona", "Cork", "Malmo", "Innsbruck", "Bologna", "Antwerp",
    "Odense", "Bremen", "Turin", "Freiburg", "Leiden", "Parma", "Basel", "Krakow",
    "Bilbao", "Uppsala", "Padua", "Modena", "Linz", "Rennes", "Nantes", "Genoa",
]
_DEPARTMENTS = [
    "coastal telemetry", "archive indexing", "grid diagnostics", "mineral cataloging",
    "freight routing", "bridge inspection", "waterway monitoring", "timber logistics",
    "rail scheduling", "acoustic mapping", "weather calibration", "port surveying",
]
_QUARTERS = ["first", "second", "third", "fourth"]


def _make_hop1_para(person: str, project: str, dept: str, quarter: str, idx: int) -> str:
    return (
        f"Record {idx:02d}: During the {quarter} quarterly staffing review for {dept}, "
        f"senior coordinator {person} was formally appointed to oversee Project {project}. "
        f"The staffing ledger confirmed that {person} assumed full operational responsibility "
        f"for Project {project} after the regional audit closed without further revisions."
    )


def _make_hop2_para(project: str, city: str, dept: str, quarter: str, idx: int) -> str:
    return (
        f"Record {idx:02d}: According to the {quarter} facilities registry for {dept}, "
        f"Project {project} maintains its primary headquarters in {city}. "
        f"All equipment manifests and annual compliance filings for Project {project} "
        f"are routed directly to the central operations office located in {city}."
    )


def _make_distractor_para(person: str, project: str, city: str, dept: str, quarter: str, idx: int) -> str:
    return (
        f"Record {idx:02d}: In the {quarter} administrative summary for {dept}, "
        f"consultant {person} reviewed archival invoices unrelated to active headquarters assignments. "
        f"A separate storage depot near {city} handled surplus hardware from legacy unit {project}, "
        f"and routine maintenance logs were filed on schedule without incident."
    )


def generate_synthetic_paired_twohop(
    n_base_docs: int = 48,
    seed: int = 20261124,
) -> list[dict[str, Any]]:
    """Build paired 2-hop QA items with both question-switching and evidence-position swapping.

    Returns 2 * n_base_docs documents (each base doc + its evidence-position-swapped twin),
    each paired with TWO questions (Q_A and Q_B) over the exact same prefix.
    Total QA items = 4 * n_base_docs (e.g., 48 base -> 96 docs -> 192 QA pairs).
    """
    rng = random.Random(seed)
    all_pairs = [(i, j) for i in range(8) for j in range(8) if i != j]
    docs: list[dict[str, Any]] = []

    for base_id in range(n_base_docs):
        # Pick 6 distinct people, 6 distinct projects, 6 distinct cities (2 chains + 4 distractors)
        people = [
            f"{fn} {ln}"
            for fn, ln in zip(rng.sample(_FIRST_NAMES, 6), rng.sample(_LAST_NAMES, 6))
        ]
        projects = rng.sample(_PROJECTS, 6)
        cities = rng.sample(_CITIES, 6)
        depts = rng.sample(_DEPARTMENTS, 8)
        quarters = [rng.choice(_QUARTERS) for _ in range(8)]

        # Stratified disjoint slot assignment for Chain A and Chain B
        perm = list(range(8))
        rng.shuffle(perm)
        a1, a2, b1, b2 = perm[0], perm[1], perm[2], perm[3]
        d_slots = perm[4:]

        # Build base version and position-swapped twin (swaps Chain A & B into the other 4 slots!)
        for variant, (va1, va2, vb1, vb2, vd) in [
            ("orig", (a1, a2, b1, b2, d_slots)),
            ("swapped", (d_slots[0], d_slots[1], d_slots[2], d_slots[3], [a1, a2, b1, b2])),
        ]:
            paras = [""] * 8
            paras[va1] = _make_hop1_para(people[0], projects[0], depts[0], quarters[0], va1 + 1)
            paras[va2] = _make_hop2_para(projects[0], cities[0], depts[1], quarters[1], va2 + 1)
            paras[vb1] = _make_hop1_para(people[1], projects[1], depts[2], quarters[2], vb1 + 1)
            paras[vb2] = _make_hop2_para(projects[1], cities[1], depts[3], quarters[3], vb2 + 1)
            for k, slot in enumerate(vd):
                paras[slot] = _make_distractor_para(
                    people[2 + k], projects[2 + k], cities[2 + k], depts[4 + k], quarters[4 + k], slot + 1
                )

            prefix_text = "\n\n".join(paras)
            # Compute character spans of each paragraph so we can map to exact token blocks
            spans: list[tuple[int, int]] = []
            cursor = 0
            for idx_p, p in enumerate(paras):
                start = cursor
                end = start + len(p)
                spans.append((start, end))
                cursor = end + (2 if idx_p < 7 else 0)

            qa_a = {
                "qa_id": f"syn_{base_id:03d}_{variant}_A",
                "doc_id": f"syn_{base_id:03d}_{variant}",
                "base_id": base_id,
                "variant": variant,
                "which_chain": "A",
                "prefix_text": prefix_text,
                "para_spans": spans,
                "evidence_slots": sorted([va1, va2]),
                "hop1_slot": va1,
                "hop2_slot": va2,
                "other_chain_slots": sorted([vb1, vb2]),
                "question_suffix": (
                    f"\n\nAnswer with only the single city name from the records above.\n"
                    f"Question: In which city is the project overseen by {people[0]} headquartered?\n"
                    f"Answer:"
                ),
                "gold_answer": f" {cities[0]}",
                "gold_answer_norm": cities[0].strip().lower(),
                "distractor_answer_norm": cities[1].strip().lower(),
            }
            qa_b = {
                "qa_id": f"syn_{base_id:03d}_{variant}_B",
                "doc_id": f"syn_{base_id:03d}_{variant}",
                "base_id": base_id,
                "variant": variant,
                "which_chain": "B",
                "prefix_text": prefix_text,
                "para_spans": spans,
                "evidence_slots": sorted([vb1, vb2]),
                "hop1_slot": vb1,
                "hop2_slot": vb2,
                "other_chain_slots": sorted([va1, va2]),
                "question_suffix": (
                    f"\n\nAnswer with only the single city name from the records above.\n"
                    f"Question: In which city is the project overseen by {people[1]} headquartered?\n"
                    f"Answer:"
                ),
                "gold_answer": f" {cities[1]}",
                "gold_answer_norm": cities[1].strip().lower(),
                "distractor_answer_norm": cities[0].strip().lower(),
            }
            docs.append(qa_a)
            docs.append(qa_b)
    return docs


# ---------------------------------------------------------------------------
# Probe 1B: Natural Multi-Hop QA (2WikiMultihopQA)
# ---------------------------------------------------------------------------

def _extract_2wiki_paragraphs(context_field: Any) -> list[tuple[str, str]]:
    """Handle both dict-of-lists and list-of-pairs representations in 2WikiMultihopQA."""
    paras: list[tuple[str, str]] = []
    if isinstance(context_field, dict):
        titles = context_field.get("title", [])
        sents_list = context_field.get("sentences", [])
        for t, sents in zip(titles, sents_list):
            text = " ".join(str(s).strip() for s in sents if str(s).strip())
            paras.append((str(t), text))
    elif isinstance(context_field, (list, tuple)):
        for item in context_field:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title = str(item[0])
                sents = item[1]
                text = " ".join(str(s).strip() for s in sents if str(s).strip()) if isinstance(sents, (list, tuple)) else str(sents)
                paras.append((title, text))
    return paras


def _extract_2wiki_supporting_titles(sf_field: Any) -> set[str]:
    titles: set[str] = set()
    if isinstance(sf_field, dict):
        for t in sf_field.get("title", []):
            titles.add(str(t))
    elif isinstance(sf_field, (list, tuple)):
        for item in sf_field:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                titles.add(str(item[0]))
            elif isinstance(item, str):
                titles.add(item)
    return titles


def load_2wiki_samples(
    dataset_dir: str | Path,
    tokenizer,
    *,
    n_samples: int = 96,
    min_prefix_tokens: int = 240,
    max_prefix_tokens: int = 1024,
    seed: int = 20261124,
) -> list[dict[str, Any]]:
    """Load deterministic validation samples from framolfese/2WikiMultihopQA."""
    p_dir = Path(dataset_dir) / "2wiki"
    candidates = list(p_dir.rglob("validation*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"2WikiMultihopQA validation parquet not found under {p_dir}")
    rows = _read_parquet_rows(candidates[0])
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)

    samples: list[dict[str, Any]] = []
    for ridx in indices:
        row = rows[ridx]
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not question or not answer:
            continue
        ans_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
        if not (1 <= len(ans_ids) <= 8):
            continue

        paras = _extract_2wiki_paragraphs(row.get("context"))
        if len(paras) < 4:
            continue
        sf_titles = _extract_2wiki_supporting_titles(row.get("supporting_facts"))

        formatted_paras: list[str] = []
        para_spans: list[tuple[int, int]] = []
        support_para_indices: list[int] = []
        cursor = 0
        for p_idx, (title, body) in enumerate(paras):
            piece = f"[{title}] {body}"
            formatted_paras.append(piece)
            start = cursor
            end = start + len(piece)
            para_spans.append((start, end))
            if title in sf_titles:
                support_para_indices.append(p_idx)
            cursor = end + 2
        prefix_text = "\n\n".join(formatted_paras)
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        if not (min_prefix_tokens <= len(prefix_ids) <= max_prefix_tokens):
            continue

        samples.append(
            {
                "qa_id": f"2wiki_{ridx}",
                "doc_id": f"2wiki_{ridx}",
                "qtype": str(row.get("type", "unknown")),
                "prefix_text": prefix_text,
                "para_spans": para_spans,
                "support_para_indices": support_para_indices,
                "question_suffix": (
                    f"\n\nAnswer the question concisely based on the passages above.\n"
                    f"Question: {question}\n"
                    f"Answer:"
                ),
                "gold_answer": f" {answer}",
                "gold_answer_norm": normalize_answer_text(answer),
            }
        )
        if len(samples) >= n_samples:
            break
    if len(samples) < n_samples:
        print(f"[data] Warning: requested {n_samples} 2Wiki samples, found {len(samples)}")
    return samples


def normalize_answer_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def map_char_spans_to_blocks(
    prefix_text: str,
    prefix_token_count: int,
    target_char_spans: list[tuple[int, int]],
    n_blocks: int = 8,
) -> list[int]:
    """Map character spans in prefix_text to the 8 equal-token block indices they overlap most."""
    total_chars = max(len(prefix_text), 1)
    block_scores = [0.0] * n_blocks
    for s_char, e_char in target_char_spans:
        s_frac = max(0.0, min(1.0, s_char / total_chars))
        e_frac = max(0.0, min(1.0, e_char / total_chars))
        for b in range(n_blocks):
            b0 = b / n_blocks
            b1 = (b + 1) / n_blocks
            ov = max(0.0, min(e_frac, b1) - max(s_frac, b0))
            block_scores[b] += ov
    ranked = sorted(range(n_blocks), key=lambda b: block_scores[b], reverse=True)
    return [b for b in ranked if block_scores[b] > 0]


# ---------------------------------------------------------------------------
# Disjoint Text + Code splits (for Probe 1C and Idea 2 calib / train / eval)
# ---------------------------------------------------------------------------

def _load_wikitext_articles(parquet_path: Path) -> list[str]:
    lines = [str(r.get("text", "")) for r in _read_parquet_rows(parquet_path)]
    articles: list[str] = []
    buf: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("= ") and stripped.endswith(" =") and not stripped.startswith("= ="):
            if buf:
                text = "\n".join(buf).strip()
                if len(text) >= 400:
                    articles.append(text)
            buf = [stripped]
        elif stripped:
            buf.append(stripped)
    if buf:
        text = "\n".join(buf).strip()
        if len(text) >= 400:
            articles.append(text)
    return articles


def _load_humaneval_docs(parquet_path: Path) -> list[str]:
    docs: list[str] = []
    for row in _read_parquet_rows(parquet_path):
        prompt = str(row.get("prompt", ""))
        sol = str(row.get("canonical_solution", ""))
        test = str(row.get("test", ""))
        full = f"{prompt}{sol}\n\n{test}".strip()
        if len(full) >= 200:
            docs.append(full)
    return docs


def _pack_token_chunks(
    texts: list[str],
    encode_fn,
    *,
    seq_len: int,
    max_chunks: int,
    domain: str,
) -> list[dict[str, Any]]:
    """Pack texts into contiguous non-overlapping token chunks of exact length seq_len."""
    chunks: list[dict[str, Any]] = []
    buf: list[int] = []
    for t in texts:
        ids = encode_fn(t)
        buf.extend(ids)
        while len(buf) >= seq_len:
            chunks.append({"tokens": buf[:seq_len], "domain": domain})
            buf = buf[seq_len:]
            if len(chunks) >= max_chunks:
                return chunks
    return chunks


def build_disjoint_text_code_splits(
    dataset_dir: str | Path,
    encode_fn,
    *,
    seq_len: int = 384,
    n_calib_a: int = 16,
    n_calib_b: int = 16,
    n_adapt_train: int = 64,
    n_eval_heldout: int = 40,
    seed: int = 20261124,
) -> dict[str, list[dict[str, Any]]]:
    """Build 4 strictly disjoint splits (calib_a, calib_b, adapt_train, eval_heldout), 50% text + 50% code.

    Zero document overlap between any two splits!
    """
    wiki_dir = Path(dataset_dir) / "wikitext"
    he_dir = Path(dataset_dir) / "humaneval"

    wiki_train = _load_wikitext_articles(next(wiki_dir.rglob("train*.parquet")))
    wiki_val = _load_wikitext_articles(next(wiki_dir.rglob("validation*.parquet")))
    wiki_test = _load_wikitext_articles(next(wiki_dir.rglob("test*.parquet")))
    he_all = _load_humaneval_docs(next(he_dir.rglob("test*.parquet")))

    rng = random.Random(seed)
    rng.shuffle(wiki_train)
    wiki_eval_pool = wiki_val + wiki_test
    rng.shuffle(wiki_eval_pool)

    # Partition HumanEval problems strictly by problem index (zero leakage)
    he_idx = list(range(len(he_all)))
    rng.shuffle(he_idx)
    n_he = len(he_idx)
    c1 = int(n_he * 0.18)
    c2 = int(n_he * 0.36)
    c3 = int(n_he * 0.76)
    he_ca = [he_all[i] for i in he_idx[:c1]]
    he_cb = [he_all[i] for i in he_idx[c1:c2]]
    he_tr = [he_all[i] for i in he_idx[c2:c3]]
    he_ev = [he_all[i] for i in he_idx[c3:]]

    # Partition WikiText train strictly into calib_a, calib_b, adapt_train; use val+test for eval_heldout
    n_wt = len(wiki_train)
    w1 = int(n_wt * 0.20)
    w2 = int(n_wt * 0.40)
    wt_ca = wiki_train[:w1]
    wt_cb = wiki_train[w1:w2]
    wt_tr = wiki_train[w2:]
    wt_ev = wiki_eval_pool

    def _make_split(wt_docs: list[str], he_docs: list[str], total_n: int) -> list[dict[str, Any]]:
        half_code = total_n // 2
        code_chunks = _pack_token_chunks(he_docs, encode_fn, seq_len=seq_len, max_chunks=half_code, domain="code")
        need_text = total_n - len(code_chunks)
        text_chunks = _pack_token_chunks(wt_docs, encode_fn, seq_len=seq_len, max_chunks=need_text, domain="text")
        combined = []
        # Interleave text and code deterministically
        max_len = max(len(text_chunks), len(code_chunks))
        for i in range(max_len):
            if i < len(text_chunks):
                combined.append(text_chunks[i])
            if i < len(code_chunks):
                combined.append(code_chunks[i])
        return combined[:total_n]

    splits = {
        "calib_a": _make_split(wt_ca, he_ca, n_calib_a),
        "calib_b": _make_split(wt_cb, he_cb, n_calib_b),
        "adapt_train": _make_split(wt_tr, he_tr, n_adapt_train),
        "eval_heldout": _make_split(wt_ev, he_ev, n_eval_heldout),
    }
    # Assert zero exact token-sequence overlap across splits
    seen_hashes: dict[str, str] = {}
    for sname, items in splits.items():
        for it in items:
            h = hashlib.sha1(json.dumps(it["tokens"]).encode()).hexdigest()
            if h in seen_hashes and seen_hashes[h] != sname:
                raise RuntimeError(f"Split leakage detected between {seen_hashes[h]} and {sname}")
            seen_hashes[h] = sname
    return splits
