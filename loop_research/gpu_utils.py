"""Safe, race-free idle GPU selection for shared multi-user servers."""

from __future__ import annotations

import fcntl
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class GPUStatus:
    index: int
    uuid: str
    name: str
    mem_used_mib: int
    mem_total_mib: int
    util_pct: int
    active_pids: list[int]

    @property
    def is_idle(self) -> bool:
        return self.mem_used_mib <= 256 and self.util_pct <= 20 and len(self.active_pids) == 0


def query_gpus() -> list[GPUStatus]:
    """Query live GPU memory, utilization, and compute processes via nvidia-smi."""
    gpu_out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    proc_out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    pids_by_uuid: dict[str, list[int]] = {}
    for line in proc_out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                pids_by_uuid.setdefault(parts[0], []).append(int(parts[1]))
            except ValueError:
                pass

    gpus: list[GPUStatus] = []
    for line in gpu_out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        idx = int(parts[0])
        uuid = parts[1]
        name = parts[2]
        mem_used = int(float(parts[3]))
        mem_total = int(float(parts[4]))
        util = int(float(parts[5]))
        gpus.append(
            GPUStatus(
                index=idx,
                uuid=uuid,
                name=name,
                mem_used_mib=mem_used,
                mem_total_mib=mem_total,
                util_pct=util,
                active_pids=pids_by_uuid.get(uuid, []),
            )
        )
    return gpus


class GPULock:
    """Advisory non-blocking flock held for the duration of a GPU experiment."""

    def __init__(self, gpu_index: int, lock_dir: str = "/tmp/loop_research_gpu_locks") -> None:
        self.gpu_index = gpu_index
        self.lock_dir = Path(lock_dir)
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.lock_dir, 0o777)
        except OSError:
            pass
        self.lock_path = self.lock_dir / f"gpu_{gpu_index}.lock"
        self._fp: Any = None

    def try_acquire(self) -> bool:
        fp = open(self.lock_path, "a+")
        try:
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fp.seek(0)
            fp.truncate()
            fp.write(f"pid={os.getpid()} run={os.environ.get('ORX_RUN_ID', 'manual')}\n")
            fp.flush()
            self._fp = fp
            return True
        except BlockingIOError:
            fp.close()
            return False

    def release(self) -> None:
        if self._fp is not None:
            try:
                fcntl.flock(self._fp.fileno(), fcntl.LOCK_UN)
                self._fp.close()
            except OSError:
                pass
            self._fp = None


def acquire_idle_gpu(preferred_index: int | None = None) -> tuple[int, GPULock, GPUStatus]:
    """Select and lock an idle GPU without ever touching another user's GPU."""
    env_pref = os.environ.get("LOOP_GPU_INDEX", "").strip()
    if preferred_index is None and env_pref:
        preferred_index = int(env_pref)

    gpus = query_gpus()
    if not gpus:
        raise RuntimeError("No GPUs found via nvidia-smi")

    by_idx = {g.index: g for g in gpus}
    order: list[int] = []
    if preferred_index is not None:
        if preferred_index not in by_idx:
            raise RuntimeError(f"Requested GPU {preferred_index} does not exist (available: {sorted(by_idx)})")
        order.append(preferred_index)
    for g in gpus:
        if g.index not in order:
            order.append(g.index)

    for idx in order:
        g = by_idx[idx]
        if not g.is_idle:
            continue
        lock = GPULock(idx)
        if lock.try_acquire():
            return idx, lock, g

    summary = ", ".join(
        f"gpu{g.index}(mem={g.mem_used_mib}MiB,util={g.util_pct}%,pids={g.active_pids})"
        for g in gpus
    )
    raise RuntimeError(f"No idle unlocked GPU available! Current status: {summary}")
