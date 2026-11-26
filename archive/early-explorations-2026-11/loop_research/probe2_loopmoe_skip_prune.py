"""Round 1 Probe 2: Looped MoE Per-Loop Redundancy Map, Dynamic Expert Skipping,
Stage-Aware Static Expert Pruning, and Complete Integer Depth Profile.

Evaluates both IFM Ouro-MoE (28 layers x 4 loops) and IFM Huginn-MoE (8+12x8+8):
  0. Exactness verification (no-op hook matches native BF16 bit-for-bit).
  1. Calibration pass (16 disjoint seqs): per-(layer, visit, expert) counts, output
     energy, and per-stage gate-ratio (w2/w1) + stay distributions, plus calib-only
     per-stage sensitivity measurement (zero eval leakage!).
  2. Part A (Per-Loop MoE Redundancy Map):
     Measure exact held-out ΔNLL of dropping 2nd expert (k=2->1, renorm & non-renorm)
     and skipping MoE entirely (k=0) for each individual loop/stage.
  3. Part B (Dynamic Expert Skipping at Matched -25%, -35%, -45% Active Expert Calls):
     Compare uniform gate-ratio (ACE-style), loop-sensitivity-weighted ratio,
     pure stage-only drop, and cross-loop stay-aware skip on NLL & ARC-150.
  4. Part C (Static Expert Pruning at 20%, 35%, 50% Sparsity):
     Compare random, global_freq, global_energy, max_stage_freq (specialist protection),
     and stage_sens_weighted (feedback/sensitivity protection) on NLL & ARC-150.
  5. Part D (Complete Integer Depth Profile):
     NLL & hidden-norm profile at R=1..8 (Ouro) and R=1..16 (Huginn) to inspect
     depth dynamics and Huginn's R=8->12->16 oscillation.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from loop_research import orx_run as orx
from loop_research.data_assets import build_disjoint_text_code_splits, load_arc_challenge_samples
from loop_research.gpu_utils import acquire_idle_gpu
from loop_research.ifm_runtime import bootstrap_xllm, load_ifm_model_for_autograd
from loop_research.vsubmoe_pilot import set_ifm_loop_depth


# ---------------------------------------------------------------------------
# Pure Logic Helpers (CPU Unit-Testable!)
# ---------------------------------------------------------------------------

def paired_bootstrap_ci(
    diffs: np.ndarray,
    n_boot: int = 2000,
    seed: int = 20261126,
) -> tuple[float, float, float]:
    arr = np.asarray(diffs, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def build_alive_expert_mask(
    scores: np.ndarray,
    prune_frac: float,
    min_alive_per_layer: int = 2,
) -> np.ndarray:
    """Build boolean alive mask [L, E] pruning prune_frac of total experts globally,
    guaranteeing at least min_alive_per_layer surviving experts in every layer.
    """
    n_layers, n_experts = scores.shape
    total = n_layers * n_experts
    n_prune = int(round(prune_frac * total))
    max_prunable = total - n_layers * min_alive_per_layer
    n_prune = max(0, min(n_prune, max_prunable))

    alive = np.ones((n_layers, n_experts), dtype=bool)
    if n_prune == 0:
        return alive

    # Protect top min_alive_per_layer experts per layer first
    protected = np.zeros((n_layers, n_experts), dtype=bool)
    for l in range(n_layers):
        top_e = sorted(range(n_experts), key=lambda e: (-float(scores[l, e]), e))[:min_alive_per_layer]
        for e in top_e:
            protected[l, e] = True

    candidates = [
        (float(scores[l, e]), l, e)
        for l in range(n_layers)
        for e in range(n_experts)
        if not protected[l, e]
    ]
    candidates.sort()  # lowest scores pruned first
    for _, l, e in candidates[:n_prune]:
        alive[l, e] = False
    return alive


def stage_id_for(lid: int, visit: int, loop_start: int, loop_end: int, n_loops: int) -> int:
    """Map (physical layer, visit index) to a contiguous stage ID.

    - For Ouro (loop_start=0, loop_end=28): stages 0..n_loops-1 are loops 0..R-1.
    - For Huginn (loop_start=8, loop_end=20):
        stage 0 = prelude (lid < 8)
        stages 1..n_loops = recurrent loops 0..R-1
        stage n_loops+1 = coda (lid >= 20)
    """
    has_prelude = loop_start > 0
    if lid < loop_start:
        return 0
    if lid < loop_end:
        return (1 if has_prelude else 0) + min(visit, n_loops - 1)
    return (1 if has_prelude else 0) + n_loops


def stage_names_for(n_layers: int, loop_start: int, loop_end: int, n_loops: int) -> list[str]:
    names: list[str] = []
    if loop_start > 0:
        names.append("prelude")
    for r in range(n_loops):
        names.append(f"loop_{r}")
    if loop_end < n_layers:
        names.append("coda")
    return names


def calibrate_drop_threshold(scores_list: list[np.ndarray], target_drop_frac_of_second: float) -> float:
    """Find threshold tau such that fraction `target_drop_frac_of_second` of tokens have score < tau."""
    if not scores_list:
        return -1.0
    all_s = np.concatenate(scores_list)
    q = float(np.clip(target_drop_frac_of_second, 0.0, 1.0))
    if q <= 0.0:
        return -1e9
    if q >= 1.0:
        return float(all_s.max()) + 1.0
    return float(np.quantile(all_s, q))


# ---------------------------------------------------------------------------
# Unified Router Intervention Engine (Exact Passthrough, Skipping, Pruning)
# ---------------------------------------------------------------------------

class LoopMoEIntervention:
    """Controls MoE routing interventions across all layers & visits cleanly."""

    def __init__(
        self,
        model,
        *,
        mode: str = "native",
        # Stage-level ablation
        target_stage: int | None = None,
        stage_action: str = "none",  # "drop2_renorm" | "drop2_noren" | "skip_moe"
        # Dynamic skipping
        dyn_threshold: float = -1.0,
        stage_weights: np.ndarray | None = None,
        stay_factor: float = 1.0,
        stage_drop2_set: set[int] | None = None,
        renorm_on_drop2: bool = True,
        # Static pruning
        alive_mask: torch.Tensor | None = None,
    ) -> None:
        self.model = model
        self.n_layers = len(model.layers)
        self.loop_start = int(model.loop_start_layer)
        self.loop_end = int(model.loop_end_layers)
        self.n_loops = int(model.loop_times)
        self.mode = mode
        self.target_stage = target_stage
        self.stage_action = stage_action
        self.dyn_threshold = float(dyn_threshold)
        self.stage_weights = stage_weights
        self.stay_factor = float(stay_factor)
        self.stage_drop2_set = stage_drop2_set or set()
        self.renorm_on_drop2 = bool(renorm_on_drop2)
        self.alive_mask = alive_mask

        self.visits = [0] * self.n_layers
        self.prev_top1: dict[int, torch.Tensor] = {}
        self.total_slots = 0
        self.active_slots = 0

    def reset_seq(self) -> None:
        self.visits = [0] * self.n_layers
        self.prev_top1.clear()

    def reset_counters(self) -> None:
        self.total_slots = 0
        self.active_slots = 0

    @property
    def active_expert_frac(self) -> float:
        return float(self.active_slots / max(self.total_slots, 1))

    @contextmanager
    def attached(self):
        saved = []
        try:
            for lid, layer in enumerate(self.model.layers):
                router = layer.moe.router
                saved.append((router, router.forward))

                def make_rtr(lid_c: int, mod, orig_fn):
                    def fwd(x, load_balancing_type=None):
                        v = self.visits[lid_c]
                        self.visits[lid_c] = v + 1
                        sid = stage_id_for(lid_c, v, self.loop_start, self.loop_end, self.n_loops)

                        if self.mode == "native":
                            weights, idx, tpe, aux = orig_fn(x, load_balancing_type)
                            self.total_slots += int(idx.numel())
                            self.active_slots += int(idx.numel())
                            return weights, idx, tpe, aux

                        if self.mode == "prune":
                            assert self.alive_mask is not None
                            logits = F.linear(x, mod.weight).float()
                            scores = logits.softmax(-1) if mod.score_func == "softmax" else logits.sigmoid()
                            sel = scores if mod.bias is None else scores + mod.bias.to(scores)
                            sel = sel.masked_fill(~self.alive_mask[lid_c], -1e9)
                            idx = sel.topk(mod.topk, dim=-1).indices
                            weights = scores.gather(-1, idx)
                            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
                            if mod.scaling_factor:
                                weights = weights * mod.scaling_factor
                            tpe = torch.bincount(idx.flatten(), minlength=mod.num_experts)
                            self.total_slots += int(idx.numel())
                            self.active_slots += int(idx.numel())
                            return weights.to(x.dtype), idx, tpe, None

                        # All skipping modes start from exact native routing
                        weights, idx, tpe, aux = orig_fn(x, load_balancing_type)
                        w_f = weights.float()
                        n_tok = int(idx.numel() // mod.topk)
                        self.total_slots += int(idx.numel())
                        scale = float(mod.scaling_factor) if mod.scaling_factor else 1.0

                        if self.mode == "stage_ablation":
                            if sid == self.target_stage:
                                if self.stage_action == "skip_moe":
                                    w_f = torch.zeros_like(w_f)
                                    self.active_slots += 0
                                elif self.stage_action == "drop2_renorm":
                                    w_f = w_f.clone()
                                    w_f[..., 0] = scale
                                    w_f[..., 1] = 0.0
                                    self.active_slots += n_tok
                                elif self.stage_action == "drop2_noren":
                                    w_f = w_f.clone()
                                    w_f[..., 1] = 0.0
                                    self.active_slots += n_tok
                                else:
                                    self.active_slots += int(idx.numel())
                            else:
                                self.active_slots += int(idx.numel())
                            return w_f.to(x.dtype), idx, tpe, aux

                        if self.mode == "dynamic_skip":
                            # Ratio w2/w1 (with weights sorted or unsorted; pick smaller of the two!)
                            w_min, min_pos = w_f.min(dim=-1)
                            w_max = w_f.amax(dim=-1).clamp_min(1e-12)
                            ratio = w_min / w_max

                            # Optional stage weight & cross-loop stay factor
                            sw = 1.0 if self.stage_weights is None else float(self.stage_weights[sid])
                            score = ratio * sw

                            top1_cur = idx.gather(-1, (1 - min_pos).unsqueeze(-1)).squeeze(-1)
                            if self.stay_factor != 1.0 and lid_c in self.prev_top1:
                                stayed = top1_cur == self.prev_top1[lid_c]
                                score = torch.where(stayed, score * self.stay_factor, score)
                            self.prev_top1[lid_c] = top1_cur.detach()

                            if sid in self.stage_drop2_set:
                                drop_mask = torch.ones_like(ratio, dtype=torch.bool)
                            else:
                                drop_mask = score < self.dyn_threshold

                            n_drop = int(drop_mask.sum().item())
                            self.active_slots += int(idx.numel()) - n_drop

                            if n_drop > 0:
                                w_new = w_f.clone()
                                # Zero out the smaller-weight slot for dropped tokens
                                drop_exp = drop_mask.unsqueeze(-1)
                                is_min_slot = F.one_hot(min_pos, num_classes=mod.topk).bool()
                                w_new = torch.where(drop_exp & is_min_slot, torch.zeros_like(w_new), w_new)
                                if self.renorm_on_drop2:
                                    w_new = torch.where(
                                        drop_exp & (~is_min_slot),
                                        torch.full_like(w_new, scale),
                                        w_new,
                                    )
                                w_f = w_new
                            return w_f.to(x.dtype), idx, tpe, aux

                        raise ValueError(f"Unknown mode: {self.mode}")

                    return fwd

                router.forward = make_rtr(lid, router, router.forward)
            yield self
        finally:
            for router, fn in saved:
                router.forward = fn


@torch.inference_mode()
def evaluate_nll(model, seqs: list[dict[str, Any]], ctrl: LoopMoEIntervention, device: torch.device) -> dict[str, Any]:
    ctrl.reset_counters()
    per_seq: list[float] = []
    txt: list[float] = []
    cod: list[float] = []
    with ctrl.attached():
        for item in seqs:
            ctrl.reset_seq()
            toks = item["tokens"]
            inp = torch.tensor([toks[:-1]], device=device, dtype=torch.long)
            tgt = torch.tensor(toks[1:], device=device, dtype=torch.long)
            logits, _, _ = model(inp, multi_segments=False, cache=None)
            loss = float(F.cross_entropy(logits[0].float(), tgt).item())
            per_seq.append(loss)
            if item["domain"] == "code":
                cod.append(loss)
            else:
                txt.append(loss)
    return {
        "nll": float(np.mean(per_seq)),
        "nll_text": float(np.mean(txt)) if txt else 0.0,
        "nll_code": float(np.mean(cod)) if cod else 0.0,
        "per_seq": per_seq,
        "active_expert_frac": ctrl.active_expert_frac,
    }


@torch.inference_mode()
def evaluate_arc(model, tokenizer, arc_samples: list[dict[str, Any]], ctrl: LoopMoEIntervention, device: torch.device) -> float:
    flags: list[int] = []
    with ctrl.attached():
        for item in arc_samples:
            prompt = f"Question: {item['question']}\nAnswer:"
            p_ids = tokenizer.encode(prompt, bos=True, eos=False)
            scores = []
            for c_txt in item["choice_texts"]:
                c_ids = tokenizer.encode(" " + c_txt, bos=False, eos=False)
                if not c_ids:
                    scores.append(-1e9)
                    continue
                full = p_ids + c_ids
                inp = torch.tensor([full[:-1]], device=device, dtype=torch.long)
                tgt = torch.tensor(full[len(p_ids) :], device=device, dtype=torch.long)
                ctrl.reset_seq()
                logits, _, _ = model(inp, multi_segments=False, cache=None)
                sl_logits = logits[0, len(p_ids) - 1 :, :].float()
                lp = F.log_softmax(sl_logits, dim=-1)
                tok_ll = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                scores.append(float(tok_ll.mean().item()))
            flags.append(1 if int(np.argmax(scores)) == item["gold_idx"] else 0)
    return float(np.mean(flags))


@torch.inference_mode()
def collect_calib_skip_and_prune_stats(
    model,
    calib_seqs: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    """Collect per-(layer, visit, expert) counts & energy, and per-stage gate-ratio & stay arrays."""
    n_layers = len(model.layers)
    n_experts = int(model.layers[0].moe.experts.n_local_experts)
    loop_start = int(model.loop_start_layer)
    loop_end = int(model.loop_end_layers)
    n_loops = int(model.loop_times)
    st_names = stage_names_for(n_layers, loop_start, loop_end, n_loops)
    n_stages = len(st_names)

    counts = np.zeros((n_layers, n_loops, n_experts), dtype=np.int64)
    out_energy = np.zeros((n_layers, n_loops, n_experts), dtype=np.float64)
    ratios_by_stage: list[list[np.ndarray]] = [[] for _ in range(n_stages)]
    stay_by_stage: list[list[np.ndarray]] = [[] for _ in range(n_stages)]

    visits = [0] * n_layers
    prev_top1: dict[int, torch.Tensor] = {}
    orig_rtr = []
    orig_exp = []
    try:
        for lid, layer in enumerate(model.layers):
            router = layer.moe.router
            exp_mod = layer.moe.experts
            orig_rtr.append((router, router.forward))
            orig_exp.append((exp_mod, exp_mod.forward))

            def make_rtr(lid_c: int, orig_fn):
                def fwd(x, load_balancing_type=None):
                    weights, idx, tpe, aux = orig_fn(x, load_balancing_type)
                    v = visits[lid_c]
                    sid = stage_id_for(lid_c, v, loop_start, loop_end, n_loops)
                    wf = weights.float()
                    w_min, min_pos = wf.min(dim=-1)
                    w_max = wf.amax(dim=-1).clamp_min(1e-12)
                    ratio = (w_min / w_max).reshape(-1).cpu().numpy()
                    ratios_by_stage[sid].append(ratio)

                    top1 = idx.gather(-1, (1 - min_pos).unsqueeze(-1)).squeeze(-1)
                    if lid_c in prev_top1:
                        stayed = (top1 == prev_top1[lid_c]).reshape(-1).cpu().numpy().astype(np.bool_)
                    else:
                        stayed = np.zeros(ratio.shape, dtype=np.bool_)
                    prev_top1[lid_c] = top1.detach()
                    stay_by_stage[sid].append(stayed)
                    return weights, idx, tpe, aux
                return fwd

            def make_exp(lid_c: int, orig_fn):
                def fwd(x, group_sizes):
                    y = orig_fn(x, group_sizes)
                    v = visits[lid_c]
                    if v < n_loops:
                        gs = (
                            group_sizes.detach().cpu().tolist()
                            if torch.is_tensor(group_sizes)
                            else [int(g) for g in group_sizes]
                        )
                        y_sq = y.float().pow(2).sum(dim=-1)
                        off = 0
                        for e in range(n_experts):
                            cnt = int(gs[e])
                            if cnt > 0:
                                counts[lid_c, v, e] += cnt
                                out_energy[lid_c, v, e] += float(y_sq[off : off + cnt].sum().item())
                                off += cnt
                    visits[lid_c] += 1
                    return y
                return fwd

            router.forward = make_rtr(lid, router.forward)
            exp_mod.forward = make_exp(lid, exp_mod.forward)

        for item in calib_seqs:
            for l in range(n_layers):
                visits[l] = 0
            prev_top1.clear()
            inp = torch.tensor([item["tokens"][:-1]], device=device, dtype=torch.long)
            _ = model(inp, multi_segments=False, cache=None)
    finally:
        for router, fn in orig_rtr:
            router.forward = fn
        for exp_mod, fn in orig_exp:
            exp_mod.forward = fn

    stage_ratios = [np.concatenate(r) if r else np.zeros(0, dtype=np.float32) for r in ratios_by_stage]
    stage_stays = [np.concatenate(s) if s else np.zeros(0, dtype=np.bool_) for s in stay_by_stage]

    # Calibrate per-stage drop2_renorm sensitivity on calib_seqs[:8] ONLY (zero eval leakage!)
    calib_sub = calib_seqs[:8]
    ctrl_base = LoopMoEIntervention(model, mode="native")
    base_cal = evaluate_nll(model, calib_sub, ctrl_base, device)["nll"]
    calib_stage_sens = np.zeros(n_stages, dtype=np.float64)
    for sid in range(n_stages):
        ctrl_s = LoopMoEIntervention(model, mode="stage_ablation", target_stage=sid, stage_action="drop2_renorm")
        res_s = evaluate_nll(model, calib_sub, ctrl_s, device)
        # Sensitivity normalized by fraction of visits in this stage
        calib_stage_sens[sid] = max(1e-5, (res_s["nll"] - base_cal) / max(1.0 - res_s["active_expert_frac"], 1e-6))

    return {
        "counts": counts,
        "out_energy": out_energy,
        "stage_names": st_names,
        "stage_ratios": stage_ratios,
        "stage_stays": stage_stays,
        "calib_stage_sens": calib_stage_sens,
    }


@torch.inference_mode()
def run_integer_depth_profile(
    model,
    seqs: list[dict[str, Any]],
    depths: list[int],
    device: torch.device,
) -> list[dict[str, Any]]:
    """Measure exact NLL and output logit entropy/norm across every integer depth R."""
    rows: list[dict[str, Any]] = []
    for d in depths:
        nlls = []
        logit_stds = []
        with set_ifm_loop_depth(model, d):
            for item in seqs:
                toks = item["tokens"]
                inp = torch.tensor([toks[:-1]], device=device, dtype=torch.long)
                tgt = torch.tensor(toks[1:], device=device, dtype=torch.long)
                logits, _, _ = model(inp, multi_segments=False, cache=None)
                lf = logits[0].float()
                nlls.append(float(F.cross_entropy(lf, tgt).item()))
                logit_stds.append(float(lf.std(dim=-1).mean().item()))
        rows.append({
            "depth": int(d),
            "nll": float(np.mean(nlls)),
            "logit_std": float(np.mean(logit_stds)),
        })
    return rows


# ---------------------------------------------------------------------------
# Full Per-Model Probe 2 Pipeline
# ---------------------------------------------------------------------------

def run_model_probe2(
    model_key: str,
    mspec: dict[str, Any],
    *,
    model_dir: Path,
    dataset_dir: Path,
    arc_samples: list[dict[str, Any]],
    cfg: dict[str, Any],
    device: torch.device,
    run_obj: Any,
) -> dict[str, Any]:
    artifact_dir = model_dir / mspec["local_dir"]
    seed = int(cfg.get("seed", 20261126))
    print(f"\n[probe2:{model_key}] Loading {artifact_dir} ...", flush=True)
    model, tokenizer, _mcfg = load_ifm_model_for_autograd(
        artifact_dir, device=str(device), attention_backend="sdpa"
    )
    n_layers = len(model.layers)
    n_experts = int(model.layers[0].moe.experts.n_local_experts)
    loop_start = int(model.loop_start_layer)
    loop_end = int(model.loop_end_layers)
    n_loops = int(model.loop_times)
    st_names = stage_names_for(n_layers, loop_start, loop_end, n_loops)
    n_stages = len(st_names)

    splits = build_disjoint_text_code_splits(
        dataset_dir,
        lambda t: tokenizer.encode(t, bos=True, eos=False),
        seq_len=int(cfg.get("seq_len", 256)),
        n_calib_a=int(cfg.get("n_calib", 16)),
        n_calib_b=4,
        n_adapt_train=4,
        n_eval_heldout=int(cfg.get("n_eval", 32)),
        seed=seed,
    )
    calib_seqs = splits["calib_a"]
    eval_seqs = splits["eval_heldout"]

    # 0) Exactness verification
    with run_obj.phase(f"{model_key}_verify", total=1, unit="check") as pv:
        inp = torch.tensor([eval_seqs[0]["tokens"][:64]], device=device, dtype=torch.long)
        with torch.inference_mode():
            ref = model(inp, multi_segments=False, cache=None)[0].float()
            ctrl_nat = LoopMoEIntervention(model, mode="native")
            with ctrl_nat.attached():
                ctrl_nat.reset_seq()
                hooked = model(inp, multi_segments=False, cache=None)[0].float()
            ctrl_all_alive = LoopMoEIntervention(
                model, mode="prune", alive_mask=torch.ones((n_layers, n_experts), device=device, dtype=torch.bool)
            )
            with ctrl_all_alive.attached():
                ctrl_all_alive.reset_seq()
                pruned_noop = model(inp, multi_segments=False, cache=None)[0].float()
        d1 = float((ref - hooked).abs().max().item())
        d2 = float((ref - pruned_noop).abs().max().item())
        if d1 > 0.0 or d2 > 1e-4:
            raise RuntimeError(f"Exactness FAILED: native_diff={d1}, prune_noop_diff={d2}")
        print(f"[probe2:{model_key}] Exactness OK: native_diff={d1:.1e}, prune_noop_diff={d2:.1e}", flush=True)
        pv.step(1)

    # 1) Calibration pass
    with run_obj.phase(f"{model_key}_calib", total=1, unit="pass") as pc:
        calib = collect_calib_skip_and_prune_stats(model, calib_seqs, device)
        print(
            f"[probe2:{model_key}] Calib per-stage sensitivity (drop2_renorm ΔNLL/drop_frac): "
            + ", ".join(f"{st_names[i]}={calib['calib_stage_sens'][i]:.3f}" for i in range(n_stages)),
            flush=True,
        )
        pc.step(1)

    # Baseline BF16
    ctrl_base = LoopMoEIntervention(model, mode="native")
    base_ev = evaluate_nll(model, eval_seqs, ctrl_base, device)
    base_arr = np.asarray(base_ev["per_seq"], dtype=np.float64)
    base_arc = evaluate_arc(model, tokenizer, arc_samples, ctrl_base, device)
    print(f"[probe2:{model_key}] Base BF16: NLL={base_ev['nll']:.4f}, ARC-150={base_arc*100:.1f}%", flush=True)

    # 2) Part A: Per-Stage MoE Redundancy Map (drop2_renorm, drop2_noren, skip_moe)
    stage_map: list[dict[str, Any]] = []
    with run_obj.phase(f"{model_key}_stage_map", total=n_stages, unit="stages") as psm:
        for sid in range(n_stages):
            row: dict[str, Any] = {"stage": st_names[sid], "stage_id": sid}
            for act in ("drop2_renorm", "drop2_noren", "skip_moe"):
                c = LoopMoEIntervention(model, mode="stage_ablation", target_stage=sid, stage_action=act)
                ev = evaluate_nll(model, eval_seqs, c, device)
                row[f"{act}_dnll"] = float(ev["nll"] - base_ev["nll"])
                row[f"{act}_active_frac"] = ev["active_expert_frac"]
            stage_map.append(row)
            print(
                f"  [stage_map:{model_key}] {st_names[sid]:8s} | "
                f"drop2_renorm={row['drop2_renorm_dnll']:+.4f} | "
                f"drop2_noren={row['drop2_noren_dnll']:+.4f} | "
                f"skip_moe={row['skip_moe_dnll']:+.4f}",
                flush=True,
            )
            psm.row(
                f"{model_key}_stages",
                stage=st_names[sid],
                drop2_renorm_dnll=row["drop2_renorm_dnll"],
                skip_moe_dnll=row["skip_moe_dnll"],
            )
            psm.step(sid + 1)

    # 3) Part B: Dynamic Expert Skipping at Matched Active-Expert Cuts (-25%, -35%, -45%)
    # Note: dropping 2nd expert on fraction q of token-visits reduces total active experts by q/2!
    # So -25% active experts => drop 2nd expert on 50% of visits; -35% => 70%; -45% => 90%!
    sens = calib["calib_stage_sens"]
    sens_norm = sens / max(float(np.mean(sens)), 1e-8)
    # Build stage-only drop ordering (drop least sensitive stages first)
    stage_order_asc = sorted(range(n_stages), key=lambda s: (float(sens[s]), s))

    skip_results: dict[str, dict[str, Any]] = {}
    skip_per_seq: dict[str, np.ndarray] = {"base": base_arr}
    target_cuts = [0.25, 0.35, 0.45]

    dyn_specs: list[dict[str, Any]] = []
    for cut in target_cuts:
        q_second = min(0.99, cut * 2.0)
        tag = f"cut{int(round(cut*100))}"
        # 1. Uniform gate-ratio (ACE-style non-looped baseline)
        tau_u = calibrate_drop_threshold(calib["stage_ratios"], q_second)
        dyn_specs.append({
            "name": f"dyn_{tag}_uniform",
            "cut": cut,
            "ctrl": LoopMoEIntervention(model, mode="dynamic_skip", dyn_threshold=tau_u, stage_weights=None),
            "arc": cut in (0.25, 0.35),
        })
        # 2. Loop-sensitivity-weighted gate-ratio (renorm)
        weighted_scores = [calib["stage_ratios"][s] * sens_norm[s] for s in range(n_stages)]
        tau_ls = calibrate_drop_threshold(weighted_scores, q_second)
        dyn_specs.append({
            "name": f"dyn_{tag}_loop_sens",
            "cut": cut,
            "ctrl": LoopMoEIntervention(model, mode="dynamic_skip", dyn_threshold=tau_ls, stage_weights=sens_norm),
            "arc": cut in (0.25, 0.35),
        })
        # 3. Cross-loop stay-aware + loop-sensitivity
        stay_factor = 0.70
        stay_scores = [
            calib["stage_ratios"][s] * sens_norm[s] * np.where(calib["stage_stays"][s], stay_factor, 1.0)
            for s in range(n_stages)
        ]
        tau_st = calibrate_drop_threshold(stay_scores, q_second)
        dyn_specs.append({
            "name": f"dyn_{tag}_stay_sens",
            "cut": cut,
            "ctrl": LoopMoEIntervention(
                model, mode="dynamic_skip", dyn_threshold=tau_st, stage_weights=sens_norm, stay_factor=stay_factor
            ),
            "arc": False,
        })
        # 4. Pure stage-only drop (drop k=2->1 on whole least-sensitive stages)
        # Approximate cut by selecting prefix of stage_order_asc
        stage_counts = np.array([len(r) for r in calib["stage_ratios"]], dtype=np.float64)
        tot_visits = max(float(stage_counts.sum()), 1.0)
        chosen_stages: set[int] = set()
        acc_frac = 0.0
        for s in stage_order_asc:
            if acc_frac + 0.5 * (stage_counts[s] / tot_visits) <= cut + 0.04:
                chosen_stages.add(s)
                acc_frac += 0.5 * (stage_counts[s] / tot_visits)
        dyn_specs.append({
            "name": f"dyn_{tag}_stage_only",
            "cut": cut,
            "ctrl": LoopMoEIntervention(
                model, mode="dynamic_skip", dyn_threshold=-1e9, stage_drop2_set=chosen_stages
            ),
            "arc": cut == 0.25,
        })

    # Also add full top-1 (50% active expert cut everywhere) for reference!
    dyn_specs.append({
        "name": "dyn_cut50_all_top1",
        "cut": 0.50,
        "ctrl": LoopMoEIntervention(model, mode="dynamic_skip", dyn_threshold=1e9),
        "arc": True,
    })

    with run_obj.phase(f"{model_key}_dyn_skip", total=len(dyn_specs), unit="conds") as pdyn:
        for i, sp in enumerate(dyn_specs, 1):
            ev = evaluate_nll(model, eval_seqs, sp["ctrl"], device)
            arr = np.asarray(ev["per_seq"], dtype=np.float64)
            skip_per_seq[sp["name"]] = arr
            dnll, lo, hi = paired_bootstrap_ci(arr - base_arr)
            arc_acc = evaluate_arc(model, tokenizer, arc_samples, sp["ctrl"], device) if sp["arc"] else None
            skip_results[sp["name"]] = {
                "target_cut": sp["cut"],
                "actual_active_frac": ev["active_expert_frac"],
                "actual_cut": 1.0 - ev["active_expert_frac"],
                "nll": ev["nll"],
                "delta_nll": dnll,
                "delta_nll_ci95": [lo, hi],
                "arc_acc": arc_acc,
            }
            arc_s = f"{arc_acc*100:.1f}%" if arc_acc is not None else "  n/a"
            print(
                f"  [dyn_skip:{model_key}] {sp['name']:22s} cut={100*(1-ev['active_expert_frac']):4.1f}% "
                f"NLL={ev['nll']:.4f} (Δ={dnll:+.4f}) ARC={arc_s}",
                flush=True,
            )
            pdyn.row(
                f"{model_key}_skip",
                policy=sp["name"],
                cut_pct=100.0 * (1.0 - ev["active_expert_frac"]),
                delta_nll=dnll,
                arc_acc=arc_acc if arc_acc is not None else 0.0,
            )
            pdyn.step(i)

    # 4) Part C: Static Expert Pruning (20%, 35%, 50% experts permanently removed)
    counts = calib["counts"]
    out_energy = calib["out_energy"]
    global_freq = counts.sum(axis=1).astype(np.float64)
    global_energy = out_energy.sum(axis=1).astype(np.float64)
    visit_tots = np.maximum(counts.sum(axis=2, keepdims=True), 1).astype(np.float64)
    max_stage_freq = (counts / visit_tots).max(axis=1)

    # Sensitivity-weighted output energy across stages
    stage_sens_mat = np.zeros((n_layers, n_loops), dtype=np.float64)
    for l in range(n_layers):
        for r in range(n_loops):
            if (l < loop_start or l >= loop_end) and r > 0:
                continue
            sid = stage_id_for(l, r, loop_start, loop_end, n_loops)
            stage_sens_mat[l, r] = sens_norm[sid]
    stage_sens_energy = (out_energy * stage_sens_mat[:, :, None]).sum(axis=1)

    rng = np.random.default_rng(seed)
    rand_scores = rng.random((n_layers, n_experts))

    prune_methods = {
        "random": rand_scores,
        "global_freq": global_freq,
        "global_energy": global_energy,
        "max_stage_freq": max_stage_freq,
        "stage_sens_energy": stage_sens_energy,
    }
    prune_fracs = [0.20, 0.35, 0.50]
    prune_specs = []
    for pf in prune_fracs:
        for mname, sc in prune_methods.items():
            prune_specs.append({
                "name": f"prune{int(round(pf*100))}_{mname}",
                "frac": pf,
                "method": mname,
                "scores": sc,
                "arc": (pf in (0.35, 0.50)) and (mname in ("global_freq", "max_stage_freq", "stage_sens_energy")),
            })

    prune_results: dict[str, dict[str, Any]] = {}
    prune_per_seq: dict[str, np.ndarray] = {"base": base_arr}
    with run_obj.phase(f"{model_key}_prune", total=len(prune_specs), unit="conds") as ppr:
        for i, sp in enumerate(prune_specs, 1):
            alive_np = build_alive_expert_mask(sp["scores"], sp["frac"], min_alive_per_layer=2)
            alive_t = torch.tensor(alive_np, device=device, dtype=torch.bool)
            ctrl_p = LoopMoEIntervention(model, mode="prune", alive_mask=alive_t)
            ev = evaluate_nll(model, eval_seqs, ctrl_p, device)
            arr = np.asarray(ev["per_seq"], dtype=np.float64)
            prune_per_seq[sp["name"]] = arr
            dnll, lo, hi = paired_bootstrap_ci(arr - base_arr)
            arc_acc = evaluate_arc(model, tokenizer, arc_samples, ctrl_p, device) if sp["arc"] else None
            actual_pruned = 1.0 - float(alive_np.mean())
            prune_results[sp["name"]] = {
                "prune_frac": actual_pruned,
                "method": sp["method"],
                "nll": ev["nll"],
                "delta_nll": dnll,
                "delta_nll_ci95": [lo, hi],
                "arc_acc": arc_acc,
            }
            arc_s = f"{arc_acc*100:.1f}%" if arc_acc is not None else "  n/a"
            print(
                f"  [prune:{model_key}] {sp['name']:26s} pruned={actual_pruned*100:4.1f}% "
                f"NLL={ev['nll']:.4f} (Δ={dnll:+.4f}) ARC={arc_s}",
                flush=True,
            )
            ppr.row(
                f"{model_key}_prune",
                policy=sp["name"],
                prune_pct=actual_pruned * 100.0,
                delta_nll=dnll,
                arc_acc=arc_acc if arc_acc is not None else 0.0,
            )
            ppr.step(i)

    # 5) Part D: Full Integer Depth Profile
    depths = list(range(1, 9)) if model_key == "ouro_moe" else list(range(1, 17))
    with run_obj.phase(f"{model_key}_depth_profile", total=1, unit="sweep") as pdp:
        depth_profile = run_integer_depth_profile(model, eval_seqs[:16], depths, device)
        print(
            f"  [depth_profile:{model_key}] "
            + " | ".join(f"R={r['depth']}:{r['nll']:.2f}" for r in depth_profile),
            flush=True,
        )
        pdp.step(1)

    # Paired contrasts (positive => first wins, lower NLL)
    contrasts: dict[str, dict[str, Any]] = {}
    for cut in target_cuts:
        tag = f"cut{int(round(cut*100))}"
        for cand in (f"dyn_{tag}_loop_sens", f"dyn_{tag}_stay_sens", f"dyn_{tag}_stage_only"):
            ref = f"dyn_{tag}_uniform"
            if cand in skip_per_seq and ref in skip_per_seq:
                d = skip_per_seq[ref] - skip_per_seq[cand]
                m, lo, hi = paired_bootstrap_ci(d)
                contrasts[f"{cand}_vs_{ref}"] = {"advantage_nll": m, "ci95": [lo, hi], "wins": int((d > 0).sum()), "n": len(d)}
    for pf in prune_fracs:
        p = int(round(pf * 100))
        for cand in (f"prune{p}_stage_sens_energy", f"prune{p}_max_stage_freq", f"prune{p}_global_energy"):
            ref = f"prune{p}_global_freq"
            if cand in prune_per_seq and ref in prune_per_seq:
                d = prune_per_seq[ref] - prune_per_seq[cand]
                m, lo, hi = paired_bootstrap_ci(d)
                contrasts[f"{cand}_vs_{ref}"] = {"advantage_nll": m, "ci95": [lo, hi], "wins": int((d > 0).sum()), "n": len(d)}

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model_key": model_key,
        "base_nll": base_ev["nll"],
        "base_arc": base_arc,
        "calib_stage_sens": {st_names[i]: float(sens[i]) for i in range(n_stages)},
        "stage_map": stage_map,
        "dynamic_skip": skip_results,
        "static_prune": prune_results,
        "depth_profile": depth_profile,
        "contrasts": contrasts,
    }


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Round 1 Probe 2: Looped MoE Dynamic Skipping, Stage Pruning & Depth Profile",
        "",
        f"- GPU: `{summary['gpu_index']}` ({summary['gpu_name']}), elapsed `{summary['elapsed_sec']:.1f}s`",
        "",
    ]
    for mk, mres in summary["models"].items():
        lines.append(f"## Model: `{mk}` (Base NLL={mres['base_nll']:.4f}, ARC-150={mres['base_arc']*100:.1f}%)")
        lines.append("")
        lines.append("### Part A: Per-Stage MoE Redundancy Map (ΔNLL vs Base)")
        lines.append("| Stage | Drop 2nd (Renorm) | Drop 2nd (NoRen) | Skip Entire MoE (k=0) |")
        lines.append("|---|---:|---:|---:|")
        for r in mres["stage_map"]:
            lines.append(
                f"| `{r['stage']}` | {r['drop2_renorm_dnll']:+.4f} | "
                f"{r['drop2_noren_dnll']:+.4f} | {r['skip_moe_dnll']:+.4f} |"
            )
        lines.append("")
        lines.append("### Part B: Dynamic Expert Skipping (Matched Active-Expert Cuts)")
        lines.append("| Policy | Actual Cut% | NLL | ΔNLL vs Base (95% CI) | ARC-150 |")
        lines.append("|---|---:|---:|---:|---:|")
        for pname, rec in mres["dynamic_skip"].items():
            lo, hi = rec["delta_nll_ci95"]
            arc_s = f"{rec['arc_acc']*100:.1f}%" if rec["arc_acc"] is not None else "—"
            lines.append(
                f"| `{pname}` | {rec['actual_cut']*100:.1f}% | {rec['nll']:.4f} | "
                f"{rec['delta_nll']:+.4f} [{lo:+.4f}, {hi:+.4f}] | {arc_s} |"
            )
        lines.append("")
        lines.append("### Part C: Static Expert Pruning")
        lines.append("| Policy | Pruned% | NLL | ΔNLL vs Base (95% CI) | ARC-150 |")
        lines.append("|---|---:|---:|---:|---:|")
        for pname, rec in mres["static_prune"].items():
            lo, hi = rec["delta_nll_ci95"]
            arc_s = f"{rec['arc_acc']*100:.1f}%" if rec["arc_acc"] is not None else "—"
            lines.append(
                f"| `{pname}` | {rec['prune_frac']*100:.1f}% | {rec['nll']:.4f} | "
                f"{rec['delta_nll']:+.4f} [{lo:+.4f}, {hi:+.4f}] | {arc_s} |"
            )
        lines.append("")
        lines.append("### Part D: Integer Depth Profile (NLL & Logit Std)")
        lines.append("| Depth R | NLL | Logit Std |")
        lines.append("|---:|---:|---:|")
        for r in mres["depth_profile"]:
            lines.append(f"| {r['depth']} | {r['nll']:.4f} | {r['logit_std']:.3f} |")
        lines.append("")
        lines.append("### Key Paired Contrasts (positive = first has LOWER NLL)")
        for ck, cv in mres["contrasts"].items():
            lo, hi = cv["ci95"]
            lines.append(f"- `{ck}`: `{cv['advantage_nll']:+.4f}` CI `[{lo:+.4f}, {hi:+.4f}]` ({cv['wins']}/{cv['n']} wins)")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--xllm-loop-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    gpu_idx, gpu_lock, gpu_status = acquire_idle_gpu(cfg.get("preferred_gpu"))
    try:
        print(
            f"[probe2] Acquired idle GPU {gpu_idx}: {gpu_status.name} "
            f"(mem={gpu_status.mem_used_mib}MiB, util={gpu_status.util_pct}%)",
            flush=True,
        )
        device = torch.device(f"cuda:{gpu_idx}")
        torch.cuda.set_device(device)

        boot = bootstrap_xllm(args.xllm_loop_dir, attention_backend="sdpa")
        print(f"[probe2] xllm bootstrap: {json.dumps(boot)}", flush=True)

        arc_samples = load_arc_challenge_samples(
            Path(args.dataset_dir), n_samples=int(cfg.get("n_arc", 150)), seed=int(cfg.get("seed", 20261126))
        )

        plan = [
            {"phase": "setup", "of": 1, "unit": "env"},
            {"phase": "download", "of": 1, "unit": "assets"},
        ]
        for mk in cfg["models"]:
            n_st = 4 if mk == "ouro_moe" else 10
            plan.extend([
                {"phase": f"{mk}_verify", "of": 1, "unit": "check"},
                {"phase": f"{mk}_calib", "of": 1, "unit": "pass"},
                {"phase": f"{mk}_stage_map", "of": n_st, "unit": "stages"},
                {"phase": f"{mk}_dyn_skip", "of": 13, "unit": "conds"},
                {"phase": f"{mk}_prune", "of": 15, "unit": "conds"},
                {"phase": f"{mk}_depth_profile", "of": 1, "unit": "sweep"},
            ])
        plan.append({"phase": "report", "of": 1, "unit": "file"})
        orx.plan(plan)
        run_obj = orx.progress(total=1, unit="run")

        models_out: dict[str, Any] = {}
        for mk, mspec in cfg["models"].items():
            models_out[mk] = run_model_probe2(
                mk,
                mspec,
                model_dir=Path(args.model_dir),
                dataset_dir=Path(args.dataset_dir),
                arc_samples=arc_samples,
                cfg=cfg,
                device=device,
                run_obj=run_obj,
            )

        summary = {
            "idea": "probe2_loopmoe_skip_prune",
            "gpu_index": gpu_idx,
            "gpu_name": gpu_status.name,
            "elapsed_sec": time.time() - t0,
            "config": cfg,
            "models": models_out,
        }
        with run_obj.phase("report", total=1, unit="file") as pr:
            (out_dir / "summary_probe2_skip_prune.json").write_text(json.dumps(summary, indent=2))
            md = render_report(summary)
            (out_dir / "report_probe2_skip_prune.md").write_text(md)
            print("========== REPORT (report_probe2_skip_prune.md) ==========", flush=True)
            print(md, flush=True)
            print("========== END REPORT ==========", flush=True)

            headline: dict[str, float] = {}
            for mk, mres in models_out.items():
                if "dyn_cut35_loop_sens_vs_dyn_cut35_uniform" in mres["contrasts"]:
                    headline[f"{mk}_skip35_loop_vs_unif"] = mres["contrasts"]["dyn_cut35_loop_sens_vs_dyn_cut35_uniform"]["advantage_nll"]
                if "prune35_stage_sens_energy_vs_prune35_global_freq" in mres["contrasts"]:
                    headline[f"{mk}_prune35_sens_vs_freq"] = mres["contrasts"]["prune35_stage_sens_energy_vs_prune35_global_freq"]["advantage_nll"]
                if "dyn_cut50_all_top1" in mres["dynamic_skip"]:
                    headline[f"{mk}_top1_dnll"] = mres["dynamic_skip"]["dyn_cut50_all_top1"]["delta_nll"]
            pr.step(1, **headline)
    finally:
        gpu_lock.release()


if __name__ == "__main__":
    main()
