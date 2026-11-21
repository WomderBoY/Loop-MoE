"""A2: Decompose routing predictability across layer, loop, and token histories.

Strictly causal: every predictor scores target (t, visit_idx) using ONLY events
observed before that target router fires, fitted exclusively on fit_request_ids.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any

from .trace import read_json, read_jsonl, write_csv


PREDICTORS = (
    "pool_stage_pop",
    "prev_token_same_stage",
    "prev_loop_copy",
    "prev_loop_transition",
    "prev_phys_layer",
    "combined_no_prev_layer",
    "combined_loop",
    "general_history",
    "shuffled_cross_loop",
)


def read_ifm_trace(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = read_jsonl(path)
    meta = next(records, None)
    if not meta or meta.get("schema_version") != 2:
        raise ValueError(f"Expected IFM v2 trace in {path}")
    requests = list(records)
    if len(requests) != meta["request_count"]:
        raise ValueError(f"Incomplete trace {path}: got {len(requests)} != {meta['request_count']}")
    return meta, requests


def _build_schedule_info(meta: dict[str, Any]) -> dict[str, Any]:
    order = [(pool, loop) for pool, loop in meta["execution_order"]]
    pool_positions: dict[str, list[int]] = defaultdict(list)
    for idx, (pool, _) in enumerate(order):
        pool_positions[pool].append(idx)
    prev_loop_visit: list[int | None] = []
    for idx, (pool, loop) in enumerate(order):
        if loop is None or loop == 0:
            prev_loop_visit.append(None)
        else:
            prev_idx = next(
                (p for p in reversed(pool_positions[pool]) if p < idx),
                None,
            )
            prev_loop_visit.append(prev_idx)
    return {
        "order": order,
        "pool_positions": dict(pool_positions),
        "prev_loop_visit": prev_loop_visit,
        "pool_visits_per_token": {p: len(pos) for p, pos in pool_positions.items()},
    }


def _bootstrap_ci(values: list[float], n_boot: int = 1000, seed: int = 20261121) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[max(0, int(0.025 * n_boot))]
    hi = means[min(n_boot - 1, int(0.975 * n_boot))]
    return lo, hi


class A2Models:
    """Empirical conditional tables fitted strictly on fit_request_ids."""

    def __init__(self, meta: dict[str, Any], sched: dict[str, Any], cfg: dict[str, Any]):
        self.meta = meta
        self.sched = sched
        self.alpha = float(cfg["score_alpha"])
        self.shrink = float(cfg["shrinkage"])
        self.max_cells = int(cfg["max_table_cells"])
        self.num_experts = int(next(iter(meta["pools"].values()))["experts"])
        self.top_k = int(meta["top_k"])
        self.uniform = self.top_k / self.num_experts

        # 1. Pool + stage popularity: (pool, target_loop) -> [count_e], total
        self.stage_counts: dict[tuple[str, int | None], list[int]] = defaultdict(lambda: [0] * self.num_experts)
        self.stage_totals: Counter = Counter()
        # Pool-only popularity (for general_history which does NOT know loop stage)
        self.pool_counts: dict[str, list[int]] = defaultdict(lambda: [0] * self.num_experts)
        self.pool_totals: Counter = Counter()

        # 2. Prev-token same-stage transition: (pool, target_loop, src_e) -> [count_tgt_e], src_total
        self.prev_tok_trans: dict[tuple[str, int | None, int], list[int]] = defaultdict(lambda: [0] * self.num_experts)
        self.prev_tok_totals: Counter = Counter()

        # 3. Prev-loop same-pool transition: (pool, src_loop, tgt_loop, src_e) -> [count_tgt_e], src_total
        self.prev_loop_trans: dict[tuple[str, int, int, int], list[int]] = defaultdict(lambda: [0] * self.num_experts)
        self.prev_loop_totals: Counter = Counter()

        # 4. Prev-physical-layer transition: (prev_pool, pool, target_loop, prev_e) -> [count_tgt_e]
        self.prev_layer_trans: dict[tuple[str, str, int | None, int], list[int]] = defaultdict(
            lambda: [0] * self.num_experts
        )
        self.prev_layer_totals: Counter = Counter()

        # 5. General history (NO loop/stage index; matched history span H + prev_pool transition)
        # Lag-1 same-pool transition without stage label: (pool, src_e) -> [count_tgt_e]
        self.gen_lag1_trans: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0] * self.num_experts)
        self.gen_lag1_totals: Counter = Counter()
        # Lag-H (period-matched, same number of past visits H without stage label): (pool, src_e) -> [count_tgt_e]
        self.gen_lagH_trans: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0] * self.num_experts)
        self.gen_lagH_totals: Counter = Counter()
        # General cross-pool transition without stage label: (prev_pool, pool, prev_e) -> [count_tgt_e]
        self.gen_prev_pool_trans: dict[tuple[str, str, int], list[int]] = defaultdict(lambda: [0] * self.num_experts)
        self.gen_prev_pool_totals: Counter = Counter()

        # 6. Shuffled cross-loop null control (within pool, src->tgt loop, category)
        self.shuf_loop_trans: dict[tuple[str, int, int, int], list[int]] = defaultdict(lambda: [0] * self.num_experts)
        self.shuf_loop_totals: Counter = Counter()

    def fit(self, fit_requests: list[dict[str, Any]], shuffle_seed: int) -> None:
        order = self.sched["order"]
        prev_loop_visit = self.sched["prev_loop_visit"]
        visits_per_tok = self.sched["pool_visits_per_token"]

        # Collect cross-loop pairs for within-(pool, src_loop, tgt_loop, category) shuffle
        loop_pairs: dict[tuple[str, int, int, str], list[tuple[list[int], list[int]]]] = defaultdict(list)

        for req in fit_requests:
            cat = req["category"]
            prev_step_visits: list[list[int]] | None = None
            pool_recent: dict[str, deque[list[int]]] = defaultdict(deque)
            last_visit_pool: str | None = None
            last_visit_ids: list[int] | None = None

            for step in req["steps"]:
                visits = step["visits"]
                for v_idx, visit in enumerate(visits):
                    pool, loop = order[v_idx]
                    tgt_ids: list[int] = visit[0]

                    # Stage & pool marginals
                    sc = self.stage_counts[pool, loop]
                    pc = self.pool_counts[pool]
                    for e in tgt_ids:
                        sc[e] += 1
                        pc[e] += 1
                    self.stage_totals[pool, loop] += 1
                    self.pool_totals[pool] += 1

                    # Prev token same stage
                    if prev_step_visits is not None:
                        pt_ids = prev_step_visits[v_idx]
                        for se in pt_ids:
                            row = self.prev_tok_trans[pool, loop, se]
                            for te in tgt_ids:
                                row[te] += 1
                            self.prev_tok_totals[pool, loop, se] += 1

                    # Prev loop same pool
                    pl_idx = prev_loop_visit[v_idx]
                    if pl_idx is not None:
                        src_loop = order[pl_idx][1]
                        pl_ids = visits[pl_idx][0]
                        loop_pairs[pool, src_loop, loop, cat].append((pl_ids, tgt_ids))
                        for se in pl_ids:
                            row = self.prev_loop_trans[pool, src_loop, loop, se]
                            for te in tgt_ids:
                                row[te] += 1
                            self.prev_loop_totals[pool, src_loop, loop, se] += 1

                    # Prev physical layer
                    if last_visit_pool is not None and last_visit_ids is not None:
                        for se in last_visit_ids:
                            row = self.prev_layer_trans[last_visit_pool, pool, loop, se]
                            row_g = self.gen_prev_pool_trans[last_visit_pool, pool, se]
                            for te in tgt_ids:
                                row[te] += 1
                                row_g[te] += 1
                            self.prev_layer_totals[last_visit_pool, pool, loop, se] += 1
                            self.gen_prev_pool_totals[last_visit_pool, pool, se] += 1

                    # General history (lag-1 and lag-H on same pool, stage-blind)
                    hist = pool_recent[pool]
                    if len(hist) >= 1:
                        for se in hist[0]:
                            row = self.gen_lag1_trans[pool, se]
                            for te in tgt_ids:
                                row[te] += 1
                            self.gen_lag1_totals[pool, se] += 1
                    h_span = visits_per_tok[pool]
                    if len(hist) >= h_span:
                        for se in hist[h_span - 1]:
                            row = self.gen_lagH_trans[pool, se]
                            for te in tgt_ids:
                                row[te] += 1
                            self.gen_lagH_totals[pool, se] += 1

                    hist.appendleft(tgt_ids)
                    if len(hist) > h_span:
                        hist.pop()
                    last_visit_pool = pool
                    last_visit_ids = tgt_ids

                prev_step_visits = [v[0] for v in visits]

        # Fit shuffled cross-loop null control (within layer, stage, task category)
        rng = random.Random(shuffle_seed)
        for (pool, src_loop, tgt_loop, _cat), pairs in sorted(loop_pairs.items()):
            srcs = [p[0] for p in pairs]
            tgts = [p[1] for p in pairs]
            rng.shuffle(tgts)
            for src_ids, tgt_ids in zip(srcs, tgts):
                for se in src_ids:
                    row = self.shuf_loop_trans[pool, src_loop, tgt_loop, se]
                    for te in tgt_ids:
                        row[te] += 1
                    self.shuf_loop_totals[pool, src_loop, tgt_loop, se] += 1

        # Cap table sizes uniformly for fairness (same max_cells budget)
        for table_attr, tot_attr in (
            ("prev_tok_trans", "prev_tok_totals"),
            ("prev_loop_trans", "prev_loop_totals"),
            ("prev_layer_trans", "prev_layer_totals"),
            ("gen_lag1_trans", "gen_lag1_totals"),
            ("gen_lagH_trans", "gen_lagH_totals"),
            ("gen_prev_pool_trans", "gen_prev_pool_totals"),
            ("shuf_loop_trans", "shuf_loop_totals"),
        ):
            table = getattr(self, table_attr)
            totals = getattr(self, tot_attr)
            if len(table) > self.max_cells:
                keep = {k for k, _ in totals.most_common(self.max_cells)}
                setattr(self, table_attr, {k: v for k, v in table.items() if k in keep})

    def _stage_prior(self, pool: str, loop: int | None) -> list[float]:
        tot = self.stage_totals[pool, loop]
        if not tot:
            return [self.uniform] * self.num_experts
        cnts = self.stage_counts[pool, loop]
        return [(c + self.shrink * self.uniform) / (tot + self.shrink) for c in cnts]

    def _pool_prior(self, pool: str) -> list[float]:
        tot = self.pool_totals[pool]
        if not tot:
            return [self.uniform] * self.num_experts
        cnts = self.pool_counts[pool]
        return [(c + self.shrink * self.uniform) / (tot + self.shrink) for c in cnts]

    def _trans_prob(
        self,
        table: dict,
        totals: Counter,
        keys: list[tuple],
        prior: list[float],
    ) -> list[float]:
        if not keys:
            return list(prior)
        acc = [0.0] * self.num_experts
        for key in keys:
            tot = totals.get(key, 0)
            row = table.get(key)
            if row is not None and tot > 0:
                denom = tot + self.shrink
                for e in range(self.num_experts):
                    acc[e] += (row[e] + self.shrink * prior[e]) / denom
            else:
                for e in range(self.num_experts):
                    acc[e] += prior[e]
        inv = 1.0 / len(keys)
        return [v * inv for v in acc]

    @staticmethod
    def _combine_probs(prior: list[float], components: list[list[float]]) -> list[float]:
        """Log-odds pooling relative to prior (no extra free parameters)."""
        eps = 1e-6
        out = []
        for e, p0 in enumerate(prior):
            p0c = min(1.0 - eps, max(eps, p0))
            logit = math.log(p0c / (1.0 - p0c))
            for comp in components:
                pc = min(1.0 - eps, max(eps, comp[e]))
                logit += math.log(pc / (1.0 - pc)) - math.log(p0c / (1.0 - p0c))
            out.append(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit)))))
        return out

    def predict_request(self, req: dict[str, Any]) -> list[dict[str, Any]]:
        """Score every decode visit causally; returns per-visit predictor scores."""
        order = self.sched["order"]
        prev_loop_visit = self.sched["prev_loop_visit"]
        visits_per_tok = self.sched["pool_visits_per_token"]

        prev_step_ids: list[list[int]] | None = None
        pool_recent_ids: dict[str, deque[list[int]]] = defaultdict(deque)
        pool_ema: dict[str, list[float]] = defaultdict(lambda: [0.0] * self.num_experts)
        stage_ema: dict[tuple[str, int | None], list[float]] = defaultdict(lambda: [0.0] * self.num_experts)
        last_pool: str | None = None
        last_ids: list[int] | None = None

        results = []
        for step_idx, step in enumerate(req["steps"]):
            visits = step["visits"]
            for v_idx, visit in enumerate(visits):
                pool, loop = order[v_idx]
                tgt_ids: list[int] = visit[0]
                stage_prior = self._stage_prior(pool, loop)
                pool_prior = self._pool_prior(pool)

                # 1. pool_stage_pop
                p_stage = stage_prior

                # 2. prev_token_same_stage (transition + stage EMA)
                if prev_step_ids is not None:
                    pt_ids = prev_step_ids[v_idx]
                    p_prev_tok = self._trans_prob(
                        self.prev_tok_trans,
                        self.prev_tok_totals,
                        [(pool, loop, se) for se in pt_ids],
                        stage_prior,
                    )
                    sema = stage_ema[pool, loop]
                    p_prev_tok = [0.75 * a + 0.25 * min(1.0, b) for a, b in zip(p_prev_tok, sema)]
                else:
                    p_prev_tok = list(stage_prior)

                # 3 & 4. prev_loop_copy & prev_loop_transition
                pl_idx = prev_loop_visit[v_idx]
                prev_loop_ids: list[int] | None = None
                src_loop: int | None = None
                if pl_idx is not None:
                    src_loop = order[pl_idx][1]
                    pl_visit = visits[pl_idx]
                    prev_loop_ids = pl_visit[0]
                    pl_cand_ids = pl_visit[3]
                    pl_cand_scores = pl_visit[4]
                    # Pure copy/retain from previous loop (tie-broken by stage popularity)
                    p_copy = [0.05 * sp for sp in stage_prior]
                    for cid, cscore in zip(pl_cand_ids, pl_cand_scores):
                        p_copy[cid] = max(p_copy[cid], 0.1 + 0.4 * float(cscore))
                    for sid, sraw in zip(pl_visit[0], pl_visit[2]):
                        p_copy[sid] = 0.6 + 0.4 * float(sraw)

                    # Transition A->B
                    p_loop_trans = self._trans_prob(
                        self.prev_loop_trans,
                        self.prev_loop_totals,
                        [(pool, src_loop, loop, se) for se in prev_loop_ids],
                        stage_prior,
                    )
                    # Shuffled cross-loop null control
                    p_shuf = self._trans_prob(
                        self.shuf_loop_trans,
                        self.shuf_loop_totals,
                        [(pool, src_loop, loop, se) for se in prev_loop_ids],
                        stage_prior,
                    )
                else:
                    p_copy = list(stage_prior)
                    p_loop_trans = list(stage_prior)
                    p_shuf = list(stage_prior)

                # 5. prev_phys_layer
                if last_pool is not None and last_ids is not None:
                    p_prev_layer = self._trans_prob(
                        self.prev_layer_trans,
                        self.prev_layer_totals,
                        [(last_pool, pool, loop, se) for se in last_ids],
                        stage_prior,
                    )
                else:
                    p_prev_layer = list(stage_prior)

                # 6. combined_no_prev_layer
                comps_no_pl = [p_prev_tok]
                if pl_idx is not None:
                    comps_no_pl.append(p_loop_trans)
                p_comb_no_pl = self._combine_probs(stage_prior, comps_no_pl)

                # 7. combined_loop
                comps_full = list(comps_no_pl)
                if last_pool is not None:
                    comps_full.append(p_prev_layer)
                p_comb_loop = self._combine_probs(stage_prior, comps_full)

                # 8. general_history (NO stage label; same lag-1, lag-H, prev_pool, pool EMA)
                hist = pool_recent_ids[pool]
                gen_comps = []
                if len(hist) >= 1:
                    gen_comps.append(
                        self._trans_prob(
                            self.gen_lag1_trans,
                            self.gen_lag1_totals,
                            [(pool, se) for se in hist[0]],
                            pool_prior,
                        )
                    )
                h_span = visits_per_tok[pool]
                if len(hist) >= h_span:
                    gen_comps.append(
                        self._trans_prob(
                            self.gen_lagH_trans,
                            self.gen_lagH_totals,
                            [(pool, se) for se in hist[h_span - 1]],
                            pool_prior,
                        )
                    )
                if last_pool is not None and last_ids is not None:
                    gen_comps.append(
                        self._trans_prob(
                            self.gen_prev_pool_trans,
                            self.gen_prev_pool_totals,
                            [(last_pool, pool, se) for se in last_ids],
                            pool_prior,
                        )
                    )
                p_gen_raw = self._combine_probs(pool_prior, gen_comps) if gen_comps else list(pool_prior)
                pema = pool_ema[pool]
                p_gen = [0.8 * a + 0.2 * min(1.0, b) for a, b in zip(p_gen_raw, pema)]

                scores = {
                    "pool_stage_pop": p_stage,
                    "prev_token_same_stage": p_prev_tok,
                    "prev_loop_copy": p_copy,
                    "prev_loop_transition": p_loop_trans,
                    "prev_phys_layer": p_prev_layer,
                    "combined_no_prev_layer": p_comb_no_pl,
                    "combined_loop": p_comb_loop,
                    "general_history": p_gen,
                    "shuffled_cross_loop": p_shuf,
                }
                results.append(
                    {
                        "step_idx": step_idx,
                        "visit_idx": v_idx,
                        "pool": pool,
                        "src_loop": src_loop,
                        "tgt_loop": loop,
                        "tgt_ids": tgt_ids,
                        "prev_loop_ids": prev_loop_ids,
                        "prev_logit_margin": visits[pl_idx][6] if pl_idx is not None else None,
                        "router_input_cosine": visit[9],
                        "logit_linf_diff": visit[10],
                        "scores": scores,
                    }
                )

                # Update online causal histories AFTER observing current visit
                hist.appendleft(tgt_ids)
                if len(hist) > h_span:
                    hist.pop()
                # EMA over observed top candidates
                pema_cur = pool_ema[pool]
                sema_cur = stage_ema[pool, loop]
                decay = 1.0 - self.alpha
                for e in range(self.num_experts):
                    pema_cur[e] *= decay
                    sema_cur[e] *= decay
                for cid, cscore in zip(visit[3], visit[4]):
                    pema_cur[cid] += self.alpha * float(cscore)
                    sema_cur[cid] += self.alpha * float(cscore)
                last_pool = pool
                last_ids = tgt_ids

            prev_step_ids = [v[0] for v in visits]

        return results


def _topk_indices(scores: list[float], m: int) -> list[int]:
    return sorted(range(len(scores)), key=lambda e: (-scores[e], e))[:m]


def run_a2(
    trace_path: Path,
    manifest: dict[str, Any],
    cfg: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    meta, requests = read_ifm_trace(trace_path)
    sched = _build_schedule_info(meta)
    req_by_id = {r["request_id"]: r for r in requests}
    fit_reqs = [req_by_id[rid] for rid in manifest["fit_request_ids"] if rid in req_by_id]
    eval_reqs = [req_by_id[rid] for rid in manifest["eval_request_ids"] if rid in req_by_id]
    if not fit_reqs or not eval_reqs:
        # Fixed-text control with small count: fall back to leave-first-half-fit if manifest ids don't match
        half = max(1, len(requests) // 2)
        fit_reqs = requests[:half]
        eval_reqs = requests[half:]

    models = A2Models(meta, sched, cfg)
    models.fit(fit_reqs, int(cfg["shuffle_seed"]))

    cand_ks: list[int] = list(cfg.get("candidate_ks", [2, 4, 6]))
    pools_meta = meta["pools"]

    # Accumulators
    req_rows: list[dict[str, Any]] = []
    layer_loop_stats: dict[tuple[str, str, int | None, int | None], dict[str, Any]] = {}
    retain_replace_acc: dict[tuple[str, str, int], Counter] = defaultdict(Counter)
    geom_acc: dict[tuple[str, str, str], Counter] = defaultdict(Counter)

    def _get_ll(pool: str, src_loop: int | None, tgt_loop: int | None) -> dict[str, Any]:
        region = pools_meta[pool]["region"]
        key = (pool, region, src_loop, tgt_loop)
        if key not in layer_loop_stats:
            layer_loop_stats[key] = {
                "n": 0,
                "top1_switches": 0,
                "overlap_sum": 0.0,
                "jaccard_sum": 0.0,
                "src_freq": Counter(),
                "tgt_freq": Counter(),
                "hits": {p: {m: 0 for m in cand_ks} for p in PREDICTORS},
                "targets": 0,
            }
        return layer_loop_stats[key]

    margin_bins = (0.05, 0.15, 0.30, 0.60, 1.20)
    cos_bins = (0.50, 0.70, 0.85, 0.92, 0.97)

    def _bin_label(val: float, edges: tuple[float, ...]) -> str:
        for i, e in enumerate(edges):
            if val < e:
                lo = "-inf" if i == 0 else f"{edges[i-1]:.2f}"
                return f"[{lo},{e:.2f})"
        return f"[{edges[-1]:.2f},+inf)"

    suff_violations = 0
    suff_total = 0

    for req in eval_reqs:
        preds = models.predict_request(req)
        req_acc: dict[tuple[str, str], Counter] = defaultdict(Counter)

        for item in preds:
            pool = item["pool"]
            region = pools_meta[pool]["region"]
            src_loop = item["src_loop"]
            tgt_loop = item["tgt_loop"]
            tgt_ids = item["tgt_ids"]
            tgt_set = set(tgt_ids)
            pl_ids = item["prev_loop_ids"]

            ll = _get_ll(pool, src_loop, tgt_loop)
            ll["n"] += 1
            ll["targets"] += len(tgt_set)
            ll["tgt_freq"].update(tgt_set)

            if pl_ids is not None:
                pl_set = set(pl_ids)
                inter = len(pl_set & tgt_set)
                ll["overlap_sum"] += inter / len(tgt_set)
                ll["jaccard_sum"] += inter / len(pl_set | tgt_set)
                ll["top1_switches"] += int(pl_ids[0] != tgt_ids[0])
                ll["src_freq"].update(pl_set)

                # Geometry & margin diagnostics
                cos = item["router_input_cosine"]
                margin = item["prev_logit_margin"]
                linf = item["logit_linf_diff"]
                any_flip = int(pl_set != tgt_set)
                top1_flip = int(pl_ids[0] != tgt_ids[0])
                if cos is not None and margin is not None and linf is not None:
                    suff = int(linf < 0.5 * margin)
                    if suff:
                        suff_total += 1
                        if any_flip:
                            suff_violations += 1
                    for axis, b in (("margin", _bin_label(margin, margin_bins)), ("cosine", _bin_label(cos, cos_bins))):
                        g = geom_acc[region, axis, b]
                        g["n"] += 1
                        g["cos_sum"] += cos
                        g["margin_sum"] += margin
                        g["linf_sum"] += linf
                        g["suff_ok"] += suff
                        g["any_flip"] += any_flip
                        g["top1_flip"] += top1_flip

            for pred_name in PREDICTORS:
                sc = item["scores"][pred_name]
                for m in cand_ks:
                    chosen = set(_topk_indices(sc, m))
                    h = len(chosen & tgt_set)
                    ll["hits"][pred_name][m] += h
                    req_acc[region, pred_name][f"hits_{m}"] += h
                    req_acc["all", pred_name][f"hits_{m}"] += h
                    if pl_ids is not None:
                        pl_set = set(pl_ids)
                        retained = tgt_set & pl_set
                        replaced = tgt_set - pl_set
                        rr_ret = retain_replace_acc[pred_name, "retained", m]
                        rr_ret["hits"] += len(chosen & retained)
                        rr_ret["targets"] += len(retained)
                        rr_rep = retain_replace_acc[pred_name, "replaced_new", m]
                        rr_rep["hits"] += len(chosen & replaced)
                        rr_rep["targets"] += len(replaced)
                req_acc[region, pred_name]["targets"] += len(tgt_set)
                req_acc["all", pred_name]["targets"] += len(tgt_set)

        for (region, pred_name), c in sorted(req_acc.items()):
            row = {
                "request_id": req["request_id"],
                "category": req["category"],
                "region": region,
                "predictor": pred_name,
                "targets": c["targets"],
            }
            for m in cand_ks:
                row[f"hits_at_{m}"] = c[f"hits_{m}"]
                row[f"recall_at_{m}"] = c[f"hits_{m}"] / c["targets"] if c["targets"] else 0.0
            req_rows.append(row)

    # Build per-layer-loop CSV
    ll_rows = []
    for (pool, region, src_loop, tgt_loop), st in sorted(
        layer_loop_stats.items(), key=lambda kv: (kv[0][0], -1 if kv[0][3] is None else kv[0][3])
    ):
        n = st["n"]
        mean_k = st["targets"] / n if n else 0.0
        marg_null = None
        if src_loop is not None and n > 0:
            indep = sum(st["src_freq"][e] * st["tgt_freq"][e] for e in st["src_freq"])
            marg_null = indep / (n * n * mean_k)
        row: dict[str, Any] = {
            "pool": pool,
            "region": region,
            "src_loop": "" if src_loop is None else src_loop,
            "tgt_loop": "" if tgt_loop is None else tgt_loop,
            "positions": n,
            "top1_switch_rate": st["top1_switches"] / n if src_loop is not None and n else "",
            "overlap": st["overlap_sum"] / n if src_loop is not None and n else "",
            "jaccard": st["jaccard_sum"] / n if src_loop is not None and n else "",
            "marginal_null_overlap": marg_null if marg_null is not None else "",
        }
        gen_r2 = st["hits"]["general_history"][2] / st["targets"] if st["targets"] else 0.0
        for p in PREDICTORS:
            for m in cand_ks:
                row[f"{p}_r{m}"] = st["hits"][p][m] / st["targets"] if st["targets"] else 0.0
        row["combined_minus_general_r2"] = row["combined_loop_r2"] - gen_r2
        row["prev_loop_trans_minus_general_r2"] = row["prev_loop_transition_r2"] - gen_r2
        ll_rows.append(row)

    # Summary by region & predictor with paired bootstrap CI vs general_history
    by_reg_pred: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in req_rows:
        by_reg_pred[r["region"], r["predictor"]].append(r)

    summary_rows = []
    for region in sorted({r["region"] for r in req_rows}):
        gen_by_req = {r["request_id"]: r for r in by_reg_pred[region, "general_history"]}
        for pred_name in PREDICTORS:
            grp = by_reg_pred[region, pred_name]
            tot_t = sum(r["targets"] for r in grp)
            row = {"region": region, "predictor": pred_name, "requests": len(grp), "targets": tot_t}
            for m in cand_ks:
                tot_h = sum(r[f"hits_at_{m}"] for r in grp)
                row[f"recall_at_{m}"] = tot_h / tot_t if tot_t else 0.0
                diffs = [
                    r[f"recall_at_{m}"] - gen_by_req[r["request_id"]][f"recall_at_{m}"]
                    for r in grp
                    if r["request_id"] in gen_by_req
                ]
                mean_d = statistics.mean(diffs) if diffs else 0.0
                lo, hi = _bootstrap_ci(diffs, int(cfg.get("bootstrap_samples", 1000)))
                row[f"delta_vs_general_r{m}"] = mean_d
                row[f"ci95_lo_r{m}"] = lo
                row[f"ci95_hi_r{m}"] = hi
                row[f"wins_vs_general_r{m}"] = sum(d > 1e-12 for d in diffs)
            summary_rows.append(row)

    # Retain vs Replace CSV
    rr_rows = []
    for pred_name in PREDICTORS:
        for split_kind in ("retained", "replaced_new"):
            row = {"predictor": pred_name, "expert_kind": split_kind}
            for m in cand_ks:
                c = retain_replace_acc[pred_name, split_kind, m]
                row[f"hits_at_{m}"] = c["hits"]
                row[f"targets_at_{m}"] = c["targets"]
                row[f"recall_at_{m}"] = c["hits"] / c["targets"] if c["targets"] else 0.0
            rr_rows.append(row)

    # Geometry & margin CSV
    geom_rows = []
    for (region, axis, b), g in sorted(geom_acc.items()):
        n = g["n"]
        geom_rows.append(
            {
                "region": region,
                "axis": axis,
                "bin": b,
                "positions": n,
                "mean_router_input_cosine": g["cos_sum"] / n,
                "mean_prev_logit_margin": g["margin_sum"] / n,
                "mean_logit_linf_diff": g["linf_sum"] / n,
                "suff_cond_rate": g["suff_ok"] / n,
                "any_flip_rate": g["any_flip"] / n,
                "top1_flip_rate": g["top1_flip"] / n,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "a2_requests.csv", list(req_rows[0]), req_rows)
    write_csv(out_dir / "a2_summary.csv", list(summary_rows[0]), summary_rows)
    write_csv(out_dir / "a2_by_layer_loop.csv", list(ll_rows[0]), ll_rows)
    if rr_rows:
        write_csv(out_dir / "a2_retain_vs_replace.csv", list(rr_rows[0]), rr_rows)
    if geom_rows:
        write_csv(out_dir / "a2_geometry_margin.csv", list(geom_rows[0]), geom_rows)

    # Compute headline stats
    rec_rows = [r for r in ll_rows if r["src_loop"] != ""]
    tot_rec_pos = sum(r["positions"] for r in rec_rows)
    mean_top1_switch = (
        sum(float(r["top1_switch_rate"]) * r["positions"] for r in rec_rows) / tot_rec_pos
        if tot_rec_pos
        else None
    )
    mean_overlap = (
        sum(float(r["overlap"]) * r["positions"] for r in rec_rows) / tot_rec_pos
        if tot_rec_pos
        else None
    )
    mean_null = (
        sum(float(r["marginal_null_overlap"]) * r["positions"] for r in rec_rows) / tot_rec_pos
        if tot_rec_pos
        else None
    )
    sum_idx = {(r["region"], r["predictor"]): r for r in summary_rows}
    headline = {
        "model": meta["model"],
        "trace": str(trace_path),
        "fit_requests": len(fit_reqs),
        "eval_requests": len(eval_reqs),
        "mean_top1_switch_rate": mean_top1_switch,
        "mean_adjacent_loop_overlap": mean_overlap,
        "mean_marginal_null_overlap": mean_null,
        "suff_cond_total": suff_total,
        "suff_cond_violations": suff_violations,
        "all_combined_loop_r2": sum_idx["all", "combined_loop"]["recall_at_2"],
        "all_general_history_r2": sum_idx["all", "general_history"]["recall_at_2"],
        "all_delta_combined_vs_general_r2": sum_idx["all", "combined_loop"]["delta_vs_general_r2"],
        "recurrent_combined_loop_r2": sum_idx.get(("recurrent", "combined_loop"), {}).get("recall_at_2"),
        "recurrent_general_history_r2": sum_idx.get(("recurrent", "general_history"), {}).get("recall_at_2"),
        "recurrent_delta_vs_general_r2": sum_idx.get(("recurrent", "combined_loop"), {}).get("delta_vs_general_r2"),
    }
    (out_dir / "a2_headline.json").write_text(json.dumps(headline, indent=2) + "\n", encoding="utf-8")
    return headline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/ifm_a1_a3.json"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    cfg = read_json(args.config)["a2"]
    manifest = read_json(args.manifest)
    headline = run_a2(args.trace, manifest, cfg, args.out_dir)
    print(json.dumps(headline, indent=2), flush=True)


if __name__ == "__main__":
    main()
