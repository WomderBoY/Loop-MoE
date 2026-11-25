"""Causal per-token route restriction pilot; no timing inferred from route counts.

The first visit's set is token-specific. Prefix-wide unions are forbidden.
Transient first-visit slots and one sequential escape slot are charged against each budget.
All restrictions run on the student's trajectory, not a teacher route replay.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from loop_research import orx_run as orx
from loop_research.data_assets import build_disjoint_text_code_splits, load_arc_challenge_samples
from loop_research.gpu_utils import acquire_idle_gpu
from loop_research.ifm_runtime import bootstrap_xllm, load_ifm_model_for_autograd


@contextmanager
def stage(name, total, unit):
    p=orx.progress(phase=name,total=total,unit=unit)
    try:
        yield p
    finally:
        p.flush()


def decide(scores, selection, first, hot, *, mode, visit, recurrent, tau):
    """scores=unbiased gate values; selection includes any native router bias.

    Escape uses log(selection-score) advantage over kth admissible expert;
    these checkpoints have no router bias. Recompute weights on selected experts.
    tau=0 is exact; tau=inf is a hard restriction. Escapes use scratch slots,
    never enlarge the persistent first-visit set.
    """
    k = 2
    native = selection.topk(k, dim=-1).indices
    if mode == "native" or not recurrent or visit == 0 or tau == 0:
        return native
    if mode == "replay":
        return first
    allowed = hot.expand_as(selection).clone()
    if mode == "load_once":
        allowed.scatter_(-1, first, True)
    elif mode != "hot_late":
        raise ValueError(mode)
    if not bool((allowed.sum(-1) >= k).all()):
        raise RuntimeError("Fewer than two admissible experts")
    masked = selection.masked_fill(~allowed, -float("inf"))
    kth = masked.topk(k, dim=-1).values[..., -1:]
    # Allow only sufficiently better cold candidates, while retaining all hot ones.
    escape = (selection.clamp_min(1e-30).log() - kth.clamp_min(1e-30).log()) > tau
    admissible = allowed | escape
    return selection.masked_fill(~admissible, -float("inf")).topk(k, dim=-1).indices


def make_hot(counts, capacity, recurrent, mode):
    """Global frequency ranking with strict peak slot accounting.

    load_once reserves 2 slots per recurrent physical pool for the entire token;
    all policies reserve one sequential scratch slot. hot_late requires two experts per
    recurrent pool, then allocates remaining slots by global frequency.
    """
    transient = 2 * len(recurrent) if mode in ("load_once", "replay") else 0
    hot_n = capacity - transient - 1
    if hot_n < 0:
        raise ValueError("Budget cannot hold first-visit working set")
    hot = torch.zeros_like(counts, dtype=torch.bool)
    if mode == "hot_late":
        for lid in recurrent:
            hot[lid, counts[lid].topk(2).indices] = True
        if int(hot.sum()) > hot_n:
            raise ValueError("Budget cannot hold two hot experts per recurrent pool")
    keys = sorted(np.ndindex(tuple(counts.shape)), key=lambda k: (-float(counts[k]), k))
    for lid, e in keys:
        if int(hot.sum()) >= hot_n:
            break
        hot[lid, e] = True
    assert int(hot.sum()) + transient + 1 <= capacity
    return hot, transient


class Routing:
    def __init__(self, model, hot, mode="native", tau=0.0, collect=True):
        self.model, self.hot, self.mode, self.tau = model, hot, mode, tau
        self.recurrent = set(range(model.loop_start_layer, model.loop_end_layers))
        self.collect = collect
        self.reset()
        self.clear_stats()

    def reset(self):
        self.visits = [0] * len(self.model.layers)
        self.first = {}

    def clear_stats(self):
        self.counts = torch.zeros_like(self.hot, dtype=torch.int64)
        self.stat = torch.zeros(6, dtype=torch.float64, device=self.hot.device)

    def report(self):
        accesses, misses, late_misses, changes, late_accesses, cold_first = self.stat.tolist()
        return dict(accesses=accesses, logical_copies=misses, late_copies=late_misses,
                    changed_fraction=changes/max(accesses, 1),
                    late_miss_fraction=late_misses/max(late_accesses, 1),
                    first_cold_per_token=cold_first, counts=self.counts.cpu().tolist())

    @contextmanager
    def attached(self):
        saved = []
        try:
            for lid, layer in enumerate(self.model.layers):
                router = layer.moe.router
                saved.append((router, router.forward))

                def make(lid, mod):
                    def forward(x, load_balancing_type=None):
                        if load_balancing_type is not None:
                            raise ValueError("Pilot runs without auxiliary loss")
                        if mod.topk != 2:
                            raise ValueError("Pilot supports top-2 checkpoints")
                        logits = F.linear(x, mod.weight).float()
                        scores = logits.softmax(-1) if mod.score_func == "softmax" else logits.sigmoid()
                        sel = scores if mod.bias is None else scores + mod.bias.to(scores)
                        visit = self.visits[lid]
                        native = sel.topk(2, dim=-1).indices
                        first = self.first.get(lid, native)
                        idx = decide(scores, sel, first, self.hot[lid], mode=self.mode,
                                     visit=visit, recurrent=lid in self.recurrent, tau=self.tau)
                        if visit == 0:
                            self.first[lid] = idx.detach()
                        self.visits[lid] += 1
                        weights = scores.gather(-1, idx)
                        weights = weights / weights.sum(-1, keepdim=True)
                        if mod.scaling_factor:
                            weights = weights * mod.scaling_factor
                        tpe = torch.bincount(idx.flatten(), minlength=mod.num_experts)
                        if self.collect:
                            self.counts[lid] += tpe.detach()
                            hot_hit = self.hot[lid][idx]
                            if visit > 0 and lid in self.recurrent and self.mode in ("load_once", "replay"):
                                first_hit = (idx.unsqueeze(-1) == first.unsqueeze(-2)).any(-1)
                                hit = hot_hit | first_hit
                            else:
                                hit = hot_hit
                            cold = (~hit).sum()
                            changed = (~(idx.unsqueeze(-1) == native.unsqueeze(-2)).any(-1)).sum()
                            late = visit > 0 and lid in self.recurrent
                            self.stat[0] += idx.numel()
                            self.stat[1] += cold
                            self.stat[2] += cold if late else 0
                            self.stat[3] += changed
                            self.stat[4] += idx.numel() if late else 0
                            self.stat[5] += cold if visit == 0 and lid in self.recurrent else 0
                        return weights.to(x.dtype), idx, tpe, None
                    return forward
                router.forward = make(lid, router)
            yield self
        finally:
            for router, fn in saved:
                router.forward = fn


def seq_logits(model, item, route, device):
    route.reset()
    ids = torch.tensor([item["tokens"][:-1]], device=device)
    return model(ids, multi_segments=False, cache=None)[0]


@torch.inference_mode()
def evaluate(model, seqs, route, device, progress=None, teacher=None):
    rows = []
    route.clear_stats()
    with route.attached():
        for i, item in enumerate(seqs):
            logits = seq_logits(model, item, route, device)[0].float()
            targets = torch.tensor(item["tokens"][1:], device=device)
            nll = float(F.cross_entropy(logits, targets))
            row = dict(index=i, domain=item["domain"], nll=nll,
                       top1=logits.argmax(-1).cpu().tolist())
            if teacher:
                row["agreement"] = float(np.mean(np.asarray(row["top1"]) == teacher[i]["top1"]))
            rows.append(row)
            if progress:
                progress.step(i+1, nll_live=nll)
    stats = route.report()
    stats["logical_copies_per_token"] = stats["logical_copies"] / sum(len(s["tokens"])-1 for s in seqs)
    return rows, stats


def bootstrap(deltas, seed=42):
    a = np.asarray(deltas)
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, len(a), (2000, len(a)))].mean(-1)
    return dict(mean=float(a.mean()), lo=float(np.quantile(means, .025)), hi=float(np.quantile(means, .975)))


@torch.inference_mode()
def verify(model, item, route, device):
    ids = torch.tensor([item["tokens"][:24]], device=device)
    native = model(ids, multi_segments=False, cache=None)[0]
    with route.attached():
        route.reset()
        got = model(ids, multi_segments=False, cache=None)[0]
    gap = float((native.float()-got.float()).abs().max())
    agree = float((native.argmax(-1) == got.argmax(-1)).float().mean())
    print("VERIFY", json.dumps(dict(max_logit_gap=gap, top1_agreement=agree)), flush=True)
    if gap > 0.08 or agree < 1.0:
        raise RuntimeError("Native-routing replacement failed numerical verification")


def row_summary(rows, ref, stats):
    delta = bootstrap([r["nll"]-b["nll"] for r,b in zip(rows,ref)])
    return dict(nll=float(np.mean([r["nll"] for r in rows])), delta_nll=delta,
                nll_text=float(np.mean([r["nll"] for r in rows if r["domain"]=="text"])),
                nll_code=float(np.mean([r["nll"] for r in rows if r["domain"]=="code"])),
                agreement=float(np.mean([r.get("agreement",1.0) for r in rows])), **stats)


@torch.inference_mode()
def arc(model, tok, samples, route, device, progress):
    flags = []
    with route.attached():
        for i, s in enumerate(samples):
            p = tok.encode(f"Question: {s['question']}\nAnswer:", bos=True, eos=False)
            scores = []
            for choice in s["choice_texts"]:
                c = tok.encode(" "+choice, bos=False, eos=False)
                ids = torch.tensor([p+c[:-1]], device=device)
                route.reset()
                logits = model(ids, multi_segments=False, cache=None)[0][0,len(p)-1:,:]
                target = torch.tensor(c, device=device)
                scores.append(-float(F.cross_entropy(logits.float(),target)))
            flags.append(int(int(np.argmax(scores))==s["gold_idx"]))
            progress.step(i+1, arc_live=float(np.mean(flags)))
    return dict(accuracy=float(np.mean(flags)), flags=flags)


def run_model(name, spec, cfg, args, device):
    model, tok, _ = load_ifm_model_for_autograd(Path(args.model_dir)/spec["local_dir"],device=str(device))
    model.eval()
    model.requires_grad_(False)
    splits = build_disjoint_text_code_splits(args.dataset_dir, lambda t: tok.encode(t,bos=False,eos=False),
        seq_len=cfg["seq_len"], n_calib_a=cfg["n_calib"], n_calib_b=cfg["n_validation"],
        n_adapt_train=cfg["n_train"], n_eval_heldout=cfg["n_test"], seed=cfg["seed"])
    out = Path(args.out_dir)/name
    out.mkdir(parents=True,exist_ok=True)
    # Tokens and manifests stay in remote run storage, not project artifacts.
    (out/"splits.json").write_text(json.dumps(splits))
    e = model.layers[0].moe.experts
    expert_bytes = (e.weight1.numel()+e.weight2.numel()+e.weight3.numel())*e.weight1.element_size()//e.n_experts
    empty = torch.zeros((len(model.layers),e.n_experts),dtype=torch.bool,device=device)
    recurrent = set(range(model.loop_start_layer,model.loop_end_layers))
    print("MODEL",json.dumps(dict(name=name,expert_bytes=expert_bytes, recurrent_layers=sorted(recurrent),
                                 loops=int(model.loop_times))),flush=True)
    with stage(name+"_verify",total=1,unit="checks") as p:
        verify(model,splits["calib_a"][0],Routing(model,empty,collect=False),device)
        p.step(1)
    with stage(name+"_calibrate",total=len(splits["calib_a"]),unit="sequences") as p:
        _,stats = evaluate(model,splits["calib_a"],Routing(model,empty),device,p)
    counts = torch.tensor(stats["counts"])
    conditions = []
    for budget in cfg["budgets_gib"]:
        capacity = math.floor(budget*(1024**3)/expert_bytes)
        for mode,taus in [("native",[0]),("replay",[float("inf")]),
                          ("load_once",cfg["taus"]),("hot_late",cfg["taus"])]:
            for tau in taus:
                hot,transient=make_hot(counts,capacity,recurrent,mode)
                key=f"{budget:g}_{mode}_{tau:g}"
                conditions.append(dict(key=key,budget=budget,capacity=capacity,mode=mode,tau=tau,
                                       hot=hot.to(device),transient=transient))
    result={"model":name,"expert_bytes":expert_bytes,"conditions":{},"arc":{},"selected":{}}
    native_refs={}
    # Validation selects candidates without reading test outcomes.
    for split,label in [("calib_b","validate"),("eval_heldout","test")]:
        seqs=splits[split]
        with stage(name+"_"+label,total=len(seqs)*len(conditions),unit="sequences") as p:
            for ci,c in enumerate(conditions):
                route=Routing(model,c["hot"],c["mode"],c["tau"])
                rows,stats=evaluate(model,seqs,route,device,teacher=native_refs.get(split))
                if split not in native_refs:
                    native_refs[split]=rows
                stats.pop("counts", None)
                summary=row_summary(rows,native_refs[split],stats)
                summary["logical_h2d_mib_per_token"]=summary["logical_copies_per_token"]*expert_bytes/1024**2
                result["conditions"].setdefault(c["key"],{})[label]=summary
                (out/f"{label}_{c['key']}.json").write_text(json.dumps(dict(rows=rows,summary=summary)))
                print("QUALITY",json.dumps(dict(model=name,split=label,key=c["key"],
                    delta_nll=summary["delta_nll"],copies=summary["logical_copies_per_token"],
                    changed=summary["changed_fraction"],hot_slots=int(c["hot"].sum()),
                    first_slots=c["transient"],scratch_slots=1,capacity=c["capacity"])),flush=True)
                p.row("quality",model=name,split=label,condition=c["key"],
                    delta_nll=summary["delta_nll"]["mean"],logical_copies=summary["logical_copies_per_token"],
                    agreement=summary["agreement"])
                p.step((ci+1)*len(seqs))
                (out/"summary.json").write_text(json.dumps(result,indent=2))
        if label=="validate":
            for budget in cfg["budgets_gib"]:
                base=next(c for c in conditions if c["budget"]==budget and c["mode"]=="native")
                basecopy=result["conditions"][base["key"]]["validate"]["logical_copies_per_token"]
                # Choose best quality among variants saving >=30% logical copies.
                for mode in ("load_once","hot_late"):
                    candidates=[c for c in conditions if c["budget"]==budget and c["mode"]==mode]
                    feasible=[c for c in candidates if result["conditions"][c["key"]]["validate"]["logical_copies_per_token"]<=.7*basecopy]
                    if feasible:
                        best=min(feasible,key=lambda c:result["conditions"][c["key"]]["validate"]["nll"])
                        result["selected"][f"{budget:g}_{mode}"]=best["key"]
            (out/"validation_selection.json").write_text(json.dumps(result["selected"],indent=2))
    # Downstream sanity check: native, first replay and validation-selected policies.
    keys=[conditions[0]["key"]]
    keys += [c["key"] for c in conditions if c["mode"]=="replay" and c["budget"]==max(cfg["budgets_gib"])]
    keys+=list(result["selected"].values())
    samples=load_arc_challenge_samples(args.dataset_dir,n_samples=cfg["n_arc"],seed=cfg["seed"]+1)
    (out/"arc_ids.json").write_text(json.dumps([s["id"] for s in samples]))
    with stage(name+"_arc",total=len(keys)*len(samples),unit="questions") as p:
        class Offset:
            def __init__(self, offset): self.offset=offset
            def step(self,n,**m): p.step(self.offset+n,**m)
        for ki,key in enumerate(keys):
            c=next(c for c in conditions if c["key"]==key)
            result["arc"][key]=arc(model,tok,samples,Routing(model,c["hot"],c["mode"],c["tau"],False),device,Offset(ki*len(samples)))
            print("ARC",json.dumps(dict(model=name,key=key,**result["arc"][key])),flush=True)
            p.row("arc",model=name,condition=key,accuracy=result["arc"][key]["accuracy"])
            (out/"summary.json").write_text(json.dumps(result,indent=2))
    print("MODEL_COMPLETE",json.dumps(dict(model=name,selected=result["selected"],summary_path=str(out/"summary.json"))),flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser=argparse.ArgumentParser()
    for key in ("config","model-dir","dataset-dir","xllm-loop-dir","out-dir"):
        parser.add_argument("--"+key,required=True)
    args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    print("CONFIG",json.dumps(cfg),flush=True)
    idx,lock,gpu=acquire_idle_gpu(cfg.get("preferred_gpu"))
    try:
        # Bind only after the shared-server idle check and advisory lock.
        device=torch.device(f"cuda:{idx}")
        torch.cuda.set_device(device)
        torch.manual_seed(cfg["seed"])
        print("GPU",json.dumps(gpu.__dict__),flush=True)
        print("RUNTIME",json.dumps(bootstrap_xllm(args.xllm_loop_dir)),flush=True)
        all_results={}
        for name,spec in cfg["models"].items():
            all_results[name]=run_model(name,spec,cfg,args,device)
        with stage("report",total=1,unit="files") as p:
            Path(args.out_dir,"summary.json").write_text(json.dumps(all_results,indent=2))
            print("FINAL_SUMMARY",json.dumps(dict(models=list(all_results),
                limitation="Logical transfer counts; no measured offload latency or compensation yet.")),flush=True)
            p.step(1)
    finally:
        lock.release()


if __name__=="__main__":
    main()
