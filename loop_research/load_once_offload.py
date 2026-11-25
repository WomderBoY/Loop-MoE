"""Real-copy incremental-KV benchmark at quality-validated operating points.

One physical expert pool is shared by all strategies, never one pool per policy.
Static hot weights, persistent first-visit experts and the demand scratch slot
all count toward that pool. Prefill is evaluated exactly but does not retain
prefix-wide first-visit unions. Decode resets its first-visit set for each token.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from loop_research import orx_run as orx
from loop_research.data_assets import build_disjoint_text_code_splits
from loop_research.gpu_utils import acquire_idle_gpu, query_gpus
from loop_research.ifm_runtime import bootstrap_xllm, load_ifm_model_for_autograd
from loop_research.load_once import Routing, evaluate, make_hot, bootstrap


def legacy_runtime():
    """Reuse the previously verified per-visit KV engine and packed expert store."""
    from loop_research.ifm_offload_runtime import (
        LoopedIncrementalEngine, FastPinnedExpertStore, FullResidentFastExecutor,
    )
    return LoopedIncrementalEngine,FastPinnedExpertStore,FullResidentFastExecutor


class SlotPlan:
    """CPU bookkeeping; no allocation, prediction or eviction in late visits."""
    def __init__(self,hot_keys,capacity,recurrent,keep_first):
        self.hot={key:i for i,key in enumerate(hot_keys)}
        self.capacity=capacity
        self.recurrent=set(recurrent)
        self.keep_first=keep_first
        self.scratch=capacity-1
        required=len(self.hot)+(2*len(self.recurrent) if keep_first else 0)+1
        if required>capacity: raise ValueError("Peak expert-slot budget exceeded")
        self.reset_token()

    def reset_token(self):
        self.first={}
        self.next_slot=len(self.hot)
        self.visits=defaultdict(int)

    def visit(self,lid):
        v=self.visits[lid]
        self.visits[lid]+=1
        return v

    def locate(self,lid,e,visit,decode):
        key=(lid,e)
        if key in self.hot: return self.hot[key],False
        if key in self.first: return self.first[key],False
        if decode and self.keep_first and visit==0 and lid in self.recurrent:
            if self.next_slot>=self.scratch: raise RuntimeError("First-visit pool overflow")
            slot=self.next_slot
            self.next_slot+=1
            self.first[key]=slot
            return slot,True
        return self.scratch,True


class OffloadExecutor:
    def __init__(self,store,pool,hot,recurrent,mode):
        self.store,self.pool,self.mode=store,pool,mode
        self.device=pool.device
        keys=[tuple(k) for k in hot.nonzero().cpu().tolist()]
        self.slots=SlotPlan(keys,len(pool),recurrent,mode=="load_once")
        for key,slot in self.slots.hot.items():
            pool[slot].copy_(store.host_experts[key[0]][key[1]],non_blocking=True)
        self.decode=False
        self.copies=0

    def start_token(self,decode):
        self.decode=decode
        self.slots.reset_token()

    def weights(self,lid,e,visit):
        slot,copy=self.slots.locate(lid,e,visit,self.decode)
        if copy:
            self.pool[slot].copy_(self.store.host_experts[lid][e],non_blocking=True)
            if self.decode:self.copies+=1
        return self.store.expert_views(self.pool[slot])

    def execute_moe(self,lid,v_idx,layer,x_in,load_balancing_type):
        moe=layer.moe
        bsz,slen,dim=x_in.shape
        if bsz!=1:raise ValueError("Batch-1 benchmark")
        x=moe.norm(x_in)
        y_shared=moe.fc2(F.silu(moe.fc1(x))*moe.fc3(x)) if moe.fc1 is not None else None
        scores,indices,_,aux=moe.router(x,load_balancing_type)
        xf=x.view(-1,dim)
        visit=self.slots.visit(lid)
        if slen==1:
            # The scratch slot is reused only after the previous expert's GEMM.
            # Copy and compute share the CUDA stream, preserving this ordering.
            idx=indices[0,0].tolist()
            y=None
            for j,e in enumerate(idx):
                w1,w2,w3=self.weights(lid,e,visit)
                ye=F.linear(F.silu(F.linear(xf,w1))*F.linear(xf,w3),w2)
                weighted=ye*scores[0,0,j].to(ye.dtype)
                y=weighted if y is None else y+weighted
            y=y.view(bsz,slen,dim)
        else:
            # Prefill routes remain token-local, but cold experts are streamed
            # through one scratch slot; no prefix-wide first-route retention.
            idx=indices[0]
            y=torch.zeros_like(xf)
            for e in torch.unique(idx).tolist():
                pos,kpos=torch.nonzero(idx==e,as_tuple=True)
                xe=xf.index_select(0,pos)
                w1,w2,w3=self.weights(lid,e,visit)
                ye=F.linear(F.silu(F.linear(xe,w1))*F.linear(xe,w3),w2)
                y.index_add_(0,pos,ye*scores[0,pos,kpos].unsqueeze(-1).to(ye.dtype))
            y=y.view(bsz,slen,dim)
        if y_shared is not None:y=y+y_shared
        return moe.residual(y,x_in,c=y),aux


def spec_for(model,arch):
    n=len(model.layers)
    s,e=int(model.loop_start_layer),int(model.loop_end_layers)
    exp=model.layers[0].moe.experts
    bpe=sum(p.numel()*p.element_size() for p in (exp.weight1,exp.weight2,exp.weight3))//exp.n_experts
    return dict(arch=arch,prelude_layers=[0,s],recurrent_layers=[s,e],coda_layers=[e,n],
                expected_pools=n,expected_visits=len(model.execution_schedule),
                expected_loops=int(model.loop_times),num_experts=int(exp.n_experts),bytes_per_expert=bpe)


@torch.inference_mode()
def generate(engine,executor,route,prompt,ntokens,device,forced=None):
    engine.reset()
    route.reset()
    if hasattr(executor,"start_token"):executor.start_token(False)
    torch.cuda.synchronize(device)
    t0=time.perf_counter()
    logits=engine.forward_step(torch.tensor([prompt],device=device),moe_executor=executor.execute_moe)
    first=int(logits[0,-1].argmax())
    del logits
    torch.cuda.synchronize(device)
    ttft_ms=(time.perf_counter()-t0)*1000
    prev=first if forced is None else int(forced[0])
    predictions=[first]
    if hasattr(executor,"copies"):executor.copies=0
    torch.cuda.synchronize(device)
    t0=time.perf_counter()
    # ntokens actual incremental forwards are timed. The first prediction comes
    # from prefill and is stored for exact full-resident/offloaded comparison.
    for step in range(ntokens):
        route.reset()
        if hasattr(executor,"start_token"):executor.start_token(True)
        logits=engine.forward_step(torch.tensor([[prev]],device=device),moe_executor=executor.execute_moe)
        nxt=int(logits[0,-1].argmax())
        predictions.append(nxt)
        prev=nxt if forced is None else int(forced[step+1])
        del logits
    torch.cuda.synchronize(device)
    elapsed=time.perf_counter()-t0
    return dict(tpot_ms=elapsed*1000/ntokens,ttft_ms=ttft_ms,tokens=predictions,
                copies_per_token=getattr(executor,"copies",0)/ntokens,kv_bytes=engine.kv_cache_bytes())


@torch.inference_mode()
def verify_engine(model,engine,executor,route,prompt,device):
    """Strict native-MoE versus direct-MoE comparison within the same KV path.

    Full-prefix versus incremental differences are recorded by diagnose_paths;
    they must not be attributed to this executor comparison.
    """
    engine.reset()
    native_engine=type(engine)(model,engine.model_spec)
    growing=list(prompt[:16])
    gaps=[]
    captured=[]
    handles=[]
    def capture(_mod,_args,output):captured.append(output[1][0,-1].tolist())
    for layer in model.layers:handles.append(layer.moe.router.register_forward_hook(capture))
    try:
        for step in range(8):
            inp=growing if step==0 else [growing[-1]]
            ids=torch.tensor([inp],device=device)
            route.reset();captured.clear()
            exact=native_engine.forward_step(ids)[0,-1]
            native_routes=list(captured)
            route.reset();captured.clear()
            got=engine.forward_step(ids,moe_executor=executor.execute_moe)[0,-1]
            require_same_execution(exact,got,native_routes,list(captured))
            gaps.append(float((exact.float()-got.float()).abs().max()))
            growing.append(int(exact.argmax()))
    finally:
        for handle in handles:handle.remove()
    print("ENGINE_VERIFY",json.dumps(dict(comparison="native_KV_vs_direct_KV",max_logit_gaps=gaps,
        greedy_exact=True,routes_exact=True)),flush=True)


def require_same_execution(reference,candidate,reference_routes,candidate_routes):
    if reference_routes!=candidate_routes:
        raise RuntimeError("Native/direct KV executor routing mismatch")
    if not bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all()):
        raise RuntimeError("Nonfinite executor verification logits")
    if not torch.equal(reference,candidate):
        gap=float((reference.float()-candidate.float()).abs().max())
        raise RuntimeError(f"Native/direct KV executor logits differ: gap={gap}")


def compare_logits(reference,candidate):
    """Describe numerical and decision differences without declaring equivalence."""
    ref,got=reference.float(),candidate.float()
    if ref.shape!=got.shape:raise ValueError("Logit shape mismatch")
    if not bool(torch.isfinite(ref).all() and torch.isfinite(got).all()):
        raise RuntimeError("Nonfinite diagnostic logits")
    diff=(ref-got).abs()
    lp,lq=F.log_softmax(ref,-1),F.log_softmax(got,-1)
    rt,ct=int(ref.argmax()),int(got.argmax())
    k=min(5,ref.numel())
    top_ref=set(ref.topk(k).indices.tolist())
    top_got=set(got.topk(k).indices.tolist())
    return dict(max_abs=float(diff.max()),mean_abs=float(diff.mean()),
                relative_l2=float((ref-got).norm()/ref.norm().clamp_min(1e-12)),
                kl=float((lp.exp()*(lp-lq)).sum()),
                ref_top1=rt,candidate_top1=ct,top1_match=int(rt==ct),
                top5_overlap=len(top_ref&top_got)/k,
                ref_token_logprob_delta=float(lq[rt]-lp[rt]))


@torch.inference_mode()
def diagnose_paths(model,Engine,full,shape,configs,prompt,steps,device,out):
    """Four paths on exactly the same growing prefix, before any CPU offload.

    A: official full prefix + native MoE; B: KV + native MoE;
    C: fresh full prefix + direct expert executor; D: KV + direct executor.
    Compare B-D separately from A-B so cache/precision differences cannot be
    attributed to the expert executor. Teacher tokens always come from A.
    This function produces diagnostics only and never authorizes timing.
    """
    policies=("native","hot_late","load_once")
    rows=[]
    progress=orx.progress(phase="diagnose",total=steps*len(policies),unit="steps")
    done=0
    for policy in policies:
        c=next(c for c in configs if c["policy"]==policy)
        route=Routing(model,c["hot"],policy,.5,False)
        kv_native,kv_direct,fresh=Engine(model,shape),Engine(model,shape),Engine(model,shape)
        growing=list(prompt[:16])
        captured=[]
        handles=[]
        def capture(_mod,_args,output):captured.append(output[1][0,-1].tolist())
        for layer in model.layers:handles.append(layer.moe.router.register_forward_hook(capture))
        try:
            with route.attached():
                for step in range(steps):
                    logits={}
                    routes={}
                    route.reset();captured.clear()
                    logits["A_full_native"]=model(torch.tensor([growing],device=device),multi_segments=False,cache=None)[0][0,-1].float()
                    routes["A_full_native"]=list(captured)
                    for key,engine,executor in [("B_kv_native",kv_native,None),("C_full_direct",fresh,full.execute_moe),("D_kv_direct",kv_direct,full.execute_moe)]:
                        route.reset();captured.clear()
                        if key=="C_full_direct":engine.reset()
                        inp=growing if step==0 or key=="C_full_direct" else [growing[-1]]
                        logits[key]=engine.forward_step(torch.tensor([inp],device=device),moe_executor=executor)[0,-1].float()
                        routes[key]=list(captured)
                    pairs=[("A_full_native","B_kv_native"),("A_full_native","C_full_direct"),
                           ("B_kv_native","D_kv_direct"),("C_full_direct","D_kv_direct")]
                    for a,b in pairs:
                        ra,rb=routes[a],routes[b]
                        if len(ra)!=len(rb):raise RuntimeError("Diagnostic visit count mismatch")
                        differing=[i for i,(x,y) in enumerate(zip(ra,rb)) if x!=y]
                        row=dict(policy=policy,budget=c["budget"],step=step,comparison=a+"__"+b,
                                 route_match=1-len(differing)/max(len(ra),1),
                                 first_different_visit=differing[0] if differing else None,
                                 **compare_logits(logits[a],logits[b]))
                        rows.append(row)
                        print("PATH_DIAGNOSTIC",json.dumps(row),flush=True)
                        progress.row("diagnostics",policy=policy,comparison=row["comparison"],
                                     max_abs=row["max_abs"],mean_abs=row["mean_abs"],kl=row["kl"],
                                     top1_match=row["top1_match"],route_match=row["route_match"])
                    growing.append(int(logits["A_full_native"].argmax()))
                    done+=1;progress.step(done)
        finally:
            for h in handles:h.remove()
        progress.flush()
    progress=orx.progress(phase="report",total=1,unit="files")
    (out/"path-diagnostics.json").write_text(json.dumps(dict(rows=rows,
        conclusion="Diagnostics only. No latency measurement or equivalence claim."),indent=2))
    print("DIAGNOSTIC_COMPLETE",json.dumps(dict(rows=len(rows),timing_started=False)),flush=True)
    progress.step(1);progress.flush()


def assert_own_gpu(idx):
    gpu=next(g for g in query_gpus() if g.index==idx)
    foreign=set(gpu.active_pids)-{os.getpid()}
    if foreign:raise RuntimeError(f"Foreign process entered GPU {idx}: {sorted(foreign)}")
    return dict(index=idx,util=gpu.util_pct,memory_used=gpu.mem_used_mib,
                cpu_load=list(os.getloadavg()))


@torch.inference_mode()
def incremental_nll(engine,executor,route,items,prefix_tokens,target_tokens,device,progress,offset):
    values=[]
    for i,item in enumerate(items):
        tokens=item["tokens"]
        prefix=min(prefix_tokens,len(tokens)-target_tokens)
        if prefix<2:raise ValueError("NLL sequence too short")
        engine.reset();route.reset()
        logits=engine.forward_step(torch.tensor([tokens[:prefix]],device=device),moe_executor=executor.execute_moe)
        losses=[]
        for j in range(target_tokens):
            target=torch.tensor([tokens[prefix+j]],device=device)
            losses.append(float(F.cross_entropy(logits[0,-1:].float(),target)))
            del logits
            if j+1<target_tokens:
                route.reset()
                logits=engine.forward_step(target.view(1,1),moe_executor=executor.execute_moe)
        values.append(dict(index=i,domain=item["domain"],nll=float(np.mean(losses))))
        progress.step(offset+i+1)
    return values


def benchmark(cfg,args,device,idx):
    Engine,Store,Full=legacy_runtime()
    name,spec=next(iter(cfg["models"].items()))
    model,tok,_=load_ifm_model_for_autograd(Path(args.model_dir)/spec["local_dir"],device=str(device))
    model.requires_grad_(False)
    model.eval()
    shape=spec_for(model,spec["arch"])
    print("MODEL",json.dumps(dict(name=name,**shape)),flush=True)
    splits=build_disjoint_text_code_splits(args.dataset_dir,lambda t:tok.encode(t,bos=False,eos=False),
        seq_len=cfg["seq_len"],n_calib_a=cfg["n_calib"],n_calib_b=cfg["n_validation"],
        n_adapt_train=cfg["n_train"],n_eval_heldout=cfg["n_test"],seed=cfg["seed"])
    out=Path(args.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    prompts=[s["tokens"][:cfg["prompt_tokens"]] for s in splits["eval_heldout"][:cfg["bench_requests"]]]
    (out/"request-prefixes.json").write_text(json.dumps(prompts))
    empty=torch.zeros((len(model.layers),shape["num_experts"]),device=device,dtype=torch.bool)
    p=orx.progress(phase="calibrate",total=cfg["n_calib"],unit="sequences")
    _,stats=evaluate(model,splits["calib_a"],Routing(model,empty),device,p)
    p.flush()
    counts=torch.tensor(stats["counts"])
    (out/"hot-counts.json").write_text(json.dumps(stats["counts"]))
    recurrent=set(range(model.loop_start_layer,model.loop_end_layers))
    policies=("native","hot_late","load_once")
    configs=[]
    for budget in cfg["bench_budgets_gib"]:
        capacity=math.floor(budget*1024**3/shape["bytes_per_expert"])
        for policy in policies:
            hot,transient=make_hot(counts,capacity,recurrent,policy)
            configs.append(dict(budget=budget,capacity=capacity,policy=policy,hot=hot.to(device),
                                transient=transient,key=f"{budget:g}_{policy}"))
    engine=Engine(model,shape)
    full=Full(model)
    if cfg.get("diagnostic_only",False):
        diagnose_paths(model,Engine,full,shape,configs,prompts[0],
                       cfg.get("diagnostic_steps",8),device,out)
        return
    p=orx.progress(phase="verify",total=len(configs)*2,unit="checks")
    for i,c in enumerate(configs):
        for j,prompt in enumerate(prompts[:2]):
            route=Routing(model,c["hot"],c["policy"],cfg["tau"],False)
            with route.attached():verify_engine(model,engine,full,route,prompt,device)
            p.step(i*2+j+1)
    p.flush()
    # Quality is rechecked on the actual incremental path before reporting any
    # latency-quality claim. Full-prefix NLL from stage one is a separate result.
    quality_items=splits["eval_heldout"][:cfg["quality_sequences"]]
    p=orx.progress(phase="incremental_quality",total=len(configs)*len(quality_items),unit="sequences")
    quality={}
    native_quality=None
    for i,c in enumerate(configs):
        route=Routing(model,c["hot"],c["policy"],cfg["tau"],False)
        with route.attached():
            values=incremental_nll(engine,full,route,quality_items,cfg["prompt_tokens"],
                cfg["quality_target_tokens"],device,p,i*len(quality_items))
        if c["policy"]=="native":native_quality=values
        delta=bootstrap([v["nll"]-b["nll"] for v,b in zip(values,native_quality)])
        quality[c["key"]]=dict(rows=values,delta_nll=delta,nll=float(np.mean([v["nll"] for v in values])))
        print("INCREMENTAL_QUALITY",json.dumps(dict(key=c["key"],delta_nll=delta,
            nll=quality[c["key"]]["nll"],sequences=len(values),targets_per_sequence=cfg["quality_target_tokens"])),flush=True)
        p.row("quality_kv",policy=c["policy"],budget=c["budget"],delta_nll_kv=delta["mean"])
    p.flush()
    (out/"incremental-quality.json").write_text(json.dumps(quality,indent=2))
    # Reference outputs for both free generation and matched-prefix replay.
    # They are produced with the same active-two-expert path as offload.
    refs={}
    native_tokens={}
    total=len(configs)*len(prompts)*2
    p=orx.progress(phase="resident_reference",total=total,unit="requests")
    done=0
    for c in configs:
        route=Routing(model,c["hot"],c["policy"],cfg["tau"],False)
        with route.attached():
            for ri,prompt in enumerate(prompts):
                free=generate(engine,full,route,prompt,cfg["decode_tokens"],device)
                if c["policy"]=="native":native_tokens[ri]=free["tokens"]
                teacher=generate(engine,full,route,prompt,cfg["decode_tokens"],device,forced=native_tokens[ri])
                refs[f"{c['key']}_{ri}_free"]=free
                refs[f"{c['key']}_{ri}_teacher"]=teacher
                done+=2
                p.step(done)
    p.flush()
    (out/"resident-reference.json").write_text(json.dumps(refs))
    del full
    engine.reset()
    gc.collect()
    torch.cuda.empty_cache()
    p=orx.progress(phase="pack_cpu",total=1,unit="stores")
    store=Store(model,device)
    assert all(layer.moe.experts.weight1.numel()==0 and layer.moe.experts.weight2.numel()==0
               and layer.moe.experts.weight3.numel()==0 for layer in model.layers)
    p.step(1)
    p.flush()
    print("PINNED_STORE",json.dumps(dict(bytes_per_expert=store.bytes_per_expert,
        gpu_allocated_bytes=torch.cuda.memory_allocated(device))),flush=True)
    records=[]
    for budget in cfg["bench_budgets_gib"]:
        choices=[c for c in configs if c["budget"]==budget]
        capacity=choices[0]["capacity"]
        pool=torch.empty((capacity,store.expert_numel),device=device,dtype=torch.bfloat16)
        assert pool.numel()*pool.element_size()<=budget*1024**3
        p=orx.progress(phase=f"offload_{budget:g}",total=len(prompts)*cfg["repeats"]*len(choices)*2,unit="requests")
        done=0
        print("PHYSICAL_BUDGET",json.dumps(dict(budget_gib=budget,pool_bytes=pool.numel()*pool.element_size(),
            slots=capacity,scratch_slots=1,environment=assert_own_gpu(idx))),flush=True)
        # Warm all policies; initialization and prefill are outside primary TPOT.
        for c in choices:
            executor=OffloadExecutor(store,pool,c["hot"],recurrent,c["policy"])
            route=Routing(model,c["hot"],c["policy"],cfg["tau"],False)
            with route.attached():generate(engine,executor,route,prompts[0],4,device)
            del executor
        for repeat in range(cfg["repeats"]):
            print("ENVIRONMENT",json.dumps(assert_own_gpu(idx)),flush=True)
            for ri,prompt in enumerate(prompts):
                # Rotate method order by request and repeat, reverse each round.
                shift=(ri+repeat)%len(choices)
                order=choices[shift:]+choices[:shift]
                if repeat%2:order=order[::-1]
                for c in order:
                    executor=OffloadExecutor(store,pool,c["hot"],recurrent,c["policy"])
                    route=Routing(model,c["hot"],c["policy"],cfg["tau"],False)
                    with route.attached():
                        for task in ("free","teacher"):
                            got=generate(engine,executor,route,prompt,cfg["decode_tokens"],device,
                                         forced=native_tokens[ri] if task=="teacher" else None)
                            ref=refs[f"{c['key']}_{ri}_{task}"]
                            agreement=float(np.mean(np.asarray(got["tokens"])==ref["tokens"]))
                            if agreement<1.0:raise RuntimeError(f"Offload changed {c['key']} tokens: {agreement}")
                            row=dict(model=name,budget=budget,policy=c["policy"],task=task,
                                     repeat=repeat,request=ri,agreement=agreement,**got)
                            records.append(row)
                            with (out/"timings.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
                            print("TIMING",json.dumps({k:v for k,v in row.items() if k!="tokens"}),flush=True)
                            p.row("timings",model=name,budget=budget,policy=c["policy"],task=task,
                                  tpot_ms=got["tpot_ms"],copies=got["copies_per_token"],agreement=agreement)
                            done+=1
                            p.step(done,tpot_live=got["tpot_ms"])
                    del executor
        p.flush()
        del pool
        gc.collect()
        torch.cuda.empty_cache()
    p=orx.progress(phase="report",total=1,unit="files")
    summaries=[]
    for budget in cfg["bench_budgets_gib"]:
        for task in ("free","teacher"):
            by={}
            for policy in policies:
                by[policy]=[np.median([r["tpot_ms"] for r in records if r["budget"]==budget and r["task"]==task
                                      and r["policy"]==policy and r["request"]==i]) for i in range(len(prompts))]
            for policy in policies:
                paired=[100*(1-a/b) for a,b in zip(by[policy],by["native"])]
                vs_hot=[100*(1-a/b) for a,b in zip(by[policy],by["hot_late"])]
                row=dict(model=name,budget=budget,task=task,policy=policy,
                         tpot_ms=float(np.mean(by[policy])),speedup_vs_static_pct=bootstrap(paired),
                         speedup_vs_hot_pct=bootstrap(vs_hot))
                summaries.append(row)
                print("LATENCY_SUMMARY",json.dumps(row),flush=True)
                p.row("summary",model=name,budget=budget,task=task,policy=policy,
                      tpot_ms=row["tpot_ms"],speedup_static=row["speedup_vs_static_pct"]["mean"],
                      speedup_hot=row["speedup_vs_hot_pct"]["mean"])
    (out/"latency-summary.json").write_text(json.dumps(dict(config=cfg,rows=summaries),indent=2))
    p.step(1)
    p.flush()


def main():
    parser=argparse.ArgumentParser()
    for k in ("config","model-dir","dataset-dir","xllm-loop-dir","out-dir"):
        parser.add_argument("--"+k,required=True)
    args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    print("CONFIG",json.dumps(cfg),flush=True)
    idx,lock,gpu=acquire_idle_gpu(cfg.get("preferred_gpu"))
    try:
        device=torch.device(f"cuda:{idx}")
        torch.cuda.set_device(device)
        torch.manual_seed(cfg["seed"])
        print("GPU",json.dumps(gpu.__dict__),flush=True)
        print("RUNTIME",json.dumps(bootstrap_xllm(args.xllm_loop_dir)),flush=True)
        with torch.inference_mode():benchmark(cfg,args,device,idx)
    finally:lock.release()


if __name__=="__main__":main()
