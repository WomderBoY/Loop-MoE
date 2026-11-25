"""Counterfactual controller ceiling with real copies and expert computation.

Routes and gate weights are recorded on each policy's causal student trajectory.
Replaying them removes online routing work, so it is a hardware-headroom test,
not an online algorithm or evidence of achievable deployment speedup.
"""
from __future__ import annotations
from contextlib import contextmanager
import argparse
import gc
import json
import math
from pathlib import Path
import numpy as np
import torch

from loop_research import orx_run as orx
from loop_research.data_assets import build_disjoint_text_code_splits
from loop_research.gpu_utils import acquire_idle_gpu
from loop_research.ifm_runtime import bootstrap_xllm,load_ifm_model_for_autograd
from loop_research.load_once import Routing,evaluate,make_hot,bootstrap
from loop_research.load_once_offload import legacy_runtime,spec_for,generate,OffloadExecutor,SlotPlan,assert_own_gpu


class RouterTape:
    """Record/replay one request; validate every physical layer and tensor shape."""
    def __init__(self,model,events=None):
        self.model=model
        self.recording=events is None
        self.events=[] if events is None else events
        self.position=0

    @contextmanager
    def attached(self):
        handles=[]
        saved=[]
        try:
            for lid,layer in enumerate(self.model.layers):
                router=layer.moe.router
                if self.recording:
                    def make_hook(lid):
                        def hook(_module,args,output):
                            # The tape owns immutable copies, independent of all
                            # later expert execution and allocator reuse.
                            self.events.append((lid,tuple(args[0].shape),tuple(
                                x.detach().clone() if isinstance(x,torch.Tensor) else x for x in output)))
                        return hook
                    handles.append(router.register_forward_hook(make_hook(lid)))
                else:
                    saved.append((router,router.forward))
                    def make_forward(lid):
                        def forward(x,load_balancing_type=None):
                            if self.position>=len(self.events):raise RuntimeError("Route tape exhausted")
                            expected,shape,output=self.events[self.position]
                            if lid!=expected or tuple(x.shape)!=shape:
                                raise RuntimeError("Route tape visit/shape mismatch")
                            self.position+=1
                            return output
                        return forward
                    router.forward=make_forward(lid)
            yield self
            if not self.recording and self.position!=len(self.events):
                raise RuntimeError("Route tape not fully consumed")
        finally:
            for handle in handles:handle.remove()
            for router,forward in saved:router.forward=forward


def tape_copy_count(events,hot_keys,capacity,recurrent,keep_first,nvisits):
    decode=events[nvisits:]
    indices=torch.stack([record[1] for _,_,record in decode]).reshape(-1,2).cpu().tolist()
    plan=SlotPlan(hot_keys,capacity,recurrent,keep_first)
    copies=0
    for i,((lid,_,_),idx) in enumerate(zip(decode,indices)):
        if i%nvisits==0:plan.reset_token()
        visit=plan.visit(lid)
        for e in idx:
            _,needed=plan.locate(lid,e,visit,True)
            copies+=int(needed)
    return copies/(len(decode)/nvisits)


def run(cfg,args,device,gpu_index):
    Engine,Store,Full=legacy_runtime()
    name,spec=next(iter(cfg["models"].items()))
    model,tok,_=load_ifm_model_for_autograd(Path(args.model_dir)/spec["local_dir"],device=str(device))
    model.eval();model.requires_grad_(False)
    shape=spec_for(model,spec["arch"])
    print("MODEL",json.dumps(dict(name=name,**shape)),flush=True)
    splits=build_disjoint_text_code_splits(args.dataset_dir,lambda s:tok.encode(s,bos=False,eos=False),
        seq_len=cfg["seq_len"],n_calib_a=cfg["n_calib"],n_calib_b=cfg["n_validation"],
        n_adapt_train=cfg["n_train"],n_eval_heldout=cfg["n_test"],seed=cfg["seed"])
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    prompts=[x["tokens"][:cfg["prompt_tokens"]] for x in splits["eval_heldout"][:cfg["bench_requests"]]]
    empty=torch.zeros((len(model.layers),shape["num_experts"]),device=device,dtype=torch.bool)
    p=orx.progress(phase="calibrate",total=cfg["n_calib"],unit="sequences")
    _,stats=evaluate(model,splits["calib_a"],Routing(model,empty),device,p);p.flush()
    counts=torch.tensor(stats["counts"])
    recurrent=set(range(model.loop_start_layer,model.loop_end_layers))
    budget=cfg["bench_budgets_gib"][0]
    capacity=math.floor(budget*1024**3/shape["bytes_per_expert"])
    choices=[]
    for policy in ("native","hot_late","load_once"):
        hot,_=make_hot(counts,capacity,recurrent,policy)
        choices.append(dict(policy=policy,hot=hot.to(device)))
    engine=Engine(model,shape);full=Full(model)
    tapes={};references={};native_tokens={};expected_copies={}
    p=orx.progress(phase="record_causal_routes",total=len(choices)*len(prompts)*2,unit="requests")
    done=0
    for c in choices:
        for ri,prompt in enumerate(prompts):
            for task in ("free","teacher"):
                route=Routing(model,c["hot"],c["policy"],cfg["tau"],False)
                tape=RouterTape(model)
                with route.attached(),tape.attached():
                    ref=generate(engine,full,route,prompt,cfg["decode_tokens"],device,
                        forced=native_tokens[ri] if task=="teacher" else None)
                key=(c["policy"],ri,task)
                tapes[key]=tape.events;references[key]=ref["tokens"]
                expected_copies[key]=tape_copy_count(tape.events,
                    [tuple(x) for x in c["hot"].nonzero().cpu().tolist()],capacity,recurrent,
                    c["policy"]=="load_once",shape["expected_visits"])
                if c["policy"]=="native" and task=="free":native_tokens[ri]=ref["tokens"]
                done+=1;p.step(done)
    p.flush()
    tape_bytes=sum(t.numel()*t.element_size() for tape in tapes.values() for _,_,record in tape
                   for t in record if isinstance(t,torch.Tensor))
    print("TAPE_STORAGE",json.dumps(dict(bytes=tape_bytes,warning="counterfactual decisions, not an online scheduler")),flush=True)
    del full;engine.reset();gc.collect();torch.cuda.empty_cache()
    p=orx.progress(phase="pack_cpu",total=1,unit="stores")
    store=Store(model,device);p.step(1);p.flush()
    assert all(sum(w.numel() for w in (layer.moe.experts.weight1,layer.moe.experts.weight2,
        layer.moe.experts.weight3))==0 for layer in model.layers)
    pool=torch.empty((capacity,store.expert_numel),device=device,dtype=torch.bfloat16)
    assert pool.numel()*pool.element_size()<=budget*1024**3
    print("PHYSICAL_BUDGET",json.dumps(dict(budget_gib=budget,expert_bytes=pool.numel()*pool.element_size(),
        tape_bytes=tape_bytes,gpu_active_bytes=torch.cuda.memory_allocated(device),
        gpu_reserved_bytes=torch.cuda.memory_reserved(device),environment=assert_own_gpu(gpu_index))),flush=True)
    arms=[(c,controller) for c in choices for controller in ("online","replay")]
    p=orx.progress(phase="warmup",total=len(arms),unit="requests")
    for ai,(c,controller) in enumerate(arms):
        executor=OffloadExecutor(store,pool,c["hot"],recurrent,c["policy"])
        route=Routing(model,c["hot"],c["policy"],cfg["tau"],False)
        key=(c["policy"],0,"free")
        with route.attached():
            if controller=="replay":
                with RouterTape(model,tapes[key]).attached():
                    got=generate(engine,executor,route,prompts[0],cfg["decode_tokens"],device)
            else:got=generate(engine,executor,route,prompts[0],cfg["decode_tokens"],device)
        if got["tokens"]!=references[key]:raise RuntimeError("Warmup did not match causal reference")
        del executor
        p.step(ai+1)
    p.flush()
    total=cfg["repeats"]*len(prompts)*len(arms)*2
    p=orx.progress(phase="offload_ceiling",total=total,unit="requests")
    rows=[];done=0
    for repeat in range(cfg["repeats"]):
        print("ENVIRONMENT",json.dumps(assert_own_gpu(gpu_index)),flush=True)
        for ri,prompt in enumerate(prompts):
            shift=(repeat+ri)%len(arms)
            order=arms[shift:]+arms[:shift]
            if repeat%2:order=order[::-1]
            for c,controller in order:
                executor=OffloadExecutor(store,pool,c["hot"],recurrent,c["policy"])
                for task in ("free","teacher"):
                    key=(c["policy"],ri,task)
                    route=Routing(model,c["hot"],c["policy"],cfg["tau"],False)
                    with route.attached():
                        if controller=="replay":
                            tape=RouterTape(model,tapes[key])
                            with tape.attached():got=generate(engine,executor,route,prompt,cfg["decode_tokens"],device,
                                forced=native_tokens[ri] if task=="teacher" else None)
                        else:
                            got=generate(engine,executor,route,prompt,cfg["decode_tokens"],device,
                                forced=native_tokens[ri] if task=="teacher" else None)
                    if got["tokens"]!=references[key]:raise RuntimeError("Live/replay offload did not match causal reference")
                    if got["copies_per_token"]!=expected_copies[key]:
                        raise RuntimeError("Live/replay expert traffic did not match causal tape")
                    row=dict(policy=c["policy"],controller=controller,task=task,request=ri,repeat=repeat,
                             tpot_ms=got["tpot_ms"],copies=got["copies_per_token"],agreement=1.0)
                    rows.append(row)
                    with (out/"ceiling-timings.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
                    print("CEILING_TIMING",json.dumps(row),flush=True)
                    p.row("ceiling",**row);done+=1;p.step(done,tpot_live=got["tpot_ms"])
                del executor
    p.flush()
    p=orx.progress(phase="report",total=1,unit="files")
    summary=[]
    for task in ("free","teacher"):
        for controller in ("online","replay"):
            baseline=[np.median([r["tpot_ms"] for r in rows if r["policy"]=="native" and
                r["controller"]==controller and r["task"]==task and r["request"]==ri]) for ri in range(len(prompts))]
            for c in choices:
                values=[np.median([r["tpot_ms"] for r in rows if r["policy"]==c["policy"] and
                    r["controller"]==controller and r["task"]==task and r["request"]==ri]) for ri in range(len(prompts))]
                contrast=bootstrap([100*(1-a/b) for a,b in zip(values,baseline)])
                row=dict(task=task,controller=controller,policy=c["policy"],tpot_ms=float(np.mean(values)),
                         speedup_vs_matched_static=contrast)
                summary.append(row);print("CEILING_SUMMARY",json.dumps(row),flush=True)
    (out/"ceiling-summary.json").write_text(json.dumps(dict(config=cfg,summary=summary,
        limitation="Replay is a counterfactual with pre-recorded decisions, not an online speedup."),indent=2))
    p.step(1);p.flush()


def main():
    parser=argparse.ArgumentParser()
    for key in ("config","model-dir","dataset-dir","xllm-loop-dir","out-dir"):
        parser.add_argument("--"+key,required=True)
    args=parser.parse_args();cfg=json.loads(Path(args.config).read_text())
    print("CONFIG",json.dumps(cfg),flush=True)
    index,lock,gpu=acquire_idle_gpu(cfg.get("preferred_gpu"))
    try:
        device=torch.device(f"cuda:{index}");torch.cuda.set_device(device);torch.manual_seed(cfg["seed"])
        print("GPU",json.dumps(gpu.__dict__),flush=True)
        print("RUNTIME",json.dumps(bootstrap_xllm(args.xllm_loop_dir)),flush=True)
        with torch.inference_mode():run(cfg,args,device,index)
    finally:lock.release()


if __name__=="__main__":main()
