# MoE CPU–GPU 专家调度研究归档（2026-11-11 至 2026-11-23）

本目录完整归档了前期围绕「利用 Looped Transformer 循环阶段信息优化 MoE CPU–GPU 专家调度」开展的代码、配置、脚本、测试与研究文档。

## 归档内容

- `loop_moe/`、`configs/`、`docs/`、`scripts/`、`tests/`、`README.md`：Foil-1/3、Loopie-6B、Looped-MoE 阶段（路由轨迹采集、19 策略驻留回放、AutoDL 复现脚本与扩样分析报告）。
- `ifm_phase/`：从冻结实验分支 `orx/b2-v2-fast-gpu-offload-vectorized-1ms-loop-predi`（commit `dc62cef`）提取的 IFM Part I Ouro MoE 与 Huginn MoE 500B 阶段完整代码（A1–A3 轨迹采集与回放、B1 KV-cache 校验、B2 真实 GPU offload 与向量化预测器端到端 TPOT 评测）。

## 已归档的冻结分支与产出报告

原有四个实验节点已在实验树中归档隐藏，其分支与日志完整保留（不可改写）：

1. `orx/a1-a3-smoke-ifm-ouro-huginn-500b`（`0f78f543`）：IFM Ouro / Huginn 500B A1–A3 Smoke。
2. `orx/a1-a3-formal-192-reqs-3-gpu-ifm-ouro-huginn-500b`（`aabc1530`）：192 请求正式采集与离线回放。
3. `orx/b1-b2-stage-b-kv-cache-verify-multi-loop-hazard`（`1ac24098`）：Stage B KV-cache 一致性验证与真实 GPU Offload 基准。
4. `orx/b2-v2-fast-gpu-offload-vectorized-1ms-loop-predi`（`1ccf7a14`）：向量化预测器与真实墙钟 TPOT 评测。

对应的完整实验报告与图表保存在项目 artifacts 目录：
- `artifacts/ifm-a1-a3-smoke-2026-11-22/`
- `artifacts/ifm-a1-a3-formal-2026-11-22/`
- `artifacts/ifm-b1-b2-2026-11-22/`
- `artifacts/ifm-b2-fast-2026-11-22/`
- `artifacts/loop-inference-2026-11-20/`

## 归档原因与转向结论

在 IFM Ouro MoE 与 Huginn MoE 500B 上，虽然循环阶段信号在 2–6.3 GiB 预算下能显著提升离线专家预测召回与 H2D 节省，但在真实端到端 GPU Offload 执行器中，受静态热门专家驻留、传输/计算重叠与 CPU 预测开销限制，相对最强通用/静态基线难以稳定转化为 10%–15% 的端到端 TPOT 降低。因此将 MoE 专家调度研究归档，全力转向更有 ICML 2027 潜力的两个核心方向（详见根目录 `docs/research-note.md` 与 `artifacts/icml-2027-loop-directions-2026-11-24/research-directions.md`）。
