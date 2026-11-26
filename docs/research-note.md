# LoopSched-MoE 研究笔记（ICML 2027 锁定主线）

更新日期：2026-11-26

前期已证伪的探索方向已完整归档至：
- `archive/moe-scheduling/`（早期 CPU–GPU 专家卸载调度）
- `archive/early-explorations-2026-11/`（前缀 KV 深度价值、跨轮解绑 / V-SubMoE、LoopQuant-MoE 混合比特量化、稠密 LoopLM 早退）

当前项目唯一聚焦 **ICML 2027** 旗舰主线：**`LoopSched-MoE` — 面向循环 MoE 大模型的正交专家收缩与非加性轨迹调度**。

---

## 一、核心科学发现（Round 1 – Round 3 已严格验证）

在 `IFM/LoopedLM-P1-moe-ouro-500b`（28 物理层 × 4 轮 = 112 访问，Top-2/64）与 `IFM/LoopedLM-P1-moe-huginn-500b`（8 prelude + 12 层 × 8 轮 + 8 coda = 112 访问，Top-2/64）上，基于严格不相交切分（`calib_a=16`, `calib_b=8`, `eval_heldout=64` Text+Code, 2,272 题零样本 MCQ：Full ARC-C 1,172 + ARC-E 600 + Full OBQA 500）完成验证：

### 1. 深度放大的正交专家收缩定律（Orthogonal Expert Shrinkage）
设 Top-$K$ 路由输出为 $y^{(K)} = \sum_{j=1}^K w_j e_j$（按权重降序），截断为 Top-$k$（$k < K$）并施加标量缩放 $\gamma$：$\hat{y}^{(k)}(\gamma) = \gamma \sum_{j=1}^k w_j e_j$。最小 MSE 闭式解为：
$$\gamma^* = 1 + \frac{\mathbb{E}\!\left[\langle s_k, r_k \rangle\right]}{\mathbb{E}\!\left[\|s_k\|^2\right]}, \quad s_k = \sum_{j=1}^k w_j e_j,\; r_k = \sum_{j=k+1}^K w_j e_j$$
对 $K=2, k=1$，传统重归一化 $\gamma_{\text{renorm}} = 1 + w_2/w_1$ 相对无重归一化（`NoRen`, $\gamma=1$）的均方误差放大倍数为：
$$\frac{\text{MSE}_{\text{renorm}}}{\text{MSE}_{\text{noren}}} = 1 + \frac{\mathbb{E}[w_2^2\|e_1\|^2]}{\mathbb{E}[w_2^2\|e_2\|^2]} - 2\,\frac{\mathbb{E}[w_2^2\langle e_1, e_2\rangle]}{\mathbb{E}[w_2^2\|e_2\|^2]}$$
- **实测验证（R3 `79965e0f`）**：循环体内 Top-1 与 Top-2 专家高度近正交（Ouro $\cos \in [+0.028, +0.086]$，Huginn $\cos \in [+0.048, +0.069]$）且主专家范数占优，`MSE_ren/noren` 在 Ouro 上随循环深度从 `2.23×`（`loop_0`）单调攀升至 **`4.94×`（`loop_3`）**，Huginn 保持 `2.11×–2.59×`；闭式最优 $\gamma_s^* \in [1.000, 1.030]$，说明**零参数 `NoRen` 已捕获 >98% 理论最优收益**。
- **下游收益**：`NoRen` 在 `-25%/-37.5%/-50%` 上于 2,272 题 MCQ 分别带来 Ouro `+1.23 / +1.76 / +1.63 pp`、Huginn `+0.04 / +1.36 / +2.24 pp`（95% CI 全部显著 > 0）。

### 2. 非加性迭代贪心置换（`IterGreedy`）打破 `-50%` 全 Top-1 天花板
- **R2-A 教训（`3c1b0d1d`）**：一阶加性背包（Additive DP）基于单访问独立敏感度同时注入 18–46 个并发 `k=0` 访问，引发严重非线性残差级联坍塌（ΔNLL `+0.87` 至 `+2.12`）。
- **R3 突破（`79965e0f`）**：逐步在真实累积前向 NLL（并经独立 `calib_b` 早停防过拟合）上执行终层 `k=1→2` 与稀疏中段 `k=1→0` 置换：
  - **精确 `-50.0%`（零动态开销）**：Ouro（`2/108/2`）达 `ΔNLL=+0.1782 / Avg MCQ=53.9%`，显著胜出 `all_top1_noren`（`+0.0503` nats / `+1.63 pp` CI `[+0.53, +2.77]`）与 `all_top1_renorm`（`+0.2244` nats / `+3.26 pp` CI `[+1.72, +4.89]`）；Huginn 仅凭 1 组置换（保护末层 `coda` `vid=111`）即削减 `37%` 全模型 ΔNLL（`+0.2161 → +0.1356`）。
  - **超稀疏 `-62.5%`**：Ouro 胜过 `unif_k01` `+0.7032` nats / `+4.93 pp` MCQ；Huginn 胜过 `unif_k01` `+0.3023` nats / `+4.40 pp` MCQ（相对 R2-A 加性 DP 提升 `+9.8 pp`）。

---

## 二、Round 4 目标：真实系统加速、分层多组置换扫描与 `Inkling-2-Mini` 扩展

针对 R3 暴露的两大核心工程与算法机会，Round 4 双轨并进：

### 轨道 A（IFM Ouro-MoE & Huginn-MoE，**已完成**：run `4e9bc3a4`）
1. **零转换 CUDA Graph + Triton Fused-MoE 彻底逆转 `-50%` 倒挂**：
   - `top1_match=True`, `cudagraph_ok=True`, `replay_diff=0.00e+00`；`decode B=1` 原生 eager `xllm` 基线延迟从 `269.22 ms -> 22.87 ms`（Ouro, **11.77×**）和 `254.55 ms -> 23.13 ms`（Huginn, **11.01×**）。
   - 相对 CUDA-Graph 基线的严格同口径算法加速：`-50% m=0` 纯 `k=1` 达 **`1.205×` (B=1) / `1.260×` (B=8) / `1.445×` (B=32) / `1.176×` (Prefill)**（Ouro）与 **`1.203× / 1.261× / 1.455× / 1.122×`**（Huginn）；`-50% m=16` 升至 **`1.252×` (B=1) / `1.497×` (B=32) / `1.292×` (Prefill)**；`-62.5%` 达 **`1.373×` (B=1) / `1.761×` (B=32, 994 tok/s) / `1.469×` (Prefill)**。
2. **分层 `k=0/k=2` 置换帕累托与 Top-2 `k=1→0` 悬崖发现**：
   - 在固定 `-50%` 激活 FLOPs 下，随 $m$ 增大延迟单调降低（`k=0` 跳过整层 router + MoE kernel）。
   - **Ouro-MoE（`best_m=2` 双赢甜点）**：`m=2` 相对 `m=0` 既更快，又同时显著优于 `m=0`（ΔNLL `+0.1782` vs `+0.2285`，CI `[+0.0376, +0.0642]`；Avg MCQ `53.9%` vs `52.3%`，`+1.63 pp` CI `[+0.53, +2.77]`，相对 `renorm` 高 `+3.26 pp`）。
   - **Huginn-MoE（`best_m=3` 揭示 Top-2 悬崖）**：`m=3` 将 ΔNLL 从 `+0.2161` 压低至 `+0.1327`（`+0.0834`，58/64 胜），但在 MCQ 上 `m=3`（`51.5%`）略低于纯 `k=1` `m=0`（`52.7%`），原因在于 Top-2 架构下 `k=1→0` 是 100% 路由专家清零断崖，仅用 WikiText 校准会强化 Coda 似然而削弱中段推理循环。

### 轨道 B（新模型 `thinkingmachines/Inkling-2-Mini`，**进行中**：`loop_research/inkling2_vllm_hook.py` + `r4b_inkling2_loopsched.py`）
- **核心动机（直接回应轨道 A 发现的 Top-2 `k=0` 悬崖）**：26.4B Looped Sparse MoE（6 prelude + 12 循环层 × 3 轮 + 6 coda = 48 逻辑层），**128 专家 Top-8，`n_shared_experts = 0`**，原生 **NVFP4 + MXFP8**（19.5 GB 单文件）。Top-8 使得每访问具备细粒度 $k_v \in \{1..8\}$ 分配空间，完全无需 `k=0` 断崖，且 0 共享专家带来更高的真实端到端加速上限。
- **生产级 vLLM 接入（SM120 Blackwell NVFP4/MXFP8 + 完整 CUDA Graph）**：
  1. 在类级别无侵入挂钩 `vllm.models.inkling2.nvidia.moe.Inkling2MoE.forward`，通过自定义 Triton 路由核（`_select_top8_keep_k_kernel` + `_scatter_general_kernel`）直接复用 vLLM 原生 `bmm_sm120` NVFP4/MXFP8 算子与 `combine`，支持任意 $k_v \in \{0..8\}$（含非 2 的幂 `3/5/6/7`）+ 精确 `NoRen` / $\gamma_{v,k}^*$ 收缩，完全兼容生产级 vLLM CUDA Graph；
  2. 全量验证 48 访问 × $k \in \{1..7\}$ 广义正交收缩几何、`NoRen` vs `Renorm` vs `Opt γ*`、物理层 `Layer-Only` vs 循环阶段 `Stage-Static` vs 非加性 `IterGreedy` 在 `-25%/-37.5%/-50%/-62.5%/-75%` 全谱上的 64 NLL + 2,272 MCQ 质量与生产级 vLLM 端到端 Decode (`B=1/8/32`) / Prefill (`B=8×256`) 加速。
