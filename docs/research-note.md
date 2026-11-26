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

### 轨道 A（IFM Ouro-MoE & Huginn-MoE）：CUDA Graph + Triton Fused-MoE + 分层 `k=0/k=2` 置换帕累托
1. **消除 Eager Python 派发瓶颈（`loop_research/ifm_fast_engine.py`）**：
   - 零权重格式转换直接复用原生 `xllm` GPU 权重；
   - 实现 **Triton 融合 GroupRMSNorm** + **Triton 零拷贝索引专家 SwiGLU 解码核（`triton_indexed_moe_swiglu`）**（直接按 `exp_ids` 原地索引 `weight1/2/3`，彻底消除 `index_select` 权重拷贝与 `.tolist()` 同步！）+ 原生零额外开销的 `k_v ∈ {0, 1, 2}` 静态路由；
   - 使用 **静态 `torch.cuda.CUDAGraph`** 捕获整次 112 轮前向，在完全公平的 CUDA Graph 基线下测量 `bf16_base` vs `LoopSched` 的真实端到端 Decode (`B=1, 8, 32`) 与 Prefill (`B=8, L=256`) 加速。
2. **修复分层候选池 + 显式扫描 `k=0/k=2` 置换对数 $m$（`loop_research/r4a_ifm_fast_and_swaps.py`）**：
   - 将候选池改为按循环阶段分层抽样（确保 Huginn `loop_2..6` 充分进入候选池，突破 R3 仅 1 组置换的局限）；
   - 在精确 `-50%` 预算下显式扫描置换组数 $m \in \{0, 2, 4, 6, 8, 12, 16\}$（含非相邻 `k=0` 约束），绘制完整的 **真实 GPU 延迟 vs 质量（NLL & 2,272 MCQ）二维帕累托前沿**；
   - 将非加性迭代贪心推广至 `-25%`、`-37.5%` 纯 `k∈{1,2}` 调度，检验在 Ouro-MoE 上是否同样全面超越 `Layer-Only`。

### 轨道 B（新模型 `thinkingmachines/Inkling-2-Mini`）：广义 Top-$8 \to k_v$ 调度与生产级 vLLM 加速
- **架构亮点**：26.4B Looped Sparse MoE（6 prelude + 12 循环层 × 3 轮 + 6 coda = 48 逻辑层），**128 专家 Top-8，`n_shared_experts = 0`**，原生 **NVFP4 + MXFP8**（19.5 GB 单文件，完美契合 Blackwell RTX PRO 5000），获得 vLLM (`>=0.35.0`) / SGLang Day-0 原生支持。
- **核心研究验证（`loop_research/r4b_inkling2_loopsched.py`）**：
  1. 将正交专家收缩定理推广至一般 Top-$K \to k$（$K=8, k \in \{1..7\}$）；
  2. 测量 48 逻辑访问的逐轮几何与敏感度分布（Prelude vs Pass 0/1/2 vs Coda）；
  3. 细粒度逐访问 $k_v \in \{0, 1, \dots, 8\}$ 非加性调度 + 在生产级推理框架（vLLM / transformers）上的真实 TTFT/TPOT 墙钟加速与下游 MCQ/NLL 评测。
