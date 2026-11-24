# Looped Transformer 研究笔记（ICML 2027 新方向）

日期：2026-11-24

前期关于「MoE CPU–GPU 专家调度」的研究已完整归档至 `archive/moe-scheduling/`（详见 `archive/moe-scheduling/ARCHIVE_README.md`）。基于 2026-11-24 的最新文献调研（见 `artifacts/icml-2027-loop-directions-2026-11-24/research-directions.md`），当前同时对前两个首选方向开展决定性的初步可行性实验。

---

## 方向 1：哪些 token 值得为了后文多算两轮？（Token KV Depth Value）

### 核心假设
> **一个当前容易预测的位置，可能是后文最需要深层 KV 的位置。** 当前预测收益（$\Delta_{\text{curr}}$）与后文 KV 写入收益（$\Delta_{\text{write}}$）可以分离测量。

### 初步可行性实验协议（`loop_research/idea1_kv_depth.py`）
- **模型**：原版冻结 `ByteDance/Ouro-1.4B`（24 层 × 4 loops，`total_ut_steps=4`，BF16）。
- **一致性校验**：首先验证自定义 Ragged-KV 循环前向在全 4 轮模式下与原生 HuggingFace `OuroForCausalLM` 的 logits 误差 $< 10^{-4}$。
- **28 组合事后最优（2/8 深块，25% 深前缀预算）**：
  - 前缀等分为 8 块；浅块仅运行前 2 轮（`ut=0,1`），其第 3、4 轮（`ut=2,3`）供后文注意力的 K/V 按 Ragged KV 语义回退到同物理层的第 2 轮 K/V；深块与问题+答案后缀固定运行完整 4 轮。
  - 穷举全部 $\binom{8}{2}=28$ 种组合，同时评估全浅（0/8）与全深（8/8）。
- **三组探针**：
  1. **Probe 1A（可控合成配对两跳问答）**：96 篇独立文档（每篇含两个互不相交的两跳事实链 + 4 条干扰事实，8 个事实分别落在 8 个块中），对**同一前缀**分别提出 Question A 与 Question B（所需证据在不同块），并对一半样本执行证据位置交换，以严格分离「内容依赖的真实证据写入」、「固定位置偏置」与「偷看未来问题的事后优势」。同时测量 Gold Answer NLL 与贪心 EM。
  2. **Probe 1B（自然多跳问答 `framolfese/2WikiMultihopQA`）**：96 条验证集样本，保留答对与答错的全部抽样样本，记录每块与 `supporting_facts` 的重合度，测量 Answer NLL 与贪心 EM/F1。
  3. **Probe 1C（当前收益 vs 后文 KV 写入收益解耦 + 因果选择器对比）**：在自然文本与代码样本上分别测量每块的 $\Delta_{\text{curr}}(b)$（全深历史下 2 轮 vs 4 轮当前块 NLL 差）与 $\Delta_{\text{write}}(b)$（仅将块 $b$ 的晚轮暴露 KV 降级为 2 轮导致后文近/中/远距 NLL 增量），检验两者相关性；并在 Probe 1A/1B 上对比全部严格因果在线基线（随机、前两块、后两块、交叉验证最优固定位置、循环2熵、FlashLoop 隐状态变化、Ouro 原生 `early_exit_gate`、loop1→loop2 KL、以及留出交叉验证的轻量因果线性预测器）。

### 内部去留门槛
1. 全浅（0/8）到全深（8/8）存在清晰的质量/NLL 差距。
2. 25% 深块预算的事后最优（`oracle_2`）至少取回该差距的 **50%**，且在配对双问中随证据块移动。
3. 因果可见特征或留出因果选择器能明显超越固定位置与低成本启发式（如至少取回事后最优相对最强简单基线优势的相当比例）。

---

## 方向 2：Loop × MoE：同一个专家在哪几轮应该继续共享？（Expert Cross-Loop Unbinding）

### 核心假设
> **大部分专家从跨轮权重共享中获益，而跨轮梯度冲突导致的共享代价集中在少数可事先辨认的专家上。** 在相同总参数预算下，将逐轮专门化容量精准分配给高冲突专家，能显著优于均匀分配、常规重要性分配、逐轮标量与仅调路由。

### 初步可行性实验协议（`loop_research/idea2_expert_unbinding.py`）
- **模型**：公开的 `IFM/LoopedLM-P1-moe-ouro-500b`（28 物理层全部循环 4 轮，25 专家/层，Top-2）与 `IFM/LoopedLM-P1-moe-huginn-500b`（8 prelude + 12 循环层 × 8 轮 + 8 coda，25 专家/层，Top-2）。
- **精确反传校验**：在 `loop_research/ifm_runtime.py` 中补充纯 PyTorch FP32 累加的 `group_rms_norm_bwd(_affine)` 与 `group_layer_norm_bwd(_affine)` 垫片，并验证拆分得到的各轮梯度之和 $\sum_r g_{e,r}$ 与原生 PyTorch `.grad` 误差 $< 10^{-4}$。
- **三组严格不相交数据切分**（真实 WikiText + Python 代码混合）：
  - `calib`（分 Split A 与 Split B 两个独立子批）：仅用于计算跨批次无偏冲突得分，不更新任何权重：
    - $S^{\text{raw}}_e = \sum_r \langle \tilde g^{(A)}_{e,r}, \tilde g^{(B)}_{e,r} \rangle_F$（消除单批采样方差偏置）；
    - $S^{\text{orth-}W}_e$：先剔除沿原权重 $W_e$ 方向的分量（排除逐轮乘性缩放即可解释的冲突）；
    - $S^{\text{orth-mean}}_e$：剔除沿跨轮均值梯度 $\bar g_e$ 方向的分量（排除「共享 Adapter + 逐轮标量」即可解释的冲突）；
    - 同时记录每专家总激活 token 数 $N_e$、逐轮分布、梯度范数与跨轮余弦矩阵。
  - `adapt_train`（独立训练集）与 `eval_heldout`（独立留出集，评估时始终走原生 router 以包含下游路由变化）。
- **两级因果检验**：
  1. **Exp 2A（单专家局部反事实散点）**：对跨层抽样的专家逐一施加相同 L2 范数预算的「逐轮独立步」vs「跨轮共享步」（梯度来自 `adapt_train`），在 `eval_heldout` 测量真实 NLL 改善差值，与 `calib` 预测冲突得分做 Spearman/Pearson 相关性检验。
  2. **Exp 2B（严格等参数量分组适配对照）**：在 6 个代表物理层（150 个专家）上固定完全一致的可训练参数预算（30 专家 × 逐轮 rank-5 = 150 专家 × 逐轮 rank-1 = 30 专家 × 共享 rank-20），对比：
     - `conflict_top`（高冲突 Top-5/层，逐轮 rank-5）
     - `matched_high` vs `matched_low`（层内按暴露量与梯度范数配对的高/低冲突组）
     - `random`（层内随机 5/层，逐轮 rank-5）
     - `gradnorm_top`（梯度范数 Top-5/层，逐轮 rank-5）
     - `freq_top`（路由频率 Top-5/层，逐轮 rank-5）
     - `shared_plus_scalar`（同高冲突 30 专家，1 个共享 rank-20 LoRA + 逐轮标量）
     - `uniform_all`（全部 150 专家，逐轮 rank-1）
     - `recon_layer`（按整层冲突选 Top 物理层、层内全专家解绑）
     - `router_loop_mod`（冻结全部专家，仅在追踪层训练逐轮路由增量 + 逐轮 norm 调制，参数更少但直接测试路由能否化解冲突）。

### 内部去留门槛
1. 跨独立批次的冲突得分 $S^{\text{orth-mean}}_e$ 具有强可复现性（高正值、显著集中于少数专家）。
2. 在单专家反事实和等参数留出适配中，高冲突组显著优于匹配低冲突组、随机组，并且明显不被 `shared_plus_scalar`、`gradnorm_top`、`uniform_all` 或 `router_loop_mod` 吃掉收益。
