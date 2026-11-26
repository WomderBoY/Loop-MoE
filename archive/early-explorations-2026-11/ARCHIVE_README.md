# 早期可行性探索与已淘汰方向归档（2026-11-24 至 2026-11-26）

本目录归档了在锁定 **ICML 2027 主线 `LoopSched-MoE`** 之前开展的早期探针与已证伪方向的代码、配置与测试文件。全部对应历史实验节点和日志均完整保留在各自冻结分支中（不可改写），此处仅将主线分支上不再维护的代码归档，保持主线仓库整洁。

## 归档清单与淘汰结论

| 归档文件 / 冻结分支 | 方向概述 | 关键实验结论 | 淘汰原因 |
|---|---|---|---|
| `loop_research/idea2_expert_unbinding.py`<br>`tests/test_idea2_logic.py` | 方向 2：专家跨轮梯度冲突解绑（IFM Ouro / Huginn） | 同参数预算下按轮切分适配输给全局共享适配 | 按轮拆分导致 $R\times$ 秩与训练样本稀释 |
| `loop_research/vsubmoe_pilot.py`<br>`tests/test_vsubmoe_logic.py` | V-SubMoE：共享子空间 + 速度调制逐轮路由 | 留出集提升微弱且不如参数等价共享适配 | 跨轮路由/子空间调制收益不足以支撑顶会主贡献 |
| `loop_research/probe2_loopmoe_skip_prune.py`<br>`tests/test_probe2_skip_prune.py`<br>`configs/probe2_*` | Round 1 Probe 2：Looped MoE 动态跳过与阶段剪枝初探（`14b9e51f`） | 首次验证 `NoRen` 收益与 U 型阶段敏感度 | 已被更严谨的 `r2a_loopsched_moe.py` 与 `r3_loopsched_v2.py` 完整替代 |
| 冻结分支 `orx/direction-1-*` | 方向 1：前缀 Token KV 深度价值（`Ouro-1.4B`） | 前缀 KV 写入敏感度比当前步弱 `80×` | 信息瓶颈在当前步而非历史前缀深浅 |
| 冻结分支 `orx/r1-probe-1-*` & `orx/r2-b-loopquant-moe-*` | `LoopQuant-MoE`：轨迹感知 AWQ 与阶段敏感度混合比特 PTQ（`04c39b4b`） | `stagesens_awq` 未赢 `pool_awq`；混合比特背包在 `2.25/2.50/3.00b` 全线输给普通全局频率基线 | 非 Loop 频率分配已隐式保护重要专家；仅保留均匀 `W3-g128 pool_awq` 作为工程压缩参考 |
| 冻结分支 `orx/r1-probe-3-*` | 稠密 `Ouro-Thinking` 分阶段早退（`957b4a3a`） | GSM8K `phase_think_kl2_ans4` 达 `85% @ R=2.72` | 稠密 LoopLM 早退文献拥挤，且偏离 MoE 专家调度主线 |

## 保留的当前 ICML 2027 主线
主线代码与完整复现脚本保留在根目录 `loop_research/`，聚焦 **`LoopSched-MoE`**（正交专家收缩 + 非加性迭代贪心轨迹调度 + CUDA Graph/Triton 与 vLLM 生产级真实加速，详见 `docs/research-note.md` 与根目录 `README.md`）。
