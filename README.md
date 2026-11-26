# LoopSched-MoE：面向 Looped Transformer 的 MoE 专家调度研究（ICML 2027）

本项目面向 **ICML 2027**，聚焦于利用 Looped Transformer 特有的跨轮（Loop）结构与误差几何信息，在不微调模型权重的前提下显著提升 **Looped Sparse MoE** 模型的推理效率与质量–计算帕累托前沿。

> 前期已证伪的探索方向均已完整归档至 `archive/moe-scheduling/`（CPU–GPU 专家卸载）与 `archive/early-explorations-2026-11/`（前缀 KV 深度价值、跨轮解绑 / V-SubMoE、LoopQuant-MoE 混合比特量化、稠密早退），详见各自目录下的 `ARCHIVE_README.md`。

---

## 核心方法与支持模型

1. **正交专家收缩（Orthogonal Expert Shrinkage, `NoRen` / $\gamma_s^*$）**：揭示循环 MoE 中 Top-$k$ 截断后传统重归一化（Renormalization）因专家近正交与主专家范数占优导致 $2.1\times\text{–}4.94\times$ 的均方误差放大，零参数去除重归一化即可在 2,272 题零样本 MCQ 上带来显著提升。
2. **非加性迭代贪心轨迹调度（`LoopSched-IterGreedy`）**：在 `(layer, loop)` 二维访问图上以真实累积前向 NLL + 独立验证集 `calib_b` 驱动非加性优化，彻底消除传统加性 DP 的并发 `k=0` 级联坍塌，支持 `-25%` 至 `-62.5%` 全预算谱及可控 `k=0/k=2` 置换组数 $m$ 扫描。
3. **双轨真实 GPU 加速后端**：
   - **IFM Ouro-MoE & Huginn-MoE（`xLLM` 格式，BF16，Top-2/64）**：零权重转换接入 **Triton Fused-MoE + 融合 GroupRMSNorm + 静态 CUDA Graph** 引擎（`loop_research/ifm_fast_engine.py`），消除 Eager Python 派发瓶颈。
   - **ThinkingMachines `Inkling-2-Mini`（NVFP4/MXFP8，Top-8/128，3 轮循环体）**：通过独立环境接入生产级 **vLLM / SGLang** 融合 Blackwell 推理框架，验证广义 Top-$8 \to k_v \in \{0..8\}$ 调度与真实端到端加速。

---

## 仓库结构

```text
.
├── archive/
│   ├── moe-scheduling/                 # 早期 CPU-GPU 专家卸载归档
│   └── early-explorations-2026-11/     # 早期已证伪方向归档（附 ARCHIVE_README.md）
├── configs/
│   ├── experiment.json                 # 当前默认运行配置（由 scripts/run_experiment.sh 读取）
│   ├── r2a_config.json / r2a_view.xml  # Round 2-A 复现配置
│   ├── r3_config.json / r3_view.xml    # Round 3 复现配置
│   ├── r4a_config.json / r4a_view.xml  # Round 4-A（IFM CUDA-Graph + Triton + 分层置换扫描）配置
│   └── r4b_inkling_config.json / r4b_view.xml # Round 4-B（Inkling-2-Mini 生产级 vLLM Top-8->k_v 调度）配置
├── docs/
│   └── research-note.md                # 核心定理、已验证实证结论与 Round 4 设计笔记
├── loop_research/
│   ├── gpu_utils.py                    # 空闲 GPU 安全选择器（严格避开他人占用显卡 + 文件锁）
│   ├── data_assets.py                  # 国内镜像模型与数据集下载（含 16 线程 ModelScope CDN）+ 确定性切分/MCQ 加载器
│   ├── ifm_runtime.py                  # IFM xllm-loop 加载器与 SM120 纯 PyTorch 垫片
│   ├── ifm_fast_engine.py              # IFM 零转换 Triton Fused-MoE + CUDA Graph 推理加速引擎
│   ├── inkling2_vllm_hook.py           # Inkling-2-Mini 生产级 vLLM SM120 NVFP4/MXFP8 Triton Top-8->k_v + NoRen 挂钩
│   ├── r2a_loopsched_moe.py            # Round 2-A 完整复现脚本
│   ├── r3_loopsched_v2.py              # Round 3 完整复现脚本
│   ├── r4a_ifm_fast_and_swaps.py       # Round 4-A：CUDA-Graph 真实加速 + 分层 IterGreedy + m 组置换扫描
│   ├── r4b_inkling2_loopsched.py       # Round 4-B：Inkling-2-Mini Top-8->k_v 调度与生产级 vLLM CUDA-Graph 加速验证
│   └── orx_run.py                      # 标准化进度与指标上报模块
├── scripts/
│   ├── setup_env.sh                    # IFM 主环境校验与依赖安装（清华镜像源）
│   ├── setup_inkling_env.sh            # Inkling-2-Mini 独立 vLLM/transformers 环境配置（清华镜像源）
│   ├── download_assets.sh              # 模型与数据集下载脚本（默认 hf-mirror.com）
│   └── run_experiment.sh               # 统一实验启动入口
├── tests/                              # 本机可运行的轻量 CPU 单元测试（不触发 GPU 训练/推理）
└── requirements-gpu.txt                # GPU 服务器基础依赖清单
```

---

## 快速开始（GPU 服务器：pku_pro5000 / pku_A800 / AutoDL / 智星云）

> **开发规范**：MacBook 本机仅用于代码编写与轻量 CPU 单元测试，请勿在本机执行模型训练或推理。GPU 脚本默认启用 `acquire_idle_gpu()`，严格跳过被他人占用的显卡。

### 1. 环境配置（使用清华 PyPI 镜像）

```bash
# 1) IFM Ouro-MoE / Huginn-MoE 主环境（复用已有 PyTorch CUDA + Triton 环境）
export PYTHON_BIN="$HOME/loop-moe-env/bin/python"
NEED_XLLM=1 bash scripts/setup_env.sh

# 若在全新 AutoDL/智星云服务器上初始化（先安装匹配驱动的 PyTorch CUDA）：
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-gpu.txt
NEED_XLLM=1 bash scripts/setup_env.sh

# 2) （可选）为 Inkling-2-Mini 创建独立 vLLM / transformers 推理环境（不干扰主环境）
bash scripts/setup_inkling_env.sh
```

### 2. 下载模型与评测数据集（默认 `https://hf-mirror.com` 镜像）

模型与数据默认缓存至 `/data1/shared/$USER/`（不可写时自动回退至 `$HOME/ifm_storage/`）：

```bash
# 下载/校验 IFM Ouro-MoE + Huginn-MoE + 全套 NLL/MCQ 评测数据集
ASSETS="ifm_ouro,ifm_huginn,datasets,arc_challenge,arc_easy,openbookqa" bash scripts/download_assets.sh

# 单独下载新模型 thinkingmachines/Inkling-2-Mini（约 19.5 GB NVFP4+MXFP8，无需 GPU）
ASSETS="inkling2_mini" bash scripts/download_assets.sh
```

### 3. 运行实验与复现命令

```bash
# 运行当前默认实验（Round 4-B：Inkling-2-Mini 生产级 vLLM Top-8->k_v 调度与加速，自动切换独立 inkling-vllm-env）
bash scripts/run_experiment.sh

# 复现 Round 4-A（IFM Ouro-MoE + Huginn-MoE CUDA-Graph + Triton + 分层 m 组置换扫描）
CONFIG_PATH=configs/r4a_config.json bash scripts/run_experiment.sh

# 仅下载新模型并检测空闲卡（若全部显卡被占用则优雅退出，不阻塞也不抢卡）
EXIT_IF_NO_IDLE_GPU=1 bash scripts/run_experiment.sh

# 复现 Round 3 全部 24 组条件结果
CONFIG_PATH=configs/r3_config.json bash scripts/run_experiment.sh

# 手动指定空闲卡号（如 GPU 4）
LOOP_GPU_INDEX=4 bash scripts/run_experiment.sh
```

### 4. 本机轻量代码与单元测试检查（MacBook CPU）

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```
