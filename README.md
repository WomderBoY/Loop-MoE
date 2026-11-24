# Looped Transformer Research（ICML 2027 新方向可行性验证）

本项目面向 **ICML 2027**，在归档前期 MoE CPU–GPU 专家调度研究（见 `archive/moe-scheduling/`）后，启动两个高潜力 Looped Transformer 方向的初步可行性实验：

1. **方向 1（Token KV 深度价值）**：*哪些 token 值得为了后文多算两轮？*
   - 核心问题：在冻结的 `ByteDance/Ouro-1.4B`（4 loops）上，将前缀切分为 8 块，仅允许 2 块走 4 轮、其余 6 块走 2 轮（晚轮采用同物理层 Ragged KV 回退），答案端固定 4 轮。
   - 穷举全部 $\binom{8}{2}=28$ 种组合得到事后最优，结合**同文档双问题配对与证据位置交换**（合成两跳问答）、**自然多跳问答**（`2WikiMultiHopQA`）以及**当前预测收益 vs 后文 KV 写入收益解耦**，检验是否存在「当前易预测但对后文价值极高」的块，以及因果可见信号能否逼近事后最优。
2. **方向 2（Loop × MoE 专家跨轮解绑）**：*同一个专家在哪几轮应该继续共享？*
   - 核心问题：在公开的 `IFM/LoopedLM-P1-moe-ouro-500b`（4 loops，整栈循环）和 `IFM/LoopedLM-P1-moe-huginn-500b`（8 loops，中间块循环）上，将同一物理专家在各访问轮次的最终语言模型梯度拆解为 $g_{e,r}$，利用跨独立批次的无偏中心化内积（原始、去权重共线、去均值共线）预测跨轮共享冲突。
   - 通过**单专家反事实干预散点**与**严格等参数量、留出集评估的分组适配对照**（对比匹配低冲突、随机、梯度范数、路由频率、共享 Adapter+逐轮标量、全专家均匀低秩、Recon 层级选择、仅调逐轮路由与调制），验证冲突预测能否提前识别真正值得解绑的少数专家。

## 仓库结构

```text
.
├── archive/moe-scheduling/     # 已归档的前期 MoE 专家调度研究代码、文档与 IFM 阶段代码
├── configs/
│   └── experiment.json         # 当前分支的实验配置（统一由 scripts/run_experiment.sh 调度）
├── docs/
│   └── research-note.md        # 当前研究笔记、实验协议与去留门槛
├── loop_research/
│   ├── gpu_utils.py            # 安全空闲 GPU 选择器（自动避开他人占用显卡 + 文件锁防冲突）
│   ├── ifm_runtime.py          # IFM xllm-loop 加载器 + 纯 PyTorch 前/反向 SM120 垫片
│   ├── data_assets.py          # 镜像源数据集下载与合成配对两跳数据生成
│   ├── idea1_kv_depth.py       # 方向 1 初步可行性实验主程序
│   ├── idea2_expert_unbinding.py # 方向 2 初步可行性实验主程序
│   └── orx_run.py              # 标准进度与指标上报工具
├── scripts/
│   ├── setup_env.sh            # 环境校验与轻量依赖安装（清华镜像源）
│   ├── download_assets.sh      # 模型与数据集下载脚本（hf-mirror.com 镜像源）
│   └── run_experiment.sh       # 统一实验入口脚本
├── tests/                      # 本机可运行的轻量单元测试（不跑 GPU 训练/推理）
└── requirements-gpu.txt        # GPU 服务器 Python 依赖清单
```

## 快速开始（GPU 服务器：pku_pro5000 / pku_A800 / AutoDL）

> 注意：不要在 MacBook 本机运行模型训练或推理；本机仅用于开发与轻量单测（`pytest tests/`）。

### 1. 环境准备

在已有 PyTorch CUDA 环境（如 `pku_pro5000` 上的 `$HOME/loop-moe-env`）中直接运行：

```bash
export PYTHON_BIN="$HOME/loop-moe-env/bin/python"   # 或激活你的 conda/venv 环境
bash scripts/setup_env.sh
```

若在全新的 AutoDL / 智星云机器上配置环境，可先安装对应 CUDA 版本的 PyTorch，再安装依赖：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-gpu.txt
bash scripts/setup_env.sh
```

### 2. 下载模型与数据集（支持国内镜像）

默认使用 `HF_ENDPOINT=https://hf-mirror.com`，模型缓存至 `/data1/shared/$USER/hf_models`（若不可写则回退到 `$HOME/ifm_storage/hf_models`）：

```bash
# 下载方向 1 所需资产（ByteDance/Ouro-1.4B + 2WikiMultihopQA + WikiText + HumanEval）
ASSETS="ouro14b,datasets" bash scripts/download_assets.sh

# 校验/下载方向 2 所需资产（IFM Ouro MoE + IFM Huginn MoE + WikiText + HumanEval）
ASSETS="ifm_ouro,ifm_huginn,datasets" bash scripts/download_assets.sh
```

### 3. 运行初步可行性实验

脚本会自动检测空闲 GPU（严格跳过显存占用 > 256 MiB、存在活跃计算进程或利用率 > 20% 的显卡），也可通过 `LOOP_GPU_INDEX` 手动指定空闲卡号：

```bash
# 运行当前分支配置的实验（读取 configs/experiment.json）
bash scripts/run_experiment.sh

# 如需指定空闲 GPU（例如 GPU 0 或 GPU 5）
LOOP_GPU_INDEX=0 bash scripts/run_experiment.sh
```

## 归档说明

前期 MoE 专家调度研究的所有代码、配置文件、实验报告和 AutoDL 脚本均已完整移至 `archive/moe-scheduling/`，详情见 `archive/moe-scheduling/ARCHIVE_README.md`。
