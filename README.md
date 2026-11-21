# Looped MoE 专家路由与缓存实验

本项目研究深度循环 MoE 在 GPU 专家缓存受限时，能否利用循环阶段和已知执行顺序减少专家权重搬运，目标是 ICML 2027。流程包括真实路由采集、离线回放和强在线基线对照。

截至 2026-11-21，四模型扩样已完成，共 **768 条请求、380,492 个生成 token**。Loopie-6B 在 **5% 专家缓存预算**下估计 H2D 字节减少 **1.140%**，129 条评估请求均受益。结果来自离线回放，工作负载的质量与适用范围见扩样报告。

| 文档 | 内容 |
| --- | --- |
| [研究笔记](docs/research-note.md) | 当前判断、相关工作和下一轮实验 |
| [扩样结果与生成质量](docs/expanded-results-2026-11-20.md) | 新旧对比、全部预算结果、消融和检查点质量 |
| [驻留方法设计](docs/loop-aware-expert-residency.md) · [开发集回放](docs/residency-pilot-2026-11-18.md) | 方法定义和旧试跑证据 |
| [扩样运行记录](docs/expanded-run-2026-11-19.md) | 固定配置、采集环境、服务器归档与 CPU 续跑 |

本地实验目录 `runs/`、演示资料 `slides/` 及演示校验输出由 Git 忽略。复核既有实验时，使用运行记录中的归档快照、配置和结果。

## 复现四模型扩样

`configs/expanded.json` 使用 Foil-1、Foil-3、Loopie-6B、Looped-MoE 四个原 checkpoint，**每模型 192 条 prompt、最多 512 个新 token**。三类各 64 条，均为新请求，来自 [Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k)、[GSM8K test](https://github.com/openai/grade-school-math)、[sanitized MBPP test](https://github.com/google-research/google-research/tree/master/mbpp)，只取题目，不向模型提供答案或测试。固定 revision、源文件 SHA256、选取规则及许可记录在 `configs/prompts_expanded.sources.json`；四个模型共享同一输入文件。Dolly 选取的内容及其添加回答后缀的版本按 CC-BY-SA-3.0 提供，归属 Databricks 及原贡献者；GSM8K 为 MIT，MBPP 为 CC-BY-4.0，归属各数据集原作者。

最多生成量由每模型 2,304 增至 98,304 token（约 42.7 倍）。保留正常 EOS 停止，实际数量以轨迹和 `analysis/requests.csv` 为准。模型精度、greedy、batch=1、循环结构、回放预算及策略参数沿用试跑设置；原 18 条 prompt 和历史结果保留。这批输入用于路由统计，不用于报告标准 benchmark 的准确率。

按[服务器与环境](#服务器与环境)配置并激活 Python 3.11 后，可在服务器上复现到新输出目录。以下命令在服务器的仓库根目录执行。已有四个固定 revision 的缓存时，可跳过下载步骤。

```bash
OUT_DIR=runs/expanded-reproduction
PYTHON=$(command -v python)
MOE_CONFIG=configs/expanded.json bash scripts/download_models.sh
# 只做 tokenizer/context 检查，不加载模型、不使用 GPU
PYTHON="$PYTHON" bash scripts/run_expanded.sh "$OUT_DIR" --preflight-only

# 先检查显存和计算进程；将 5、7 换成实际空闲卡号
nvidia-smi
mkdir -p "$OUT_DIR/logs"
nohup env PYTHON="$PYTHON" \
  bash scripts/run_expanded.sh "$OUT_DIR" --gpus 5 7 \
  >"$OUT_DIR/logs/driver.log" 2>&1 < /dev/null &

# 查看总状态和逐批日志
cat "$OUT_DIR/status.json"
tail -n 20 "$OUT_DIR/logs/driver.log"
```

脚本最多使用两张明确指定的卡，启动时、每批前及采集中检查计算进程；发现其他进程占卡会停止自己的采集，不终止他人进程，也不等待空闲后自动重试。每批 24 条（各类 8 条），两个工作进程共享 32 批任务；已完成批次可复用，同样的命令可恢复，未完成批次重跑。恢复要求配置、输入和采集/分析代码指纹一致；改动后使用新输出目录。

### AutoDL：无卡准备，单张 A800 80GB 四模型并发

本次租用实例的代码目录为 `/root/autodl-tmp/Loop-MoE`，环境为 `/root/miniconda3/envs/loop-moe`，权重缓存为 `/root/autodl-tmp/huggingface`。`scripts/autodl_env.sh` 设置这些路径与 Hugging Face 镜像。无卡模式可以完成依赖安装、权重下载、文件校验、合成测试和 tokenizer 上下文检查，不运行模型 forward。

```bash
cd /root/autodl-tmp/Loop-MoE
source scripts/autodl_env.sh
OUT_DIR=runs/autodl-a800-reproduction
# 首次准备时创建环境并安装依赖
conda create --prefix "$MOE_ENV_PREFIX" python=3.11 -y
bash scripts/setup_gpu.sh cu130
# 无卡模式的准备与检查
MOE_CONFIG=configs/expanded.json bash scripts/download_models.sh
python scripts/verify_model_cache.py --output runs/setup/verified-models.json
python -m unittest discover -s tests -v
HF_HUB_OFFLINE=1 python -m loop_moe.expanded \
  --out-dir "$OUT_DIR" --preflight-only
```

切换为**单 GPU 模式、A800 80GB** 后启动：

```bash
cd /root/autodl-tmp/Loop-MoE
OUT_DIR=runs/autodl-a800-reproduction
mkdir -p "$OUT_DIR/logs"
nohup bash scripts/run_autodl_single_gpu.sh "$OUT_DIR" \
  >"$OUT_DIR/logs/driver.log" 2>&1 < /dev/null &
cat "$OUT_DIR/status.json"
tail -n 20 "$OUT_DIR/logs/driver.log"
```

该入口使用 `--parallel-models --exclusive-gpu`，每个模型独立运行一个采集任务队列，按顺序完成自己的 8 批请求。默认选择唯一可见 GPU 的 UUID，可用 `GPU_ID` 指定卡号或 UUID。启动会核对 A800 80GB 型号、至少 60 GiB 空闲显存和 GPU 占用情况。`--exclusive-gpu` 仅用于租用平台独占分配的整卡，运行期间依赖平台隔离。共享校内服务器使用默认的持续占卡检查。四进程共享 GPU 的运行用于路由采样。

通用入口也支持 `bash scripts/run_expanded.sh OUT_DIR --gpus 0 --parallel-models`；省略 `--parallel-models` 则沿用每张卡一个进程的任务队列方式。

采集完成后释放 GPU，自动合并各模型的 8 批轨迹，然后运行路由统计、基础回放与驻留回放。合并后按请求划分，每模型 **63 条拟合、129 条评估**。输出目录包含各模型的 `trace.jsonl.gz`、`analysis/`、`replay/`、`residency/`，以及记录输入和代码指纹的 `manifest.json` 与上下文检查结果 `preflight.json`。已完成的 11 月扩样归档位置和 CPU 续跑命令见[运行记录](docs/expanded-run-2026-11-19.md#归档与-cpu-后处理)。

如需重新构建同一批输入，先指定一个未存在的输出文件；这个步骤不调用模型：

```bash
python scripts/prepare_expanded_prompts.py --cache /tmp/loop-moe-prompt-sources \
  --out /tmp/prompts_expanded.jsonl
```

采集仍按每步重算完整前缀的方式进行，512 token 会明显增加计算量。增加样本量能改善路由与回放统计，不能据此得出真实 KV-cached decode 或端到端加速结论。

## 服务器与环境

### 租用单卡并传代码

以下为两台校内服务器的只读查询结果，适合对照租用实例的实际 GPU 型号和驱动。

| SSH 别名 | GPU | `nvidia-smi` 显存 | Compute capability | 驱动 |
| --- | --- | ---: | --- | --- |
| `pku_A800` | NVIDIA A800-SXM4-80GB | 81920 MiB | 8.0 | 580.126.20 |
| `pku_pro5000` | NVIDIA RTX PRO 5000 72GB Blackwell | 73415 MiB | 12.0 | 580.126.20 |

本实验用一张 GPU。租用时选 Linux x86_64、带 Conda 的镜像，查看 `nvidia-smi`。推荐驱动 `580.95.05` 或更新，用 `cu130` 安装 PyTorch。A800 若只有 CUDA 12.x 驱动支持且兼容 CUDA 12.6，可改用 `cu126`。Blackwell 的原生支持从 CUDA 12.8 开始，本项目在 Blackwell 上用 `cu130`。预编译的 PyTorch 安装包已包含所需的 CUDA 用户态库，安装脚本无需 `sudo` 或 `nvcc`。 [Foil 作者环境](https://github.com/SR-A-W/how-to-loop-moe#installation) · [NVIDIA 架构与 CUDA 对照表](https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html) · [NVIDIA CUDA 13.0 驱动对照表](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-toolkit-release-notes/index.html)

代码可从 Git 仓库拉取。若改动仍在本地工作区，从 Mac 的仓库根目录直接传到**租用实例**。将 `user@rented-host` 换成平台提供的 SSH 地址或本机 SSH 别名。

```bash
GPU_HOST=user@rented-host
rsync -av --exclude='__pycache__/' README.md requirements-gpu.txt environment-replay.yml configs loop_moe scripts tests "$GPU_HOST:loop-moe/"
ssh "$GPU_HOST"
```

如果平台分配了单独的 SSH 端口，`rsync` 加 `-e 'ssh -p 端口'`，`ssh` 加 `-p 端口`。

### 在租用实例配置依赖

如果平台提供以下两种基础镜像，优先选择 **PyTorch 2.12.1 / Python 3.12 / Ubuntu 22.04 / CUDA 13.0**。无需自制镜像；进入后新建下面的 Python 3.11 环境，按脚本安装本项目已验证的版本。基础镜像预装的 Python/PyTorch 与实验环境可以不同，不直接复用其 2.12.1 环境。

另一种 **PyTorch 2.8.0 / Python 3.12 / Ubuntu 22.04 / CUDA 12.8** 也可作为系统底座，但不是当前默认选择；同样需要新建环境并安装项目依赖。能否使用 `cu130` 取决于实例宿主机 NVIDIA 驱动，而不是仅看镜像名称；`nvidia-smi` 中的 CUDA Version 表示驱动支持的 CUDA 上限，不是当前 Python 的 PyTorch CUDA 构建。当前 A800 上已实际验证的是：

| 组件 | 已验证版本 |
| --- | --- |
| Python | 3.11.18（脚本要求 3.11） |
| PyTorch | 2.13.0+cu130 |
| Transformers | 5.15.0 |
| einops | 0.8.2 |
| A800 驱动 | 580.126.20 |

租用 CUDA 13.0 实例时优先选驱动 `580.126.20` 或更新以贴近已验证环境；CUDA 13.x 要求 R580 或更新的驱动系列。PyTorch 官方提供 2.13.0 的 `cu130` 安装包，使用下面的脚本即可，无需手动编译 PyTorch 或单独安装系统 CUDA Toolkit。[PyTorch 官方安装版本](https://pytorch.org/get-started/previous-versions/#v2130) · [NVIDIA CUDA 兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

```bash
cd ~/loop-moe
nvidia-smi
conda create -n loop-moe python=3.11 -y
conda activate loop-moe
bash scripts/setup_gpu.sh cu130
python -m unittest discover -s tests -v
```

脚本安装 `torch==2.13.0` 对应的 CUDA 构建、`transformers==5.15.0` 和 `einops==0.8.2`。`cu130` 的原生 CUDA 组件另按 `configs/native-cu130-validated.txt` 固定为已验证版本。模型代码与权重固定在 `configs/foil_pilot.json` 所列的 Hugging Face revision，首次运行自动下载所选模型的文件。两个模型的权重各约 1.1 GB，另需为 PyTorch 安装和缓存留空间。若系统盘空间有限，先将 `HF_HOME` 指到实例的可写数据盘。模型由发布者提供的自定义 Python 实现加载。 [Foil 模型卡](https://huggingface.co/ShourenWSR/how-to-loop-moe)

## 试跑与路由采集

试跑配置默认使用 Foil-1 和 Foil-3，包含文本、数学、代码各 6 条请求，采用 BF16、batch=1、greedy decoding，最多生成 128 个新 token。采集使用第一个可见 GPU。使用共享服务器时先确认对应 GPU 空闲，使用有作业调度器的实例时保留调度器分配的 `CUDA_VISIBLE_DEVICES`。

```bash
# 两条 prompt、最多生成 4 token，验证下载、加载、GPU 算子、导出和回放
bash scripts/run_pilot.sh runs/smoke --model Foil-1 --limit 2 --max-new-tokens 4

# 两个模型，使用内置的 18 条 prompt
bash scripts/run_pilot.sh runs/pilot

# 换成自己的输入，每行需要 id、category、text 三个非空字段，id 必须唯一
bash scripts/run_pilot.sh runs/expanded --prompts /path/to/prompts.jsonl
```

输入格式示例见 `configs/prompts_pilot.jsonl`。可以重复传 `--model` 选择多个模型。模型已下载时可加 `--local-files-only`。采集完成的文件会保留。若第二个模型中途中断，用同一个输出目录和 `--model Foil-3` 只补跑第二个模型。使用自定义配置时给 `FOIL_CONFIG` 指定配置文件。

### Loopie 和 Looped-MoE

| `--model` | 权重与精度 | 专家层的实际访问顺序 |
| --- | --- | --- |
| `Loopie-6B` | [Loopie-6B-A0.6B-preview](https://huggingface.co/zitian-gao/Loopie-6B-A0.6B-preview)，BF16 | 18 个物理层，每层连续执行两次，每层 128 选 8 |
| `Looped-MoE` | [1e18-d704-seed42](https://huggingface.co/ml-ryanlee/seedvar-looped-moe-1e18-d704-seed42)，FP32 | 8 层整块执行两轮，每层 8 选 2 |

Loopie 使用模型的 chat template，按作者的 [vLLM 实现](https://github.com/zitian-gao/vllm/blob/9a9703d9918115cac2c6abfdb84aca411468c16c/vllm/model_executor/models/qwen3_moe.py) 在 Transformers 内重复调用同一个物理层。嵌入表和输出层按 checkpoint 中的 152,064 行加载，greedy decoding 使用发布配置声明的 151,936 个词表位置。Looped-MoE 直接使用作者发布的模型代码和 checkpoint 精度。两者关闭 KV cache，每步重算前缀，采集真实 router 的 top-k。回放使用同一套策略和各模型专家权重的实际字节数。两个 checkpoint 合计约 15 GB，可先在不使用 GPU 的情况下下载。

```bash
export MOE_CONFIG=configs/loop_models_pilot.json
bash scripts/download_models.sh
# 直连 Hugging Face 较慢时可用镜像，仍下载配置所固定的 revision
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 bash scripts/download_models.sh

# 从 nvidia-smi 确认空闲卡后，将示例编号 7 改为该卡的编号或 UUID
GPU_ID=7
CUDA_VISIBLE_DEVICES="$GPU_ID" bash scripts/run_pilot.sh runs/loop-smoke --limit 2 --max-new-tokens 4 --local-files-only
CUDA_VISIBLE_DEVICES="$GPU_ID" bash scripts/run_pilot.sh runs/loop-pilot --local-files-only
```

分析与回放可以脱离 GPU 单独运行，只用 Python 标准库。

```bash
python -m loop_moe.analyze runs/pilot/Foil-1/trace.jsonl.gz --out-dir runs/pilot/Foil-1/analysis
python -m loop_moe.replay runs/pilot/Foil-1/trace.jsonl.gz --out-dir runs/pilot/Foil-1/replay
```

## 输出与基础回放

每个 checkpoint 有自己的输出子目录。

| 文件 | 内容 |
| --- | --- |
| `trace.jsonl.gz` | 首行记录模型、环境、物理专家池和按各模型实际精度计算的权重字节数。之后每行一个请求，含输入/生成 token 和按 forward、展开深度记录的路由。 |
| `analysis/routing.csv` | 按 prefill/decode、类别、物理层和 loop 转移统计 overlap、Jaccard、交集大小分布，以及相邻 loop 的 block 输出 cosine。`marginal_null_overlap` 是保持两边边缘频率后独立重配对的期望。 |
| `analysis/routing_heatmap.html` | 浏览器可直接打开，按物理层和 loop 转移查看 overlap 和 Jaccard 热图。 |
| `analysis/confidence.csv` | 上一轮 gate margin 或归一化熵分桶后，对应下一轮 overlap。margin 切点为 `.01/.025/.05/.1/.2`，熵切点为 `.2/.4/.6/.8/.9`，桶号从 0 开始。 |
| `analysis/requests.csv`、`reuse_distance.csv`、`reuse_lengths.csv` | 逐请求工作集字节数、串行专家访问间隔内不同专家的个数、同 token 同专家连续出现的轮数。 |
| `replay/replay_summary.csv`、`replay_requests.csv` | 同预算下各策略的 miss、重复加载、字节命中率和估计 H2D 字节数，分别提供汇总与逐请求数据。 |
| `replay/replay_config.json` | 记录拟合/评估请求 ID、预算、策略和参数。 |

采集时每生成一步都会重算整个前缀。首个 forward 保留完整 prefill 路由并预测首个输出 token。后续 forward 的 `routes` 只保留新加入前缀的最后一个输入位置，`prefix_expert_token_counts` 另记本次实际计算的完整前缀负载。每条 `routes` 按位置依次存 `[top-k专家, 原始softmax分数, 专家合并权重, margin, 熵, 上一轮同物理层的block输出cosine, top-p专家与分数]`。绝对位置从 `first_position` 开始，`p=2k`。 [Foil 生成与采集接口](https://huggingface.co/ShourenWSR/how-to-loop-moe/raw/b3d128b44cbefd54e2c8784224a110186c8931f1/modeling_loop_lm.py)

回放使用上述 **新位置 decode 路由**，每个请求冷启动，按实际层访问顺序展开，同一层内按专家 ID 串行处理。一个物理专家跨 loop 共用缓存条目。预算为全部专家权重字节数的 5/10/20/40/80/100%，向下取整到可放入的完整专家数。允许在临时计算缓冲中处理 miss 后立即释放，缓存预算只计驻留权重。Belady 在这个等大小、允许绕过缓存的串行模型下给出最少 miss。这里的 H2D 字节数是按 miss 和权重大小推算的流量，端到端速度需要后续单独测量。

策略包括 LRU、LFU、静态热门专家、优先保留上一轮专家的 LRU、OLED 的常数 confidence gate 保留近似、HybriMoE 启发的 top-p 分数 EMA、在相同 EMA 上只增加「本 token 还有下一轮」保留加分的版本，以及 Belady。静态热门专家使用分开的拟合请求，所有策略在同一评估请求集上运行。OLED 近似实现其 gate 保留信号，HybriMoE 近似实现其分数平滑。当前回放只模拟缓存替换。EMA 系数 `0.2`、loop 加分 `0.1` 是试跑配置。Foil-1 与 Foil-3 同时改变了层数、轮数和每层专家数，跨模型比较适合看机制是否一致。 [OLED-MoE §4.3](https://arxiv.org/html/2609.33385v1#S4.SS3) · [HybriMoE §IV](https://arxiv.org/html/2504.05897)

## 本地检查

```bash
python3 -m unittest discover -s tests -v
for script in scripts/*.sh; do bash -n "$script" || break; done
```

在 Mac 上先激活 Python 3.11 环境，再运行合成测试和语法检查。模型小跑在服务器执行。

## 下一次同池访问的驻留实验

[方法设计](docs/loop-aware-expert-residency.md)对应的新入口是 `loop_moe.residency`。它只读已采集轨迹，**不加载模型、不需要 GPU**，使用 Python 3.11 和标准库。原来的 `loop_moe.replay` 保留用于复查历史结果。

### 环境与运行命令

```bash
conda env create -f environment-replay.yml
conda activate loop-moe-replay
python -m unittest discover -s tests -v

# 服务器已有 GPU 环境也可以直接复用，不需要安装额外依赖
cd ~/Loop-MoE
PYTHON=~/loop-moe-env/bin/python bash scripts/run_residency.sh runs/residency-pilot \
  runs/a800-20261117/pilot/Looped-MoE/trace.jsonl.gz \
  runs/a800-20261117/pilot/Loopie-6B/trace.jsonl.gz

# Foil 结构边界检查；请换成自己机器上的轨迹路径
PYTHON=~/loop-moe-env/bin/python bash scripts/run_residency.sh runs/residency-foil \
  runs/pilot_2026-11-13/pilot/Foil-1/trace.jsonl.gz \
  runs/pilot_2026-11-13/pilot/Foil-3/trace.jsonl.gz

# 单独运行一个模型
python -m loop_moe.residency /path/to/trace.jsonl.gz \
  --config configs/residency.json --out-dir runs/residency-single/Looped-MoE
```

默认六档预算为 5/10/20/40/80/100%。`run_residency.sh` 会拒绝覆盖已有完成结果；改变配置时使用新的输出目录。Mac 只运行合成轨迹测试；正式回放建议使用服务器 CPU。需要新轨迹时，继续使用上面的下载、GPU 环境和采集脚本，在运行前重新确认空闲显卡。

### 预测器和执行规则

- 每个请求冷启动。路由历史按物理池和阶段保存，不要求对应专家一直驻留。每次 source 访问只预测下一次同池访问；距离来自固定执行表，包含 Foil head/tail，不读取未来实际 miss 或路由。新采集会直接记录 `execution_order`；旧轨迹只从首步提取池/阶段顺序，并检查所有请求每步结构一致。
- A 是本 token 上一轮，B 是目标阶段最近一次已完成访问。跨 token 首轮时 A 缺失，上一 token 末轮作为单独边界特征。当前 gate 已产生后更新当前池估值，所有方法都在同样的专家 ID 升序下逐个服务；未驻留专家只在实际被路由选中时加载。
- 每次拟合机会包含池内**全部**专家，榜外专家也计入分母。gate 排名分为第 1、其余 top-k、k+1 至 2k、榜外；EMA 分为零、低于 1/N、其余三档。计数概率依次向共享特征、同池特征收缩，稀疏处回退到拟合热度和已观测 gate EMA。默认收缩强度 32，不做自动调参。
- `gate_ema`、`history_a`、`history_b`、`phase_ab` 比较历史增量；`pooled_ab` 合并阶段；`boundary_ab` 只区分跨 token 与内部转移。`generic_lag` 不读阶段标签，记录同池最近 m 次访问的完整布尔模式，m 为一 token 的同池访问次数，和双历史覆盖相同跨度；稀疏模式向“最近一次/更早是否出现”退回。每层统计表统一最多 4096 个 cell，按拟合机会数裁剪，实际 cell 数写入配置，**各级 cell 上限相同，实际使用量可能不同；这不等同于严格匹配已拟合参数数目**。
- 四格主对照为 `phase_ab` / `generic_lag`，分别加或不加 `_distance`。折扣为 `p/(1+d)`，每次访问重新计算尚余的 d；优化 H2D 字节时 H=b，字节因子抵消。没有为当前层或上一轮专家设置硬保留名额。
- `phase_ab_shuffled` 只打乱拟合集中同池同目标阶段的 source→target 配对，整体置换目标 top-k 集合，保留阶段专家边缘频率；评估顺序保持真实顺序，随机种子固定为 20261118。

### 显存口径与基线

默认 `memory_mode=peak`，预算同时容纳驻留专家和正在传输/计算的一个专家。miss 时先腾出槽位再加载，至少要容纳一个专家。`peak_weight_bytes` 不含固定的 attention、激活等非专家开销，也不代表实测 GPU 峰值。等大小专家下，Belady 在完全相同的服务顺序、准入和缓冲规则下运行；穷举测试验证其最优性。

将配置复制到新文件并设 `memory_mode=resident` 可使用旧的“允许额外一个专家临时缓冲、服务后立即绕过缓存”模型，适合与历史分析对照；输出显式记录额外缓冲后的峰值。当前 gate 对所有策略都在服务前可见；旧 `replay` 在服务后才更新分数，因此新实现的分数基线与旧数字不要求完全相同。**跨显存口径不要直接比较命中率。**

除了 LRU/LFU/静态热门/固定 loop 奖励/gate-only/Score EMA，增加 `oled_adaptive` 驻留适配版：当前激活专家归一化 gate 分数、未激活专家置零；用平滑激活比率与缓存利用率的乘积分配层配额，每 5 个完整 decode 位置更新一次，按整数比例分配并限制池大小。空池利用率取中性值 1，便于零配额恢复；EMA 沿用 0.2。这些是单位置回放的明确适配选择，**没有复现 OLED 的完整执行系统或 CPU 补偿**。参考 [OLED-MoE §4.3](https://arxiv.org/html/2609.33385v1#S4.SS3)。

### 输出与解释

| 文件 | 内容 |
| --- | --- |
| `residency_summary.csv` | 各预算/策略的字节命中率、每 decode 位置 H2D、相对最强在线基线的流量变化。最强基线包含传统策略、gate/EMA 预测、合并/边界阶段以及两种通用历史对照，是评估集整体上的回顾性参照，不用于选参；Belady 不参加在线排名。 |
| `residency_requests.csv` | 逐请求流量、命中、重载、专家权重峰值和 OLED 配额更新数。 |
| `residency_positions.csv` | 每个 decode 位置的实际访问数、miss 和推算 H2D。 |
| `residency_phases.csv` | 按物理池、目标阶段、跨 token 首轮/内部转移分开统计，包含冷启动首轮。 |
| `paired_requests.csv` / `paired_summary.csv` | 每个新策略对每个在线基线的额外命中字节、损失命中字节和净节省；逐请求配对差值、样本标准差及改善/退化请求数。增减是同一次访问上的命中差，满足 gained−lost=净节省，不表示唯一的因果淘汰链。 |
| `prediction_requests.csv` | 各 source→target 转移的 top-k 召回。完整物理池候选取目标 k 个；驻留候选统一取 source 时刻 **Score EMA 的同一驻留集合**，固定保留 2 个，另报候选中可达目标数和条件召回。最后一次没有实际后继的 source 不进入指标。 |
| `residency_config.json` | 拟合/验证/评估 ID、执行表、固定参数、显存口径、轨迹与源文件 SHA256、Python 版本、拟合表实际大小及运行时间。 |
| `predictors.json` | 拟合得到的选中次数/机会次数表和专家热度，方便检查分母与稀疏程度。 |

默认沿用旧 pilot 的按类别哈希划分，18 条请求中 6 条拟合、12 条评估；这些评估请求此前已用于确定方向，所以结果只能作为开发对照。最终实验应另采此前未查看的新请求，用显式 ID 划分：

```json
{
  "fit_request_ids": ["fit-01", "fit-02"],
  "validation_request_ids": ["val-01"],
  "evaluation_request_ids": ["test-01", "test-02"]
}
```

三个集合必须互斥并完整覆盖输入轨迹。验证集留出供外部选参，本程序不自动调参，也不把验证请求混入拟合或测试。运行时传 `--split /path/to/split.json`，或给批处理脚本设 `RESIDENCY_SPLIT`；参数选择完成后固定配置再运行测试。回放不测量端到端 TPOT、真实传输重叠或 GPU 调度开销。

完成后可以验证结果收支并生成中文表格报告（只读结果，不重新拟合）：

```bash
python -m loop_moe.residency_report runs/residency-pilot/Looped-MoE \
  runs/residency-pilot/Loopie-6B --output runs/residency-pilot/report.md
```

已完成的四模型开发对照见 [2026-11-18 回放报告](docs/residency-pilot-2026-11-18.md)，本机与 `pku_A800:~/Loop-MoE/runs/residency-20261118-v1/` 均保存原始结果。
