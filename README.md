# Looped MoE 专家路由与缓存实验

第一轮用 Foil 的公开权重采集生成时的专家路由，再离线比较相同显存预算下的缓存策略。默认使用 **Foil-1 和 Foil-3、BF16、batch=1、greedy decoding、最多 128 个新 token**。项目附带 18 条英文试跑 prompt，文本、数学、代码各 6 条。扩大实验时用 `--prompts` 指定自己的 JSONL，可按 research note 每类准备约 64 条。

## 1. 租用单卡并传代码

以下为两台校内服务器的只读查询结果，适合对照租用实例的实际 GPU 型号和驱动。

| SSH 别名 | GPU | `nvidia-smi` 显存 | Compute capability | 驱动 |
| --- | --- | ---: | --- | --- |
| `pku_A800` | NVIDIA A800-SXM4-80GB | 81920 MiB | 8.0 | 580.126.20 |
| `pku_pro5000` | NVIDIA RTX PRO 5000 72GB Blackwell | 73415 MiB | 12.0 | 580.126.20 |

本实验用一张 GPU。租用时选 Linux x86_64、带 Conda 的镜像，查看 `nvidia-smi`。推荐驱动 `580.95.05` 或更新，用 `cu130` 安装 PyTorch。A800 若只有 CUDA 12.x 驱动支持且兼容 CUDA 12.6，可改用 `cu126`。Blackwell 的原生支持从 CUDA 12.8 开始，本项目在 Blackwell 上用 `cu130`。预编译的 PyTorch 安装包已包含所需的 CUDA 用户态库，安装脚本无需 `sudo` 或 `nvcc`。 [Foil 作者环境](https://github.com/SR-A-W/how-to-loop-moe#installation) · [NVIDIA 架构与 CUDA 对照表](https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html) · [NVIDIA CUDA 13.0 驱动对照表](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-toolkit-release-notes/index.html)

代码可从 Git 仓库拉取。若改动仍在本地工作区，从 Mac 的仓库根目录直接传到**租用实例**。将 `user@rented-host` 换成平台提供的 SSH 地址或本机 SSH 别名。

```bash
GPU_HOST=user@rented-host
rsync -av --exclude='__pycache__/' README.md requirements-gpu.txt configs loop_moe scripts "$GPU_HOST:loop-moe/"
ssh "$GPU_HOST"
```

如果平台分配了单独的 SSH 端口，`rsync` 加 `-e 'ssh -p 端口'`，`ssh` 加 `-p 端口`。

## 2. 在租用实例配置环境

```bash
cd ~/loop-moe
nvidia-smi
conda create -n loop-moe python=3.11 -y
conda activate loop-moe
bash scripts/setup_gpu.sh cu130
```

脚本安装 `torch==2.13.0` 对应的 CUDA 构建、`transformers==5.15.0` 和 `einops==0.8.2`。模型代码与权重固定在 `configs/foil_pilot.json` 所列的 Hugging Face revision，首次运行自动下载所选模型的文件。两个模型的权重各约 1.1 GB，另需为 PyTorch 安装和缓存留空间。若系统盘空间有限，先将 `HF_HOME` 指到实例的可写数据盘。模型由发布者提供的自定义 Python 实现加载。 [Foil 模型卡](https://huggingface.co/ShourenWSR/how-to-loop-moe)

## 3. 先小跑，再跑试验

以下命令使用第一个可见 GPU。使用有作业调度器的实例时，保留调度器分配的 `CUDA_VISIBLE_DEVICES`。

```bash
# 两条 prompt、最多生成 4 token，验证下载、加载、GPU 算子、导出和回放
bash scripts/run_pilot.sh runs/smoke --model Foil-1 --limit 2 --max-new-tokens 4

# 两个模型，使用内置的 18 条 prompt
bash scripts/run_pilot.sh runs/pilot

# 换成自己的输入，每行需要 id、category、text 三个非空字段，id 必须唯一
bash scripts/run_pilot.sh runs/expanded --prompts /path/to/prompts.jsonl
```

输入格式示例见 `configs/prompts_pilot.jsonl`。可以重复传 `--model` 选择多个模型。模型已下载时可加 `--local-files-only`。采集完成的文件会保留。若第二个模型中途中断，用同一个输出目录和 `--model Foil-3` 只补跑第二个模型。使用自定义配置时给 `FOIL_CONFIG` 指定配置文件。

分析与回放可以脱离 GPU 单独运行，只用 Python 标准库。

```bash
python -m loop_moe.analyze runs/pilot/Foil-1/trace.jsonl.gz --out-dir runs/pilot/Foil-1/analysis
python -m loop_moe.replay runs/pilot/Foil-1/trace.jsonl.gz --out-dir runs/pilot/Foil-1/replay
```

## 4. 看结果

每个 checkpoint 有自己的输出子目录。

| 文件 | 内容 |
| --- | --- |
| `trace.jsonl.gz` | 首行记录模型、环境、物理专家池和真实 BF16 权重字节数。之后每行一个请求，含输入/生成 token 和按 forward、展开深度记录的路由。 |
| `analysis/routing.csv` | 按 prefill/decode、类别、物理层和 loop 转移统计 overlap、Jaccard、交集大小分布，以及相邻 loop 的 block 输出 cosine。`marginal_null_overlap` 是保持两边边缘频率后独立重配对的期望。 |
| `analysis/routing_heatmap.html` | 浏览器可直接打开，按物理层和 loop 转移查看 overlap 和 Jaccard 热图。 |
| `analysis/confidence.csv` | 上一轮 gate margin 或归一化熵分桶后，对应下一轮 overlap。margin 切点为 `.01/.025/.05/.1/.2`，熵切点为 `.2/.4/.6/.8/.9`，桶号从 0 开始。 |
| `analysis/requests.csv`、`reuse_distance.csv`、`reuse_lengths.csv` | 逐请求工作集字节数、串行专家访问间隔内不同专家的个数、同 token 同专家连续出现的轮数。 |
| `replay/replay_summary.csv`、`replay_requests.csv` | 同预算下各策略的 miss、重复加载、字节命中率和估计 H2D 字节数，分别提供汇总与逐请求数据。 |
| `replay/replay_config.json` | 记录拟合/评估请求 ID、预算、策略和参数。 |

Foil 发布版每生成一步都会重算整个前缀。首个 forward 保留完整 prefill 路由并预测首个输出 token。后续 forward 的 `routes` 只保留新加入前缀的最后一个输入位置，`prefix_expert_token_counts` 另记本次实际计算的完整前缀负载。每条 `routes` 按位置依次存 `[top-k专家, 原始softmax分数, 专家合并权重, margin, 熵, 上一轮同物理层的block输出cosine, top-p专家与分数]`。绝对位置从 `first_position` 开始，`p=2k`。 [Foil 生成与采集接口](https://huggingface.co/ShourenWSR/how-to-loop-moe/raw/b3d128b44cbefd54e2c8784224a110186c8931f1/modeling_loop_lm.py)

回放使用上述 **新位置 decode 路由**，每个请求冷启动，按实际层访问顺序展开，同一层内按专家 ID 串行处理。一个物理专家跨 loop 共用缓存条目。预算为全部专家权重字节数的 5/10/20/40/80/100%，向下取整到可放入的完整专家数。允许在临时计算缓冲中处理 miss 后立即释放，缓存预算只计驻留权重。Belady 在这个等大小、允许绕过缓存的串行模型下给出最少 miss。这里的 H2D 字节数是按 miss 和权重大小推算的流量，端到端速度需要后续单独测量。

策略包括 LRU、LFU、静态热门专家、优先保留上一轮专家的 LRU、OLED 的常数 confidence gate 保留近似、HybriMoE 启发的 top-p 分数 EMA、在相同 EMA 上只增加「本 token 还有下一轮」保留加分的版本，以及 Belady。静态热门专家使用分开的拟合请求，所有策略在同一评估请求集上运行。OLED 近似实现其 gate 保留信号，HybriMoE 近似实现其分数平滑。当前回放只模拟缓存替换。EMA 系数 `0.2`、loop 加分 `0.1` 是试跑配置。Foil-1 与 Foil-3 同时改变了层数、轮数和每层专家数，跨模型比较适合看机制是否一致。 [OLED-MoE §4.3](https://arxiv.org/html/2609.33385v1#S4.SS3) · [HybriMoE §IV](https://arxiv.org/html/2504.05897)

## 本地检查

```bash
python3 -m unittest discover -s tests -v
bash -n scripts/setup_gpu.sh
bash -n scripts/run_pilot.sh
```

这些检查只读合成轨迹和脚本。第一次真实生成由租用实例上的 `runs/smoke` 验证。
