# 2026-11-19 四模型扩样运行记录

最终批次在 AutoDL 单张 A800 80GB 上采集，32 个分片迁至 `pku_A800`，统计和回放于北京时间 11 月 20 日 05:56:45 完成。复现实验使用 [README 的运行命令](../README.md#复现四模型扩样)，研究结论与工作负载限制见[扩样结果](expanded-results-2026-11-20.md)。

## 输入与采集配置

| 项目 | 实际设置 |
| --- | --- |
| 模型 | Foil-1、Foil-3、Loopie-6B、Looped-MoE，固定 revision 与旧试跑一致 |
| 输入 | 每模型 192 条，文本、数学、代码各 64 条，四模型共用输入 |
| 解码 | batch=1、greedy、每步重算完整前缀，最多 512 个新 token，保留 EOS |
| 精度 | Foil 和 Loopie 使用 BF16，Looped-MoE 使用 FP32 |
| 分批 | 每模型 8 批，每批 24 条，四模型各用一个采集进程 |
| 回放 | 5/10/20/40/80/100% 专家权重预算，沿用旧策略参数 |
| 划分 | 合并后按类别及请求哈希划分，每类 21 条拟合、43 条评估，共 63/129 |

输入为 [prompts_expanded.jsonl](../configs/prompts_expanded.jsonl)，SHA256 为 `bab6fd9ec1938dd61323b08c4c0c3540c77e59849ae4cc04913dd07d05822b83`，与旧 18 条试跑提示词没有完全相同的文本。[输入来源记录](../configs/prompts_expanded.sources.json)列出 Dolly、GSM8K test、sanitized MBPP test 的上游 revision、记录 ID、抽样规则和许可。

| 模型 | 输入 token 范围 | 输入加 512 的最大长度 | 上下文上限 |
| --- | ---: | ---: | ---: |
| Foil-1 | 8–171 | 683 | 4,096 |
| Foil-3 | 8–171 | 683 | 4,096 |
| Loopie-6B | 15–176 | 688 | 131,072 |
| Looped-MoE | 8–168 | 680 | 1,024 |

## 采集环境与完整性检查

| 项目 | 实际路径或版本 |
| --- | --- |
| AutoDL 代码 | `/root/autodl-tmp/Loop-MoE` |
| Conda 环境 | `/root/miniconda3/envs/loop-moe` |
| 权重缓存 | `/root/autodl-tmp/huggingface` |
| 基础提交 | `8802f3d`，实际运行代码保存在归档快照 |
| 依赖 | Python 3.11.18、PyTorch 2.13.0+cu130、Transformers 5.15.0、einops 0.8.2 |
| CUDA 原生组件 | Triton 3.7.1、CUDA Python bindings 13.4.4，其余见[环境约束](../configs/native-cu130-validated.txt) |

AutoDL 使用 Git bundle 传入代码，通过镜像断点续传权重，少量 tokenizer 和配置文件由校内服务器同一 revision 的缓存补齐。六个权重文件合计 **17,352,518,912 字节**。LFS 文件校验 SHA256，普通文件校验 Git blob SHA1，同时检查 safetensors 头部、张量偏移、分片索引和 JSON。

实例的 `runs/setup/` 保存 `requirements-installed.txt`、`verified-models.json`、`model-imports.json` 和 `setup-ready.json`。四类模型通过离线导入，`pip check` 与 22 项合成测试通过。上下文预检在 `runs/autodl-a800-4models/preflight.json`。

当天先在共享 `pku_A800` 的 5、7 号卡试采。北京时间 10:29 检测到其他计算进程后，驱动停止自己的采集。当时未发布完整分片，部分输出保留在 `runs/expanded-20261119-v1/`，最终报告使用随后在 AutoDL 完成的批次。

## 归档与 CPU 后处理

11 月 19 日 18:34 完成接收校验，18:35 启动后处理。32 个压缩分片合计 **2,319,268,999 字节**。整包共 124 个文件，包含分片、输入、配置、代码、日志和环境记录。归档大小 **2,319,953,920 字节**，整包和每个原始文件的 SHA256 均已验证。整包 SHA256 为 `72406e3fbfee512e43922931fa521e12afa7adfa06e3f26d3d60cd47ed4a7676`。

归档根目录位于 `pku_A800:~/Loop-MoE/runs/autodl-20261119-handoff/`。

| 相对路径 | 内容 |
| --- | --- |
| `transfer.tar`、`archive.sha256` | 原始归档与校验和 |
| `verification-receipt.json` | 接收和完整性校验凭证 |
| `snapshot/` | 采集时的代码、输入与续跑目录 |
| `snapshot/runs/autodl-a800-4models/` | 原始分片、合并轨迹、统计与两套回放结果 |

在 `pku_A800` 上激活原 Python 3.11 环境后，按需查看状态、校验分片或补做 CPU 后处理。

```bash
cd ~/Loop-MoE/runs/autodl-20261119-handoff/snapshot
OUT_DIR=runs/autodl-a800-4models
cat "$OUT_DIR/status.json"
tail -n 30 "$OUT_DIR/logs/cpu-driver.log"
python scripts/resume_postprocess.py --out-dir "$OUT_DIR" --check-only
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  nice -n 10 python -u scripts/resume_postprocess.py --out-dir "$OUT_DIR"
```

续跑脚本在 CPU 上执行合并、路由统计和回放，复用已完成阶段。各模型的阶段日志位于 `logs/postprocess-<模型>.log`。启动后的状态文件会随运行更新，复核接收时的原始文件可读取 `transfer.tar`。本批使用校内服务器完成后处理，AutoDL 无卡模式当时只提供 0.5 核和 2 GiB 内存。
