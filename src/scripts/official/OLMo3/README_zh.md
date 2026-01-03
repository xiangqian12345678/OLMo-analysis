# Olmo 3 模型训练

我们推出了 Olmo 3，这是一个包含 7B 和 32B 模型的新系列。该系列包括 Base、Instruct 和 Think 变体。基础模型采用分段训练方法进行训练。

Olmo 是一系列**O**pen **l**anguage **mo**dels（开源语言模型），旨在促进语言模型的科学研究。这些模型在 Dolma 3 数据集上进行训练。我们发布了所有代码、检查点、日志（即将推出）以及相关的训练细节。

| 大小 | 训练令牌数 | 层数 | 隐藏层大小 | Q 头 | KV 头 | 上下文长度 |
|--------|-----------------|--------|-------------|---------|----------|----------------|
| [OLMo 3 7B](https://huggingface.co/allenai/Olmo-3-1025-7B) | 5.93 万亿 | 32 | 4096 | 32 | 32 | 65,536 |
| [OLMo 3 32B](https://huggingface.co/allenai/Olmo-3-1125-32B) | 5.50 万亿 | 64 | 5120 | 40 | 8 | 65,536 |

本批次发布的核心模型包括以下内容：

| 阶段 | [Olmo 3 7B Think] | [Olmo 3 32B Think] | [Olmo 3 7B Instruct] | [Olmo 3 32B Instruct] |
|-------|-------------------|--------------------|----------------------|-----------------------|
| 基础模型 | [Olmo-3-7B](https://huggingface.co/allenai/Olmo-3-1025-7B) | [Olmo-3-32B](https://huggingface.co/allenai/Olmo-3-1125-32B) |  |  |
| SFT | [Olmo-3-7B-Think-SFT](https://huggingface.co/allenai/Olmo-3-7B-Think-SFT) | [Olmo-3-32B-Think-SFT](https://huggingface.co/allenai/Olmo-3-32B-Think-SFT) | [Olmo-3-7B-Instruct-SFT](https://huggingface.co/allenai/Olmo-3-7B-Instruct-SFT) | [Olmo-3-32B-Instruct-SFT](https://huggingface.co/allenai/Olmo-3-32B-Instruct-SFT) |
| DPO | [Olmo-3-7B-Think-DPO](https://huggingface.co/allenai/Olmo-3-7B-Think-DPO) | [Olmo-3-32B-Think-DPO](https://huggingface.co/allenai/Olmo-3-32B-Think-DPO) | [Olmo-3-7B-Instruct-DPO](https://huggingface.co/allenai/Olmo-3-7B-Instruct-DPO) | [Olmo-3-32B-Instruct-DPO](https://huggingface.co/allenai/Olmo-3-32B-Instruct-DPO) |
| 最终模型 (RLVR) | [Olmo-3-7B-Think](https://huggingface.co/allenai/Olmo-3-7B-Think) | [Olmo-3-32B-Think](https://huggingface.co/allenai/Olmo-3-32B-Think) | [Olmo-3-7B-Instruct](https://huggingface.co/allenai/Olmo-3-7B-Instruct) | [Olmo-3-32B-Instruct](https://huggingface.co/allenai/Olmo-3-32B-Instruct) |

## 训练数据

Olmo 3 7B 预训练采用三阶段流程。
在第一阶段，我们在大量基于网络的数据上进行训练：[dolma3](https://huggingface.co/datasets/allenai/dolma3)。
在第二阶段，我们在较小量的高质量、针对性数据上进行训练：[dolma3-dolmino](https://huggingface.co/datasets/allenai/dolma3_dolmino)。
在第三阶段，我们在包含部分长文档的高质量数据上进行训练：[dolma3-longmino](https://huggingface.co/datasets/allenai/dolma3_longmino)。

更多详细信息请参阅 [dolma3](https://github.com/allenai/dolma3) 仓库。

使用 [allenai/dolma3-tokenizer](https://huggingface.co/allenai/dolma2-tokenizer)（与 `allenai/dolma2-tokenizer` 相同）进行预分词的数据集版本可在 https://olmo-data.org/ 上获取，manifest 在以下 mix 文件中定义：

| 模型 | 阶段 | 数据 Mix |
|-------|-------|-----|
| Olmo 3 7B | 阶段 1 (预训练) | [OLMo-mix-0625-official.txt](https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/data/mixes/OLMo-mix-0625-official.txt) |
| Olmo 3 7B | 阶段 2 (中期训练) | [OLMo-midtraining-mix-0625-100B.txt](https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/data/mixes/OLMo-midtraining-mix-0625-100B.txt) |
| Olmo 3 7B | 阶段 3 (长上下文) | [OLMo-longmino-mix-0625.txt](https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/data/mixes/OLMo-longmino-mix-0625.txt) |
| Olmo 3 32B | 阶段 1 (预训练) | dolma3 -> [OLMo-mix-0925-official.txt](https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/data/mixes/OLMo-mix-0925-official.txt) |
| Olmo 3 32B | 阶段 2 (中期训练) | dolma3-dolmino -> [OLMo-midtraining-mix-0925-ingredient1-100B.txt](https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/data/mixes/OLMo-midtraining-mix-0925-ingredient1-100B.txt) <br> [OLMo-midtraining-mix-0925-ingredient2-100B.txt](https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/data/mixes/OLMo-midtraining-mix-0925-ingredient2-100B.txt) |
| Olmo 3 32B | 阶段 3 (长上下文) | dolma3-longmino -> [OLMo-longmino-mix-0925.txt](https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/data/mixes/OLMo-longmino-mix-0925.txt) |

总体而言，我们推荐使用为 Olmo 3 32B 定义的 mix，因为它们稍微更加精细。

例如，可以通过以下命令获取包含分词数据的 numpy 文件：

```bash
wget https://olmo-data.org/preprocessed/dolma3-0625/v0.1-official/allenai/dolma3-tokenizer/olmocr_science_pdfs/science_math_and_technology/000000.npy
```

## Olmo 3 7B 模型训练

Olmo 3 7B 预训练过程的官方训练脚本、检查点和监控日志可在下表中找到。

| 阶段 | 令牌数  | GPU | 脚本 | 监控 |
|-------|-----------|------|--------|------------|
| 阶段 1 (预训练) | 5.93 万亿 | 512 H100s | [OLMo-3-1025-7B-pretrain-1.py](https://github.com/allenai/OLMo-core/blob/main/src/scripts/official/OLMo3/OLMo-3-1025-7B-pretrain-1.py) <br> [OLMo-3-1025-7B-pretrain-2.py](https://github.com/allenai/OLMo-core/blob/main/src/scripts/official/OLMo3/OLMo-3-1025-7B-pretrain-2.py) | [wandb.ai/Olmo3-7B](https://wandb.ai/ai2-llm/Olmo-3-1025-7B/reports/Olmo-3-7B-October-2025--VmlldzoxNDcwOTM0NA) |
| 阶段 2 (中期训练) | 1000 亿 | 128 H100s | [OLMo-3-1025-7B-midtrain.py](https://github.com/allenai/OLMo-core/blob/main/src/scripts/official/OLMo3/OLMo-3-1025-7B-midtrain.py) | [wandb.ai/Olmo3-7B](https://wandb.ai/ai2-llm/Olmo-3-1025-7B/reports/Olmo-3-7B-October-2025--VmlldzoxNDcwOTM0NA) |
| 阶段 3 (长上下文) | 500 亿 | 256 H100s | [OLMo-3-1025-7B-long-context.py](https://github.com/allenai/OLMo-core/blob/main/src/scripts/official/OLMo3/OLMo-3-1025-7B-long-context.py) | [wandb.ai/Olmo3-7B](https://wandb.ai/ai2-llm/Olmo-3-1025-7B/reports/Olmo-3-7B-October-2025--VmlldzoxNDcwOTM0NA) |

Olmo 3 7B 的 Olmo-core 格式检查点完整列表可在 [OLMo-3-1025-7B.csv](https://github.com/allenai/OLMo-core/blob/main/src/scripts/official/OLMo3/OLMo-3-1025-7B.csv) 中找到。

可以通过枚举 HF 仓库引用找到 Olmo 3 32B 的 HF 格式检查点完整列表：

```python
from huggingface_hub import list_repo_refs
out = list_repo_refs("allenai/Olmo-3-1025-7B")
branches = [b.name for b in out.branches]
```

## Olmo 3 32B 模型训练

Olmo 3 32B 预训练过程的官方训练脚本、检查点和监控日志可在下表中找到。与 Olmo 3 7B 不同，我们在预训练过程中的多个点使用模型合并（"souping"）。具体而言，我们对两个独立的中期训练运行的输出进行"soup"（简单参数平均），并对长上下文阶段产生的最后三个检查点进行"soup"。

| 阶段 | 令牌数  | GPU | 脚本 | 监控 |
|-------|-----------|------|--------|------------|
| 阶段 1 (预训练) | 5.50 万亿 | 1024 H100s | [OLMo-3-1025-32B-pretrain.py](https://github.com/allenai/OLMo-core/blob/main/src/scripts/official/OLMo3/OLMo-3-1025-32B-pretrain.py) | [wandb.ai/Olmo3-32B](https://wandb.ai/ai2-llm/Olmo-3-1125-32B/reports/Olmo-3-32B-November-2025--VmlldzoxNTA4NzAxMw) |
| 阶段 2 (中期训练) | 1000 亿 x2 | 512 H100s | [OLMo-3-1025-32B-midtrain-ingredient-1.py](https://github.com/allenai/OLMo-core/blob/main/src/scripts/official/OLMo3/OLMo-3-1025-32B-midtrain-ingredient-1.py) <br> [OLMo-3-1025-32B-midtrain-ingredient-2.py](https://github.com/allenai/OLMo-core/blob/main/src/scripts/official/OLMo3/OLMo-3-1025-32B-midtrain-ingredient-2.py) | [wandb.ai/Olmo3-32B](https://wandb.ai/ai2-llm/Olmo-3-1125-32B/reports/Olmo-3-32B-November-2025--VmlldzoxNTA4NzAxMw) |
| 阶段 3 (长上下文) | 1000 亿 | 1024 H100s | [OLMo-3-1025-32B-long-context.py](https://github.com/allenai/OLMo-core/blob/main/src/scripts/official/OLMo3/OLMo-3-1025-32B-long-context.py) | [wandb.ai/Olmo3-32B](https://wandb.ai/ai2-llm/Olmo-3-1125-32B/reports/Olmo-3-32B-November-2025--VmlldzoxNTA4NzAxMw) |

Olmo 3 32B 的 Olmo-core 格式检查点完整列表可在 [OLMo-3-1025-32B.csv](https://github.com/allenai/OLMo-core/blob/main/src/scripts/official/OLMo3/OLMo-3-1025-32B.csv) 中找到。

可以通过枚举 HF 仓库引用找到 Olmo 3 32B 的 HF 格式检查点完整列表：

```python
from huggingface_hub import list_repo_refs
out = list_repo_refs("allenai/Olmo-3-1125-32B")
branches = [b.name for b in out.branches]
```
