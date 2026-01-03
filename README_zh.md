<div align="center">
  <!-- <img src="https://github.com/allenai/OLMo/assets/8812459/774ac485-a535-4768-8f7c-db7be20f5cc3" width="300"/> -->
  <img src="https://huggingface.co/datasets/allenai/blog-images/resolve/main/olmo2/olmo.png" alt="OLMo Logo" width="280" style="margin-left:'auto' margin-right:'auto' display:'block'"/>
  <br>
  <h1>OLMo-core</h1>
  <h4>OLMo 建模和训练的构建模块</h4>
</div>
<p align="center">
  <a href="https://olmo-core.readthedocs.io/en/latest/">
    <img alt="Docs" src="https://img.shields.io/badge/API-docs-red">
  </a>
  <a href="https://github.com/allenai/OLMo-core/tree/main/src/examples">
    <img alt="Examples" src="https://img.shields.io/badge/API-examples-994B00">
  </a>
  <a href="https://github.com/allenai/OLMo-core/releases/tag/v1.9.0">
    <img alt="Pypi" src="https://img.shields.io/pypi/v/ai2-olmo-core.svg">
  </a>
  <a href="https://github.com/allenai/OLMo-core/blob/main/LICENSE">
    <img alt="GitHub License" src="https://img.shields.io/github/license/allenai/OLMo">
  </a>
  <a href="https://arxiv.org/pdf/2501.00656.pdf">
    <img alt="Paper URL" src="https://img.shields.io/badge/arxiv-2402.00838-orange">
  </a>
  <a href="https://playground.allenai.org">
    <img alt="Playground" src="https://img.shields.io/badge/Ai2-Playground-F0529C">
  </a>
  <a href="https://discord.gg/sZq3jTNVNG">
    <img alt="Discord" src="https://img.shields.io/badge/Discord%20-%20blue?style=flat&logo=discord&label=Ai2&color=%235B65E9">
  </a>
</p>

## 安装

首先根据您的操作系统和硬件的具体说明安装 [PyTorch](https://pytorch.org)。

对于开发，我们建议从源代码安装：

```bash
git clone https://github.com/allenai/OLMo-core.git
cd OLMo-core
pip install -e .[all]
```
或者您可以从 PyPI 安装：

```bash
pip install ai2-olmo-core
```

要使用某些功能，还需要安装许多可选依赖项，包括：

- [flash-attn](https://github.com/Dao-AILab/flash-attention)、[ring-flash-attn](https://github.com/zhuzilin/ring-flash-attention) 和 [TransformerEngine](https://github.com/NVIDIA/TransformerEngine)，用于相应的注意力后端。
- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel)，用于低内存"融合线性"损失实现。
- [torchao](https://github.com/pytorch/ao)，用于 float8 训练。
- [grouped_gemm](https://github.com/tgale96/grouped_gemm)，用于无丢弃混合专家（MoE）模型。在 [PR #21](https://github.com/tgale96/grouped_gemm/pull/21) 发布之前（v0.1.6 之后），您可能需要从源代码编译。

发布的 [Docker 镜像](https://github.com/orgs/allenai/packages?repo_name=OLMo-core)包含所有核心和可选依赖项，并定期在我们的内部 H100 集群上进行测试。
但如果您打算使用这些镜像，有几件事需要记住：

- 它们不附带安装的 OLMo-core 包，仅包含其依赖项，以适应常规代码更改。
- 如果您的硬件或驱动程序/CUDA 版本不同，它们可能无法在您自己的集群上运行。

如果由于上述任何原因发布的镜像不适合您的用例，您可以调整我们的 [Dockerfile](https://github.com/allenai/OLMo-core/blob/main/src/Dockerfile) 来构建您自己的镜像。

## 官方训练脚本

已发布模型的官方训练脚本可以在 [`src/scripts/official/`](https://github.com/allenai/OLMo-core/tree/main/src/scripts/official) 中找到。

这些脚本旨在使用 ``torchrun`` 启动，或者如果您有权访问 Beaker，则使用 OLMo-core 的 Beaker 启动 CLI。

例如：

```bash
torchrun --nproc-per-node=8 src/scripts/official/OLMo2/OLMo-2-0325-32B-train.py \
  --save-folder=/path/to/save/checkpoints
```

您可以从命令行覆盖大多数配置选项。例如，要覆盖学习率，您可以这样启动脚本：

```bash
torchrun --nproc-per-node=8 src/scripts/official/OLMo2/OLMo-2-0325-32B-train.py \
  --save-folder=/path/to/save/checkpoints \
  --train_module.optim.lr=6e-3
```

要从检查点继续退火，我们使用一个单独的脚本，可以像这样启动：

```bash
torchrun --nproc-per-node=8 src/scripts/official/OLMo2/OLMo-2-0325-32B-anneal.py \
  --save-folder=/path/to/save/checkpoints \
  --checkpoint=https://olmo-checkpoints.org/ai2-llm/peteish32/step721901
```

### 可用的训练脚本

| 模型系列 | 目录 | 描述 |
|--------------|-----------|-------------|
| **OLMo-2** | [`src/scripts/official/OLMo2/`](https://github.com/allenai/OLMo-core/tree/main/src/scripts/official/OLMo2) | OLMo-2 32B 模型的训练脚本和模型卡 |
| **OLMo-3** | [`src/scripts/official/OLMo3/`](https://github.com/allenai/OLMo-core/tree/main/src/scripts/official/OLMo3) | OLMo-3 7B 和 32B 模型的训练脚本和模型卡 |

## 推理

### 使用 Hugging Face Transformers

您可以使用我们的 Hugging Face [transformers](https://github.com/huggingface/transformers) 集成在 OLMo 检查点上运行推理：

```bash
pip install transformers>=4.57.0
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
olmo = AutoModelForCausalLM.from_pretrained("allenai/Olmo-3-1125-32B")
tokenizer = AutoTokenizer.from_pretrained("allenai/Olmo-3-1125-32B")
message = ["Language modeling is "]
inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
# inputs = {k: v.to('cuda') for k,v in inputs.items()} # 可选验证 cuda
# olmo = olmo.to('cuda')
response = olmo.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=1.0, top_p=0.7)
print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])
```

或者，使用 Hugging Face pipeline 抽象：

```python
from transformers import pipeline
olmo_pipe = pipeline("text-generation", model="allenai/Olmo-3-1125-32B")
print(olmo_pipe("Language modeling is"))
```

### 使用 vLLM

[vLLM](https://docs.vllm.ai/en/latest/) 为 OLMo 模型提供高吞吐量推理。您可以使用它进行离线批量推理：

```bash
pip install vllm>=0.11.0
```

```python
from vllm import LLM, SamplingParams
llm = LLM(model="allenai/Olmo-3-1125-32B")
sampling_params = SamplingParams(temperature=1.0, top_p=0.7)
prompts = ["Language modeling is"]
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

有关更多详细信息，请参阅 [vLLM 文档](https://docs.vllm.ai/en/latest/getting_started/quickstart/#offline-batched-inference)。

### 使用 Olmo-core（测试版）

Olmo-core 中直接支持自回归生成。使用此功能，我们提供了一个聊天循环演示，可用于在交互式聊天会话中与模型交互：

```bash
python -m olmo_core.generate.chat https://olmo-checkpoints.org/ai2-llm/Olmo-3-1025-7B/stage3/step11921/ --max-new-tokens 512
```

## 评估

用于评估 OLMo 模型的其他工具可在 [OLMo Eval](https://github.com/allenai/OLMo-eval) 和 [olmes](https://github.com/allenai/olmes) 存储库中找到。

## 开发

Python 库源代码位于 `src/olmo_core`。相应的测试位于 `src/test`。库文档位于 `docs`。您可以使用 `make docs` 在本地构建文档。

代码检查：

- 我们使用 `pytest` 运行测试。您可以使用 `pytest -v src/test` 运行所有测试。您还可以将 `pytest` 指向特定的测试文件以单独运行它。
- 我们使用 `isort` 和 `black` 进行代码格式化。理想情况下，您应该将这些集成到您的编辑器中，但您也可以手动运行它们或使用 pre-commit hook 进行配置。要验证所有文件格式正确，请运行 `make style-check`。
- 我们使用 `ruff` 作为主要的 linter。您可以使用 `make lint-check` 运行它。
- 我们使用 `mypy` 作为类型检查器。您可以使用 `make type-check` 运行它。

## 引用

```bibtex
@misc{olmo20242olmo2furious,
      title={{2 OLMo 2 Furious}},
      author={{Team OLMo} and Pete Walsh and Luca Soldaini and Dirk Groeneveld and Kyle Lo and Shane Arora and Akshita Bhagia and Yuling Gu and Shengyi Huang and Matt Jordan and Nathan Lambert and Dustin Schwenk and Oyvind Tafjord and Taira Anderson and David Atkinson and Faeze Brahman and Christopher Clark and Pradeep Dasigi and Nouha Dziri and Michal Guerquin and Hamish Ivison and Pang Wei Koh and Jiacheng Liu and Saumya Malik and William Merrill and Lester James V. Miranda and Jacob Morrison and Tyler Murray and Crystal Nam and Valentina Pyatkin and Aman Rangapur and Michael Schmitz and Sam Skjonsberg and David Wadden and Christopher Wilhelm and Michael Wilson and Luke Zettlemoyer and Ali Farhadi and Noah A. Smith and Hannaneh Hajishirzi},
      year={2024},
      eprint={2501.00656},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.00656},
}
```
