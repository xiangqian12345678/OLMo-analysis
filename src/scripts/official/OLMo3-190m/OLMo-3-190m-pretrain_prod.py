"""
Official pre-training script for OLMo-3-190m.

Training configuration for the 190M parameter model.
"""

import argparse
from typing import List

from olmo_core.config import DType
from olmo_core.data import (
    DataMix,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.script_utils import ExperimentConfig, main
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    LMEvaluatorCallbackConfig,
    MonkeyPatcherCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

# 模型的上下文窗口大小，即每次输入的最大 token 数量
DEFAULT_SEQUENCE_LENGTH = 2048
'''
所有 GPU 并行处理的全局批次大小，以 token 数为单位 ~262K tokens 
为什么选择这个值：
    与小模型规模匹配：
    训练 tokens 总目标：50B（见 max_duration=Duration.tokens(int(50e9)) ）
    每步更新：262K tokens
    估计总步数：(50B / 262K ≈ 190,000) 步
分布式训练设计：
    rank_microbatch_size = 2 * 4096  # 8192 tokens/每GPU
    # 需要的 GPU 数 = GLOBAL_BATCH_SIZE / rank_microbatch_size
    # = 262,144 / 8,192 ≈ 32 GPUs
Chinchilla 缩放定律：对于小模型，较大的批次有助于训练稳定性，但需权衡收敛速度
'''
GLOBAL_BATCH_SIZE = 512 * 512
'''
优化器的初始学习率
为什么小模型使用更高学习率?
理论依据：
1.参数量级差异：
    小模型（190M）：参数较少，梯度方差相对较大
    大模型（7B+）：参数众多，单个参数的梯度更新应更谨慎
2.优化器动量：
    betas=(0.9, 0.95)  # AdamW 动量参数
    较强的动量（beta2=0.95）配合更高学习率，能加快小模型收敛
3.与模型深度的关系：
    小模型层数少（通常 < 24 层），梯度消失/爆炸风险较低
    可承受更大的学习率步长
'''
LR = 5e-4


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    """
    构建 OLMo-3 190M 模型的完整训练 ExperimentConfig。

    该配置函数负责：
    - 模型结构与注意力后端
    - 数据集与数据加载策略
    - 优化器 / 学习率调度
    - 并行训练策略（HSDP）
    - 训练循环、日志、评估与 checkpoint

    Args:
        opts: 命令行参数（路径、实验名、序列长度等）
        overrides: 通过 CLI 传入的配置覆盖项（key=value）

    Returns:
        ExperimentConfig: 可直接用于 Trainer 启动训练的完整配置
    """

    # =========================
    # 基础超参 & 分词器配置
    # =========================

    # 若命令行未指定 sequence_length，则使用默认上下文长度
    sequence_length = opts.sequence_length or DEFAULT_SEQUENCE_LENGTH

    # 使用 Dolma2 官方分词器配置（与 OLMo 预训练语料严格对齐）
    tokenizer_config = TokenizerConfig.dolma2()

    # =========================
    # 模型配置
    # =========================

    model_config = TransformerConfig.olmo3_190M(
        # 词表大小：通常会 padding 到 128 的倍数，利于 Tensor Core / FlashAttention
        vocab_size=tokenizer_config.padded_vocab_size(),

        # 使用 FlashAttention-2：
        # - 显著降低显存占用
        # - 提升长序列 attention 的吞吐
        attn_backend=AttentionBackendName.flash_2,
    )

    # =========================
    # 数据集配置
    # =========================

    # 测试环境：使用较小的验证数据集（避免大内存占用）
    # 生产环境：改回 DataMix.OLMo_mix_0625_official
    dataset_config = NumpyFSLDatasetConfig.from_data_mix(
        # 使用小规模验证集进行测试（避免 28 亿 tokens 导致的内存问题）
        # DataMix.OLMo_mix_0625_official,  # 生产环境使用此行
        DataMix.v3_small_ppl_validation,  # 测试环境使用此行（约几百万 tokens）

        # 使用与模型完全一致的 tokenizer
        tokenizer=tokenizer_config,

        # 数据混合集根目录（通常是预处理后的 numpy/token 文件）
        mix_base_dir=opts.data_root,

        # 模型实际输入的上下文长度
        sequence_length=sequence_length,

        # target 序列最大长度：
        # - 通常 >= sequence_length
        # - 用于 padding / shift / loss 计算
        max_target_sequence_length=max(8192, sequence_length),

        # 工作目录（缓存数据索引、临时文件等）
        work_dir=opts.work_dir,
    )

    # =========================
    # 数据加载器配置
    # =========================

    data_loader_config = NumpyDataLoaderConfig(
        # 全局 batch size（token 级别）：
        # 实际 = global_batch_size / (world_size)
        global_batch_size=GLOBAL_BATCH_SIZE,

        # 固定随机种子，确保多卡 & 多次训练可复现
        seed=34521,

        # DataLoader worker 数量：
        # 小模型 + numpy dataset 通常不需要太多
        num_workers=4,
    )

    # =========================
    # 训练模块配置（模型 + 优化器 + 并行）
    # =========================

    train_module_config = TransformerTrainModuleConfig(
        # 每张 GPU 的 micro-batch（token 数）
        # 单 GPU 训练: 减小 batch size 以适应 16GB 显存
        # 原 8192 tokens,改为 2048 tokens (约 8MB + 开销)
        rank_microbatch_size=2048,

        # 模型允许的最大输入长度
        max_sequence_length=sequence_length,

        # -------- 优化器配置 --------
        optim=SkipStepAdamWConfig(
            # 学习率：
            # 190M 小模型通常使用更大的 LR，加快收敛
            lr=LR,

            # 权重衰减（AdamW 标准配置）
            weight_decay=0.1,

            # Adam 动量参数（beta2 偏小有助于适配大 batch）
            betas=(0.9, 0.95),

            # 参数分组覆盖：
            # embedding 层通常不做 weight decay（经验最佳实践）
            group_overrides=[
                OptimGroupOverride(
                    params=["embeddings.weight"],
                    opts=dict(weight_decay=0.0)
                )
            ],
        ),

        # -------- 学习率调度 --------
        scheduler=CosWithWarmup(
            # 余弦退火 + warmup
            # 小模型 warmup 步数可以更短
            warmup_steps=1000
        ),

        # 使用 torch.compile：
        # - 对长时间预训练收益明显
        # - 首次 compile 会慢一些
        compile_model=True,

        # -------- 并行训练配置 --------
        dp_config=TransformerDataParallelConfig(
            # HSDP（Hybrid Sharded Data Parallel）：
            # - 参数分片 + 数据并行
            # - 适合中小模型规模
            name=DataParallelType.hsdp,

            # 模型参数 dtype（节省显存）
            param_dtype=DType.bfloat16,

            # 梯度归约 dtype（提高数值稳定性）
            reduce_dtype=DType.float32,

            # wrapping 粒度：
            # 以 Transformer block 为单位进行 FSDP 包裹
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
        ),

        # 是否启用 Float8 训练（此处关闭）
        float8_config=Float8Config(enabled=False),

        # Z-loss：
        # 防止 softmax logits 爆炸，常用于大词表语言模型
        z_loss_multiplier=1e-4,

        # 梯度裁剪，防止不稳定更新
        max_grad_norm=1.0,
    )

    # =========================
    # Trainer / Callback 配置
    # =========================

    trainer_config = (
        TrainerConfig(
            # checkpoint 保存路径
            save_folder=opts.save_folder,

            # 允许覆盖已有实验目录
            save_overwrite=True,

            # 指标收集频率（step）
            metrics_collect_interval=10,

            # 训练取消信号检查频率
            cancel_check_interval=10,

            # 最大训练 token 数（主要用于长期预训练）
            max_duration=Duration.tokens(int(50e9)),

            # 硬停止步数（调试 / 小规模实验常用）
            # 测试环境：100 步验证流程可行
            # 生产环境：恢复为 95,000
            hard_stop=Duration.steps(int(100)),  # Duration.steps(int(95_000))
        )

        # -------- 回调：运行时补丁 --------
        .with_callback(
            "monkey_patcher",
            MonkeyPatcherCallback()
        )

        # -------- 回调：Checkpoint --------
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                # 每 1000 step 保存一次
                save_interval=1000,

                # 不保存临时（ephemeral）checkpoint
                ephemeral_save_interval=None,

                # 同步保存，避免分布式下的潜在问题
                save_async=False,
            ),
        )

        # -------- 日志：Comet --------
        .with_callback(
            "comet",
            CometCallback(
                name=opts.name,
                cancel_check_interval=10,
                enabled=False,  # 默认关闭
            ),
        )

        # -------- 日志：WandB --------
        .with_callback(
            "wandb",
            WandBCallback(
                name=opts.name,
                cancel_check_interval=10,
                enabled=False,  # 默认关闭
            ),
        )

        # -------- 保存最终配置 --------
        .with_callback(
            "config_saver",
            ConfigSaverCallback()
        )

        # -------- 困惑度评估 --------
        .with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyPaddedFSLDatasetConfig.from_data_mix(
                    # 小规模验证集，用于 ppl sanity check
                    DataMix.v3_small_ppl_validation,
                    mix_base_dir=opts.data_root,
                    sequence_length=sequence_length,
                    tokenizer=tokenizer_config,
                    work_dir=opts.work_dir,
                ),
                # 小模型评估可以更频繁
                eval_interval=5_000,
            ),
        )

        # -------- 下游任务评估 --------
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=sorted([
                    # 推理 / 多选
                    "arc_challenge_test_mc_5shot_fast",
                    "arc_easy_test_mc_5shot_fast",
                    "hellaswag_bpb_5shot",

                    # MMLU 子集
                    "mmlu_humanities_test_mc_5shot_fast",
                    "mmlu_other_test_mc_5shot_fast",
                    "mmlu_social_sciences_test_mc_5shot_fast",
                    "mmlu_stem_test_mc_5shot_fast",

                    # 基础能力
                    "basic_skills_arithmetic_rc_5shot",
                    "basic_skills_coding_rc_5shot",
                    "basic_skills_common_knowledge_rc_5shot",
                    "basic_skills_logical_reasoning_rc_5shot",
                    "basic_skills_pattern_rc_5shot",
                    "basic_skills_string_operations_rc_5shot",

                    # 生成类（BPB）
                    "codex_humaneval_gold_bpb_3shot",
                    "minerva_math_500_gold_bpb_0shot",

                    # sanity check
                    "copycolors_10way_fast",
                ]),
                tokenizer=tokenizer_config,
                eval_interval=5_000,
            ),
        )
    )

    # =========================
    # 汇总 ExperimentConfig
    # =========================

    return ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
    ).merge(overrides)  # 合并 CLI 覆盖参数


if __name__ == "__main__":
    # 入口：调用 main 函数启动训练
    main(build_config)
