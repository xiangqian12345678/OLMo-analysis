"""
Official pre-training script for OLMo-3-1025-7B.

Part 1 of 2. See OLMo-3-1025-7B-pretrain-2.py for part 2.
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

DEFAULT_SEQUENCE_LENGTH = 8192
GLOBAL_BATCH_SIZE = 8192 * 512  # ~4M tokens
LR = 3e-4


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    sequence_length = opts.sequence_length or DEFAULT_SEQUENCE_LENGTH
    tokenizer_config = TokenizerConfig.dolma2()

    model_config = TransformerConfig.olmo3_7B(
        vocab_size=tokenizer_config.padded_vocab_size(),  # 词表大小（补齐到128的倍数）
        attn_backend=AttentionBackendName.flash_2,  # 使用 FlashAttention-2
    )

    dataset_config = NumpyFSLDatasetConfig.from_data_mix(
        DataMix.OLMo_mix_0625_official,  # 官方数据混合集
        tokenizer=tokenizer_config,  # Dolma2 分词器
        mix_base_dir=opts.data_root,  # 数据根目录
        sequence_length=sequence_length,  # 序列长度
        max_target_sequence_length=max(8192, sequence_length),
        work_dir=opts.work_dir,  # 工作目录
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,  # 全局批次大小
        seed=34521,  # 随机种子
        num_workers=8,  # 数据加载进程数
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=2 * 8192,
        max_sequence_length=sequence_length,
        optim=SkipStepAdamWConfig(
            lr=LR,  # 学习率 3e-4
            weight_decay=0.1,  # 权重衰减
            betas=(0.9, 0.95),  # Adam beta 参数
            group_overrides=[
                # embedding 参数不应用权重衰减
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=CosWithWarmup(warmup_steps=2000),  # 余弦退火 + 2000步预热
        compile_model=True,  # torch.compile 编译
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,  # HSDP (混合序列+数据并行)
            param_dtype=DType.bfloat16,  # 参数使用 bfloat16
            reduce_dtype=DType.float32,  # 梯度聚合用 float32
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
        ),

        float8_config=Float8Config(enabled=False),  # Float8 关闭
        z_loss_multiplier=1e-5,  # Z-loss 防止崩溃
        max_grad_norm=1.0,  # 梯度裁剪
    )

    trainer_config = (
        TrainerConfig(
            save_folder=opts.save_folder,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.tokens(int(5e12)),  # Originally scheduled for 5T
            hard_stop=Duration.steps(
                # But at this step we decided to extend schedule to 7T. See OLMo-3-1025-7B-pretrain-2.py
                int(597046)
            ),
        )
        .with_callback("monkey_patcher", MonkeyPatcherCallback())
        .with_callback( # 保存检查点
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=None,
                save_async=False,
            ),
        )
        .with_callback( # Comet 日志
            "comet",
            CometCallback(
                name=opts.name,
                cancel_check_interval=10,
                enabled=False,  # NOTE: change to true to enable
            ),
        )
        .with_callback( # WandB 日志
            "wandb",
            WandBCallback(
                name=opts.name,
                cancel_check_interval=10,
                enabled=False,  # NOTE: change to true to enable
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback()) # 保存配置
        .with_callback( # 	困惑度评估
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyPaddedFSLDatasetConfig.from_data_mix(
                    DataMix.v3_small_ppl_validation,
                    mix_base_dir=opts.data_root,
                    sequence_length=sequence_length,
                    tokenizer=tokenizer_config,
                    work_dir=opts.work_dir,
                ),
                eval_interval=10_000,
            ),
        )
        .with_callback( # 下游任务评估
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=sorted(
                    [  # "fast" task set
                        # Subset of OLMES
                        "arc_challenge_test_bpb_5shot",
                        "arc_challenge_test_mc_5shot_fast",
                        "arc_easy_test_bpb_5shot",
                        "arc_easy_test_mc_5shot_fast",
                        "hellaswag_bpb_5shot",
                        "mmlu_humanities_test_bpb_5shot",
                        "mmlu_humanities_test_mc_5shot_fast",
                        "mmlu_other_test_bpb_5shot",
                        "mmlu_other_test_mc_5shot_fast",
                        "mmlu_social_sciences_test_bpb_5shot",
                        "mmlu_social_sciences_test_mc_5shot_fast",
                        "mmlu_stem_test_bpb_5shot",
                        "mmlu_stem_test_mc_5shot_fast",
                        # Basic Skills
                        "basic_skills_arithmetic_rc_5shot",
                        "basic_skills_coding_rc_5shot",
                        "basic_skills_common_knowledge_rc_5shot",
                        "basic_skills_logical_reasoning_rc_5shot",
                        "basic_skills_pattern_rc_5shot",
                        "basic_skills_string_operations_rc_5shot",
                        # Gen tasks BPB
                        "codex_humaneval_gold_bpb_3shot",
                        "codex_mbpp_gold_bpb_3shot",
                        "minerva_math_500_gold_bpb_0shot",
                        "mt_mbpp_cpp_gold_bpb_3shot",
                        "mt_mbpp_java_gold_bpb_3shot",
                        "mt_mbpp_rust_gold_bpb_3shot",
                        # Sanity check for MCQA ability
                        "copycolors_10way_fast",
                    ]
                ),
                tokenizer=tokenizer_config,
                eval_interval=10_000,
            ),
        )
    )

    return ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
    ).merge(overrides)


if __name__ == "__main__":
    main(build_config)
