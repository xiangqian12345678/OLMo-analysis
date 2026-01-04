import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from olmo_core.data import (
    NumpyDatasetConfig,
    NumpyPaddedFSLDataset,
    NumpyVSLDatasetConfig,
    TextDataLoaderBase,
    TokenizerConfig,
)
from olmo_core.data.utils import get_labels
from olmo_core.distributed.utils import get_rank, get_world_size, is_distributed
from olmo_core.eval import Evaluator
from olmo_core.eval.lm_evaluator import LMEvaluator
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.utils import (
    cuda_sync_debug_mode,
    format_float,
    get_default_device,
    move_to_device,
)

from ..common import Duration, MetricMergeStrategy
from ..train_module import EvalBatchSizeUnit, EvalBatchSpec, TransformerTrainModule
from .callback import Callback, CallbackConfig

if TYPE_CHECKING:
    from olmo_eval import HFTokenizer

    from ..trainer import Trainer

log = logging.getLogger(__name__)


@dataclass
class EvaluatorCallback(Callback):
    """
    在训练循环中定期对 :class:`~olmo_core.train.train_module.TransformerTrainModule`
    进行评估的回调。

    该回调支持两种评估触发方式：
    1. 按固定间隔（eval_interval）定期评估
    2. 在特定步数（fixed_steps）进行评估

    评估过程包括：
    - 遍历所有配置的评估器
    - 对每个批次进行前向传播
    - 计算并记录评估指标
    - 记录评估速度和吞吐量
    """

    evaluators: List[Evaluator] = field(default_factory=list)
    """
    要运行的评估器列表。
    可以包含语言模型评估器（LMEvaluator）或下游任务评估器（DownstreamEvaluator）。
    """

    eval_interval: Optional[int] = 1000
    """
    运行评估器的间隔（以步数为单位）。
    默认每 1000 步运行一次评估。设置为 None 则禁用定期评估。
    """

    fixed_steps: Optional[List[int]] = None
    """
    固定的评估步数列表。
    在这些特定步数时运行评估器。例如 [1000, 10000] 表示在这些步数时进行评估。
    """

    eval_on_startup: bool = False
    """
    是否在训练启动时运行一次评估。
    设置为 True 会在训练开始时立即执行评估。
    """

    cancel_after_first_eval: bool = False
    """
    是否在第一次评估后取消训练。
    如果为 True，在运行完第一次评估后取消整个训练流程。
    配合 eval_on_startup=True 使用时，可以只运行评估而不进行训练。
    """

    eval_duration: Duration = field(default_factory=lambda: Duration.epochs(1))
    """
    每个评估器的运行时长。
    默认为 1 个 epoch，即遍历完整评估数据集一次。
    可以通过 Duration.epochs()、Duration.steps()、Duration.tokens() 等方法配置。
    """

    log_interval: int = 5
    """
    评估循环中日志输出的间隔（以步数为单位）。
    默认每 5 个批次输出一次评估进度日志。
    """

    def post_attach(self):
        """
        回调附加到训练器后的初始化检查。

        验证训练模块是否为 TransformerTrainModule 类型。
        如果类型不匹配，抛出 OLMoConfigurationError 异常。
        """
        if not isinstance(self.trainer.train_module, TransformerTrainModule):
            raise OLMoConfigurationError(
                f"'{self.__class__.__name__}' only supports the '{TransformerTrainModule.__name__}' train module"
            )

    def pre_train(self):
        """
        训练开始前的钩子方法。

        如果 eval_on_startup 为 True，则在训练开始前运行一次评估。
        这可以用于验证模型的初始性能。
        """
        if self.eval_on_startup:
            self._perform_eval()

    def post_step(self):
        """
        每个训练步骤后的钩子方法。

        检查是否需要运行评估，根据以下条件：
        1. 当前步数是 eval_interval 的倍数
        2. 当前步数在 fixed_steps 列表中

        如果满足任一条件，则运行评估。
        """
        if self.step <= 1:
            return

        if (self.eval_interval is not None and self.step % self.eval_interval == 0) or (
                self.fixed_steps is not None and self.step in self.fixed_steps
        ):
            self._perform_eval()

    def _perform_eval(self):
        """
        执行所有评估器的评估流程。

        主要步骤：
        1. 获取数据并行的世界大小
        2. 遍历每个评估器
        3. 对每个批次进行前向传播
        4. 更新评估指标
        5. 计算并记录最终指标
        6. 记录评估速度和吞吐量

        注意：
        - 评估过程中使用 torch.no_grad() 禁用梯度计算
        - 评估结果会被记录到训练器的指标系统中
        - 支持分布式评估，所有 rank 同步进行评估
        """
        # 将模型设置为评估模式
        # TODO: 确保梯度在此点已被清零
        #  self.trainer.optim.zero_grad(set_to_none=True)
        #  self.trainer.model.eval()
        dp_world_size = get_world_size(self.trainer.dp_process_group)

        evaluator_times = []
        evaluator_names = []
        evaluator_bs = []

        # 遍历所有评估器，逐个执行评估
        for evaluator in self.evaluators:
            log.info(f"Running {evaluator.name} evals...")
            start_time = time.monotonic()  # 记录评估开始时间（单调时钟，不受系统时间调整影响）
            evaluator.reset_metrics()  # 重置评估器的所有指标计数器
            eval_step = 0
            eval_tokens = 0
            for batch in evaluator:
                eval_step += 1
                # 累计处理的总 token 数 = 当前批次 token 数 × 数据并行进程数
                eval_tokens += batch["input_ids"].numel() * dp_world_size

                # 将批次数据移动到默认设备（GPU）
                batch = move_to_device(batch, get_default_device())
                with torch.no_grad():  # 禁用梯度计算，减少内存占用和计算开销
                    # 运行前向传播，获取 logits 和未归约的 CE 损失
                    labels = get_labels(batch)
                    # 使用训练模块对批次进行评估推理
                    # eval_batch: 执行模型前向传播，计算 logits 和损失
                    # labels: 已处理好的标签张量，形状与 input_ids 相同
                    output = self.trainer.train_module.eval_batch(batch, labels=labels)

                    # 断言输出是 LMOutputWithLoss 类型
                    # LMOutputWithLoss: 包含 logits、hidden_states、loss 和其他指标的命名元组/数据类
                    assert isinstance(output, LMOutputWithLoss)

                    # 解包输出，获取需要的字段
                    logits, _, ce_loss, _ = output
                    #  ↑    ↑    ↑      ↑
                    #  |    |    |      └─ 其他指标（如 perplexity 等）
                    #  |    |    └─ 交叉熵损失（Cross Entropy Loss）
                    #  |    └─ 隐藏状态（hidden states，这里用下划线表示不使用）
                    #  └─ Logits: 模型输出的预测概率分布，shape: (batch_size, seq_len, vocab_size)

                    # 注意：这里可能有主机-设备同步，但这是可以接受的
                    # cuda_sync_debug_mode(0) 表示禁用 CUDA 同步调试模式
                    with cuda_sync_debug_mode(0):
                        evaluator.update_metrics(batch, ce_loss, logits)

                # 检查评估时长是否达到限制（基于步数、token 数或 epoch）
                # due() 方法判断是否已达到配置的评估时长
                if self.eval_duration.due(step=eval_step, tokens=eval_tokens, epoch=1):
                    self._log_progress(evaluator, eval_step)
                    break  # 达到时长限制，提前结束当前评估器

                # 定期记录评估进度（按间隔或到最后一个批次）
                if eval_step % self.log_interval == 0 or eval_step == evaluator.total_batches:
                    self._log_progress(evaluator, eval_step)

            # 注意：这里会有主机-设备同步，但可以接受。每个评估器只同步一次
            metrics_str = []
            evaluation_names = []
            with cuda_sync_debug_mode(0):
                # 计算并获取所有评估指标的最终值
                metrics = evaluator.compute_metrics()
                for name, value in metrics.items():
                    evaluation_names.append(name)
                    metrics_str.append(f"    {name}={format_float(value.item())}")
                    # 将指标记录到训练器，格式为 eval/evaluator_name/metric_name
                    self.trainer.record_metric(f"eval/{evaluator.name}/{name}", value)

            # 记录评估器执行时间、指标名称和批次计数
            evaluator_times.append(time.monotonic() - start_time)
            evaluator_names.append(evaluation_names)
            evaluator_bs.append(eval_step)

            # 输出评估完成日志，包含耗时和所有指标值
            log.info(
                f"Finished {evaluator.name} evals in {time.monotonic() - start_time:.1f} seconds. Metrics:\n"
                + "\n".join(metrics_str)
            )

        # 按评估时间升序排序评估器（便于快速查看最慢的评估器）
        sorted_evaluators = sorted(
            zip(evaluator_names, evaluator_bs, evaluator_times), key=lambda x: x[2]
        )

        # 记录评估速度统计信息
        eval_speeds = []
        for names, bs, t in sorted_evaluators:
            name = names[0]  # 取第一个指标名称作为评估器标识
            eval_speeds.append(f"    {name} (+variants): {t:.1f} sec ({bs} batches)")
        # 计算总评估时间和总批次
        total_time = sum(evaluator_times)
        total_bs = sum(int(bs) if bs is not None else 0 for bs in evaluator_bs)
        eval_speeds.append(
            f"    Total evaluation time: {total_time:.1f} seconds ({total_bs} batches)"
        )
        log.info("Evaluation speed:\n" + "\n".join(eval_speeds))

        # 记录到训练器指标系统，使用 sum 策略在分布式环境中聚合
        self.trainer.record_metric(
            "throughput/in-loop eval time (s)", total_time, merge_strategy=MetricMergeStrategy.sum
        )
        self.trainer.record_metric(
            "throughput/in-loop eval batches", total_bs, merge_strategy=MetricMergeStrategy.sum
        )

        # 如果配置为第一次评估后取消，则取消训练
        if self.cancel_after_first_eval:
            self.trainer.cancel_run(
                "canceled from evaluator callback since 'cancel_after_first_eval' is set",
                no_sync=True,  # 'no_sync' 因为我们同时从所有 rank 调用此方法
            )

    def _log_progress(self, evaluator: Evaluator, eval_step: int):
        """
        记录评估进度日志。
        参数:
            evaluator: 评估器实例
            eval_step: 当前评估步数
        日志格式：
            - 如果评估器有总批次数：[eval=evaluator_name,step=current/total]
            - 如果评估器没有总批次数：[eval=evaluator_name,step=current]
        """
        if evaluator.total_batches is not None:
            log.info(f"[eval={evaluator.name},step={eval_step}/{evaluator.total_batches}]")
        else:
            log.info(f"[eval={evaluator.name},step={eval_step}]")


@dataclass
class LMEvaluatorCallbackConfig(CallbackConfig):
    """
    语言模型评估回调配置类。

    用于配置在训练过程中定期运行语言模型评估的回调。
    该配置会构建一个 LMEvaluator 实例，用于评估模型的语言建模能力。

    属性:
        eval_dataset: 评估数据集配置，必须是 NumpyDatasetConfig 的子类
        eval_interval: 评估间隔（步数），默认每 1000 步评估一次
        fixed_steps: 固定的评估步数列表，在这些步数时也会进行评估
        eval_on_startup: 是否在训练启动时运行一次评估
        cancel_after_first_eval: 是否在第一次评估后取消训练，配合 eval_on_startup=True 使用时可在不训练的情况下仅运行评估
        eval_duration: 每个评估器的运行时长，默认 1 个 epoch
        log_interval: 评估过程中的日志输出间隔（步数）
        enabled: 是否启用此回调
    """

    eval_dataset: NumpyDatasetConfig
    """评估数据集配置"""
    eval_interval: Optional[int] = 1000
    """评估间隔（步数）"""
    fixed_steps: Optional[List[int]] = None
    """固定的评估步数列表"""
    eval_on_startup: bool = False
    """是否在启动时评估"""
    cancel_after_first_eval: bool = False
    """是否在第一次评估后取消训练"""
    eval_duration: Duration = field(default_factory=lambda: Duration.epochs(1))
    """每个评估器的运行时长"""
    log_interval: int = 5
    """评估日志输出间隔"""
    enabled: bool = True
    """是否启用此回调"""

    def build(self, trainer: "Trainer") -> Optional[Callback]:
        """
        构建评估回调实例。

        参数:
            trainer: 训练器实例，提供访问模型、设备、分布式进程组等的接口

        返回:
            构建的 EvaluatorCallback 实例，如果 enabled 为 False 则返回 None

        异常:
            OLMoConfigurationError: 当配置不符合要求时抛出，如序列长度超出限制、数据集类型不正确等
        """
        # 如果未启用，返回 None
        if not self.enabled:
            return None

        # 获取数据集的最大序列长度
        if isinstance(self.eval_dataset, NumpyVSLDatasetConfig):
            # NumpyVSLDatasetConfig 使用 max_sequence_length 属性
            dataset_max_sequence_length = self.eval_dataset.max_sequence_length
        else:
            # 其他配置使用 sequence_length 属性
            assert hasattr(self.eval_dataset, "sequence_length")
            dataset_max_sequence_length = self.eval_dataset.sequence_length

        # 获取训练模块的评估批处理规范
        batch_spec = trainer.train_module.eval_batch_spec

        # 验证数据集的最大序列长度不超过模型的限制
        if (
                batch_spec.max_sequence_length is not None
                and dataset_max_sequence_length > batch_spec.max_sequence_length
        ):
            raise OLMoConfigurationError(
                f"The maximum sequence length for the LM eval dataset ({dataset_max_sequence_length:,d} tokens) "
                f"is too long for the train module's maximum eval sequence length ({batch_spec.max_sequence_length:,d} tokens)"
            )

        # 计算全局评估批大小
        global_eval_batch_size: int
        if batch_spec.batch_size_unit == EvalBatchSizeUnit.tokens:
            # 批大小单位为 token 数，全局批大小 = 单 rank 批大小 * DP 进程数
            global_eval_batch_size = batch_spec.rank_batch_size * get_world_size(
                trainer.dp_process_group
            )
        elif batch_spec.batch_size_unit == EvalBatchSizeUnit.instances:
            # 批大小单位为实例数，全局批大小 = 单 rank 批大小 * 序列长度 * DP 进程数
            global_eval_batch_size = (
                    batch_spec.rank_batch_size
                    * dataset_max_sequence_length
                    * get_world_size(trainer.dp_process_group)
            )
        else:
            raise NotImplementedError(batch_spec.batch_size_unit)

        # 构建数据集实例
        dataset = self.eval_dataset.build()
        # 验证数据集类型必须是 NumpyPaddedFSLDataset（填充的 FSL 数据集）
        if not isinstance(dataset, NumpyPaddedFSLDataset):
            raise OLMoConfigurationError(
                f"Expected a padded FSL dataset, got '{dataset.__class__.__name__}' instead"
            )

        # 验证数据加载器类型必须是 TextDataLoaderBase（基于文本的数据加载器）
        if not isinstance(trainer.data_loader, TextDataLoaderBase):
            raise OLMoConfigurationError(
                f"Expected a text-based data loader, got '{dataset.__class__.__name__}' instead"
            )

        # 从 NumPy 数据集构建语言模型评估器
        evaluator = LMEvaluator.from_numpy_dataset(
            dataset,
            name="lm",
            global_batch_size=global_eval_batch_size,
            collator=trainer.data_loader.collator,
            device=trainer.device,
            dp_process_group=trainer.dp_process_group,
        )

        # 构建并返回评估回调
        return EvaluatorCallback(
            evaluators=[evaluator],
            eval_interval=self.eval_interval,
            fixed_steps=self.fixed_steps,
            log_interval=self.log_interval,
            eval_on_startup=self.eval_on_startup,
            cancel_after_first_eval=self.cancel_after_first_eval,
            eval_duration=self.eval_duration,
        )


class DownstreamEvaluator(Evaluator):
    """
    下游任务评估器。

    用于评估模型在特定下游任务（如 Hellaswag、ARC 等）上的表现。
    支持多种评估指标类型，包括 F1 分数、准确率、CE 损失等。

    属性:
        metric_type_to_label: 指标类型到可读标签的映射字典
    """

    metric_type_to_label = {
        # ===== 精确率/召回率/F1 相关指标 =====
        "f1_v1": "F1 score",  # F1分数: 2 * (precision * recall) / (precision + recall)
        "f1_v2": "F1 score v2",  # F1分数版本2

        # ===== 准确率相关指标 =====
        "acc_v1": "accuracy",  # 准确率: (TP + TN) / (TP + TN + FP + FN)
        "acc_v2": "accuracy v2",  # 准确率版本2

        # ===== 长度归一化准确率 =====
        "len_norm_v1": "length-normalized accuracy",  # 长度归一化准确率: accuracy / sequence_length 或其他归一化方式
        "len_norm_v2": "length-normalized accuracy v2",  # 长度归一化准确率版本2

        # ===== PMI-DC 准确率 =====
        "pmi_dc_v1": "PMI-DC accuracy",  # 基于点互信息-去相关性的准确率，结合 PMI 和去相关技术的准确率计算
        "pmi_dc_v2": "PMI-DC accuracy v2",  # PMI-DC 准确率版本2

        # ===== 损失相关指标 =====
        "ce_loss_v1": "CE loss",  # 交叉熵损失: -Σ(y * log(p))，y为真实标签，p为预测概率
        "ce_loss_v2": "CE loss v2",  # 交叉熵损失版本2

        # ===== BPB（Bits Per Byte，每字节数比特）=====
        "bpb_v1": "BPB",  # 每字节数比特: ce_loss / ln(2)，衡量数据压缩效率
        "bpb_v2": "BPB v2",  # BPB版本2

        # ===== Soft Loss 相关指标 =====
        "soft_v1": "soft loss",  # 软损失: 基于软标签的损失计算，如 KL 散度
        "soft_v2": "soft loss v2",  # 软损失版本2

        # ===== Log Soft Loss 相关指标 =====
        "soft_log_v1": "log soft loss",  # 对数软损失: log(soft_loss) 或基于 log_softmax 的损失
        "soft_log_v2": "log soft loss v2",  # 对数软损失版本2
    }

    def __init__(
            self,
            *,
            name: str,
            task: str,
            batch_spec: EvalBatchSpec,
            tokenizer: "HFTokenizer",
            device: Optional[torch.device] = None,
            dp_process_group: Optional[dist.ProcessGroup] = None,
    ):
        """
        初始化下游任务评估器。

        参数:
            name: 评估器名称
            task: 下游任务名称（如 "hellaswag", "arc_easy" 等）
            batch_spec: 评估批处理规范，包含批大小、序列长度等配置
            tokenizer: 用于文本编码的 tokenizer
            device: 计算设备，默认为 get_default_device()
            dp_process_group: 数据并行进程组，用于分布式评估
        """
        from olmo_eval import ICLMetric, ICLMultiChoiceTaskDataset, build_task

        # 构建任务数据集
        task_dataset: ICLMultiChoiceTaskDataset
        if batch_spec.fixed_sequence_length:
            # 固定序列长度模式
            assert batch_spec.max_sequence_length is not None
            task_dataset = build_task(
                task, tokenizer, model_ctx_len=batch_spec.max_sequence_length, fixed_ctx_len=True
            )
        elif batch_spec.max_sequence_length is not None:
            # 指定最大序列长度
            task_dataset = build_task(task, tokenizer, model_ctx_len=batch_spec.max_sequence_length)
        else:
            # 使用默认序列长度
            task_dataset = build_task(task, tokenizer)

        # 保存任务标签和数据集
        self.label = task
        self.task = task_dataset

        # 初始化评估指标计算器
        self.metric = ICLMetric(metric_type=self.task.metric_type).to(
            device or get_default_device()
        )

        # 创建分布式采样器（如果在分布式环境中）
        sampler: Optional[DistributedSampler] = None
        if is_distributed():
            sampler = DistributedSampler(
                self.task,  # type: ignore  # 要采样的数据集
                drop_last=False,  # 保留不足一个批次的数据，不丢弃最后的剩余样本
                shuffle=False,  # 不打乱数据顺序，确保评估结果的可重复性和确定性
                num_replicas=get_world_size(dp_process_group),  # 数据并行的进程总数（GPU数量）
                rank=get_rank(dp_process_group),  # 当前进程在数据并行组中的rank ID
            )

        # 验证任务的最大序列长度不超过模型的限制
        if (
                batch_spec.max_sequence_length is not None
                and self.task.max_sequence_length > batch_spec.max_sequence_length
        ):
            raise OLMoConfigurationError(
                f"The maximum sequence length for downstream eval task '{task}' ({self.task.max_sequence_length:,d} tokens) "
                f"is too long for the train module's maximum eval sequence length ({batch_spec.max_sequence_length:,d} tokens)"
            )

        # 计算每个 rank 的批大小（以实例为单位）
        rank_batch_size_instances: int
        if batch_spec.batch_size_unit == EvalBatchSizeUnit.instances:
            # 批大小单位为实例数
            rank_batch_size_instances = batch_spec.rank_batch_size
        elif batch_spec.batch_size_unit == EvalBatchSizeUnit.tokens:
            # 批大小单位为 token 数，需要转换为实例数
            if batch_spec.fixed_sequence_length:
                assert batch_spec.max_sequence_length is not None
                # 验证 token 数可以被序列长度整除
                if batch_spec.rank_batch_size % batch_spec.max_sequence_length != 0:
                    raise OLMoConfigurationError(
                        f"The eval batch size ({batch_spec.rank_batch_size} tokens) must be divisible "
                        f"by the maximum eval sequence length ({batch_spec.max_sequence_length:,d} tokens)"
                    )
                rank_batch_size_instances = (
                        batch_spec.rank_batch_size // batch_spec.max_sequence_length
                )
            else:
                # 使用任务的最大序列长度进行转换
                rank_batch_size_instances = (
                        batch_spec.rank_batch_size // self.task.max_sequence_length
                )
        else:
            raise NotImplementedError(batch_spec.batch_size_unit)

        # 记录批大小信息
        log.info(
            f"Using per-rank batch size of {rank_batch_size_instances} instances "
            f"for downstream eval task '{task}' with max sequence length {self.task.max_sequence_length:,d} tokens"
        )

        # 创建数据加载器
        data_loader = DataLoader(
            self.task,  # type: ignore
            batch_size=rank_batch_size_instances,
            collate_fn=self.task.collate_fn,
            drop_last=False,
            shuffle=False,
            num_workers=0,
            sampler=sampler,
        )

        # 调用父类初始化
        super().__init__(name=name, batches=data_loader, device=device)

    def update_metrics(
            self, batch: Dict[str, Any], ce_loss: Optional[torch.Tensor], logits: Optional[torch.Tensor]
    ) -> None:
        """
        使用当前批次的预测结果更新评估指标。

        参数:
            batch: 输入批次数据
            ce_loss: 交叉熵损失（下游评估不使用此参数）
            logits: 模型输出的 logits
        """
        del ce_loss  # 下游评估不需要 CE 损失
        self.metric.update(batch, logits)

    def compute_metrics(self) -> Dict[str, torch.Tensor]:
        """
        计算并返回所有评估指标。

        返回:
            字典，键为指标名称（包含任务名和可读标签），值为指标值
        """
        # 从指标计算器获取原始指标值
        metric_type_to_value = self.metric.compute()
        outputs = {}
        # 将指标类型映射到可读标签
        for metric_type, value in metric_type_to_value.items():
            key = f"{self.label} ({self.metric_type_to_label[metric_type]})"
            outputs[key] = value
        return outputs

    def reset_metrics(self) -> None:
        """重置所有评估指标，准备开始新一轮评估。"""
        self.metric.reset()


@dataclass
class DownstreamEvaluatorCallbackConfig(CallbackConfig):
    """
    下游评估回调配置类。

    用于配置在训练过程中定期运行下游任务评估的回调。
    支持多个下游任务的评估，使用指定的 tokenizer 进行文本编码。

    属性:
        tasks: 下游任务列表，例如 ["hellaswag", "arc_easy", "arc_challenge"]
        tokenizer: Tokenizer 配置，用于文本编码
        eval_interval: 评估间隔（步数），默认每 1000 步评估一次
        fixed_steps: 固定的评估步数列表，在这些步数时也会进行评估
        eval_duration: 每个评估器的运行时长，默认 1 个 epoch
        eval_on_startup: 是否在训练启动时运行一次评估
        cancel_after_first_eval: 是否在第一次评估后取消训练，配合 eval_on_startup=True 使用时可在不训练的情况下仅运行评估
        log_interval: 评估过程中的日志输出间隔（步数）
        enabled: 是否启用此回调
    """

    tasks: List[str]
    """下游任务列表"""
    tokenizer: TokenizerConfig
    """Tokenizer 配置"""
    eval_interval: Optional[int] = 1000
    """评估间隔（步数）"""
    fixed_steps: Optional[List[int]] = None
    """固定的评估步数列表"""
    eval_duration: Duration = field(default_factory=lambda: Duration.epochs(1))
    """每个评估器的运行时长"""
    eval_on_startup: bool = False
    """是否在启动时评估"""
    cancel_after_first_eval: bool = False
    """是否在第一次评估后取消训练"""
    log_interval: int = 5
    """评估日志输出间隔"""
    enabled: bool = True
    """是否启用此回调"""

    def build(self, trainer: "Trainer") -> Optional[Callback]:
        """
        构建评估回调实例。
        参数:
            trainer: 训练器实例，提供访问模型、设备、分布式进程组等的接口
        返回:
            构建的 EvaluatorCallback 实例，如果 enabled 为 False 则返回 None
        异常:
            OLMoConfigurationError: 当 tokenizer.identifier 为 None 时抛出
        """
        # 如果未启用，返回 None
        if not self.enabled:
            return None

        # 导入 HF tokenizer
        from olmo_eval import HFTokenizer

        # 验证 tokenizer 标识符是否存在
        if self.tokenizer.identifier is None:
            raise OLMoConfigurationError(
                "Tokenizer 'identifier' required to build a concrete tokenizer"
            )

        # 构建 HF tokenizer 实例
        tokenizer = HFTokenizer(
            self.tokenizer.identifier,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
        )

        # 为每个任务创建下游评估器
        evaluators: List[Evaluator] = []
        for task in sorted(self.tasks):
            evaluators.append(
                DownstreamEvaluator(
                    name="downstream",
                    task=task,
                    batch_spec=trainer.train_module.eval_batch_spec,
                    tokenizer=tokenizer,
                    device=trainer.device,
                    dp_process_group=trainer.dp_process_group,
                )
            )

        # 构建并返回评估回调
        return EvaluatorCallback(
            evaluators=evaluators,
            eval_interval=self.eval_interval,
            fixed_steps=self.fixed_steps,
            eval_on_startup=self.eval_on_startup,
            cancel_after_first_eval=self.cancel_after_first_eval,
            log_interval=self.log_interval,
            eval_duration=self.eval_duration,
        )
