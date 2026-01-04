import os
import tempfile
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.distributed as dist

from olmo_core.eval.task_groups import TASK_GROUPS

from ..config import Config
from ..data import DataLoaderBase, TokenizerConfig
from ..exceptions import OLMoConfigurationError
from ..io import is_url
from ..utils import get_default_device
from .callbacks import Callback, CallbackConfig
from .checkpoint import CheckpointerConfig
from .common import Duration, LoadStrategy, StepSkipRange
from .train_module import TrainModule
from .trainer import Trainer


@dataclass
class TrainerConfig(Config):
    """
    用于方便构建 :class:`Trainer` 实例的配置类。

    该类封装了训练器所需的所有配置参数，支持灵活的初始化和构建方式。

    .. seealso::
        有关字段的详细说明，请参阅 :class:`Trainer` 文档。
    """

    # ========== 路径与保存相关配置 ==========
    save_folder: str
    """保存检查点和其他训练输出文件的主文件夹路径。支持本地路径或远程 URL。"""

    work_dir: Optional[str] = None
    """工作目录，用于存放临时文件和中间结果。如果为 None 且 save_folder 不是 URL，则使用 save_folder。"""

    load_path: Optional[str] = None
    """用于加载检查点的路径。如果为 None，则不加载检查点。"""

    load_strategy: LoadStrategy = LoadStrategy.if_available
    """加载策略，决定如何在检查点不存在时的行为。默认为 if_available（如果可用则加载）。"""

    load_optim_state: Optional[bool] = None
    """是否加载优化器状态。如果为 None，则根据 load_strategy 自动决定。"""

    load_trainer_state: Optional[bool] = None
    """是否加载训练器状态（如学习率调度器状态）。如果为 None，则根据 load_strategy 自动决定。"""

    checkpointer: CheckpointerConfig = field(default_factory=CheckpointerConfig)
    """检查点器配置，控制检查点的保存、加载和管理行为。"""

    # ========== 设备与训练配置 ==========
    device: Optional[str] = None
    """训练设备，如 'cuda', 'cpu' 等。如果为 None，则使用默认设备。"""

    save_overwrite: bool = False
    """是否覆盖已存在的检查点文件。"""

    max_duration: Duration = field(default_factory=lambda: Duration.epochs(1))
    """最大训练时长，可以是步数、轮次或时间。默认为 1 个 epoch。"""

    cancel_check_interval: int = 25
    """检查取消请求的间隔步数。默认每 25 步检查一次。"""

    hard_stop: Optional[Duration] = None
    """硬性停止训练的时间限制。达到此限制后立即停止训练，忽略其他条件。"""

    metrics_collect_interval: int = 5
    """收集和记录指标的间隔步数。默认每 5 步收集一次。"""

    # ========== 回调与高级功能 ==========
    callbacks: Dict[str, Callback] = field(default_factory=dict)
    """训练回调函数字典，键为回调名称，值为回调实例。"""

    async_bookkeeping: Optional[bool] = None
    """是否异步执行簿记操作（如日志记录、指标收集）。如果为 None，则根据系统配置自动决定。"""

    bookkeeping_soft_timeout: int = 30
    """簿记操作的软超时时间（秒），超过此时间会发出警告但继续执行。默认为 30 秒。"""

    no_checkpoints: bool = False
    """是否禁用检查点保存。设置为 True 则不保存任何检查点。"""

    no_evals: bool = False
    """是否禁用评估。设置为 True 则不执行任何评估回调。"""

    steps_to_skip: Optional[List[StepSkipRange]] = None
    """要跳过的训练步数范围列表。可用于跳过特定步骤的训练。"""

    def add_callback(self, name: str, callback: Callback):
        """
        添加一个回调函数。

        :param name: 回调名称，必须唯一。
        :param callback: 要添加的回调实例。
        :raises OLMoConfigurationError: 如果已存在同名回调。
        """
        if name in self.callbacks:
            raise OLMoConfigurationError(f"A callback with name '{name}' already exists")
        self.callbacks[name] = callback

    def add_callbacks(self, callbacks: Dict[str, Callback]):
        """
        添加一组回调函数。

        :param callbacks: 要添加的回调字典，键为回调名称，值为回调实例。
        """
        for name, callback in callbacks.items():
            self.add_callback(name, callback)

    def with_callback(self, name: str, callback: Callback) -> "TrainerConfig":
        """
        返回添加了额外回调的新训练器配置。

        此方法不修改当前配置对象，而是返回一个新的配置对象。

        :param name: 要分配给回调的名称，必须唯一。
        :param callback: 要添加的回调实例。
        :return: 包含新增回调的新 TrainerConfig 实例。
        """
        # 使用 replace 创建当前配置的副本，并深拷贝 callbacks 字典
        # 这样可以确保原配置对象不被修改
        # self: TrainerConfig 类型
        # out: TrainerConfig 类型（新实例，与 self 独立）
        out = replace(self, callbacks=deepcopy(self.callbacks))

        # 将新回调添加到副本的 callbacks 字典中
        # out.add_callback() 调用 TrainerConfig.add_callback()
        out.add_callback(name, callback)
        return out

    def with_callbacks(self, callbacks: Dict[str, Callback]) -> "TrainerConfig":
        """
        返回添加了多个回调的新训练器配置。

        此方法不修改当前配置对象，而是返回一个新的配置对象。

        :param callbacks: 要添加的回调字典，键为回调名称，值为回调实例。键必须唯一。
        :return: 包含新增回调的新 TrainerConfig 实例。
        """
        # 使用 replace 创建当前配置的副本，并深拷贝 callbacks 字典
        # 这样可以确保原配置对象不被修改
        out = replace(self, callbacks=deepcopy(self.callbacks))
        # 将多个回调批量添加到副本的 callbacks 字典中
        out.add_callbacks(callbacks)
        return out

    def with_recommended_evals(
            self,
            tokenizer: TokenizerConfig,
            sequence_length: int,
            cluster: str,
            task_set: str = "full",
            eval_interval: int = 10_000,
    ) -> "TrainerConfig":
        """
        返回添加了推荐评估回调的新训练器配置。

        该方法会添加两个评估回调：
        1. 下游评估器（downstream_evaluator）：用于评估下游任务性能
        2. 语言模型评估器（lm_evaluator）：用于验证集评估

        :param tokenizer: 分词器配置。
        :param sequence_length: 序列长度。
        :param cluster: 集群名称，用于确定数据路径。
        :param task_set: 任务集名称，默认为 "full"。
        :param eval_interval: 评估间隔步数，默认为 10,000 步。
        :return: 包含评估回调的新 TrainerConfig 实例。
        """
        from olmo_core.data import DataMix, NumpyPaddedFSLDatasetConfig
        from olmo_core.internal.common import get_root_dir, get_work_dir
        from olmo_core.train.callbacks import (
            DownstreamEvaluatorCallbackConfig,
            LMEvaluatorCallbackConfig,
        )

        try:
            # 从预定义的任务组中获取指定的任务集
            tasks = TASK_GROUPS[task_set]
        except KeyError as e:
            # 如果任务集不存在，抛出 ValueError，并保留原始 KeyError 信息
            raise ValueError(f"Task set not recognized: {task_set}") from e

        # 对任务进行排序，确保执行顺序的一致性
        tasks.sort()

        # 使用链式调用添加两个评估回调
        # 第一次调用添加下游评估器
        # 第二次调用添加语言模型评估器
        # 每次调用都返回一个新的 TrainerConfig 实例
        return self.with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=tasks, tokenizer=tokenizer, eval_interval=eval_interval
            ),
        ).with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyPaddedFSLDatasetConfig.from_data_mix(
                    DataMix.v3_small_ppl_validation,
                    mix_base_dir=get_root_dir(cluster),
                    sequence_length=sequence_length,
                    tokenizer=tokenizer,
                    work_dir=get_work_dir(get_root_dir(cluster)),
                ),
                eval_interval=eval_interval,
            ),
        )

    def build(
            self,
            train_module: TrainModule,
            data_loader: DataLoaderBase,
            *,
            dp_process_group: Optional[dist.ProcessGroup] = None,
            checkpointer_pg: Optional[dist.ProcessGroup] = None,
    ) -> Trainer:
        """
        构建对应的 Trainer 实例。

        :param train_module: 要训练的模块。
        :param data_loader: 用于训练的数据加载器。
        :param dp_process_group: 数据并行进程组。默认为
            :data:`olmo_core.train.train_module.TrainModule.dp_process_group`。
        :param checkpointer_pg: 检查点进程组。
        :return: 构建好的 Trainer 实例。
        """
        # 将配置转换为字典，排除 None 值，不递归处理嵌套对象
        kwargs = self.as_dict(exclude_none=True, recurse=False)

        # 如果未提供 dp_process_group，则使用 train_module 的默认值
        if dp_process_group is None:
            dp_process_group = train_module.dp_process_group

        # 从 kwargs 中提取设备配置
        device = kwargs.pop("device", None)

        # 处理工作目录配置
        work_dir = kwargs.pop("work_dir", None)
        if work_dir is None:
            # 如果 work_dir 未设置，使用 save_folder 作为 work_dir
            if not is_url(self.save_folder):
                work_dir = self.save_folder
            else:
                # 如果 save_folder 是 URL，使用临时目录
                work_dir = os.path.join(tempfile.gettempdir(), os.path.basename(self.save_folder))
        elif is_url(work_dir):
            # work_dir 不能是 URL
            raise OLMoConfigurationError(
                f"Trainer 'work_dir' must be a local path, not a URL ('{work_dir}')"
            )

        # 构建检查点器的配置参数
        checkpointer_kwargs = {}
        if self.checkpointer.save_overwrite is None:
            checkpointer_kwargs["save_overwrite"] = self.save_overwrite
        if self.checkpointer.work_dir is None:
            checkpointer_kwargs["work_dir"] = work_dir
        # 构建检查点器
        checkpointer = kwargs.pop("checkpointer").build(
            process_group=checkpointer_pg, **checkpointer_kwargs
        )

        # 分离已实例化的回调和配置形式的回调
        all_callbacks = kwargs.pop("callbacks")
        callbacks = {k: cb for k, cb in all_callbacks.items() if not isinstance(cb, CallbackConfig)}
        callback_configs = {
            k: cb for k, cb in all_callbacks.items() if isinstance(cb, CallbackConfig)
        }

        # 创建 Trainer 实例
        trainer = Trainer(
            train_module=train_module,
            data_loader=data_loader,
            checkpointer=checkpointer,
            work_dir=Path(work_dir),
            device=torch.device(device) if device is not None else get_default_device(),
            dp_process_group=dp_process_group,
            callbacks=callbacks,
            **kwargs,
        )

        # 构建并添加配置形式的回调
        for cb_name, cb_config in callback_configs.items():
            cb = cb_config.build(trainer)
            if cb is not None:
                trainer.add_callback(cb_name, cb)

        return trainer
