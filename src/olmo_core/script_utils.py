import argparse
import logging
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import rich

from olmo_core.aliases import PathOrStr
from olmo_core.config import Config
from olmo_core.data import NumpyDataLoaderConfig, NumpyDatasetConfig
from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_state_dict
from olmo_core.io import is_url, join_path, normalize_path
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.train import (
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import ConfigSaverCallback
from olmo_core.train.train_module import TransformerTrainModuleConfig
from olmo_core.utils import prepare_cli_environment, seed_all

log = logging.getLogger(__name__)


@dataclass
class ExperimentConfig(Config):
    """
    完整的训练实验配置类。

    该类封装了训练所需的所有组件配置，提供了一个统一的配置接口。
    所有的训练脚本都需要返回一个 ExperimentConfig 实例，该实例会被 main() 函数使用。

    Attributes:
        model: Transformer 模型配置，定义模型结构（层数、隐藏层大小、注意力机制等）
        dataset: 数据集配置，定义数据来源、预处理方式、序列长度等
        data_loader: 数据加载器配置，定义批次大小、worker 数量、随机种子等
        train_module: 训练模块配置，包含优化器、学习率调度器、微批次大小等训练参数
        trainer: 训练器配置，定义训练循环、检查点保存、日志记录、回调函数等
        init_seed: 初始化随机种子，确保模型初始化、数据加载等可复现（默认 12536）
        load_path: 可选的检查点加载路径，用于从指定路径加载预训练模型或断点续训
                 如果未设置，则尝试从 save_folder 中加载检查点
    """
    model: TransformerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = 12536
    load_path: Optional[str] = None


def get_cli_parser() -> argparse.ArgumentParser:
    """
    创建并返回用于解析训练脚本命令行参数的 ArgumentParser 对象。

    该函数定义了所有训练脚本共用的命令行参数接口，包括：
    - --name: 训练运行名称
    - --sequence-length: 训练序列长度
    - --data-root: 数据源根目录/URL
    - --save-folder: 检查点保存目录（必需参数）
    - --work-dir: 本地工作目录
    - --dry-run: 干运行模式

    Returns:
        argparse.ArgumentParser: 配置好的参数解析器对象，可用于解析命令行参数

    Note:
        该解析器支持在命令行末尾传递配置覆盖项（如 model.hidden_size=768），
        这些未识别的参数会被 parse_known_args() 返回，用于后续的配置覆盖
    """
    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        usage=f"python {sys.argv[0]} [OPTIONS...] [CONFIG_OVERRIDES...]",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --name: 训练运行名称，用于日志记录和标识
    parser.add_argument(
        "--name",
        type=str,
        help="""A name to assign the run for logging.""",
    )
    # --sequence-length: 训练和评估的序列长度
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="""The sequence length to train and eval on. Different scripts have different default
        sequence-length values. If a value is not specified here, the default value is used.""",
    )
    # --data-root: 数据源文件的根目录/URL，默认使用公共数据源（速度较慢）
    parser.add_argument(
        "--data-root",
        type=str,
        default="https://olmo-data.org",
        help="""The root directory/URL of the data source files.
        The default 'https://olmo-data.org' is public, but potentially very slow.
        Ai2 employees should prefer '/weka/oe-training-default/ai2-llm' when using a cluster with weka access,
        otherwise 'gs://ai2-llm' or 's3://ai2-llm'.""",
    )
    # --save-folder: 保存检查点的目录路径（本地或远程），所有训练节点需可访问
    parser.add_argument(
        "--save-folder",
        type=str,
        required=True,
        help="""A local or remote directory to save checkpoints to.
        All ranks should have access to this directory, so when training in a multi-node setup
        this could either be a path to a folder on a shared filesystem (such as NFS) or a URL
        to cloud storage, like 's3://...' or 'gs://...'.""",
    )
    # --work-dir: 数据集预处理的本地工作目录，未设置时从 save_folder 自动推断
    parser.add_argument(
        "--work-dir",
        type=str,
        help="""A local directory to use as a working directory for dataset preprocessing.
        If not set this will be inferred from the save folder.""",
    )
    # --dry-run: 干运行模式，打印配置后退出，用于验证配置
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="""Print the config and exit.""",
    )
    return parser


def _parse_args(
    parser: Optional[argparse.ArgumentParser] = None,
) -> Tuple[argparse.Namespace, List[str]]:
    """
    解析命令行参数和配置覆盖项。

    Args:
        parser: 可选的自定义参数解析器。如果未提供，则使用默认的 get_cli_parser() 创建解析器

    Returns:
        Tuple[argparse.Namespace, List[str]]: 包含两个元素的元组
            - opts: 解析后的命名空间对象，包含所有已知的命令行参数
                    例如：name, sequence_length, data_root, save_folder, work_dir, dry_run
            - overrides: 未被解析的剩余参数列表，通常用于配置覆盖
                         例如：["model.hidden_size=768", "trainer.max_steps=10000"]

    工作流程：
        1. 确定使用哪个解析器（传入的 parser 或默认的 get_cli_parser()）
        2. 解析命令行参数，将已知参数存入 opts，未知参数存入 overrides
        3. 如果 work_dir 未设置，根据 save_folder 的类型自动推断 work_dir
           - 如果 save_folder 是 URL（如 s3://、gs://、https://），使用本地临时目录
           - 否则将 work_dir 设置为与 save_folder 相同
    """
    # 确定使用哪个参数解析器：如果未提供，则使用默认的 get_cli_parser()
    parser = parser if parser is not None else get_cli_parser()

    # 解析命令行参数
    # parse_known_args() 会将已知的命令行参数解析到 opts（Namespace 对象）
    # 未被识别的参数（如配置覆盖项 "model.hidden_size=768"）会返回到 overrides 列表
    opts, overrides = parser.parse_known_args()

    # 如果 work_dir 未指定，根据 save_folder 的类型自动推断
    if opts.work_dir is None:
        # save_folder 是远程 URL（如 s3://、gs://、https://）时，无法作为本地工作目录
        # 因此使用本地临时目录作为数据集预处理的工作目录
        if is_url(opts.save_folder):
            opts.work_dir = "/tmp/olmo-core/dataset-cache"
        else:
            # save_folder 是本地路径时，直接将其作为工作目录
            # 这样检查点和临时数据文件会保存在同一位置
            opts.work_dir = opts.save_folder

    return opts, overrides


def main(
    config_builder: Callable[[argparse.Namespace, List[str]], ExperimentConfig],
    parser: Optional[argparse.ArgumentParser] = None,
) -> None:
    """
    训练脚本的主入口函数。

    Args:
        config_builder: 配置构建器函数，接收命令行参数和配置覆盖项，返回完整的实验配置
        parser: 可选的自定义参数解析器，如果未提供则使用默认的 get_cli_parser()

    工作流程：
        1. 解析命令行参数和配置覆盖项
        2. 构建完整配置对象
        3. 准备训练环境（分布式后端初始化等）
        4. 设置随机种子以确保可复现性
        5. 构建模型、训练模块、数据集、数据加载器和训练器
        6. 将配置保存到 W&B 和检查点目录
        7. 加载检查点（如果存在）
        8. 开始训练
        9. 清理分布式训练环境
    """
    # 解析命令行参数和配置覆盖项
    # opts: 解析后的命名空间参数对象，包含 --name, --sequence-length, --save-folder 等命令行选项
    # overrides: 未被解析的剩余参数，通常用于配置覆盖，如 "model.hidden_size=768"
    opts, overrides = _parse_args(parser)

    # 如果是干运行模式（--dry-run），则只打印配置信息，不进行实际训练
    if opts.dry_run:
        # 准备 CLI 环境，设置日志格式等
        prepare_cli_environment()

    # 使用配置构建器创建完整的实验配置对象
    # config_builder 会根据 opts 和 overrides 构建包含模型、数据、训练器等所有组件的配置
    config = config_builder(opts, overrides)

    # 干运行模式：打印配置并退出，用于验证配置是否正确
    if opts.dry_run:
        rich.print(config)
        return

    # 准备训练环境，包括：
    # - 初始化分布式通信后端（如 NCCL、GLOO）
    # - 设置进程组（world_size, rank, local_rank）
    # - 设置设备（CUDA/XLA）
    # shared_filesystem 参数：save_folder 是本地文件系统（非 URL）时启用共享文件系统模式
    prepare_training_environment(shared_filesystem=not is_url(opts.save_folder))

    # 在所有设备上设置随机数生成器（RNG）状态
    # 这包括 Python 的 random、NumPy、PyTorch、CUDA 等
    # 确保训练过程的可复现性（在相同配置下）
    seed_all(config.init_seed)

    # ========== 构建训练组件 ==========

    # 构建模型
    # init_device="meta": 使用 "meta" 设备初始化模型（不分配实际内存），节省显存
    # 实际的参数会在加载检查点或前向传播时分配内存
    model = config.model.build(init_device="meta")

    # 构建训练模块
    # 训练模块封装了模型，并添加了训练所需的功能：
    # - 前向传播和损失计算
    # - 梯度累积
    # - 优化器集成
    # - 数据并行（DP）进程组
    train_module = config.train_module.build(model)

    # 构建数据集
    # 根据配置创建数据集实例，负责：
    # - 读取数据文件
    # - 数据预处理（tokenization、padding、truncation 等）
    # - 提供数据访问接口
    dataset = config.dataset.build()

    # 构建数据加载器
    # 数据加载器负责：
    # - 批量数据加载（batching）
    # - 数据预处理和转换
    # - 多进程数据加载
    # - 分布式数据采样（确保各进程获得不同的数据）
    # dp_process_group: 数据并行进程组，用于确保各进程加载不同的数据分片
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)

    # 构建训练器
    # 训练器是训练流程的核心控制器，负责：
    # - 管理训练循环（epoch、iteration）
    # - 前向和反向传播
    # - 优化器步骤和梯度裁剪
    # - 学习率调度
    # - 检查点保存和加载
    # - 日志记录（W&B、TensorBoard）
    # - 回调管理（早停、混合精度训练等）
    trainer = config.trainer.build(train_module, data_loader)

    # ========== 保存配置信息 ==========

    # 将配置保存到 W&B 和每个检查点目录
    # 这样可以在训练后复现实验配置
    for callback in trainer.callbacks.values():
        if isinstance(callback, ConfigSaverCallback):
            # 将 ExperimentConfig 转换为字典格式并赋值给回调
            # ConfigSaverCallback 会在保存检查点时将配置一起保存
            callback.config = config.as_config_dict()
            break

    # ========== 加载检查点（如果有） ==========

    # 如果设置了加载路径且保存文件夹中没有找到检查点，则从加载路径加载检查点
    # 这是一个容错机制：
    # 1. 先尝试从保存文件夹加载检查点（maybe_load_checkpoint）
    # 2. 如果没有找到且配置了 load_path，则从 load_path 加载
    # 3. load_trainer_state=False: 只加载模型和优化器状态，不加载训练器状态（如 iteration、epoch）
    #    这通常用于从不同的检查点继续训练
    if not trainer.no_checkpoints and not trainer.maybe_load_checkpoint() and config.load_path:
        log.info(
            f"Loading checkpoint from {config.load_path} since no checkpoints were found in the save folder..."
        )
        trainer.load_checkpoint(config.load_path, load_trainer_state=False)

    # ========== 开始训练 ==========

    # 启动训练循环
    # trainer.fit() 会：
    # 1. 遍历数据集进行多轮训练（epochs）
    # 2. 在每个 iteration 中执行前向传播、计算损失、反向传播、更新参数
    # 3. 定期保存检查点
    # 4. 定期进行评估（如果有验证集）
    # 5. 记录训练指标到日志
    trainer.fit()

    # ========== 清理分布式训练环境 ==========

    # 清理分布式训练后端
    # 包括：
    # - 销毁进程组
    # - 释放分布式通信资源
    # - 清理 CUDA 缓存等
    teardown_training_environment()


def get_lr_from_checkpoint(
    path: PathOrStr, param: Optional[str] = None, param_group: Optional[int] = None
) -> float:
    """
    从检查点中提取学习率。

    该函数用于从已保存的检查点中获取优化器的学习率。根据检查点中优化器状态的存储格式
   （flattened 或 unflattened），需要提供不同的参数来定位学习率。

    Args:
        path: 检查点路径，可以是完整路径或检查点目录的根路径
        param: 模型参数名称（如 "embeddings.weight"），仅在 flattened 格式下使用
               如果未指定，默认使用 "embeddings.weight"
        param_group: 参数组索引（如 0, 1, 2...），仅在 unflattened 格式下使用
                     如果未指定，默认使用第 0 个参数组

    Returns:
        float: 检查点中保存的学习率值

    Raises:
        RuntimeError: 当检查点使用 flattened 格式但指定了 param_group 时抛出

    工作流程：
        1. 规范化路径，确保指向 model_and_optim 子目录
        2. 检查检查点元数据，判断优化器状态是 flattened 还是 unflattened 格式
        3. 根据格式类型选择正确的键来获取学习率：
           - unflattened 格式：使用 f"optim.param_groups.{param_group}.lr"
           - flattened 格式：使用 f"optim.param_groups.{param}.lr"
        4. 加载指定键的值并返回

    Note:
        - flattened 格式：优化器状态按参数平铺，每个参数有独立的学习率
        - unflattened 格式：优化器状态按参数组组织，参数组内所有参数共享同一学习率
    """
    # 规范化路径，确保路径格式统一
    path = normalize_path(path)
    # 如果路径不以 "/model_and_optim" 结尾，则添加该后缀
    # 检查点通常包含 model_and_optim 子目录，存储模型和优化器状态
    if not path.endswith("/model_and_optim"):
        path = join_path(path, "model_and_optim")

    # 获取检查点元数据，用于检查状态字典的结构
    metadata = get_checkpoint_metadata(path)

    # 检查优化器状态是否为 unflattened 格式
    # unflattened 格式的特征是存在 "optim.param_groups.0.params" 键
    if "optim.param_groups.0.params" in metadata.state_dict_metadata:
        # unflattened 格式：学习率按参数组存储
        if param is not None:
            log.warning(
                "'param' will be ignored since the optimizer state in the checkpoint to load is in unflattened format"
            )
        # 如果未指定参数组，默认使用第 0 个参数组
        if param_group is None:
            param_group = 0
        # 构建学习率键：optim.param_groups.{param_group}.lr
        key = f"optim.param_groups.{param_group}.lr"
    else:
        # flattened 格式：学习率按参数存储
        # 在这种格式下指定 param_group 是无效的，会抛出错误
        if param_group is not None:
            raise RuntimeError(
                "'param_group' is required since the optimizer state in the checkpoint to load is in flattened format"
            )
        # 如果未指定参数名称，默认使用 embeddings.weight
        if param is None:
            param = "embeddings.weight"
        # 构建学习率键：optim.param_groups.{param}.lr
        key = f"optim.param_groups.{param}.lr"

    # 创建只包含学习率键的状态字典，用于选择性加载
    state_dict = {key: None}
    # 从检查点加载学习率值
    load_state_dict(path, state_dict)
    # 确保学习率已成功加载
    assert state_dict[key] is not None
    # 返回学习率的浮点数值
    return float(state_dict[key])  # type: ignore
