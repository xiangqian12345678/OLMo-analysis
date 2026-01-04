from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Optional, Tuple

import torch

from ..config import StrEnum
from ..data.utils import get_labels
from ..utils import format_timedelta

TRAIN_CE_LOSS_METRIC = "train/CE loss"
TRAIN_PPL_METRIC = "train/PPL"
TRAIN_Z_LOSS_METRIC = "train/Z loss"


class DurationUnit(StrEnum):
    """
    Units that can be used to define a :class:`Duration`.
    """

    steps = "steps"
    """
    Steps (batches).
    """
    epochs = "epochs"
    """
    Epochs.
    """
    tokens = "tokens"
    """
    Tokens.
    """


@dataclass
class Duration:
    value: int
    """
    The value of the duration.
    """
    unit: DurationUnit
    """
    The unit associated with the :data:`value`.
    """

    @classmethod
    def steps(cls, steps: int) -> "Duration":
        """
        Define a duration from a number of steps.
        """
        return cls(value=steps, unit=DurationUnit.steps)

    @classmethod
    def epochs(cls, epochs: int) -> "Duration":
        """
        Define a duration from a number of epochs.
        """
        return cls(value=epochs, unit=DurationUnit.epochs)

    @classmethod
    def tokens(cls, tokens: int) -> "Duration":
        """
        Define a duration from a number of tokens.
        """
        return cls(value=tokens, unit=DurationUnit.tokens)

    @classmethod
    def chinchilla_tokens(
        cls, multiple: float, *, model_params: int, _tok_per_param: int = 20
    ) -> "Duration":
        """
        Define a duration based on a multiple of the Chinchilla-optimal number of tokens.

        The rule of thumb for Chinchilla compute optimality is 20 tokens-per-parameter
        for decoder-only natural language models trained with AdamW on dataset mixtures
        similar to the Pile.

        Chinchilla optimality refers to training-time compute only, and does not account for
        inference-time compute. In practice, models are often trained with more tokens than
        the Chinchilla optimal value ("overtrained") to improve inference-time performance.

        Chinchilla: https://arxiv.org/abs/2203.15556
        Chinchilla replication: https://arxiv.org/abs/2404.10102

        :param multiple: The Chinchilla multiplier. 1.0 is the Chinchilla optimal value.
            Values less than 1.0 will undertrain relative to Chinchilla, and values greater
            than 1.0 will overtrain relative to Chinchilla.
        :param model_params: The number of *active, non-embedding* parameters in the target model.
        """
        tokens = int(_tok_per_param * model_params * multiple)
        return Duration.tokens(tokens)

    def due(self, *, step: int, tokens: int, epoch: int) -> bool:
        """
        检查是否达到预定的持续时间/阈值
        参数:
            step: 当前训练步数（training steps）
            tokens: 当前处理的 token 总数
            epoch: 当前训练轮数（epochs）
        返回:
            bool: True 表示已达到阈值，False 表示未达到
        注意:
            使用关键字参数（*）强制调用者明确指定参数名
        """
        # 如果单位是步数，检查当前步数是否达到目标值
        if self.unit == DurationUnit.steps:
            return step >= self.value

        # 如果单位是 token 数，检查当前 token 总数是否达到目标值
        elif self.unit == DurationUnit.tokens:
            return tokens >= self.value

        # 如果单位是轮数，检查当前轮数是否超过目标值
        # 注意: 这里用 > 而不是 >=，表示在完成该轮数后才算达到
        elif self.unit == DurationUnit.epochs:
            return epoch > self.value

        # 未知单位类型
        else:
            raise NotImplementedError


class LoadStrategy(StrEnum):
    """
    Determines the strategy for loading checkpoints prior to training.
    """

    if_available = "if_available"
    """
    The trainer will attempt to load a checkpoint from the save folder or load path (in that order)
    but will train from scratch if no checkoint is found.
    """

    always = "always"
    """
    The trainer will attempt to load a checkpoint from the save folder or load path (in that order)
    and raise an error if no checkpoint is found.
    """

    never = "never"
    """
    The trainer will never load a checkpoint even if one exists in the save folder or load path.
    """


class ReduceType(StrEnum):
    """
    An enumeration of the allowed ways to reduce a metric across ranks.
    """

    mean = "mean"
    """
    Average across the process group.
    """

    sum = "sum"
    """
    Add across the process group.
    """

    max = "max"
    """
    Take the max across the process group.
    """

    l2_norm = "l2_norm"
    """
    For metrics that are computed as L2 norms on each rank, this will correctly reduce the norm
    across the process group to produce the global L2 norm.
    """


class MetricMergeStrategy(StrEnum):
    """
    Determines how duplicate metrics are merged.
    """

    warn = "warn"
    """
    Warn when a duplicate is logged, keeping the current value.
    """

    latest = "latest"
    """
    The latest is used.
    """

    oldest = "oldest"
    """
    The oldest (first logged) is used.
    """

    mean = "mean"
    """
    When a duplicate is logged we take the average with the last value.
    """

    sum = "sum"
    """
    The sum of the duplicates is used.
    """

    max = "max"
    """
    Take the maximum value of the duplicates.
    """

    min = "min"
    """
    Take the minimum value of the duplicates.
    """


def reshape_inputs_for_loss(
    logits: torch.Tensor, labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # shape: (B * S, V)
    logits_for_loss = logits.view(-1, logits.size(-1))
    # shape: (B, S) -> (B * S,)
    labels_for_loss = labels.view(-1)
    return logits_for_loss, labels_for_loss


def get_inputs_for_loss(
    batch: Dict[str, Any], logits: torch.Tensor, label_ignore_index: int = -100
) -> Tuple[torch.Tensor, torch.Tensor]:
    return reshape_inputs_for_loss(
        logits, batch.get("labels", get_labels(batch, label_ignore_index=label_ignore_index))
    )


@dataclass
class TrainingProgress:
    current_step: int
    """
    The current training step.
    """
    total_steps: Optional[int] = None
    """
    The step that training will stop at.
    """
    time_remaining: Optional[timedelta] = None
    """
    Estimated time remaining.
    """

    def __str__(self) -> str:
        if self.total_steps is not None:
            progress_perc = min(100, int(100 * self.current_step / self.total_steps))
            progress_str = (
                f"{progress_perc}% complete (step {self.current_step:,d}/{self.total_steps:,d})"
            )
        else:
            progress_str = f"step {self.current_step:,d}/???"
        if self.time_remaining is not None:
            progress_str += f", eta {format_timedelta(self.time_remaining)}"
        return progress_str


@dataclass
class StepSkipRange:
    """Defines a range of steps to skip during training."""

    start: int
    """The first step to skip (steps start at 1, not 0)."""
    stop: int
    """The endpoint of the range (exclusive)."""
