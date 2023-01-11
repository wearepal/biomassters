from typing import Any, Optional, Protocol

import attr
from conduit.types import Stage
from torch import Tensor
import torch.nn.functional as F
from typing_extensions import override

from src.algorithms.base import Algorithm
from src.loss import stable_mse_loss
from src.types import TrainSample
from src.utils import to_item

__all__ = ["Erm"]


class _Loss(Protocol):
    def __call__(self, input: Tensor, *, target: Tensor) -> Tensor:
        ...


@attr.define(kw_only=True, eq=False)
class Erm(Algorithm):
    _loss_fn: Optional[_Loss] = None

    @property
    def loss_fn(self) -> _Loss:
        if self._loss_fn is None:
            return stable_mse_loss
        return self._loss_fn

    @override
    def training_step(self, batch: TrainSample, batch_idx: int) -> Tensor:
        logits = self.forward(batch["image"])
        loss = self.loss_fn(input=logits, target=batch["label"])
        results_dict = {f"{str(Stage.FIT)}/batch_loss": to_item(loss)}
        self.log_dict(results_dict)

        return loss
