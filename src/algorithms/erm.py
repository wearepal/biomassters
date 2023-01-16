from pathlib import Path
from typing import Optional, Protocol

from conduit.types import Stage
from omegaconf.dictconfig import DictConfig
from torch import Tensor
from typing_extensions import override

from src.algorithms.base import Algorithm
from src.loss import stable_mse_loss
from src.types import TrainSample
from src.utils import to_item

__all__ = ["Erm"]


class _Loss(Protocol):
    def __call__(self, input: Tensor, *, target: Tensor) -> Tensor:
        ...


class Erm(Algorithm):
    def __init__(
        self,
        *,
        lr: float = 5.0e-5,
        weight_decay: float = 0.0,
        optimizer_cls: str = "torch.optim.AdamW",
        optimizer_kwargs: Optional[DictConfig] = None,
        scheduler_cls: Optional[str] = None,
        scheduler_kwargs: Optional[DictConfig] = None,
        lr_sched_freq: int = 1,
        test_on_best: bool = False,
        loss_fn: Optional[_Loss] = None,
        ckpt_path: Optional[Path] = None,
        pred_dir: Optional[Path] = None,
    ) -> None:
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_cls=scheduler_cls,
            scheduler_kwargs=scheduler_kwargs,
            lr_sched_freq=lr_sched_freq,
            test_on_best=test_on_best,
            ckpt_path=ckpt_path,
            pred_dir=pred_dir,
        )
        self._loss_fn = loss_fn

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
