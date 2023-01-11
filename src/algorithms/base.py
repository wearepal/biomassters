from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union, final

from PIL import Image
import attr
from conduit.data.structures import TernarySample
from conduit.types import LRScheduler, Stage
from hydra.utils import instantiate
from loguru import logger
from omegaconf.dictconfig import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from ranzen.torch.data import TrainingMode
import torch
from torch import Tensor, optim
import torch.nn as nn
from torch.types import Number
from typing_extensions import Self, TypeAlias, override

from src.data import SentinelDataModule
from src.loss import stable_mse_loss
from src.types import TestSample, TrainSample
from src.utils import to_item, to_numpy

__all__ = [
    "Algorithm",
    "MetricDict",
]

MetricDict: TypeAlias = Dict[str, Number]


@attr.define(kw_only=True, eq=False)
class Algorithm(pl.LightningModule):
    model: nn.Module = attr.field(init=False)
    pred_dir: Optional[Path] = attr.field(default=None, init=False)
    lr: float = 5.0e-5
    weight_decay: float = 0.0
    optimizer_cls: str = "torch.optim.AdamW"
    optimizer_kwargs: Optional[DictConfig] = None
    scheduler_cls: Optional[str] = None
    scheduler_kwargs: Optional[DictConfig] = None
    lr_sched_freq: int = 1
    test_on_best: bool = False

    def __new__(cls: type[Self], *args: Any, **kwargs: Any) -> Self:
        obj = object.__new__(cls)
        pl.LightningModule.__init__(obj)
        return obj

    def training_step(
        self,
        batch: TernarySample[Tensor],
        batch_idx: int,
    ) -> STEP_OUTPUT:
        ...

    @torch.no_grad()
    def eval_step(self, batch: TrainSample) -> Tensor:
        preds = self.forward(batch["image"])
        # Collect the sample-wise mean-squared errors in the outputs
        return stable_mse_loss(
            # double precision is required here to avoid overflow when handling
            # the unnormalised targets.
            input=preds.to(torch.double),
            target=batch["label"].to(torch.double),
            samplewise=True,
        ).cpu()

    @override
    @torch.no_grad()
    def validation_step(
        self,
        batch: TrainSample,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> Tensor:
        return self.eval_step(batch=batch)

    def _eval_epoch_end(self, outputs: List[Tensor], *, stage: Stage) -> MetricDict:
        mse = torch.cat(outputs, dim=0).mean()
        rmse = mse.sqrt()
        return {f"{str(stage)}": to_item(rmse)}

    @override
    @torch.no_grad()
    def validation_epoch_end(self, outputs: List[Tensor]) -> None:
        results_dict = self._eval_epoch_end(outputs=outputs, stage=Stage.VALIDATE)
        self.log_dict(results_dict, sync_dist=True)

    @override
    @torch.no_grad()
    def test_step(
        self,
        batch: TrainSample,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> Tensor:
        return self.eval_step(batch=batch)

    @override
    @torch.no_grad()
    def test_epoch_end(self, outputs: List[Tensor]) -> None:
        results_dict = self._eval_epoch_end(outputs=outputs, stage=Stage.TEST)
        self.log_dict(results_dict, sync_dist=True)

    @override
    def predict_step(
        self, batch: TestSample[List[str]], batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> None:
        preds = self.forward(batch["image"])
        preds_np = to_numpy(preds)
        for pred, chip in zip(preds_np, batch["chip"]):
            im = Image.fromarray(pred)
            assert self.pred_dir is not None
            save_path = self.pred_dir / f"{chip}_agbm.tif"
            im.save(save_path, format="TIFF", save_all=True)

    @override
    def configure_optimizers(
        self,
    ) -> Union[
        Tuple[
            Union[
                list[optim.Optimizer],
                optim.Optimizer,
            ],
            list[Mapping[str, Union[LRScheduler, int, TrainingMode]]],
        ],
        list[optim.Optimizer],
        optim.Optimizer,
    ]:
        optimizer_config = DictConfig({"_target_": self.optimizer_cls})
        if self.optimizer_kwargs is not None:
            optimizer_config.update(self.optimizer_kwargs)

        optimizer = instantiate(
            optimizer_config,
            params=self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        if self.scheduler_cls is not None:
            scheduler_config = DictConfig({"_target_": self.scheduler_cls})
            if self.scheduler_kwargs is not None:
                scheduler_config.update(self.scheduler_kwargs)
            scheduler = instantiate(scheduler_config, optimizer=optimizer)
            scheduler_config = {
                "scheduler": scheduler,
                "interval": TrainingMode.step.name,
                "frequency": self.lr_sched_freq,
            }
            return [optimizer], [scheduler_config]
        return optimizer

    @override
    def forward(self, x: Any) -> Tensor:
        return self.model(x)

    def _run_internal(
        self, dm: SentinelDataModule, *, trainer: pl.Trainer, test: bool = True
    ) -> Self:
        # Run routines to tune hyperparameters before training.
        trainer.tune(model=self, datamodule=dm)
        # Train the model
        trainer.fit(model=self, datamodule=dm)
        if test:
            # Test the model if desired
            trainer.test(
                model=self,
                ckpt_path="best" if self.test_on_best else None,
                datamodule=dm,
            )
        if self.pred_dir is not None:
            self.pred_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Generating predictions and saving them to {self.pred_dir.resolve()}.")
            trainer.predict(model=self, datamodule=dm)

        return self

    def _setup(self, *, dm: SentinelDataModule, model: nn.Module, pred_dir: Optional[Path]) -> None:
        self.model = model

    @final
    def run(
        self,
        dm: SentinelDataModule,
        *,
        model: nn.Module,
        trainer: pl.Trainer,
        test: bool = True,
        pred_dir: Optional[Path] = None,
    ) -> Self:
        self._setup(dm=dm, model=model, pred_dir=pred_dir)
        return self._run_internal(dm=dm, trainer=trainer, test=test)
