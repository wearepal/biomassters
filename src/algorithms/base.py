from itertools import product
from pathlib import Path
import shutil
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
    final,
)

from PIL import Image
from conduit.data.structures import TernarySample
from conduit.types import LRScheduler, Stage
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info  # type: ignore
from pytorch_lightning.utilities.rank_zero import rank_zero_only  # type: ignore
from pytorch_lightning.utilities.types import STEP_OUTPUT
from ranzen.torch.data import TrainingMode
import torch
from torch import Tensor, optim
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.types import Number
from typing_extensions import Self, TypeAlias, override

from src.data import SentinelDataModule, SentinelDataset
from src.loss import stable_mse_loss
from src.models import ModelPipeline
from src.types import ImageSample, TestSample, TrainSample
from src.utils import some, to_item, to_numpy, to_targz

__all__ = [
    "Algorithm",
    "MetricDict",
]

MetricDict: TypeAlias = Dict[str, Number]


def filter_params(
    named_params: Iterable[Tuple[str, Parameter]],
    weight_decay: float = 0.0,
    exclusion_patterns: Tuple[str, ...] = ("bias",),
) -> List[Dict[str, Union[List[Parameter], float]]]:
    params: List[Parameter] = []
    excluded_params: List[Parameter] = []

    for name, param in named_params:
        if not param.requires_grad:
            continue
        elif any(pattern in name for pattern in exclusion_patterns):
            excluded_params.append(param)
        else:
            params.append(param)

    return [
        {"params": params, "weight_decay": weight_decay},
        {
            "params": excluded_params,
            "weight_decay": 0.0,
        },
    ]


S = TypeVar("S", bound=ImageSample)


class Algorithm(pl.LightningModule):
    model: ModelPipeline

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
        predict_with_best: bool = True,
        ckpt_path: Optional[Path] = None,
        pred_dir: Optional[Path] = None,
        tta: bool = False,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_kwargs
        self.lr_sched_freq = lr_sched_freq
        self.test_on_best = test_on_best
        self.predict_with_best = predict_with_best
        self.pred_dir = pred_dir
        self.ckpt_path = ckpt_path if ckpt_path is None else str(ckpt_path.resolve())
        self.tta = tta

    @override
    def training_step(
        self,
        batch: TernarySample[Tensor],
        batch_idx: int,
    ) -> STEP_OUTPUT:
        ...

    @torch.no_grad()
    def eval_step(self, batch: TrainSample) -> Tensor:
        preds = (
            self._forward_with_tta(batch)
            if self.tta
            else self.forward(batch["image"], mask=batch["mask"])
        )
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
        return {f"{str(stage)}/RMSE": to_item(rmse)}

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
        x = batch["image"]
        preds = self._forward_with_tta(batch) if self.tta else self.forward(x, mask=batch["mask"])
        preds_np = to_numpy(
            preds.to(torch.float32).view(-1, SentinelDataset.RESOLUTION, SentinelDataset.RESOLUTION)
        )
        for pred, chip in zip(preds_np, batch["chip"]):
            im = Image.fromarray(pred)
            assert im.size == (256, 256)
            assert some(self.pred_dir)
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
        if some(self.optimizer_kwargs):
            optimizer_config.update(self.optimizer_kwargs)

        params = filter_params(self.named_parameters(), weight_decay=self.weight_decay)
        optimizer = instantiate(optimizer_config, _partial_=True)(params=params, lr=self.lr)

        if some(self.scheduler_cls):
            scheduler_config = DictConfig({"_target_": self.scheduler_cls})
            if some(self.scheduler_kwargs):
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
    def forward(self, x: Tensor, *, mask: Tensor) -> Tensor:
        return self.model.forward(x, mask=mask)

    def _forward_with_tta(self, inputs: ImageSample) -> Tensor:
        """
        Forward method with test-time augmentation (TTA), where
        the augmentation takes the form of all transformations from
        the D4 group.
        """
        preds_avg = 0
        # Average the predictions over the orbit generated by the D4 group.
        combinations = list(product([True, False], [0, 1, 2, 3]))
        for flip, k in combinations:
            x_aug = inputs["image"]
            # Apply horizontal fliping.
            if flip:
                x_aug = torch.flip(x_aug, dims=(-1,))
            # Apply a k*90-degree rotation.
            x_aug = torch.rot90(x_aug, k=k, dims=(-2, -1))
            preds = self.forward(x_aug, mask=inputs["mask"])
            # Apply the inverse transforms to the predictions so that
            # they're in the canonical orientation.
            preds = torch.rot90(preds, k=-k, dims=(-2, -1))
            if flip:
                preds = torch.flip(preds, dims=(-1,))
            preds_avg += preds
        preds_avg /= len(combinations)
        preds_avg = cast(Tensor, preds_avg)
        return preds_avg

    def _run_internal(
        self, dm: SentinelDataModule, *, trainer: pl.Trainer, test: bool = True
    ) -> Self:
        if some(self.ckpt_path):
            # We do this to avoid loading the trainer/scheduler/optimizer states,
            # which is usually the desired behaviour due to learning-rate scheduling.
            rank_zero_info(f"Loading model weights from checkpoint '{self.ckpt_path}'")
            state_dict = torch.load(self.ckpt_path)["state_dict"]
            # Excise the nuisance "model" prefix.
            state_dict = {name.removeprefix("model."): param for name, param in state_dict.items()}
            self.model.load_state_dict(state_dict)
        # Train the model
        trainer.fit(model=self, datamodule=dm)
        if test:
            # Test the model if desired
            trainer.test(
                model=self,
                ckpt_path="best" if self.test_on_best else None,
                datamodule=dm,
            )
        if some(self.pred_dir):
            self.pred_dir.mkdir(parents=True, exist_ok=True)
            rank_zero_info(
                f"Generating predictions and saving them to '{self.pred_dir.resolve()}'."
            )
            try:
                trainer.predict(
                    model=self, datamodule=dm, ckpt_path="best" if self.predict_with_best else None
                )
            # setting ckpt_path to 'best' without any validation loops having been performed
            # will result in an error
            except ValueError:
                trainer.predict(model=self, datamodule=dm)
            except KeyboardInterrupt:
                shutil.rmtree(self.pred_dir)

        return self

    @rank_zero_only
    def _archive_predictions(self):
        if some(self.pred_dir):
            # Tar pred_dir so that it's in a submission-ready state.
            self.print("Archiving the generated predictions so they're ready-for-submission.")
            output_path = to_targz(source=self.pred_dir)
            self.print(f"Predictions archived to {output_path.resolve()}")

    def _setup(self, *, dm: SentinelDataModule, model: ModelPipeline) -> None:
        self.model = model

    @final
    def run(
        self,
        dm: SentinelDataModule,
        *,
        model: ModelPipeline,
        trainer: pl.Trainer,
        test: bool = True,
    ) -> Self:
        self._setup(dm=dm, model=model)
        return self._run_internal(dm=dm, trainer=trainer, test=test)
