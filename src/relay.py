from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from conduit.progress import CdtProgressBar
from hydra.utils import instantiate
import loguru
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from ranzen.hydra import Option, Options, Relay
import torch.nn as nn
from torch.types import Number
from typing_extensions import override

from src.algorithms.base import Algorithm
from src.conf import WandbLoggerConf
from src.data import DenormalizeModule, SentinelDataModule
from src.models import ModelFactory

__all__ = ["SentinelRelay"]


@dataclass(unsafe_hash=True)
class SentinelRelay(Relay):
    dm: DictConfig
    alg: DictConfig
    model: DictConfig
    trainer: DictConfig
    logger: DictConfig
    checkpointer: DictConfig

    score: bool = False
    seed: Optional[int] = 0
    progbar_theme: CdtProgressBar.Theme = CdtProgressBar.Theme.CYBERPUNK
    pred_dir: Optional[Path] = None

    @classmethod
    @override
    def with_hydra(
        cls,
        root: Union[Path, str],
        *,
        alg: Options[Algorithm],
        model: Options[ModelFactory],
        clear_cache: bool = False,
    ) -> None:
        configs = dict(
            alg=alg,
            checkpointer=[Option(class_=ModelCheckpoint, name="base")],
            dm=[Option(class_=SentinelDataModule, name="base")],
            logger=[Option(class_=WandbLoggerConf, name="base")],
            model=model,
            trainer=[Option(class_=pl.Trainer, name="base")],
        )
        super().with_hydra(
            root=root,
            instantiate_recursively=False,
            clear_cache=clear_cache,
            **configs,
        )

    @override
    def run(self, raw_config: Dict[str, Any]) -> Optional[Number]:
        loguru.logger.info(f"Current working directory: '{os.getcwd()}'")
        pl.seed_everything(self.seed, workers=True)
        dm: SentinelDataModule = instantiate(self.dm)
        dm.prepare_data()
        dm.setup()

        model_fn: ModelFactory = instantiate(self.model)
        model = model_fn(in_channels=dm.in_channels(model_fn.IS_TEMPORAL))
        # Merge the temporal dimension with the channeld dimension if the
        # model is not spatiotemporal.
        if not model_fn.IS_TEMPORAL:
            model = nn.Sequential(nn.Flatten(start_dim=1, end_dim=2), model)
        model = nn.Sequential(model, DenormalizeModule(*dm.target_normalizers))

        if self.logger.get("group", None) is None:
            default_group = "_".join(
                dict_conf["_target_"].split(".")[-1].lower() for dict_conf in (self.model, self.alg)
            )
            self.logger["group"] = default_group
        logger: WandbLogger = instantiate(self.logger, reinit=True)

        # pred_dir_full = self.pred_dir
        # if self.pred_dir is not None:
        #     # Guard against multiprocessing shenaningans
        #         if logger._offline:
        #             pred_subdir = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        #         else:
        #             pred_subdir = logger.experiment.name
        #         pred_dir_full = self.pred_dir / pred_subdir
        #         loguru.logger.info(f"Predictions will be saved to '{pred_dir_full.resolve()}'.")
        #         raw_config["path_to_predictions"] = str(pred_dir_full.resolve())

        logger.log_hyperparams(raw_config)  # type: ignore
        progbar = CdtProgressBar(theme=self.progbar_theme)
        # checkpointer: ModelCheckpoint = instantiate(self.checkpointer)
        # trainer_callbacks = [checkpointer, progbar]
        trainer_callbacks = [progbar]

        # Set up and execute the training algorithm.
        trainer: pl.Trainer = instantiate(
            self.trainer,
            logger=logger,
            callbacks=trainer_callbacks,
        )
        alg: Algorithm = instantiate(self.alg)
        alg.run(dm=dm, model=model, trainer=trainer, pred_dir=self.pred_dir)

        if self.score:
            loguru.logger.info("Scoring model.")
            scores = trainer.test(model=alg, dataloaders=dm.val_dataloader())[0]
            return sum(scores.values())
