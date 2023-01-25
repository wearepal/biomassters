import contextlib
import copy
import os
import threading
from typing import Any, Dict, Generator, Optional, Tuple

import pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.exceptions import (
    MisconfigurationException,  # type: ignore
)
from pytorch_lightning.utilities.rank_zero import rank_zero_info  # type: ignore
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import Tensor
from typing_extensions import TypeAlias, override  # type: ignore

from src.utils import some

__all__ = ["EMA", "EMACheckpointer"]


TensorTuple: TypeAlias = Tuple[Tensor, ...]


@torch.no_grad()
def ema_update(
    ema_model_tuple: TensorTuple, current_model_tuple: TensorTuple, decay: float
) -> None:
    torch._foreach_mul_(ema_model_tuple, decay)
    torch._foreach_add_(
        ema_model_tuple,
        current_model_tuple,
        alpha=(1.0 - decay),
    )


def run_ema_update_cpu(
    ema_model_tuple: TensorTuple,
    current_model_tuple: TensorTuple,
    decay: float,
    pre_sync_stream: Optional[torch.cuda.Stream] = None,
) -> None:
    if some(pre_sync_stream):
        pre_sync_stream.synchronize()
    ema_update(ema_model_tuple, current_model_tuple, decay)


class EMA(Callback):
    """
    Implements Exponential Moving Averaging (EMA).
    When training a model, this callback will maintain moving averages of the trained parameters.
    When evaluating, we use the moving averages copy of the trained parameters.
    When saving, we save an additional set of parameters with the prefix `ema`.
    """

    def __init__(
        self,
        decay: float,
        *,
        validate_original_weights: bool = False,
        every_n_steps: int = 1,
        cpu_offload: bool = False,
    ) -> None:
        """
        :param decay: The exponential decay used when calculating the moving average. Has to be between 0-1.
        :param validate_original_weights: Validate the original weights, as apposed to the EMA weights.
        :param every_n_steps: Apply EMA every N steps.
        :param cpu_offload: Offload weights to CPU.
        """
        if not (0 < decay < 1):
            raise MisconfigurationException("EMA decay value must be in the range (0, 1)")
        self.decay = decay
        self.validate_original_weights = validate_original_weights
        self.every_n_steps = every_n_steps
        self.cpu_offload = cpu_offload

        self.current_step = 0
        # Attributes to be initialized at the start of training:
        self.ema_params: Optional[Tuple[Tensor]] = None
        self.device: Optional[torch.device] = None
        self.thread: Optional[threading.Thread] = None
        self.stream: Optional[torch.cuda.Stream] = None

    def _initialize(self, pl_module: "pl.LightningModule") -> None:
        self.device = pl_module.device if not self.cpu_offload else torch.device("cpu")
        self.ema_params = tuple(
            copy.deepcopy(param.data.detach()).to(self.device) for param in pl_module.parameters()
        )
        if any(p.is_cuda for p in pl_module.parameters()):
            self.stream = torch.cuda.Stream()

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._initialize(pl_module)

    def swap_tensors(self, tensor1: Tensor, tensor2: Tensor) -> None:
        tmp = torch.empty_like(tensor1)
        tmp.copy_(tensor1)
        tensor1.copy_(tensor2)
        tensor2.copy_(tmp)

    def swap_model_weights(
        self, pl_module: "pl.LightningModule", saving_ema_model: bool = False
    ) -> None:
        assert some(self.ema_params)
        self.in_saving_ema_model_context = saving_ema_model
        for param, ema_param in zip(pl_module.parameters(), self.ema_params):
            self.swap_tensors(param.data, ema_param)

    @override
    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(pl_module=pl_module)

    @override
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(pl_module)

    @override
    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(pl_module)

    @override
    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(pl_module)

    @override
    def on_predict_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(pl_module)

    @override
    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(pl_module)

    def _should_validate_ema_weights(self, trainer: "pl.Trainer") -> bool:
        return (not self.validate_original_weights) and some(self.ema_params)

    @override
    def on_load_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> None:
        super().on_load_checkpoint
        checkpoint_callback = trainer.checkpoint_callback
        connector = trainer._checkpoint_connector
        ckpt_path = connector.resume_checkpoint_path

        if some(ckpt_path) and isinstance(checkpoint_callback, EMACheckpointer):
            ckpt_path = str(ckpt_path)
            ext = checkpoint_callback.FILE_EXTENSION
            if ckpt_path.endswith(f"-EMA{ext}"):
                rank_zero_info(
                    "loading EMA based weights. "
                    "The callback will treat the loaded EMA weights as the main weights"
                    " and create a new EMA copy when training."
                )
                return
            ema_path = ckpt_path.replace(ext, f"-EMA{ext}")
            if os.path.exists(ema_path):
                ema_state_dict = torch.load(ema_path, map_location=torch.device("cpu"))
                checkpoint["optimizer_states"] = ema_state_dict["optimizer_states"]
                del ema_state_dict
                rank_zero_info("EMA state has been restored.")
            else:
                raise MisconfigurationException(
                    "Unable to find the associated EMA weights when re-loading, "
                    f"training will start with new EMA weights. Expected them to be at: {ema_path}",
                )

    @contextlib.contextmanager
    def save_ema_model(self, trainer: "pl.Trainer") -> Generator[None, None, None]:
        """
        Saves an EMA copy of the model + EMA optimizer states for resume.
        """
        pl_module = trainer.lightning_module
        self.swap_model_weights(pl_module, saving_ema_model=True)
        try:
            yield
        finally:
            self.swap_model_weights(pl_module, saving_ema_model=False)

    def join(self) -> None:
        if some(self.stream):
            self.stream.synchronize()
        if some(self.thread):
            self.thread.join()

    def _should_update_at_step(self) -> bool:
        return self.current_step % self.every_n_steps == 0

    @torch.no_grad()
    def update(self, pl_module: "pl.LightningModule") -> None:
        assert some(self.ema_params)
        assert some(self.device)

        if some(self.stream):
            self.stream.wait_stream(torch.cuda.current_stream())  # type: ignore

        with torch.cuda.stream(self.stream):  # type: ignore
            current_model_state = tuple(
                param.data.to(self.device, non_blocking=True) for param in pl_module.parameters()
            )
            if self.device.type == "cuda":
                ema_update(self.ema_params, current_model_state, self.decay)

        if self.device.type == "cpu":
            self.thread = threading.Thread(
                target=run_ema_update_cpu,
                args=(
                    self.ema_params,
                    current_model_state,
                    self.decay,
                    self.stream,
                ),
            )
            self.thread.start()

    @override
    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self._should_update_at_step():
            self.update(pl_module)
        self.current_step += 1


class EMACheckpointer(ModelCheckpoint):
    def _ema_callback(self, trainer: "pytorch_lightning.Trainer") -> Optional[EMA]:
        ema_callback = None
        for callback in trainer.callbacks:  # type: ignore
            if isinstance(callback, EMA):
                ema_callback = callback
                break
        return ema_callback

    def _ema_format_filepath(self, filepath: str) -> str:
        return filepath.replace(self.FILE_EXTENSION, f"-EMA{self.FILE_EXTENSION}")

    @override
    def _save_checkpoint(self, trainer: "pytorch_lightning.Trainer", filepath: str) -> None:
        ema_callback = self._ema_callback(trainer)
        if some(ema_callback):
            super()._save_checkpoint(trainer, filepath)
            # save EMA copy of the model as well.
            with ema_callback.save_ema_model(trainer):
                filepath = self._ema_format_filepath(filepath)
                if self.verbose:
                    rank_zero_info(f"Saving EMA weights to separate checkpoint {filepath}")
                super()._save_checkpoint(trainer, filepath)
        else:
            super()._save_checkpoint(trainer, filepath)
