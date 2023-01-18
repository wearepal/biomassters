from typing import Tuple, TypeVar
import warnings

from omegaconf import OmegaConf
from ranzen.hydra import Option
import torch.multiprocessing

from src.algorithms import Erm
from src.models import Unet3dImagenFn, Unet3dVdFn, UnetFn, UnetPlusPlusFn
from src.relay import SentinelRelay

T = TypeVar("T")


def resolve_tuple(*args: T) -> Tuple[T, ...]:
    return tuple(args)


OmegaConf.register_new_resolver("tuple", resolve_tuple)
OmegaConf.register_new_resolver("eval", eval)

torch.multiprocessing.set_sharing_strategy("file_system")
TO_IGNORE = (
    "lightning_lite.plugins.environments",
    "pytorch_lightning.loops.epoch.prediction_epoch_loop",
    "pytorch_lightning.trainer.connectors.data_connector",
    "torch.distributed.distributed_c10d",
    "torch.nn.modules.module",
    "torch.optim.lr_scheduler",
)
for module in TO_IGNORE:
    warnings.filterwarnings("ignore", module=module)

if __name__ == "__main__":
    alg_ops = [
        Option(Erm, "erm"),
    ]
    model_ops = [
        Option(Unet3dVdFn, "unet3d"),
        Option(Unet3dImagenFn, "unet3d_imagen"),
        Option(UnetFn, "unet"),
        Option(UnetPlusPlusFn, "unetpp"),
    ]

    SentinelRelay.with_hydra(
        root="conf",
        alg=alg_ops,
        model=model_ops,
        clear_cache=True,
    )
