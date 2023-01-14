import warnings

from ranzen.hydra import Option
import torch.multiprocessing

from src.algorithms import Erm
from src.models import Unet3dVdFn, UnetFn, UnetPlusPlusFn
from src.relay import SentinelRelay

torch.multiprocessing.set_sharing_strategy("file_system")
TO_IGNORE = (
    "torch.nn.modules.module",
    "torch.distributed.distributed_c10d",
    "pytorch_lightning.trainer.connectors.data_connector",
    "lightning_lite.plugins.environments",
)
for module in TO_IGNORE:
    warnings.filterwarnings("ignore", module=module)

if __name__ == "__main__":
    alg_ops = [
        Option(Erm, "erm"),
    ]
    model_ops = [
        Option(Unet3dVdFn, "unet3d"),
        Option(UnetFn, "unet"),
        Option(UnetPlusPlusFn, "unetpp"),
    ]

    SentinelRelay.with_hydra(
        root="conf",
        alg=alg_ops,
        model=model_ops,
        clear_cache=True,
    )
