
import segmentation_models_pytorch as smp # type: ignore
from dataclasses import dataclass
from typing import Generic, Tuple, TypeVar, Optional

import torch.nn as nn
from typing_extensions import override

__all__ = [
    "ModelFactory",
    "Unet",
    "UnetPlusPlus",
]

M = TypeVar("M", bound=nn.Module)


@dataclass(unsafe_hash=True)
class ModelFactory(Generic[M]):
    def __call__(self, in_channels: int) -> M:
        ...


@dataclass(unsafe_hash=True)
class Unet(ModelFactory[smp.Unet]):
    encoder_name: str = "resnet50"
    encoder_depth: int = 5
    encoder_weights: Optional[str] = None
    decoder_use_batchnorm: bool = True
    decoder_channels: Tuple[int, ...] = (256, 128, 64, 32, 16)
    decoder_attention: bool = True

    @override
    def __call__(self, in_channels):
        return smp.Unet(
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weights,
            in_channels=in_channels,                 
            classes=1,                     
            encoder_depth=self.encoder_depth,
            decoder_attention_type="scse" if self.decoder_attention else None,
            activation=None
        )
@dataclass(unsafe_hash=True)
class UnetPlusPlus(ModelFactory[smp.UnetPlusPlus]):
    encoder_name: str = "resnet50"
    encoder_depth: int = 5
    encoder_weights: Optional[str] = None
    decoder_use_batchnorm: bool = True
    decoder_channels: Tuple[int, ...] = (256, 128, 64, 32, 16)
    decoder_attention: bool = True

    @override
    def __call__(self, in_channels):
        return smp.UnetPlusPlus(
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weights,
            in_channels=in_channels,                 
            classes=1,                     
            encoder_depth=self.encoder_depth,
            decoder_attention_type="scse" if self.decoder_attention else None,
            activation=None
        )
