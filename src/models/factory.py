from dataclasses import dataclass
from typing import Generic, Optional, Tuple, TypeVar

import segmentation_models_pytorch as smp  # type: ignore
import torch.nn as nn
from typing_extensions import override

from src.models.unet3d_vd import Unet3dVd

__all__ = [
    "ModelFactory",
    "UnetFn",
    "UnetPlusPlusFn",
    "Unet3dVdFn",
]

M = TypeVar("M", bound=nn.Module)


@dataclass(unsafe_hash=True)
class ModelFactory(Generic[M]):
    def __call__(self, in_channels: int) -> M:
        ...


@dataclass(unsafe_hash=True)
class UnetFn(ModelFactory[smp.Unet]):
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
            activation=None,
        )


@dataclass(unsafe_hash=True)
class UnetPlusPlusFn(ModelFactory[smp.UnetPlusPlus]):
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
            activation=None,
        )


@dataclass(unsafe_hash=True)
class Unet3dVdFn(ModelFactory[Unet3dVd]):
    dim: int = 64
    multipliers: Tuple[int, ...] = (1, 2, 4, 8)
    n_attn_heads: int = 8
    attn_head_dim: int = 32
    init_dim: Optional[int] = None
    init_kernel_size: int = 7
    use_sparse_linear_attn: bool = False
    apply_mid_spatial_attn: bool = True
    resnet_groups: int = 8
    spatial_decoder: bool = True

    @override
    def __call__(self, in_channels):
        return Unet3dVd(
            in_channels=in_channels,
            out_channels=1,
            dim=self.dim,
            multipliers=self.multipliers,
            n_attn_heads=self.n_attn_heads,
            attn_head_dim=self.attn_head_dim,
            init_dim=self.init_dim,
            init_kernel_size=self.init_kernel_size,
            use_sparse_linear_attn=self.use_sparse_linear_attn,
            apply_mid_spatial_attn=self.apply_mid_spatial_attn,
            spatial_decoder=self.spatial_decoder,
            resnet_groups=self.resnet_groups,
        )
