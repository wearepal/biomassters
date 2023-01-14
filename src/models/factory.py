from dataclasses import dataclass
from typing import ClassVar, Generic, Optional, Protocol, Tuple, TypeVar

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

M = TypeVar("M", bound=nn.Module, covariant=True)


@dataclass(unsafe_hash=True)
class ModelFactory(Protocol, Generic[M]):
    IS_TEMPORAL: ClassVar[bool]

    def __call__(self, in_channels: int) -> M:
        ...


@dataclass(unsafe_hash=True)
class UnetFn(ModelFactory[smp.Unet]):
    IS_TEMPORAL: ClassVar[bool] = False

    encoder_name: str = "resnet50"
    encoder_depth: int = 5
    encoder_weights: Optional[str] = None
    decoder_use_batchnorm: bool = True
    decoder_channels: Tuple[int, ...] = (256, 128, 64, 32, 16)
    decoder_attention: bool = False

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
    IS_TEMPORAL: ClassVar[bool] = False

    encoder_name: str = "resnet50"
    encoder_depth: int = 5
    encoder_weights: Optional[str] = None
    decoder_use_batchnorm: bool = True
    decoder_channels: Tuple[int, ...] = (256, 128, 64, 32, 16)
    decoder_attention: bool = False

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
    IS_TEMPORAL: ClassVar[bool] = True

    dim: int = 64
    multipliers: Tuple[int, ...] = (1, 2, 4, 8)
    n_attn_heads: int = 8
    attn_head_dim: int = 32
    init_dim: Optional[int] = None
    init_kernel_size: int = 7
    use_sparse_linear_attn: bool = False
    resnet_groups: int = 8
    spatial_decoder: bool = True
    use_gca: bool = False
    cosine_sim_attn: bool = False

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
            spatial_decoder=self.spatial_decoder,
            resnet_groups=self.resnet_groups,
            use_gca=self.use_gca,
            cosine_sim_attn=self.cosine_sim_attn,
        )
