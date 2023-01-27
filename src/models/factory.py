from dataclasses import dataclass
from typing import ClassVar, Generic, Optional, Protocol, Tuple, TypeVar

import segmentation_models_pytorch as smp  # type: ignore
import torch.nn as nn
from typing_extensions import override

from src.models.unet3d import Unet3DImagen, Unet3dVd, Unet3dVd2

__all__ = [
    "ModelFactory",
    "Unet3dImagenFn",
    "Unet3dVdFn",
    "Unet3dVd2Fn",
    "UnetFn",
    "UnetPlusPlusFn",
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

    apply_mid_spatial_attn: bool = True
    attn_head_dim: int = 64
    cosine_sim_attn: bool = False
    dim: int = 128
    dim_mults: Tuple[int, ...] = (1, 2, 4, 8)
    init_dim: Optional[int] = None
    init_kernel_size: int = 7
    max_distance: int = 32
    memory_efficient: bool = False
    num_attn_heads: int = 8
    num_resnet_blocks: int = 1
    pixel_shuffle: bool = False
    resnet_groups: int = 8
    scale_skip_connection: bool = True
    spatial_decoder: bool = False
    use_gca: bool = True
    use_sparse_linear_attn: bool = False
    combine_upsample_fmaps: bool = False
    rel_pos_embedding: bool = True
    sd_rate: float = 0.0

    @override
    def __call__(self, in_channels):
        return Unet3dVd(
            apply_mid_spatial_attn=self.apply_mid_spatial_attn,
            attn_head_dim=self.attn_head_dim,
            combine_upsample_fmaps=self.combine_upsample_fmaps,
            cosine_sim_attn=self.cosine_sim_attn,
            dim=self.dim,
            dim_mults=self.dim_mults,
            in_channels=in_channels,
            init_dim=self.init_dim,
            init_kernel_size=self.init_kernel_size,
            max_distance=self.max_distance,
            memory_efficient=self.memory_efficient,
            num_attn_heads=self.num_attn_heads,
            num_resnet_blocks=self.num_resnet_blocks,
            out_channels=1,
            pixel_shuffle=self.pixel_shuffle,
            resnet_groups=self.resnet_groups,
            scale_skip_connection=self.scale_skip_connection,
            spatial_decoder=self.spatial_decoder,
            use_gca=self.use_gca,
            use_sparse_linear_attn=self.use_sparse_linear_attn,
            rel_pos_embedding=self.rel_pos_embedding,
            sd_rate=self.sd_rate,
        )


@dataclass(unsafe_hash=True)
class Unet3dImagenFn(ModelFactory[Unet3DImagen]):
    IS_TEMPORAL: ClassVar[bool] = True
    attend_at_middle: bool = True
    attn_head_dim: int = 64
    combine_upsample_fmaps: bool = False
    cosine_sim_attn: bool = False
    dim: int = 128
    dim_mults: Tuple[int, ...] = (1, 2, 4, 8)
    ff_mult: int = 2
    final_conv_kernel_size: int = 3
    final_resnet_block: bool = True
    init_conv_to_final_conv_residual: bool = False
    init_dim: Optional[int] = None
    init_kernel_size: int = 7
    layer_attn: bool = False
    layer_attn_depth: int = 1
    memory_efficient: bool = False
    num_attn_heads: int = 8
    num_resnet_blocks: int = 1
    pixel_shuffle_upsample: bool = True
    resnet_groups: int = 8
    scale_skip_connection: bool = True
    time_rel_pos_bias_depth: int = 2
    use_gca: bool = True
    use_linear_attn: bool = False

    @override
    def __call__(self, in_channels):
        return Unet3DImagen(
            attend_at_middle=self.attend_at_middle,
            attn_head_dim=self.attn_head_dim,
            combine_upsample_fmaps=self.combine_upsample_fmaps,
            cosine_sim_attn=self.cosine_sim_attn,
            dim=self.dim,
            dim_mults=self.dim_mults,
            ff_mult=self.ff_mult,
            final_conv_kernel_size=self.final_conv_kernel_size,
            final_resnet_block=self.final_resnet_block,
            in_channels=in_channels,
            init_conv_to_final_conv_residual=self.init_conv_to_final_conv_residual,
            init_dim=self.init_dim,
            init_kernel_size=self.init_kernel_size,
            layer_attn=self.layer_attn,
            layer_attn_depth=self.layer_attn_depth,
            memory_efficient=self.memory_efficient,
            num_attn_heads=self.num_attn_heads,
            out_channels=1,
            pixel_shuffle_upsample=self.pixel_shuffle_upsample,
            resnet_groups=self.resnet_groups,
            scale_skip_connection=self.scale_skip_connection,
            time_rel_pos_bias_depth=self.time_rel_pos_bias_depth,
            use_gca=self.use_gca,
            use_linear_attn=self.use_linear_attn,
        )


@dataclass(unsafe_hash=True)
class Unet3dVd2Fn(ModelFactory[Unet3dVd2]):
    IS_TEMPORAL: ClassVar[bool] = True

    apply_mid_spatial_attn: bool = True
    attn_head_dim: int = 64
    cosine_sim_attn: bool = False
    dim: int = 128
    dim_mults: Tuple[int, ...] = (1, 2, 4, 8)
    init_dim: Optional[int] = None
    init_kernel_size: int = 7
    max_distance: int = 32
    memory_efficient: bool = False
    num_attn_heads: int = 8
    num_resnet_blocks: int = 1
    pixel_shuffle: bool = False
    resnet_groups: int = 8
    scale_skip_connection: bool = True
    spatial_decoder: bool = False
    use_gca: bool = True
    use_sparse_linear_attn: bool = False
    combine_upsample_fmaps: bool = False
    rel_pos_embedding: bool = True
    sd_rate: float = 0.0

    @override
    def __call__(self, in_channels):
        return Unet3dVd2(
            apply_mid_spatial_attn=self.apply_mid_spatial_attn,
            attn_head_dim=self.attn_head_dim,
            combine_upsample_fmaps=self.combine_upsample_fmaps,
            cosine_sim_attn=self.cosine_sim_attn,
            dim=self.dim,
            dim_mults=self.dim_mults,
            in_channels=in_channels,
            init_dim=self.init_dim,
            init_kernel_size=self.init_kernel_size,
            max_distance=self.max_distance,
            memory_efficient=self.memory_efficient,
            num_attn_heads=self.num_attn_heads,
            num_resnet_blocks=self.num_resnet_blocks,
            out_channels=1,
            pixel_shuffle=self.pixel_shuffle,
            resnet_groups=self.resnet_groups,
            scale_skip_connection=self.scale_skip_connection,
            spatial_decoder=self.spatial_decoder,
            use_gca=self.use_gca,
            use_sparse_linear_attn=self.use_sparse_linear_attn,
            rel_pos_embedding=self.rel_pos_embedding,
            sd_rate=self.sd_rate,
        )
