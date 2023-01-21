from typing import Any, List, Literal, Optional, Tuple, TypeVar, Union

from einops import rearrange, reduce, repeat  # type: ignore
from einops_exts import rearrange_many  # type: ignore
import torch
from torch import Tensor, einsum, nn
from typing_extensions import override

from src.utils import default_if_none, some, unwrap_or

__all__ = [
    "AxialConv3d",
    "ChanLayerNorm",
    "EtaPool",
    "GlobalContextAttention",
    "LayerNorm",
    "Residual",
    "cast_tuple",
    "pseudo_conv2d",
    "PixelShuffleUpsample",
]


class ChanLayerNorm(nn.Module):
    def __init__(
        self, dim: int, *, eps: float = 1e-5, stable: bool = False, init_zero: bool = False
    ) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(dim) if init_zero else torch.ones(dim))
        self.stable = stable

    @override
    def forward(self, x: Tensor) -> Tensor:
        if self.stable:
            x = x / x.amax(dim=1, keepdim=True).detach()
        var, mean = torch.var_mean(x, dim=1, unbiased=False, keepdim=True)
        gamma = self.gamma.view(1, -1, *((1,) * (x.ndim - 2)))
        return (x - mean) * (var + self.eps).rsqrt() * gamma


class LayerNorm(nn.Module):
    def __init__(
        self, dim: int, *, eps: float = 1e-5, stable: bool = False, init_zero: bool = False
    ) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(dim) if init_zero else torch.ones(dim))
        self.stable = stable

    @override
    def forward(self, x: Tensor) -> Tensor:
        if self.stable:
            x = x / x.amax(dim=1, keepdim=True).detach()
        var, mean = torch.var_mean(x, dim=1, unbiased=False, keepdim=True)
        return (x - mean) * (var + self.eps).rsqrt() * self.gamma


V = TypeVar("V")


def cast_tuple(
    val: Union[V, Tuple[V, ...], List[V]], *, length: Optional[int] = None
) -> Tuple[V, ...]:
    if isinstance(val, list):
        val = tuple(val)

    output = val if isinstance(val, tuple) else ((val,) * default_if_none(length, default=1))

    if some(length):
        assert len(output) == length

    return output


def pseudo_conv2d(
    in_channels: int,
    *,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    groups: int = 1,
    bias: bool = False,
) -> nn.Conv3d:
    """
    Pseudo Conv2d that uses Conv3d but with kernel size of 1 across frames dimension.
    """
    kernel_size = cast_tuple(kernel_size, length=2)
    stride = cast_tuple(stride, length=2)
    padding = cast_tuple(padding, length=2)

    if len(kernel_size) == 2:
        kernel_size = (1, *kernel_size)

    if len(stride) == 2:
        stride = (1, *stride)

    if len(padding) == 2:
        padding = (0, *padding)

    return nn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups,
    )


class GlobalContextAttention(nn.Module):
    """basically a superior form of squeeze-excitation that is attention-esque"""

    def __init__(self, *, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.to_context = pseudo_conv2d(in_channels=in_dim, out_channels=1, kernel_size=1)
        hidden_dim = max(3, out_dim // 2)

        self.net = nn.Sequential(
            pseudo_conv2d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=1),
            nn.SiLU(),
            pseudo_conv2d(in_channels=hidden_dim, out_channels=out_dim, kernel_size=1),
            nn.Sigmoid(),
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        context = self.to_context(x)
        x, context = rearrange_many((x, context), "b n ... -> b n (...)")
        out = einsum("b i n, b c n -> b c i", context.softmax(dim=-1), x)
        out = rearrange(out, "... -> ... 1 1")
        return self.net(out)


class Residual(nn.Module):
    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    @override
    def forward(self, x: nn.Module, **kwargs: Any) -> Tensor:
        return self.fn(x, **kwargs) + x


class AxialConv3d(nn.Module):
    """
    main contribution from make-a-video - pseudo conv3d axial space-time convolutions
    """

    def __init__(
        self,
        in_channels: int,
        *,
        out_channels: Optional[int] = None,
        kernel_size: int = 3,
        temporal_kernel_size: Optional[int] = None,
        padding: Optional[Union[int, Tuple[int, int]]] = None,
    ) -> None:
        super().__init__()
        out_channels = default_if_none(out_channels, default=in_channels)
        temporal_kernel_size = default_if_none(temporal_kernel_size, default=kernel_size)

        self.spatial_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=default_if_none(padding, default="same"),
        )
        if kernel_size > 1:
            self.temporal_conv = nn.Conv1d(
                out_channels,
                out_channels=out_channels,
                kernel_size=temporal_kernel_size,
                padding="same",
            )
            nn.init.dirac_(self.temporal_conv.weight.data)  # initialized to be identity
            if some(self.temporal_conv.bias):
                nn.init.zeros_(self.temporal_conv.bias.data)
        else:
            self.temporal_conv = None
        self.kernel_size = kernel_size

    @override
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, h, w = x.shape
        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = self.spatial_conv(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", b=b)
        x = x

        if self.temporal_conv is None:
            return x

        x = rearrange(x, "b c f h w -> (b h w) c f")
        x = self.temporal_conv(x)
        x = rearrange(x, "(b h w) c f -> b c f h w", h=h, w=w)

        return x


class EtaPool(nn.Module):
    """
    Efficient-temporal-attention pooling.
    """

    def __init__(
        self,
        in_dim: int,
        *,
        kernel_size: int = 1,
        attn_fn: Literal["softmax", "sigmoid"] = "softmax",
    ) -> None:
        super().__init__()
        self.net = nn.Conv1d(
            in_channels=in_dim,
            out_channels=1,
            kernel_size=kernel_size,
            padding="same",
            bias=False,
        )
        self.attn_fn = nn.Softmax(-1) if attn_fn == "softmax" else nn.Sigmoid()

    @override
    def forward(self, x: Tensor) -> Tensor:
        spatial_descriptors = reduce(x, "b c f h w -> b c f", reduction="mean")
        attn = self.attn_fn(self.net(spatial_descriptors))
        attn = rearrange(attn, "b 1 f -> b 1 f 1 1")
        return reduce(attn * x, "b c f h w -> b c 1 h w", reduction="sum")


class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_dim: int, *, out_dim: Optional[int] = None) -> None:
        super().__init__()
        out_dim = unwrap_or(out_dim, default=in_dim)
        conv = pseudo_conv2d(in_dim, out_channels=out_dim * 4, kernel_size=1)
        self._init_conv(conv)

        self.net = nn.Sequential(conv, nn.SiLU())

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

    def _init_conv(self, conv: nn.Conv3d) -> None:
        o, i, f, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 4, i, f, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, "o ... -> (o 4) ...")
        conv.weight.data.copy_(conv_weight)
        if some(conv.bias):
            nn.init.zeros_(conv.bias.data)

    @override
    def forward(self, x: Tensor) -> Tensor:
        out = self.net(x)
        frames = x.shape[2]
        out = rearrange(out, "b c f h w -> (b f) c h w")
        out = self.pixel_shuffle(out)
        return rearrange(out, "(b f) c h w -> b c f h w", f=frames)
