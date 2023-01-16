from typing import Any, Optional, Tuple, TypeVar, Union, List

from einops import rearrange  # type: ignore
from einops_exts import rearrange_many  # type: ignore
import torch
from torch import Tensor, einsum, nn
from typing_extensions import override

from src.utils import default_if_none, some

__all__ = [
    "AxialConv3d",
    "GlobalContextAttention",
    "LayerNorm",
    "Residual",
    "cast_tuple",
    "pseudo_conv2d",
]


class LayerNorm(nn.Module):
    def __init__(
        self, dim: int, *, eps=1e-5, stable: bool = False, init_zero: bool = False
    ) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(
            torch.zeros(1, dim, 1, 1, 1) if init_zero else torch.ones(1, dim, 1, 1, 1)
        )
        self.stable = stable

    @override
    def forward(self, x: Tensor) -> Tensor:
        if self.stable:
            x = x / x.amax(dim=1, keepdim=True).detach()
        var, mean = torch.var_mean(x, dim=1, unbiased=False, keepdim=True)
        return (x - mean) * (var + self.eps).rsqrt() * self.gamma


V = TypeVar("V")


def cast_tuple(val: Union[V, Tuple[V, ...], List[V]], *, length: Optional[int] = None) -> Tuple[V, ...]:
    if isinstance(val, list):
        val = tuple(val)

    output = val if isinstance(val, tuple) else ((val,) * default_if_none(length, default=1))

    if some(length):
        assert len(output) == length

    return output


# pseudo conv2d that uses conv3d but with kernel size of 1 across frames dimension
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
            padding=default_if_none(padding, default=kernel_size // 2),
        )
        if kernel_size > 1:
            self.temporal_conv = nn.Conv1d(
                out_channels, out_channels, kernel_size=temporal_kernel_size
            )
            nn.init.dirac_(self.temporal_conv.weight.data)  # initialized to be identity
            if some(self.temporal_conv.bias):
                nn.init.zeros_(self.temporal_conv.bias.data)
        else:
            self.temporal_conv = nn.Identity()
        self.kernel_size = kernel_size

    @override
    def forward(self, x: Tensor) -> Tensor:
        b, _, *_, h, w = x.shape
        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = self.spatial_conv(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", b=b)
        x = rearrange(x, "b c f h w -> (b h w) c f")
        x = self.temporal_conv(x)

        x = rearrange(x, "(b h w) c f -> b c f h w", h=h, w=w)

        return x
