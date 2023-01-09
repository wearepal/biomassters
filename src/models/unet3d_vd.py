# Adapted shamelessly from https://github.com/lucidrains/video-diffusion-pytorch
from typing_extensions import override
import math
from torch import Tensor
import torch
from torch import nn, einsum
from typing import Optional, Tuple, Any


from einops import rearrange  # type: ignore
from einops_exts import rearrange_many  # type: ignore

from rotary_embedding_torch import RotaryEmbedding  # type: ignore

__all__ = ["Unet3dVd"]

# relative positional bias
class RelativePositionBias(nn.Module):
    def __init__(self, heads: int = 8, *, num_buckets: int = 32, max_distance: int = 128) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position: Tensor, *, num_buckets: int = 32, max_distance: int = 128
    ) -> Tensor:
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = (
            max_exact
            + (
                torch.log(n.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    @override
    def forward(self, seq_length: int, *, device: torch.device) -> Tensor:
        q_pos = torch.arange(seq_length, dtype=torch.long, device=device)
        k_pos = torch.arange(seq_length, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, "j -> 1 j") - rearrange(q_pos, "i -> i 1")
        rp_bucket = self._relative_position_bucket(
            rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance
        )
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, "i j h -> h i j")


class Residual(nn.Module):
    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    @override
    def forward(self, x: nn.Module, *args: Any, **kwargs: Any) -> Tensor:
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    @override
    def forward(self, x: Tensor) -> Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def up_conv(dim: int) -> nn.ConvTranspose3d:
    return nn.ConvTranspose3d(
        in_channels=dim,
        out_channels=dim,
        kernel_size=(1, 4, 4),
        stride=(1, 2, 2),
        padding=(0, 1, 1),
    )


def down_conv(dim) -> nn.Conv3d:
    return nn.Conv3d(
        in_channels=dim,
        out_channels=dim,
        kernel_size=(1, 4, 4),
        stride=(1, 2, 2),
        padding=(0, 1, 1),
    )


class LayerNorm(nn.Module):
    def __init__(self, dim: int, *, eps=1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    @override
    def forward(self, x: Tensor) -> Tensor:
        var, mean = torch.var_mean(x, dim=1, unbiased=False, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class PreNorm(nn.Module):
    def __init__(self, dim: int, *, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    @override
    def forward(self, x: Tensor, **kwargs: Any):
        x = self.norm(x)
        return self.fn(x, **kwargs)


# building block modules


class Block2d(nn.Module):
    def __init__(self, in_dim: int, *, out_dim: int, groups: int = 8) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_channels=out_dim, kernel_size=(3, 3), padding=(1, 1))
        self.norm = nn.GroupNorm(groups, num_channels=out_dim)
        self.act = nn.SiLU()

    @override
    def forward(self, x: Tensor, scale_shift: Optional[Tuple[Tensor, Tensor]] = None) -> Tensor:
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)


class Block3d(nn.Module):
    def __init__(self, in_dim: int, *, out_dim: int, groups: int = 8) -> None:
        super().__init__()
        self.proj = nn.Conv3d(
            in_dim, out_channels=out_dim, kernel_size=(1, 3, 3), padding=(0, 1, 1)
        )
        self.norm = nn.GroupNorm(groups, num_channels=out_dim)
        self.act = nn.SiLU()

    @override
    def forward(self, x: Tensor, scale_shift: Optional[Tuple[Tensor, Tensor]] = None) -> Tensor:
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)


class ResnetBlock2d(nn.Module):
    def __init__(self, in_dim, *, out_dim, groups=8) -> None:
        super().__init__()
        self.block1 = Block2d(in_dim, out_dim=out_dim, groups=groups)
        self.block2 = Block2d(out_dim, out_dim=out_dim, groups=groups)
        self.res_conv = (
            nn.Conv2d(in_dim, out_channels=out_dim, kernel_size=1)
            if in_dim != out_dim
            else nn.Identity()
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)


class ResnetBlock3d(nn.Module):
    def __init__(self, in_dim, *, out_dim, groups=8) -> None:
        super().__init__()
        self.block1 = Block3d(in_dim, out_dim=out_dim, groups=groups)
        self.block2 = Block3d(out_dim, out_dim=out_dim, groups=groups)
        self.res_conv = (
            nn.Conv3d(in_dim, out_channels=out_dim, kernel_size=1)
            if in_dim != out_dim
            else nn.Identity()
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)


class SpatialLinearAttention(nn.Module):
    def __init__(self, dim: int, *, heads: int = 4, dim_head: int = 32) -> None:
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, out_channels=hidden_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, out_channels=dim, kernel_size=1)

    @override
    def forward(self, x: Tensor) -> Tensor:
        b, c, f, h, w = x.shape
        x = rearrange(x, "b c f h w -> (b f) c h w")

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = rearrange_many(qkv, "b (h c) x y -> b h c (x y)", h=self.heads)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        out = self.to_out(out)
        return rearrange(out, "(b f) c h w -> b c f h w", b=b)


class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops: str, *, to_einops: str, fn: nn.Module):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    @override
    def forward(self, x: Tensor, **kwargs: Any) -> Tensor:
        shape = x.shape
        reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(" "), shape)))
        x = rearrange(x, f"{self.from_einops} -> {self.to_einops}")
        x = self.fn(x, **kwargs)
        x = rearrange(x, f"{self.to_einops} -> {self.from_einops}", **reconstitute_kwargs)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        heads: int = 4,
        dim_head: int = 32,
        rotary_emb: Optional[RotaryEmbedding] = None,
    ) -> None:
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    @override
    def forward(
        self,
        x: Tensor,
        *,
        pos_bias: Optional[Tensor] = None,
    ) -> Tensor:
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # split out heads
        q, k, v = rearrange_many(qkv, "... n (h d) -> ... h n d", h=self.heads)
        # scale
        q = q * self.scale

        # rotate positions into queries and keys for time attention
        if self.rotary_emb is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity
        sim = einsum("... h i d, ... h j d -> ... h i j", q, k)

        # relative positional bias
        if pos_bias is not None:
            sim = sim + pos_bias

        # numerical stability
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values
        out = einsum("... h i j, ... h j d -> ... h i d", attn, v)
        out = rearrange(out, "... h n d -> ... n (h d)")
        return self.to_out(out)


# model
class Unet3dVd(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        dim: int = 64,
        out_channels: Optional[int] = None,
        multipliers: Tuple[int, ...] = (1, 2, 4, 8),
        n_attn_heads: int = 8,
        attn_head_dim: int = 32,
        init_dim: Optional[int] = None,
        init_kernel_size: int = 7,
        use_sparse_linear_attn: bool = False,
        mid_spatial_attn: bool = False,
        resnet_groups: int = 8,
        max_distance: int = 32,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        # temporal attention and its relative positional encoding
        rotary_emb = RotaryEmbedding(min(32, attn_head_dim))
        temporal_attn = lambda dim: EinopsToAndFrom(
            from_einops="b c f h w",
            to_einops="b (h w) f c",
            fn=Attention(dim, heads=n_attn_heads, dim_head=attn_head_dim, rotary_emb=rotary_emb),
        )

        self.time_rel_pos_bias = RelativePositionBias(
            heads=n_attn_heads, max_distance=max_distance
        )

        # initial conv
        init_dim = dim if init_dim is None else init_dim
        assert init_kernel_size % 2 == 1

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=init_dim,
            kernel_size=(1, init_kernel_size, init_kernel_size),
            padding=(0, init_padding, init_padding),
        )

        self.init_temporal_attn = Residual(PreNorm(init_dim, fn=temporal_attn(init_dim)))

        # dimensions
        dims = [init_dim, *map(lambda m: dim * m, multipliers)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # layers
        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        num_resolutions = len(in_out)
        # modules for all layers
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.down_blocks.append(
                nn.ModuleList(
                    [
                        ResnetBlock3d(in_dim=dim_in, out_dim=dim_out, groups=resnet_groups),
                        ResnetBlock3d(in_dim=dim_out, out_dim=dim_out, groups=resnet_groups),
                        Residual(
                            PreNorm(
                                dim_out, fn=SpatialLinearAttention(dim_out, heads=n_attn_heads)
                            )
                        )
                        if use_sparse_linear_attn
                        else nn.Identity(),
                        Residual(PreNorm(dim_out, fn=temporal_attn(dim_out))),
                        down_conv(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock3d(in_dim=mid_dim, out_dim=mid_dim)

        if mid_spatial_attn:
            spatial_attn = EinopsToAndFrom(
                from_einops="b c f h w",
                to_einops="b f (h w) c",
                fn=Attention(mid_dim, heads=n_attn_heads),
            )

            self.mid_spatial_attn = Residual(PreNorm(mid_dim, fn=spatial_attn))
        else:
            self.mid_spatial_attn = nn.Identity()
        self.mid_temporal_attn = Residual(PreNorm(mid_dim, fn=temporal_attn(mid_dim)))

        self.mid_block2 = ResnetBlock3d(in_dim=mid_dim, out_dim=mid_dim, groups=resnet_groups)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.up_blocks.append(
                nn.ModuleList(
                    [
                        ResnetBlock3d(in_dim=dim_out * 2, out_dim=dim_in, groups=resnet_groups),
                        ResnetBlock3d(in_dim=dim_in, out_dim=dim_in, groups=resnet_groups),
                        Residual(
                            PreNorm(dim_in, fn=SpatialLinearAttention(dim_in, heads=n_attn_heads))
                        )
                        if use_sparse_linear_attn
                        else nn.Identity(),
                        Residual(PreNorm(dim_in, fn=temporal_attn(dim_in))),
                        up_conv(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_channels = in_channels if out_channels is None else out_channels
        # Final (2d) conv block operating on temporally-pooled features
        self.final_conv = nn.Sequential(
            # ResnetBlock3d(in_dim=dim * 2, out_dim=dim, groups=resnet_groups),
            # nn.Conv3d(in_channels=dim, out_channels=out_channels, kernel_size=1),
            ResnetBlock2d(in_dim=dim * 2, out_dim=dim, groups=resnet_groups),
            nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=1),
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        time_rel_pos_bias = self.time_rel_pos_bias(x.size(2), device=x.device)
        x = self.init_conv(x)
        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)
        r = x.clone()
        h = []
        for block1, block2, spatial_attn, temporal_attn, downsample in self.down_blocks:  # type: ignore
            x = block1(x)
            x = block2(x)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(x, pos_bias=time_rel_pos_bias)
        x = self.mid_block2(x)

        for block1, block2, spatial_attn, temporal_attn, upsample in self.up_blocks:  # type: ignore
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x)
            x = block2(x)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        # temporal pooling
        x = x.mean(dim=2)
        return self.final_conv(x)
