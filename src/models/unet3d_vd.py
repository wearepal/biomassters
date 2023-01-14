# Adapted shamelessly from https://github.com/lucidrains/video-diffusion-pytorch
import math
from typing import Any, Optional, Tuple, TypeVar, Union

from einops import rearrange, repeat  # type: ignore
from einops.layers.torch import Rearrange  # type: ignore
from einops_exts import rearrange_many  # type: ignore
from rotary_embedding_torch import RotaryEmbedding  # type: ignore
import torch
from torch import Tensor, einsum, nn
import torch.nn.functional as F
from typing_extensions import override

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
    def forward(self, x: nn.Module, **kwargs: Any) -> Tensor:
        return self.fn(x, **kwargs) + x


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


V = TypeVar("V")


def cast_tuple(val: Union[V, Tuple[V, ...]], *, length: Optional[int] = None) -> Tuple[V, ...]:
    if isinstance(val, list):
        val = tuple(val)

    output = val if isinstance(val, tuple) else ((val,) * (1 if length is None else length))

    if length is not None:
        assert len(output) == length

    return output


# axial space-time convolutions, but made causal to keep in line with the design decisions of imagen-video paper


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
        out_channels = in_channels if out_channels is None else in_channels
        temporal_kernel_size = kernel_size if temporal_kernel_size is None else temporal_kernel_size

        self.spatial_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2 if padding is None else padding,
        )
        if kernel_size > 1:
            self.temporal_conv = nn.Conv1d(
                out_channels, out_channels, kernel_size=temporal_kernel_size
            )
            nn.init.dirac_(self.temporal_conv.weight.data)  # initialized to be identity
            if self.temporal_conv.bias is not None:
                nn.init.zeros_(self.temporal_conv.bias.data)
        else:
            self.temporal_conv = nn.Identity()
        self.kernel_size = kernel_size

    @override
    def forward(self, x: Tensor) -> Tensor:
        b, c, *_, h, w = x.shape
        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = self.spatial_conv(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", b=b)
        x = rearrange(x, "b c f h w -> (b h w) c f")
        x = self.temporal_conv(x)

        x = rearrange(x, "(b h w) c f -> b c f h w", h=h, w=w)

        return x


# pseudo conv2d that uses conv3d but with kernel size of 1 across frames dimension
def pseudo_conv2d(
    in_channels: int,
    *,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = 1,
    padding=0,
    **kwargs,
):
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
        **kwargs,
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
    def __init__(self, dim: int, *, eps=1e-5, stable: bool = False) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))
        self.stable = stable

    @override
    def forward(self, x: Tensor) -> Tensor:
        if self.stable:
            x = x / x.amax(dim=1, keepdim=True).detach()
        var, mean = torch.var_mean(x, dim=1, unbiased=False, keepdim=True)
        return (x - mean) * (var + self.eps).rsqrt() * self.gamma


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
    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        x = self.norm(x)
        return self.act(x)


class Block3d(nn.Module):
    def __init__(self, in_dim: int, *, out_dim: int, groups: int = 8) -> None:
        super().__init__()
        self.proj = nn.Conv3d(
            in_dim, out_channels=out_dim, kernel_size=(1, 3, 3), padding=(0, 1, 1)
        )
        # self.proj = AxialConv3d(in_dim, out_channels=out_dim, kernel_size=3, padding=(1, 1))
        self.norm = nn.GroupNorm(groups, num_channels=out_dim)
        self.act = nn.SiLU()

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        x = self.norm(x)
        return self.act(x)


class Always:
    def __init__(self, val: float) -> None:
        super().__init__()
        self.val = val

    def __call__(self, x: Tensor) -> float:
        return self.val


class ResnetBlock3d(nn.Module):
    def __init__(
        self, in_dim: int, *, out_dim: int, groups: int = 8, use_gca: bool = False
    ) -> None:
        super().__init__()
        self.block1 = Block3d(in_dim, out_dim=out_dim, groups=groups)
        self.block2 = Block3d(out_dim, out_dim=out_dim, groups=groups)
        self.res_conv = (
            nn.Conv3d(in_dim, out_channels=out_dim, kernel_size=1)
            if in_dim != out_dim
            else nn.Identity()
        )
        self.gca = GlobalContext(dim_in=out_dim, dim_out=out_dim) if use_gca else Always(1)

    @override
    def forward(self, x: Tensor) -> Tensor:
        h = self.block1(x)
        h = self.block2(h)
        h = h * self.gca(h)
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


class GlobalContext(nn.Module):
    """basically a superior form of squeeze-excitation that is attention-esque"""

    def __init__(self, *, dim_in, dim_out):
        super().__init__()
        self.to_k = pseudo_conv2d(in_channels=dim_in, out_channels=1, kernel_size=1)
        hidden_dim = max(3, dim_out // 2)

        self.net = nn.Sequential(
            pseudo_conv2d(in_channels=dim_in, out_channels=hidden_dim, kernel_size=1),
            nn.SiLU(),
            pseudo_conv2d(in_channels=hidden_dim, out_channels=dim_out, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        context = self.to_k(x)
        x, context = rearrange_many((x, context), "b n ... -> b n (...)")
        out = einsum("b i n, b c n -> b c i", context.softmax(dim=-1), x)
        out = rearrange(out, "... -> ... 1 1")
        return self.net(out)


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
        cosine_sim_attn=False,
    ) -> None:
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)
        self.cosine_sim_attn = cosine_sim_attn
        self.scale = 1 if self.cosine_sim_attn else dim_head**-0.5
        self.cosine_sim_scale = 16 if cosine_sim_attn else 1

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

        if self.cosine_sim_attn:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)

        # similarity
        sim = einsum("... h i d, ... h j d -> ... h i j", q, k) * self.cosine_sim_scale

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


class PerceiverAttention(nn.Module):
    def __init__(
        self, *, dim: int, dim_head: int = 64, heads: int = 8, cosine_sim_attn: bool = False
    ) -> None:
        super().__init__()
        self.scale = dim_head**-0.5 if not cosine_sim_attn else 1
        self.cosine_sim_attn = cosine_sim_attn
        self.cosine_sim_scale = 16 if cosine_sim_attn else 1

        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim, bias=False), nn.LayerNorm(dim))

    @override
    def forward(self, x, *, latents: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = self.norm(x)
        latents = self.norm_latents(latents)

        b, h = x.shape[0], self.heads

        q = self.to_q(latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)

        q = q * self.scale

        # cosine sim attention

        if self.cosine_sim_attn:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)

        # similarities and masking

        sim = einsum("... i d, ... j d  -> ... i j", q, k) * self.cosine_sim_scale

        if mask is not None:
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = F.pad(mask, (0, latents.shape[-2]), value=True)
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, max_neg_value)

        # attention

        attn = sim.softmax(dim=-1)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        return self.to_out(out)


def feedforward_block(dim: int, *, mult=2) -> nn.Sequential:
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, hidden_dim, bias=False),
        nn.GELU(),
        LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, dim, bias=False),
    )


def masked_mean(x, *, dim: int, mask: Optional[Tensor] = None) -> Tensor:
    if mask is None:
        return x.mean(dim=dim)

    denom = mask.sum(dim=dim, keepdim=True)
    mask = rearrange(mask, "b n -> b n 1")
    masked_t = x.masked_fill(~mask, 0.0)
    return masked_t.sum(dim=dim) / denom.clamp(min=1e-5)


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        dim_head: int = 64,
        heads: int = 8,
        num_latents: int = 64,
        num_latents_mean_pooled: int = 4,  # number of latents derived from mean pooled representation of the sequence
        max_seq_len: int = 512,
        ff_mult: int = 4,
        cosine_sim_attn: bool = False,
    ) -> None:
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        self.to_latents_from_mean_pooled_seq = None

        if num_latents_mean_pooled > 0:
            self.to_latents_from_mean_pooled_seq = nn.Sequential(
                LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange("b (n d) -> b n d", n=num_latents_mean_pooled),
            )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(
                            dim=dim, dim_head=dim_head, heads=heads, cosine_sim_attn=cosine_sim_attn
                        ),
                        feedforward_block(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x, mask=None):
        n, device = x.shape[1], x.device
        pos_emb = self.pos_emb(torch.arange(n, device=device))

        x_with_pos = x + pos_emb

        latents = repeat(self.latents, "n d -> b n d", b=x.shape[0])

        if self.to_latents_from_mean_pooled_seq is not None:
            meanpooled_seq = masked_mean(
                x, dim=1, mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
            )
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents, latents), dim=-2)

        for attn, ff in self.layers:  # type: ignore
            latents = attn(x_with_pos, latents, mask=mask) + latents
            latents = ff(latents) + latents

        return latents


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
        use_sparse_linear_attn: bool = True,
        resnet_groups: int = 8,
        max_distance: int = 32,
        spatial_decoder: bool = True,
        use_gca: bool = False,
        cosine_sim_attn: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        # Reduce along the temporal dimension prior to decoding.
        # For simplicity of implementation, we retain the 3d convs, applying
        # them over single-frame inputs.
        self.spatial_decoder = spatial_decoder

        # temporal attention and its relative positional encoding
        rotary_emb = RotaryEmbedding(min(32, attn_head_dim))
        temporal_attn = lambda dim: EinopsToAndFrom(
            from_einops="b c f h w",
            to_einops="b (h w) f c",
            fn=Attention(
                dim,
                heads=n_attn_heads,
                dim_head=attn_head_dim,
                rotary_emb=rotary_emb,
                cosine_sim_attn=cosine_sim_attn,
            ),
        )

        self.time_rel_pos_bias = RelativePositionBias(heads=n_attn_heads, max_distance=max_distance)

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
        self.encoder_blocks = nn.ModuleList([])
        self.decoder_blocks = nn.ModuleList([])

        num_resolutions = len(in_out)
        # modules for all layers
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.encoder_blocks.append(
                nn.ModuleList(
                    [
                        ResnetBlock3d(
                            in_dim=dim_in, out_dim=dim_out, groups=resnet_groups, use_gca=use_gca
                        ),
                        ResnetBlock3d(
                            in_dim=dim_out, out_dim=dim_out, groups=resnet_groups, use_gca=use_gca
                        ),
                        Residual(
                            PreNorm(dim_out, fn=SpatialLinearAttention(dim_out, heads=n_attn_heads))
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

        spatial_attn = EinopsToAndFrom(
            from_einops="b c f h w",
            to_einops="b f (h w) c",
            fn=Attention(mid_dim, heads=n_attn_heads),
        )

        self.mid_spatial_attn = Residual(PreNorm(mid_dim, fn=spatial_attn))
        self.mid_temporal_attn = Residual(PreNorm(mid_dim, fn=temporal_attn(mid_dim)))

        self.mid_block2 = ResnetBlock3d(
            in_dim=mid_dim, out_dim=mid_dim, groups=resnet_groups, use_gca=use_gca
        )

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.decoder_blocks.append(
                nn.ModuleList(
                    [
                        ResnetBlock3d(
                            in_dim=dim_out * 2,
                            out_dim=dim_in,
                            groups=resnet_groups,
                            use_gca=use_gca,
                        ),
                        ResnetBlock3d(
                            in_dim=dim_in, out_dim=dim_in, groups=resnet_groups, use_gca=use_gca
                        ),
                        Residual(
                            PreNorm(dim_in, fn=SpatialLinearAttention(dim_in, heads=n_attn_heads))
                        )
                        if use_sparse_linear_attn
                        else nn.Identity(),
                        nn.Identity()
                        if self.spatial_decoder
                        else Residual(PreNorm(dim_in, fn=temporal_attn(dim_in))),
                        up_conv(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Final (2d) conv block operating on temporally-pooled features
        self.final_conv = nn.Sequential(
            ResnetBlock3d(in_dim=dim * 2, out_dim=dim, groups=resnet_groups, use_gca=True),
            nn.Conv3d(in_channels=dim, out_channels=self.out_channels, kernel_size=1),
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        time_rel_pos_bias = self.time_rel_pos_bias(x.size(2), device=x.device)
        x = self.init_conv(x)
        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)
        init_conv_residual = x.clone()
        if self.spatial_decoder:
            init_conv_residual = init_conv_residual.mean(dim=2, keepdim=True)
        hiddens = []
        for block1, block2, spatial_attn, temporal_attn, downsample in self.encoder_blocks:  # type: ignore
            x = block1(x)
            x = block2(x)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias)
            if self.spatial_decoder:
                hiddens.append(x.mean(dim=2, keepdim=True))
            else:
                hiddens.append(x)
            x = downsample(x)

        x = self.mid_block1(x)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(x, pos_bias=time_rel_pos_bias)
        x = self.mid_block2(x)

        if self.spatial_decoder:
            x = x.mean(dim=2, keepdim=True)

        for block1, block2, spatial_attn, temporal_attn, upsample in self.decoder_blocks:  # type: ignore
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = block1(x)
            x = block2(x)
            x = spatial_attn(x)
            x = (
                temporal_attn(x)
                if self.spatial_decoder
                else temporal_attn(x, pos_bias=time_rel_pos_bias)
            )
            x = upsample(x)

        x = torch.cat((x, init_conv_residual), dim=1)
        # temporal pooling
        x = x.mean(dim=2, keepdim=True)
        return self.final_conv(x).squeeze(dim=2)
