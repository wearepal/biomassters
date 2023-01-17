import math
from typing import Any, List, Optional, Sequence, Tuple, Union, cast

from einops import rearrange, repeat  # type: ignore
from einops.layers.torch import Rearrange  # type: ignore
from einops_exts import rearrange_many  # type: ignore
from einops_exts.torch import EinopsToAndFrom  # type: ignore
import torch
from torch import Tensor, einsum, nn
import torch.nn.functional as F
from typing_extensions import override

from src.utils import some, unwrap_or

from .common import (
    AxialConv3d,
    ChanLayerNorm,
    GlobalContextAttention,
    LayerNorm,
    Residual,
    cast_tuple,
    pseudo_conv2d,
)

__all__ = ["Unet3DImagen"]


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    @override
    def forward(self, x: Tensor) -> Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = rearrange(x, "i -> i 1") * rearrange(emb, "j -> 1 j")
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class DynamicPositionBias(nn.Module):
    def __init__(self, dim: int, *, heads: int, depth: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Sequential(nn.Linear(1, dim), LayerNorm(dim), nn.SiLU()))
        for _ in range(max(depth - 1, 0)):
            self.mlp.append(nn.Sequential(nn.Linear(dim, dim), LayerNorm(dim), nn.SiLU()))
        self.mlp.append(nn.Linear(dim, heads))

    @override
    def forward(self, n: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        i = torch.arange(n, device=device)
        j = torch.arange(n, device=device)

        indices = rearrange(i, "i -> i 1") - rearrange(j, "j -> 1 j")
        indices += n - 1

        pos = torch.arange(-n + 1, n, device=device, dtype=dtype)
        pos = rearrange(pos, "... -> ... 1")
        pos = self.mlp(pos)
        bias = pos[indices]
        bias = rearrange(bias, "i j h -> h i j")
        return bias


def fc_block(dim: int, mult: int = 2) -> nn.Sequential:
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, hidden_dim, bias=False),
        nn.GELU(),
        LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, dim, bias=False),
    )


def pseudo_conv2d_block(dim: int, *, mult: int = 2):
    # in paper, it seems for self attention layers they did feedforwards with twice channel width
    hidden_dim = dim * mult
    return nn.Sequential(
        ChanLayerNorm(dim),
        pseudo_conv2d(in_channels=dim, out_channels=hidden_dim, kernel_size=1, bias=False),
        nn.GELU(),
        ChanLayerNorm(hidden_dim),
        pseudo_conv2d(in_channels=hidden_dim, out_channels=dim, kernel_size=1, bias=False),
    )


class Block3d(nn.Module):
    def __init__(self, in_dim: int, *, out_dim: int, groups: int = 8, norm: bool = True) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(groups, num_channels=in_dim) if norm else nn.Identity()
        self.act = nn.SiLU()
        self.proj = AxialConv3d(in_dim, out_channels=out_dim, kernel_size=3, padding=1)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.act(x)
        return self.proj(x)


class ResnetBlock3d(nn.Module):
    def __init__(
        self, in_dim: int, *, out_dim: int, groups: int = 8, use_gca: bool = False
    ) -> None:
        super().__init__()
        self.block1 = Block3d(in_dim, out_dim=out_dim, groups=groups)
        self.block2 = Block3d(out_dim, out_dim=out_dim, groups=groups)
        self.res_conv = (
            pseudo_conv2d(in_dim, out_channels=out_dim, kernel_size=1)
            if in_dim != out_dim
            else nn.Identity()
        )
        self.gca = GlobalContextAttention(in_dim=out_dim, out_dim=out_dim) if use_gca else None

    @override
    def forward(self, x: Tensor) -> Tensor:
        h = self.block1(x)
        h = self.block2(h)
        if self.gca:
            h = h * self.gca(h)
        return h + self.res_conv(x)


class FinalResnetBlock(nn.Module):
    def __init__(
        self, in_dim: int, *, out_dim: int, groups: int = 8, use_gca: bool = False
    ) -> None:
        super().__init__()

        def _block(_in_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.GroupNorm(groups, num_channels=_in_dim),
                nn.SiLU(),
                pseudo_conv2d(_in_dim, out_channels=out_dim, kernel_size=3, padding=1),
            )

        self.block1 = _block(in_dim)
        self.block2 = _block(out_dim)
        self.res_conv = (
            nn.Conv3d(in_dim, out_channels=out_dim, kernel_size=1)
            if in_dim != out_dim
            else nn.Identity()
        )
        self.gca = GlobalContextAttention(in_dim=out_dim, out_dim=out_dim) if use_gca else None

    @override
    def forward(self, x: Tensor) -> Tensor:
        h = self.block1(x)
        h = self.block2(h)
        if self.gca:
            h = h * self.gca(h)
        return h + self.res_conv(x)


def downsample(in_dim: int, *, out_dim: Optional[int] = None) -> nn.Sequential:
    out_dim = unwrap_or(out_dim, default=in_dim)
    return nn.Sequential(
        Rearrange("b c f (h p1) (w p2) -> b (c p1 p2) f h w", p1=2, p2=2),
        pseudo_conv2d(in_channels=in_dim * 4, out_channels=out_dim, kernel_size=1),
    )


class Pad(nn.Module):
    def __init__(self, padding: Sequence[int], value: float = 0.0) -> None:
        super().__init__()
        self.padding = padding
        self.value = value

    @override
    def forward(self, x: Tensor) -> Tensor:
        return F.pad(x, self.padding, value=self.value)


def upsample(in_dim: int, *, out_dim: Optional[int] = None) -> nn.Sequential:
    out_dim = unwrap_or(out_dim, default=in_dim)

    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        pseudo_conv2d(in_dim, out_channels=out_dim, kernel_size=3, padding=1),
    )


class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_dim: int, *, out_dim: Optional[int] = None) -> None:
        super().__init__()
        out_dim = unwrap_or(out_dim, default=in_dim)
        conv = pseudo_conv2d(in_dim, out_channels=out_dim * 4, kernel_size=1)

        self.net = nn.Sequential(conv, nn.SiLU())

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self._init_conv(conv)

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


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        heads: int = 4,
        dim_head: int = 32,
        cosine_sim_attn: bool = False,
        init_zero: bool = False,
    ) -> None:
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.norm = LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, out_features=dim, bias=False), LayerNorm(dim, init_zero=init_zero)
        )
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
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # split out heads
        q, k, v = rearrange_many(qkv, "... n (h d) -> ... h n d", h=self.heads)
        # scale
        q = q * self.scale

        if self.cosine_sim_attn:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)

        # similarity
        sim = einsum("... h i d, ... h j d -> ... h i j", q, k) * self.cosine_sim_scale

        # relative positional bias
        if some(pos_bias):
            sim = sim + pos_bias

        # numerical stability
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values
        out = einsum("... h i j, ... h j d -> ... h i d", attn, v)
        out = rearrange(out, "... h n d -> ... n (h d)")
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        dim_head: int = 32,
        heads: int = 8,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = ChanLayerNorm(dim)

        self.nonlin = nn.SiLU()

        def _kvq_block() -> nn.Sequential:
            return nn.Sequential(
                nn.Dropout(dropout),
                pseudo_conv2d(dim, out_channels=inner_dim, kernel_size=1, bias=False),
                pseudo_conv2d(
                    inner_dim,
                    out_channels=inner_dim,
                    kernel_size=3,
                    bias=False,
                    padding=1,
                    groups=inner_dim,
                ),
            )

        self.to_q = _kvq_block()
        self.to_k = _kvq_block()
        self.to_v = _kvq_block()

        self.to_out = nn.Sequential(
            pseudo_conv2d(inner_dim, out_channels=dim, kernel_size=1, bias=False),
            ChanLayerNorm(dim),
        )

    @override
    def forward(self, fmap: Tensor) -> Tensor:
        h, x, y = self.heads, *fmap.shape[-2:]

        fmap = self.norm(fmap)
        q, k, v = (fn(fmap) for fn in (self.to_q, self.to_k, self.to_v))
        q, k, v = rearrange_many((q, k, v), "b (h c) x y -> (b h) (x y) c", h=h)

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)

        q = q * self.scale

        context = einsum("b n d, b n e -> b d e", k, v)
        out = einsum("b n d, b d e -> b n e", q, context)
        out = rearrange(out, "(b h) (x y) d -> b (h d) x y", h=h, x=x, y=y)

        out = self.nonlin(out)
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        depth: int = 1,
        heads: int = 8,
        dim_head: int = 32,
        ff_mult: int = 2,
        cosine_sim_attn: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([])

        def _block() -> nn.Sequential:
            return nn.Sequential(
                Residual(
                    EinopsToAndFrom(
                        "b c f h w",
                        "b (f h w) c",
                        Attention(
                            dim=dim,
                            heads=heads,
                            dim_head=dim_head,
                            cosine_sim_attn=cosine_sim_attn,
                        ),
                    )
                ),
                Residual(pseudo_conv2d_block(dim=dim, mult=ff_mult)),
            )

        for _ in range(depth):
            self.blocks.append(_block())

    @override
    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            block = cast(nn.Sequential, block)
            x = block(x)
        return x


class LinearAttentionTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        depth: int = 1,
        heads: int = 8,
        dim_head: int = 32,
        ff_mult: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([])

        def _block() -> nn.Sequential:
            return nn.Sequential(
                Residual(
                    LinearAttention(dim=dim, heads=heads, dim_head=dim_head),
                ),
                Residual(pseudo_conv2d_block(dim=dim, mult=ff_mult)),
            )

        for _ in range(depth):
            self.blocks.append(_block())

    @override
    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            block = cast(nn.Sequential, block)
            x = block(x)
        return x


def resize_video_to(
    video: Tensor,
    *,
    target_image_size: Union[int, Tuple[int, int]],
    clamp_range: Optional[Tuple[int, int]] = None,
) -> Tensor:
    orig_video_size = video.shape[-1]

    if orig_video_size == target_image_size:
        return video

    frames = video.shape[2]
    video = rearrange(video, "b c f h w -> (b f) c h w")

    out = F.interpolate(video, target_image_size, mode="nearest")

    if some(clamp_range):
        out = out.clamp(*clamp_range)

    out = rearrange(out, "(b f) c h w -> b c f h w", f=frames)

    return out


class UpsampleCombiner(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        enabled: bool = False,
        dim_ins: Tuple[int, ...] = (),
        dim_outs: Union[int, Tuple[int, ...]] = (),
    ) -> None:
        super().__init__()
        dim_outs = cast_tuple(dim_outs, length=len(dim_ins))
        assert len(dim_ins) == len(dim_outs)

        self.enabled = enabled

        if not self.enabled:
            self.dim_out = dim
            return

        self.fmap_convs = nn.ModuleList(
            [Block3d(in_dim=dim_in, out_dim=dim_out) for dim_in, dim_out in zip(dim_ins, dim_outs)]
        )
        self.dim_out = dim + (sum(dim_outs) if len(dim_outs) > 0 else 0)

    @override
    def forward(self, x: Tensor, *, fmaps: Optional[Sequence[Tensor]] = None) -> Tensor:
        target_size = x.shape[-1]

        fmaps = unwrap_or(fmaps, default=())

        if not self.enabled or len(fmaps) == 0 or len(self.fmap_convs) == 0:
            return x

        fmaps = [resize_video_to(fmap, target_image_size=target_size) for fmap in fmaps]
        outs = [conv(fmap) for fmap, conv in zip(fmaps, self.fmap_convs)]
        return torch.cat((x, *outs), dim=1)


class Identity(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # type: ignore
        super().__init__()

    @override
    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:  # type: ignore
        return x


class IterSum(nn.Module):
    def __init__(self, *fns: nn.Module) -> None:
        super().__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x: Tensor) -> Tensor:
        outputs = [fn(x) for fn in self.fns]
        return cast(Tensor, sum(outputs))


class Unet3DImagen(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_resnet_blocks: int = 1,
        dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
        in_channels: int = 3,
        out_channels: Optional[int] = None,
        attn_head_dim: int = 64,
        n_attn_heads: int = 8,
        ff_mult: int = 2,
        layer_attn: bool = False,
        layer_attn_depth: int = 1,
        attend_at_middle: bool = True,
        time_rel_pos_bias_depth: int = 2,
        use_linear_attn: bool = False,
        init_dim: Optional[int] = None,
        resnet_groups: int = 8,
        init_kernel_size: int = 7,
        memory_efficient: bool = False,
        init_conv_to_final_conv_residual=False,
        use_gca: bool = True,
        scale_skip_connection: bool = True,
        final_resnet_block: bool = True,
        final_conv_kernel_size: int = 3,
        cosine_sim_attn: bool = False,
        # combine feature maps from all upsample blocks, used in unet squared
        # successfully
        combine_upsample_fmaps: bool = False,
        # may address checkboard artifacts
        pixel_shuffle_upsample: bool = True,
    ) -> None:
        super().__init__()
        # determine dimensions
        self.channels = in_channels
        self.channels_out = unwrap_or(out_channels, default=in_channels)

        init_channels = in_channels
        init_dim = unwrap_or(init_dim, default=dim)

        # initial convolution

        self.init_conv = pseudo_conv2d(
            in_channels=init_channels,
            out_channels=init_dim,
            kernel_size=init_kernel_size,
            padding=init_kernel_size // 2,
        )

        dims = [init_dim] + [dim * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))

        # attention related params
        attn_kwargs = dict(
            heads=n_attn_heads, dim_head=attn_head_dim, cosine_sim_attn=cosine_sim_attn
        )
        # temporal attention - attention across video frames
        def _temporal_peg(dim: int) -> Residual:
            return Residual(
                nn.Sequential(
                    Pad((0, 0, 0, 0, 1, 1)),
                    nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=(3, 1, 1), groups=dim),
                )
            )

        def _temporal_attn(dim: int) -> EinopsToAndFrom:
            return EinopsToAndFrom(
                from_einops="b c f h w",
                to_einops="(b h w) f c",
                fn=Residual(
                    Attention(
                        dim,
                        **{**attn_kwargs, "init_zero": True},
                    )
                ),
            )

        # temporal attention relative positional encoding

        self.time_rel_pos_bias = DynamicPositionBias(
            dim=dim * 2, heads=n_attn_heads, depth=time_rel_pos_bias_depth
        )

        # initial resnet block (for memory efficient unet)
        self.init_resnet_block = (
            ResnetBlock3d(
                init_dim,
                out_dim=init_dim,
                groups=resnet_groups,
                use_gca=use_gca,
            )
            if memory_efficient
            else None
        )

        self.init_temporal_peg = _temporal_peg(init_dim)
        self.init_temporal_attn = _temporal_attn(init_dim)

        # scale for resnet skip connections

        self.skip_connect_scale = 1.0 if not scale_skip_connection else (2**-0.5)

        # layers

        self.encoder_blocks = nn.ModuleList([])
        self.decoder_blocks = nn.ModuleList([])
        num_resolutions = len(in_out)

        # downsampling layers

        skip_connect_dims = []  # keep track of skip connection dimensions

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            transformer_block_cls = (
                TransformerBlock
                if layer_attn
                else (LinearAttentionTransformerBlock if use_linear_attn else Identity)
            )

            current_dim = dim_in

            # whether to pre-downsample, from memory efficient unet
            if memory_efficient:
                pre_downsample = downsample(dim_in, out_dim=dim_out)
                current_dim = dim_out
            else:
                pre_downsample = nn.Identity()

            skip_connect_dims.append(current_dim)

            # whether to do post-downsample, for non-memory efficient unet

            if memory_efficient:
                post_downsample = nn.Identity()
            else:
                post_downsample = (
                    downsample(in_dim=current_dim, out_dim=dim_out)
                    if not is_last
                    else IterSum(
                        pseudo_conv2d(dim_in, out_channels=dim_out, kernel_size=3, padding=1),
                        pseudo_conv2d(dim_in, out_channels=dim_out, kernel_size=1),
                    )
                )

            self.encoder_blocks.append(
                nn.ModuleList(
                    [
                        pre_downsample,
                        ResnetBlock3d(
                            in_dim=current_dim,
                            out_dim=current_dim,
                            groups=resnet_groups,
                        ),
                        nn.ModuleList(
                            [
                                ResnetBlock3d(
                                    in_dim=current_dim,
                                    out_dim=current_dim,
                                    groups=resnet_groups,
                                    use_gca=use_gca,
                                )
                                for _ in range(num_resnet_blocks)
                            ]
                        ),
                        transformer_block_cls(
                            dim=current_dim,
                            depth=layer_attn_depth,
                            ff_mult=ff_mult,
                            **attn_kwargs,
                        ),
                        _temporal_peg(current_dim),
                        _temporal_attn(current_dim),
                        post_downsample,
                    ]
                )
            )

        # middle layers

        mid_dim = dims[-1]

        self.mid_block1 = ResnetBlock3d(
            in_dim=mid_dim,
            out_dim=mid_dim,
            groups=resnet_groups,
        )
        self.mid_attn = (
            EinopsToAndFrom("b c f h w", "b (f h w) c", Residual(Attention(mid_dim, **attn_kwargs)))
            if attend_at_middle
            else nn.Identity()
        )
        self.mid_temporal_peg = _temporal_peg(mid_dim)
        self.mid_temporal_attn = _temporal_attn(mid_dim)
        self.mid_block2 = ResnetBlock3d(
            in_dim=mid_dim,
            out_dim=mid_dim,
            groups=resnet_groups,
        )

        upsample_cls = PixelShuffleUpsample if pixel_shuffle_upsample else upsample

        # upsampling layers

        upsample_fmap_dims: List[int] = []

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            transformer_block_cls = (
                TransformerBlock
                if layer_attn
                else (LinearAttentionTransformerBlock if use_linear_attn else Identity)
            )

            skip_connect_dim = skip_connect_dims.pop()

            upsample_fmap_dims.append(dim_out)

            self.decoder_blocks.append(
                nn.ModuleList(
                    [
                        ResnetBlock3d(
                            in_dim=dim_out + skip_connect_dim,
                            out_dim=dim_out,
                            groups=resnet_groups,
                        ),
                        nn.ModuleList(
                            [
                                ResnetBlock3d(
                                    in_dim=dim_out + skip_connect_dim,
                                    out_dim=dim_out,
                                    groups=resnet_groups,
                                    use_gca=use_gca,
                                )
                                for _ in range(num_resnet_blocks)
                            ]
                        ),
                        transformer_block_cls(
                            dim=dim_out,
                            depth=layer_attn_depth,
                            ff_mult=ff_mult,
                            **attn_kwargs,
                        ),
                        _temporal_peg(dim_out),
                        _temporal_attn(dim_out),
                        upsample_cls(in_dim=dim_out, out_dim=dim_in)
                        if not is_last or memory_efficient
                        else Identity(),
                    ]
                )
            )

        # whether to combine feature maps from all upsample blocks before final resnet block out
        self.upsample_combiner = UpsampleCombiner(
            dim=dim,
            enabled=combine_upsample_fmaps,
            dim_ins=tuple(upsample_fmap_dims),
            dim_outs=dim,
        )

        # whether to do a final residual from initial conv to the final resnet block out
        self.init_conv_to_final_conv_residual = init_conv_to_final_conv_residual
        final_conv_dim = self.upsample_combiner.dim_out + (
            dim if init_conv_to_final_conv_residual else 0
        )

        # Poolign over the time axis.
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        # final optional resnet block and convolution out
        self.final_res_block = (
            FinalResnetBlock(
                in_dim=final_conv_dim,
                out_dim=dim,
                groups=resnet_groups,
                use_gca=True,
            )
            if final_resnet_block
            else nn.Identity()
        )

        final_conv_dim_in = dim if final_resnet_block else final_conv_dim
        self.final_conv = pseudo_conv2d(
            in_channels=final_conv_dim_in,
            out_channels=self.channels_out,
            kernel_size=final_conv_kernel_size,
            padding=final_conv_kernel_size // 2,
        )

        nn.init.zeros_(self.final_conv.weight)
        if some(self.final_conv.bias):
            nn.init.zeros_(self.final_conv.bias)

    def _add_skip_connection(self, x: Tensor, *, hiddens: List[Tensor]) -> Tensor:
        return torch.cat((x, hiddens.pop() * self.skip_connect_scale), dim=1)

    @override
    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        assert (
            x.ndim == 5
        ), "input to 3d unet must have 5 dimensions (batch, channels, time, height, width)"

        frames, device, dtype = x.size(2), x.device, x.dtype

        # get time relative positions
        time_attn_bias = self.time_rel_pos_bias.forward(frames, device=device, dtype=dtype)
        x = self.init_conv(x)
        x = self.init_temporal_peg(x)
        x = self.init_temporal_attn(x, pos_bias=time_attn_bias)

        # init conv residual
        if self.init_conv_to_final_conv_residual:
            init_conv_residual = x.clone()
        else:
            init_conv_residual = None

        # initial resnet block (for memory efficient unet)
        if some(self.init_resnet_block):
            x = self.init_resnet_block(x)

        # go through the layers of the unet, down and up
        hiddens = []

        for (  # type: ignore
            pre_downsample,
            init_block,
            resnet_blocks,
            attn_block,
            temporal_peg,
            temporal_attn,
            post_downsample,
        ) in self.encoder_blocks:
            x = pre_downsample(x)
            x = init_block(x)

            for resnet_block in resnet_blocks:
                x = resnet_block(x)
                hiddens.append(x)

            x = attn_block(x)
            x = temporal_peg(x)
            x = temporal_attn(x, pos_bias=time_attn_bias)
            hiddens.append(x)
            x = post_downsample(x)

        x = self.mid_block1(x)

        x = self.mid_attn(x)

        x = self.mid_temporal_peg(x)
        x = self.mid_temporal_attn(x, pos_bias=time_attn_bias)

        x = self.mid_block2(x)

        enc_hiddens = []

        for (  # type: ignore
            init_block,
            resnet_blocks,
            attn_block,
            temporal_peg,
            temporal_attn,
            upsample,
        ) in self.decoder_blocks:
            x = self._add_skip_connection(x, hiddens=hiddens)
            x = init_block(x)

            for resnet_block in resnet_blocks:
                x = self._add_skip_connection(x, hiddens=hiddens)
                x = resnet_block(x)

            x = attn_block(x)
            x = temporal_peg(x)
            x = temporal_attn(x, pos_bias=time_attn_bias)

            enc_hiddens.append(x.contiguous())
            x = upsample(x)

        # whether to combine all feature maps from upsample blocks
        x = self.upsample_combiner.forward(x, fmaps=enc_hiddens)

        # final top-most residual if needed

        if some(init_conv_residual):
            x = torch.cat((x, init_conv_residual), dim=1)

        x = self.temporal_pool(x)
        x = self.final_res_block(x)

        return self.final_conv(x)
