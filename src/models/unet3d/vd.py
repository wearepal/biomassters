# Adapted shamelessly from https://github.com/lucidrains/video-diffusion-pytorch
import math
from typing import Any, List, Literal, Optional, Sequence, Tuple, Union, cast

from einops import rearrange  # type: ignore
from einops_exts import rearrange_many  # type: ignore
from rotary_embedding_torch import RotaryEmbedding  # type: ignore
import torch
from torch import Tensor, einsum, nn
import torch.nn.functional as F
from typing_extensions import override

from src.utils import some, unwrap_or

from .common import (
    ChanLayerNorm,
    EtaPool,
    GlobalContextAttention,
    PixelShuffleUpsample,
    Residual,
    cast_tuple,
)

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


def down_conv(in_channels: int, *, out_channels: Optional[int] = None) -> nn.Conv3d:
    out_channels = unwrap_or(out_channels, default=in_channels)
    return nn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(1, 4, 4),
        stride=(1, 2, 2),
        padding=(0, 1, 1),
    )


class PreNorm(nn.Module):
    def __init__(self, dim: int, *, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = ChanLayerNorm(dim)

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
        self.gca = GlobalContextAttention(in_dim=out_dim, out_dim=out_dim) if use_gca else None

    @override
    def forward(self, x: Tensor) -> Tensor:
        h = self.block1(x)
        h = self.block2(h)
        if self.gca:
            h = h * self.gca(h)
        return h + self.res_conv(x)


class SpatialLinearAttention(nn.Module):
    def __init__(
        self, dim: int, *, heads: int = 4, dim_head: int = 32, dropout: float = 0.05
    ) -> None:
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(dim, out_channels=hidden_dim * 3, kernel_size=1, bias=False),
        )
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
        if some(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

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


def temporal_attention(
    dim: int,
    n_attn_heads: int,
    attn_head_dim: int,
    rotary_emb: Optional[RotaryEmbedding],
    cosine_sim_attn: bool = False,
) -> EinopsToAndFrom:
    return EinopsToAndFrom(
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


def resize_video_to(
    video: Tensor,
    *,
    target_image_size: Union[int, Tuple[int, int]],
    clamp_range: Optional[Tuple[int, int]] = None,
    mode: Literal["nearest", "linear", "bilinear", "bicubic"] = "bilinear",
) -> Tensor:
    orig_video_size = video.shape[-1]

    if orig_video_size == target_image_size:
        return video

    frames = video.shape[2]
    video = rearrange(video, "b c f h w -> (b f) c h w")

    out = F.interpolate(video, target_image_size, mode=mode)

    if some(clamp_range):
        out = out.clamp(*clamp_range)

    out = rearrange(out, "(b f) c h w -> b c f h w", f=frames)

    return out


class UpsampleCombiner(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        dim_ins: Tuple[int, ...] = (),
        dim_outs: Union[int, Tuple[int, ...]] = (),
    ) -> None:
        super().__init__()
        dim_outs = cast_tuple(dim_outs, length=len(dim_ins))
        assert len(dim_ins) == len(dim_outs)

        self.fmap_convs = nn.ModuleList(
            [Block3d(in_dim=dim_in, out_dim=dim_out) for dim_in, dim_out in zip(dim_ins, dim_outs)]
        )
        self.dim_out = dim + (sum(dim_outs) if len(dim_outs) > 0 else 0)

    @override
    def forward(self, x: Tensor, *, fmaps: Optional[Sequence[Tensor]] = None) -> Tensor:
        target_size = x.shape[-1]

        fmaps = unwrap_or(fmaps, default=())

        if (len(fmaps) == 0) or (len(self.fmap_convs) == 0):
            return x

        fmaps = [resize_video_to(fmap, target_image_size=target_size) for fmap in fmaps]
        outs = [conv(fmap) for fmap, conv in zip(fmaps, self.fmap_convs)]
        return torch.cat((x, *outs), dim=1)


class EncoderStage(nn.Module):
    def __init__(
        self,
        *,
        attn_head_dim: int,
        cosine_sim_attn: bool,
        dim_in: int,
        dim_out: int,
        is_last: bool,
        groups: int,
        n_attn_heads: int,
        rotary_emb: Optional[RotaryEmbedding],
        temporal_pooling: bool,
        use_gca: bool,
        use_sparse_linear_attn: bool,
        num_resnet_blocks: int,
        memory_efficient: bool,
    ) -> None:
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        current_dim = dim_in
        if memory_efficient:
            self.pre_downsample = (
                down_conv(dim_in, out_channels=dim_out) if memory_efficient else nn.Identity()
            )
            current_dim = dim_out
            self.post_downsample = nn.Identity()
        else:
            self.pre_downsample = nn.Identity()
            self.post_downsample = nn.Identity() if is_last else down_conv(dim_out)

        self.init_rn_block = ResnetBlock3d(
            in_dim=current_dim,
            out_dim=dim_out,
            groups=groups,
            use_gca=use_gca,
        )

        def _block(_in_dim: int):
            return nn.Sequential(
                ResnetBlock3d(
                    in_dim=_in_dim,
                    out_dim=dim_out,
                    groups=groups,
                    use_gca=use_gca,
                ),
                EtaPool(dim_out, kernel_size=3) if temporal_pooling else nn.Identity(),
            )

        self.rn_blocks = nn.ModuleList([_block(dim_out) for _ in range(num_resnet_blocks)])

        self.spatial_attn = (
            Residual(PreNorm(dim_out, fn=SpatialLinearAttention(dim_out, heads=n_attn_heads)))
            if use_sparse_linear_attn
            else nn.Identity()
        )

        self.temporal_attn = Residual(
            PreNorm(
                dim_out,
                fn=temporal_attention(
                    dim=dim_out,
                    n_attn_heads=n_attn_heads,
                    attn_head_dim=attn_head_dim,
                    rotary_emb=rotary_emb,
                    cosine_sim_attn=cosine_sim_attn,
                ),
            )
        )
        self.final_pool = EtaPool(dim_out, kernel_size=3) if temporal_pooling else nn.Identity()

    @override
    def forward(self, x: Tensor, *, pos_bias: Tensor) -> Tuple[Tensor, List[Tensor]]:
        x = self.pre_downsample(x)
        x = self.init_rn_block(x)
        shortcuts = []
        for block in self.rn_blocks:
            block = cast(nn.Sequential, block)
            x = block.forward(x)
            shortcuts.append(x)
        x = self.spatial_attn(x)
        x = self.temporal_attn(x, pos_bias=pos_bias)
        shortcuts.append(self.final_pool.forward(x))
        return self.post_downsample(x), shortcuts


def _add_skip_connection(
    x: Tensor, *, skip_connections: List[Tensor], scale: float = 1.0
) -> Tensor:
    return torch.cat((x, skip_connections.pop() * scale), dim=1)


class DecoderStage(nn.Module):
    def __init__(
        self,
        *,
        attn_head_dim: int,
        cosine_sim_attn: bool,
        dim_in: int,
        dim_out: int,
        groups: int,
        is_last: bool,
        memory_efficient: bool,
        n_attn_heads: int,
        num_resnet_blocks: int,
        pixel_shuffle: bool,
        rotary_emb: Optional[RotaryEmbedding],
        skip_connect_scale: float,
        skip_connect_dim: int,
        spatial_only: bool,
        use_gca: bool,
        use_sparse_linear_attn: bool,
    ) -> None:
        super().__init__()
        self.skip_connect_scale = skip_connect_scale
        self.init_rn_block = ResnetBlock3d(
            # each block receives the concateation of the
            # previous stage's output and the skip connection
            # from the encoder.
            in_dim=dim_in + skip_connect_dim,
            out_dim=dim_out,
            groups=groups,
            use_gca=use_gca,
        )

        self.rn_blocks = nn.ModuleList(
            [
                ResnetBlock3d(
                    in_dim=dim_out + skip_connect_dim,
                    out_dim=dim_out,
                    groups=groups,
                    use_gca=use_gca,
                )
                for _ in range(num_resnet_blocks)
            ]
        )
        self.spatial_attn = (
            Residual(PreNorm(dim_out, fn=SpatialLinearAttention(dim_out, heads=n_attn_heads)))
            if use_sparse_linear_attn
            else nn.Identity()
        )

        self.temporal_attn = (
            None
            if spatial_only
            else Residual(
                PreNorm(
                    dim_out,
                    fn=temporal_attention(
                        dim=dim_out,
                        n_attn_heads=n_attn_heads,
                        attn_head_dim=attn_head_dim,
                        rotary_emb=rotary_emb,
                        cosine_sim_attn=cosine_sim_attn,
                    ),
                )
            )
        )
        if (not is_last) or memory_efficient:
            self.upsample = PixelShuffleUpsample(dim_out) if pixel_shuffle else up_conv(dim_out)
        else:
            self.upsample = nn.Identity()

    @override
    def forward(
        self, x: Tensor, *, skip_connections: List[Tensor], pos_bias: Tensor
    ) -> Tuple[Tensor, Tensor]:
        x = _add_skip_connection(
            x, skip_connections=skip_connections, scale=self.skip_connect_scale
        )
        x = self.init_rn_block(x)
        for block in self.rn_blocks:
            block = cast(ResnetBlock3d, block)
            x = _add_skip_connection(
                x, skip_connections=skip_connections, scale=self.skip_connect_scale
            )
            x = block.forward(x)
        x = self.spatial_attn(x)
        if some(self.temporal_attn):
            x = self.temporal_attn(x, pos_bias=pos_bias)
        return self.upsample(x), x


class MiddleStage(nn.Module):
    def __init__(
        self,
        *,
        apply_mid_spatial_attn: bool,
        attn_head_dim: int,
        cosine_sim_attn: bool,
        dim: int,
        groups: int,
        n_attn_heads: int,
        rotary_emb: Optional[RotaryEmbedding],
        temporal_pooling: bool,
        use_gca: bool,
    ) -> None:
        super().__init__()

        self.mid_block1 = ResnetBlock3d(in_dim=dim, out_dim=dim)

        spatial_attn = EinopsToAndFrom(
            from_einops="b c f h w",
            to_einops="b f (h w) c",
            fn=Attention(dim, heads=n_attn_heads),
        )

        self.mid_spatial_attn = (
            Residual(PreNorm(dim, fn=spatial_attn)) if apply_mid_spatial_attn else nn.Identity()
        )

        self.mid_temporal_attn = Residual(
            PreNorm(
                dim,
                fn=temporal_attention(
                    dim=dim,
                    n_attn_heads=n_attn_heads,
                    attn_head_dim=attn_head_dim,
                    rotary_emb=rotary_emb,
                    cosine_sim_attn=cosine_sim_attn,
                ),
            )
        )

        self.mid_block2 = ResnetBlock3d(
            in_dim=dim,
            out_dim=dim,
            groups=groups,
            use_gca=use_gca,
        )
        self.temporal_pool = EtaPool(dim, kernel_size=3) if temporal_pooling else nn.Identity()

    @override
    def forward(self, x: Tensor, *, pos_bias: Tensor) -> Tensor:
        x = self.mid_block1(x)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(x, pos_bias=pos_bias)
        x = self.temporal_pool(x)
        return x


# model
class Unet3dVd(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        dim: int = 64,
        out_channels: Optional[int] = None,
        dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_attn_heads: int = 8,
        attn_head_dim: int = 32,
        init_dim: Optional[int] = None,
        init_kernel_size: int = 7,
        use_sparse_linear_attn: bool = True,
        resnet_groups: int = 8,
        max_distance: int = 11,
        spatial_decoder: bool = False,
        use_gca: bool = False,
        cosine_sim_attn: bool = False,
        apply_mid_spatial_attn: bool = True,
        num_resnet_blocks: int = 1,
        pixel_shuffle: bool = False,
        scale_skip_connection: bool = True,
        memory_efficient: bool = False,
        combine_upsample_fmaps: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = unwrap_or(out_channels, default=in_channels)
        # Reduce along the temporal dimension prior to decoding.
        # For simplicity of implementation, we retain the 3d convs, applying
        # them over single-frame inputs.
        self.spatial_decoder = spatial_decoder
        self.memory_efficient = memory_efficient
        self.skip_connect_scale = 1.0 if not scale_skip_connection else (2**-0.5)

        # temporal attention and its relative positional encoding
        rotary_emb = RotaryEmbedding(min(32, attn_head_dim))
        temporal_attn = lambda dim: EinopsToAndFrom(
            from_einops="b c f h w",
            to_einops="b (h w) f c",
            fn=Attention(
                dim,
                heads=num_attn_heads,
                dim_head=attn_head_dim,
                rotary_emb=rotary_emb,
                cosine_sim_attn=cosine_sim_attn,
            ),
        )

        self.time_rel_pos_bias = RelativePositionBias(
            heads=num_attn_heads, max_distance=max_distance
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

        self.init_downsampling_block = (
            ResnetBlock3d(
                init_dim,
                out_dim=init_dim,
                groups=resnet_groups,
                use_gca=use_gca,
            )
            if memory_efficient
            else nn.Identity()
        )

        self.init_temporal_attn = Residual(PreNorm(init_dim, fn=temporal_attn(init_dim)))
        self.init_temporal_pool = (
            EtaPool(init_dim, kernel_size=3) if spatial_decoder else nn.Identity()
        )

        # dimensions
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # layers
        self.encoder_stages = nn.ModuleList([])
        self.decoder_stages = nn.ModuleList([])
        # keep track of skip connection dimensions
        skip_connect_dims = []

        num_resolutions = len(in_out)
        # modules for all layers
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            stage = EncoderStage(
                dim_in=dim_in,
                dim_out=dim_out,
                groups=resnet_groups,
                use_gca=use_gca,
                temporal_pooling=spatial_decoder,
                is_last=is_last,
                n_attn_heads=num_attn_heads,
                attn_head_dim=attn_head_dim,
                rotary_emb=rotary_emb,
                cosine_sim_attn=cosine_sim_attn,
                use_sparse_linear_attn=use_sparse_linear_attn,
                num_resnet_blocks=num_resnet_blocks,
                memory_efficient=memory_efficient,
            )
            skip_connect_dims.append(dim_out)
            self.encoder_stages.append(stage)

        mid_dim = dims[-1]
        self.middle_stage = MiddleStage(
            dim=mid_dim,
            groups=resnet_groups,
            use_gca=use_gca,
            n_attn_heads=num_attn_heads,
            attn_head_dim=attn_head_dim,
            rotary_emb=rotary_emb,
            cosine_sim_attn=cosine_sim_attn,
            apply_mid_spatial_attn=apply_mid_spatial_attn,
            temporal_pooling=spatial_decoder,
        )

        upsample_fmap_dims = []
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)
            upsample_fmap_dims.append(dim_in)
            self.decoder_stages.append(
                DecoderStage(
                    dim_in=dim_out,
                    dim_out=dim_in,
                    groups=resnet_groups,
                    use_gca=use_gca,
                    is_last=is_last,
                    n_attn_heads=num_attn_heads,
                    attn_head_dim=attn_head_dim,
                    rotary_emb=rotary_emb,
                    cosine_sim_attn=cosine_sim_attn,
                    spatial_only=spatial_decoder,
                    use_sparse_linear_attn=use_sparse_linear_attn,
                    num_resnet_blocks=num_resnet_blocks,
                    pixel_shuffle=pixel_shuffle,
                    skip_connect_scale=self.skip_connect_scale,
                    skip_connect_dim=skip_connect_dims.pop(),
                    memory_efficient=memory_efficient,
                )
            )

        if combine_upsample_fmaps:
            self.upsample_combiner = UpsampleCombiner(
                dim=dim,
                dim_ins=tuple(upsample_fmap_dims),
                dim_outs=dim,
            )
            # init residual dim + dim of the tensor resulting from the
            # concatenation of the upsampled (and convolved) fmaps and
            # decoder's output.
            final_conv_dim = dim + self.upsample_combiner.dim_out
        else:
            self.upsample_combiner = None
            # dim of decoder's output + init residual dim
            final_conv_dim = dim * 2

        self.final_temporal_pool = EtaPool(final_conv_dim, kernel_size=3)

        # Final (2d) conv block operating on temporally-pooled features
        self.final_conv = nn.Sequential(
            ResnetBlock3d(in_dim=final_conv_dim, out_dim=dim, groups=resnet_groups, use_gca=True),
            nn.Conv3d(in_channels=dim, out_channels=self.out_channels, kernel_size=1),
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        time_rel_pos_bias = self.time_rel_pos_bias(x.size(2), device=x.device)
        x = self.init_conv(x)
        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)
        init_conv_residual = x.clone()
        init_conv_residual = self.init_temporal_pool(init_conv_residual)
        x = self.init_downsampling_block(x)

        skip_connections = []
        for stage in self.encoder_stages:
            stage = cast(EncoderStage, stage)
            x, stage_scs = stage.forward(x, pos_bias=time_rel_pos_bias)
            skip_connections.extend(stage_scs)

        x = self.middle_stage.forward(x, pos_bias=time_rel_pos_bias)

        fmaps = []
        for stage in self.decoder_stages:
            stage = cast(DecoderStage, stage)
            x, stage_fmap = stage.forward(
                x, skip_connections=skip_connections, pos_bias=time_rel_pos_bias
            )
            fmaps.append(stage_fmap)

        # whether to combine all feature maps from upsample blocks
        if some(self.upsample_combiner):
            x = self.upsample_combiner.forward(x, fmaps=fmaps)

        x = torch.cat((x, init_conv_residual), dim=1)
        # temporal pooling
        x = self.final_temporal_pool(x)
        return self.final_conv(x).squeeze(dim=2)
