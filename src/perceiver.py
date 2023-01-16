from typing import Optional

from einops import rearrange, repeat  # type: ignore
from einops.layers.torch import Rearrange  # type: ignore
from einops_exts import rearrange_many  # type: ignore
import torch
from torch import Tensor, einsum, nn
import torch.nn.functional as F
from typing_extensions import override

from src.utils import some

__all__ = [
    "PerceiverAttention",
    "PerceiverResampler",
]


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

        h = self.heads

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

        if some(mask):
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

    @override
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        n, device = x.shape[1], x.device
        pos_emb = self.pos_emb(torch.arange(n, device=device))

        x_with_pos = x + pos_emb

        latents = repeat(self.latents, "n d -> b n d", b=x.shape[0])

        if some(self.to_latents_from_mean_pooled_seq):
            meanpooled_seq = masked_mean(
                x, dim=1, mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
            )
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents, latents), dim=-2)

        for attn, ff in self.layers:  # type: ignore
            latents = attn(x_with_pos, latents, mask=mask) + latents
            latents = ff(latents) + latents

        return latents
