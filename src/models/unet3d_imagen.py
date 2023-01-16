import math
from typing import Any, Optional, Tuple, TypeVar, Union, cast

from einops import rearrange, repeat  # type: ignore
from einops.layers.torch import Rearrange  # type: ignore
from einops_exts import rearrange_many  # type: ignore
from rotary_embedding_torch import RotaryEmbedding  # type: ignore
import torch
from torch import Tensor, einsum, nn
import torch.nn.functional as F
from typing_extensions import override

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


V = TypeVar("V")


def cast_tuple(val: Union[V, Tuple[V, ...]], *, length: Optional[int] = None) -> Tuple[V, ...]:
    if isinstance(val, list):
        val = tuple(val)

    output = val if isinstance(val, tuple) else ((val,) * (1 if length is None else length))

    if length is not None:
        assert len(output) == length

    return output


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


class Unet3DImagen(nn.Module):
    def __init__(
        self,
        *,
        dim,
        image_embed_dim=1024,
        num_resnet_blocks=1,
        num_image_tokens=4,
        num_time_tokens=2,
        learned_sinu_pos_emb_dim=16,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        channels_out=None,
        attn_dim_head=64,
        attn_heads=8,
        ff_mult=2.0,
        layer_attns=False,
        layer_attns_depth=1,
        attend_at_middle=True,  # whether to have a layer of attention at the bottleneck (can turn off for higher resolution in cascading DDPM, before bringing in efficient attention)
        time_rel_pos_bias_depth=2,
        use_linear_attn=False,
        init_dim=None,
        resnet_groups=8,
        init_conv_kernel_size=7,  # kernel size of initial conv, if not using cross embed
        init_cross_embed=True,
        init_cross_embed_kernel_sizes=(3, 7, 15),
        dropout=0.0,
        memory_efficient=False,
        init_conv_to_final_conv_residual=False,
        use_global_context_attn=True,
        scale_skip_connection=True,
        final_resnet_block=True,
        final_conv_kernel_size=3,
        cosine_sim_attn=False,
        combine_upsample_fmaps=False,  # combine feature maps from all upsample blocks, used in unet squared successfully
        pixel_shuffle_upsample=True,  # may address checkboard artifacts
    ):
        super().__init__()

        # guide researchers

        assert (
            attn_heads > 1
        ), "you need to have more than 1 attention head, ideally at least 4 or 8"

        if dim < 128:
            print_once(
                "The base dimension of your u-net should ideally be no smaller than 128, as recommended by a professional DDPM trainer https://nonint.com/2022/05/04/friends-dont-let-friends-train-small-diffusion-models/"
            )

        # save locals to take care of some hyperparameters for cascading DDPM

        self._locals = locals()
        self._locals.pop("self", None)
        self._locals.pop("__class__", None)

        self.self_cond = self_cond

        # determine dimensions

        self.channels = channels
        self.channels_out = default(channels_out, channels)

        # (1) in cascading diffusion, one concats the low resolution image, blurred, for conditioning the higher resolution synthesis
        # (2) in self conditioning, one appends the predict x0 (x_start)
        init_channels = channels * (1 + int(lowres_cond) + int(self_cond))
        init_dim = default(init_dim, dim)

        # optional image conditioning

        self.has_cond_image = cond_images_channels > 0
        self.cond_images_channels = cond_images_channels

        init_channels += cond_images_channels

        # initial convolution

        self.init_conv = (
            CrossEmbedLayer(
                init_channels,
                dim_out=init_dim,
                kernel_sizes=init_cross_embed_kernel_sizes,
                stride=1,
            )
            if init_cross_embed
            else Conv2d(
                init_channels, init_dim, init_conv_kernel_size, padding=init_conv_kernel_size // 2
            )
        )

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time conditioning

        cond_dim = default(cond_dim, dim)
        time_cond_dim = dim * 4 * (2 if lowres_cond else 1)

        # embedding time for log(snr) noise from continuous version

        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim)
        sinu_pos_emb_input_dim = learned_sinu_pos_emb_dim + 1

        self.to_time_hiddens = nn.Sequential(
            sinu_pos_emb, nn.Linear(sinu_pos_emb_input_dim, time_cond_dim), nn.SiLU()
        )

        self.to_time_cond = nn.Sequential(nn.Linear(time_cond_dim, time_cond_dim))

        # project to time tokens as well as time hiddens

        self.to_time_tokens = nn.Sequential(
            nn.Linear(time_cond_dim, cond_dim * num_time_tokens),
            Rearrange("b (r d) -> b r d", r=num_time_tokens),
        )

        # low res aug noise conditioning

        self.lowres_cond = lowres_cond

        if lowres_cond:
            self.to_lowres_time_hiddens = nn.Sequential(
                LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim),
                nn.Linear(learned_sinu_pos_emb_dim + 1, time_cond_dim),
                nn.SiLU(),
            )

            self.to_lowres_time_cond = nn.Sequential(nn.Linear(time_cond_dim, time_cond_dim))

            self.to_lowres_time_tokens = nn.Sequential(
                nn.Linear(time_cond_dim, cond_dim * num_time_tokens),
                Rearrange("b (r d) -> b r d", r=num_time_tokens),
            )

        # normalizations

        self.norm_cond = nn.LayerNorm(cond_dim)

        # text encoding conditioning (optional)

        self.text_to_cond = None

        if cond_on_text:
            assert exists(
                text_embed_dim
            ), "text_embed_dim must be given to the unet if cond_on_text is True"
            self.text_to_cond = nn.Linear(text_embed_dim, cond_dim)

        # finer control over whether to condition on text encodings

        self.cond_on_text = cond_on_text

        # attention pooling

        self.attn_pool = (
            PerceiverResampler(
                dim=cond_dim,
                depth=2,
                dim_head=attn_dim_head,
                heads=attn_heads,
                num_latents=attn_pool_num_latents,
                cosine_sim_attn=cosine_sim_attn,
            )
            if attn_pool_text
            else None
        )

        # for classifier free guidance

        self.max_text_len = max_text_len

        self.null_text_embed = nn.Parameter(torch.randn(1, max_text_len, cond_dim))
        self.null_text_hidden = nn.Parameter(torch.randn(1, time_cond_dim))

        # for non-attention based text conditioning at all points in the network where time is also conditioned

        self.to_text_non_attn_cond = None

        if cond_on_text:
            self.to_text_non_attn_cond = nn.Sequential(
                nn.LayerNorm(cond_dim),
                nn.Linear(cond_dim, time_cond_dim),
                nn.SiLU(),
                nn.Linear(time_cond_dim, time_cond_dim),
            )

        # attention related params

        attn_kwargs = dict(
            heads=attn_heads, dim_head=attn_dim_head, cosine_sim_attn=cosine_sim_attn
        )

        num_layers = len(in_out)

        # temporal attention - attention across video frames

        temporal_peg_padding = (0, 0, 0, 0, 2, 0) if time_causal_attn else (0, 0, 0, 0, 1, 1)
        temporal_peg = lambda dim: Residual(
            nn.Sequential(Pad(temporal_peg_padding), nn.Conv3d(dim, dim, (3, 1, 1), groups=dim))
        )

        temporal_attn = lambda dim: EinopsToAndFrom(
            "b c f h w",
            "(b h w) f c",
            Residual(
                Attention(dim, **{**attn_kwargs, "causal": time_causal_attn, "init_zero": True})
            ),
        )

        # temporal attention relative positional encoding

        self.time_rel_pos_bias = DynamicPositionBias(
            dim=dim * 2, heads=attn_heads, depth=time_rel_pos_bias_depth
        )

        # resnet block klass

        num_resnet_blocks = cast_tuple(num_resnet_blocks, num_layers)
        resnet_groups = cast_tuple(resnet_groups, num_layers)

        resnet_klass = partial(ResnetBlock, **attn_kwargs)

        layer_attns = cast_tuple(layer_attns, num_layers)
        layer_attns_depth = cast_tuple(layer_attns_depth, num_layers)
        layer_cross_attns = cast_tuple(layer_cross_attns, num_layers)

        assert all(
            [
                layers == num_layers
                for layers in list(map(len, (resnet_groups, layer_attns, layer_cross_attns)))
            ]
        )

        # downsample klass

        downsample_klass = Downsample

        if cross_embed_downsample:
            downsample_klass = partial(
                CrossEmbedLayer, kernel_sizes=cross_embed_downsample_kernel_sizes
            )

        # initial resnet block (for memory efficient unet)

        self.init_resnet_block = (
            resnet_klass(
                init_dim,
                init_dim,
                time_cond_dim=time_cond_dim,
                groups=resnet_groups[0],
                use_gca=use_global_context_attn,
            )
            if memory_efficient
            else None
        )

        self.init_temporal_peg = temporal_peg(init_dim)
        self.init_temporal_attn = temporal_attn(init_dim)

        # scale for resnet skip connections

        self.skip_connect_scale = 1.0 if not scale_skip_connection else (2**-0.5)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        layer_params = [
            num_resnet_blocks,
            resnet_groups,
            layer_attns,
            layer_attns_depth,
            layer_cross_attns,
        ]
        reversed_layer_params = list(map(reversed, layer_params))

        # downsampling layers

        skip_connect_dims = []  # keep track of skip connection dimensions

        for ind, (
            (dim_in, dim_out),
            layer_num_resnet_blocks,
            groups,
            layer_attn,
            layer_attn_depth,
            layer_cross_attn,
        ) in enumerate(zip(in_out, *layer_params)):
            is_last = ind >= (num_resolutions - 1)

            layer_use_linear_cross_attn = not layer_cross_attn and use_linear_cross_attn
            layer_cond_dim = cond_dim if layer_cross_attn or layer_use_linear_cross_attn else None

            transformer_block_klass = (
                TransformerBlock
                if layer_attn
                else (LinearAttentionTransformerBlock if use_linear_attn else Identity)
            )

            current_dim = dim_in

            # whether to pre-downsample, from memory efficient unet

            pre_downsample = None

            if memory_efficient:
                pre_downsample = downsample_klass(dim_in, dim_out)
                current_dim = dim_out

            skip_connect_dims.append(current_dim)

            # whether to do post-downsample, for non-memory efficient unet

            post_downsample = None
            if not memory_efficient:
                post_downsample = (
                    downsample_klass(current_dim, dim_out)
                    if not is_last
                    else Parallel(Conv2d(dim_in, dim_out, 3, padding=1), Conv2d(dim_in, dim_out, 1))
                )

            self.downs.append(
                nn.ModuleList(
                    [
                        pre_downsample,
                        resnet_klass(
                            current_dim,
                            current_dim,
                            cond_dim=layer_cond_dim,
                            linear_attn=layer_use_linear_cross_attn,
                            time_cond_dim=time_cond_dim,
                            groups=groups,
                        ),
                        nn.ModuleList(
                            [
                                ResnetBlock(
                                    current_dim,
                                    current_dim,
                                    time_cond_dim=time_cond_dim,
                                    groups=groups,
                                    use_gca=use_global_context_attn,
                                )
                                for _ in range(layer_num_resnet_blocks)
                            ]
                        ),
                        transformer_block_klass(
                            dim=current_dim,
                            depth=layer_attn_depth,
                            ff_mult=ff_mult,
                            context_dim=cond_dim,
                            **attn_kwargs,
                        ),
                        temporal_peg(current_dim),
                        temporal_attn(current_dim),
                        post_downsample,
                    ]
                )
            )

        # middle layers

        mid_dim = dims[-1]

        self.mid_block1 = ResnetBlock(
            mid_dim,
            mid_dim,
            cond_dim=cond_dim,
            time_cond_dim=time_cond_dim,
            groups=resnet_groups[-1],
        )
        self.mid_attn = (
            EinopsToAndFrom("b c f h w", "b (f h w) c", Residual(Attention(mid_dim, **attn_kwargs)))
            if attend_at_middle
            else None
        )
        self.mid_temporal_peg = temporal_peg(mid_dim)
        self.mid_temporal_attn = temporal_attn(mid_dim)
        self.mid_block2 = ResnetBlock(
            mid_dim,
            mid_dim,
            cond_dim=cond_dim,
            time_cond_dim=time_cond_dim,
            groups=resnet_groups[-1],
        )

        # upsample klass

        upsample_klass = Upsample if not pixel_shuffle_upsample else PixelShuffleUpsample

        # upsampling layers

        upsample_fmap_dims = []

        for ind, (
            (dim_in, dim_out),
            layer_num_resnet_blocks,
            groups,
            layer_attn,
            layer_attn_depth,
            layer_cross_attn,
        ) in enumerate(zip(reversed(in_out), *reversed_layer_params)):
            is_last = ind == (len(in_out) - 1)
            layer_use_linear_cross_attn = not layer_cross_attn and use_linear_cross_attn
            layer_cond_dim = cond_dim if layer_cross_attn or layer_use_linear_cross_attn else None
            transformer_block_klass = (
                TransformerBlock
                if layer_attn
                else (LinearAttentionTransformerBlock if use_linear_attn else Identity)
            )

            skip_connect_dim = skip_connect_dims.pop()

            upsample_fmap_dims.append(dim_out)

            self.ups.append(
                nn.ModuleList(
                    [
                        resnet_klass(
                            dim_out + skip_connect_dim,
                            dim_out,
                            cond_dim=layer_cond_dim,
                            linear_attn=layer_use_linear_cross_attn,
                            time_cond_dim=time_cond_dim,
                            groups=groups,
                        ),
                        nn.ModuleList(
                            [
                                ResnetBlock(
                                    dim_out + skip_connect_dim,
                                    dim_out,
                                    time_cond_dim=time_cond_dim,
                                    groups=groups,
                                    use_gca=use_global_context_attn,
                                )
                                for _ in range(layer_num_resnet_blocks)
                            ]
                        ),
                        transformer_block_klass(
                            dim=dim_out,
                            depth=layer_attn_depth,
                            ff_mult=ff_mult,
                            context_dim=cond_dim,
                            **attn_kwargs,
                        ),
                        temporal_peg(dim_out),
                        temporal_attn(dim_out),
                        upsample_klass(dim_out, dim_in)
                        if not is_last or memory_efficient
                        else Identity(),
                    ]
                )
            )

        # whether to combine feature maps from all upsample blocks before final resnet block out

        self.upsample_combiner = UpsampleCombiner(
            dim=dim, enabled=combine_upsample_fmaps, dim_ins=upsample_fmap_dims, dim_outs=dim
        )

        # whether to do a final residual from initial conv to the final resnet block out

        self.init_conv_to_final_conv_residual = init_conv_to_final_conv_residual
        final_conv_dim = self.upsample_combiner.dim_out + (
            dim if init_conv_to_final_conv_residual else 0
        )

        # final optional resnet block and convolution out

        self.final_res_block = (
            ResnetBlock(
                final_conv_dim,
                dim,
                time_cond_dim=time_cond_dim,
                groups=resnet_groups[0],
                use_gca=True,
            )
            if final_resnet_block
            else None
        )

        final_conv_dim_in = dim if final_resnet_block else final_conv_dim
        final_conv_dim_in += channels if lowres_cond else 0

        self.final_conv = Conv2d(
            final_conv_dim_in,
            self.channels_out,
            final_conv_kernel_size,
            padding=final_conv_kernel_size // 2,
        )

        zero_init_(self.final_conv)

    # if the current settings for the unet are not correct
    # for cascading DDPM, then reinit the unet with the right settings
    def cast_model_parameters(
        self, *, lowres_cond, text_embed_dim, channels, channels_out, cond_on_text
    ):
        if (
            lowres_cond == self.lowres_cond
            and channels == self.channels
            and cond_on_text == self.cond_on_text
            and text_embed_dim == self._locals["text_embed_dim"]
            and channels_out == self.channels_out
        ):
            return self

        updated_kwargs = dict(
            lowres_cond=lowres_cond,
            text_embed_dim=text_embed_dim,
            channels=channels,
            channels_out=channels_out,
            cond_on_text=cond_on_text,
        )

        return self.__class__(**{**self._locals, **updated_kwargs})

    # methods for returning the full unet config as well as its parameter state

    def to_config_and_state_dict(self):
        return self._locals, self.state_dict()

    # class method for rehydrating the unet from its config and state dict

    @classmethod
    def from_config_and_state_dict(klass, config, state_dict):
        unet = klass(**config)
        unet.load_state_dict(state_dict)
        return unet

    # methods for persisting unet to disk

    def persist_to_file(self, path):
        path = Path(path)
        path.parents[0].mkdir(exist_ok=True, parents=True)

        config, state_dict = self.to_config_and_state_dict()
        pkg = dict(config=config, state_dict=state_dict)
        torch.save(pkg, str(path))

    # class method for rehydrating the unet from file saved with `persist_to_file`

    @classmethod
    def hydrate_from_file(klass, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path))

        assert "config" in pkg and "state_dict" in pkg
        config, state_dict = pkg["config"], pkg["state_dict"]

        return Unet.from_config_and_state_dict(config, state_dict)

    # forward with classifier free guidance

    def forward_with_cond_scale(self, *args, cond_scale=1.0, **kwargs):
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob=1.0, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        *,
        cond_drop_prob=0.0,
        ignore_time=False,
    ):
        assert (
            x.ndim == 5
        ), "input to 3d unet must have 5 dimensions (batch, channels, time, height, width)"

        batch_size, frames, device, dtype = x.size(0), x.shape(2), x.device, x.dtype

        # get time relative positions

        time_attn_bias = self.time_rel_pos_bias(frames, device=device, dtype=dtype)

        # ignoring time in pseudo 3d resnet blocks

        conv_kwargs = dict(ignore_time=ignore_time)

        # initial convolution

        x = self.init_conv(x)

        if not ignore_time:
            x = self.init_temporal_peg(x)
            x = self.init_temporal_attn(x, attn_bias=time_attn_bias)

        # init conv residual

        if self.init_conv_to_final_conv_residual:
            init_conv_residual = x.clone()

        text_tokens = None

        # initial resnet block (for memory efficient unet)

        if self.init_resnet_block is not None:
            x = self.init_resnet_block(x, t, **conv_kwargs)

        # go through the layers of the unet, down and up

        hiddens = []

        for (
            pre_downsample,
            init_block,
            resnet_blocks,
            attn_block,
            temporal_peg,
            temporal_attn,
            post_downsample,
        ) in self.downs:
            if exists(pre_downsample):
                x = pre_downsample(x)

            x = init_block(x, t, c, **conv_kwargs)

            for resnet_block in resnet_blocks:
                x = resnet_block(x, t, **conv_kwargs)
                hiddens.append(x)

            x = attn_block(x, c)

            if not ignore_time:
                x = temporal_peg(x)
                x = temporal_attn(x, attn_bias=time_attn_bias)

            hiddens.append(x)

            if exists(post_downsample):
                x = post_downsample(x)

        x = self.mid_block1(x, t, c, **conv_kwargs)

        if exists(self.mid_attn):
            x = self.mid_attn(x)

        if not ignore_time:
            x = self.mid_temporal_peg(x)
            x = self.mid_temporal_attn(x, attn_bias=time_attn_bias)

        x = self.mid_block2(x, t, c, **conv_kwargs)

        add_skip_connection = lambda x: torch.cat(
            (x, hiddens.pop() * self.skip_connect_scale), dim=1
        )

        up_hiddens = []

        for (
            init_block,
            resnet_blocks,
            attn_block,
            temporal_peg,
            temporal_attn,
            upsample,
        ) in self.ups:
            x = add_skip_connection(x)
            x = init_block(x, t, c, **conv_kwargs)

            for resnet_block in resnet_blocks:
                x = add_skip_connection(x)
                x = resnet_block(x, t, **conv_kwargs)

            x = attn_block(x, c)

            if not ignore_time:
                x = temporal_peg(x)
                x = temporal_attn(x, attn_bias=time_attn_bias)

            up_hiddens.append(x.contiguous())
            x = upsample(x)

        # whether to combine all feature maps from upsample blocks

        x = self.upsample_combiner(x, up_hiddens)

        # final top-most residual if needed

        if self.init_conv_to_final_conv_residual:
            x = torch.cat((x, init_conv_residual), dim=1)

        if exists(self.final_res_block):
            x = self.final_res_block(x, t, **conv_kwargs)

        if exists(lowres_cond_img):
            x = torch.cat((x, lowres_cond_img), dim=1)

        return self.final_conv(x)
