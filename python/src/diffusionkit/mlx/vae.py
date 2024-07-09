# reference: https://github.com/ml-explore/mlx-examples/tree/main/stable_diffusion

#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Argmax, Inc. All Rights Reserved.
#

import math
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
from argmaxtools.utils import get_logger

from .config import AutoencoderConfig

logger = get_logger(__name__)


def upsample_nearest(x, scale: int = 2):
    B, H, W, C = x.shape
    x = mx.broadcast_to(x[:, :, None, :, None, :], (B, H, scale, W, scale, C))
    x = x.reshape(B, H * scale, W * scale, C)

    return x


class Attention(nn.Module):
    """A single head unmasked attention for use with the VAE."""

    def __init__(self, dims: int, norm_groups: int = 32):
        super().__init__()

        self.group_norm = nn.GroupNorm(norm_groups, dims, pytorch_compatible=True)
        self.query_proj = nn.Linear(dims, dims)
        self.key_proj = nn.Linear(dims, dims)
        self.value_proj = nn.Linear(dims, dims)
        self.out_proj = nn.Linear(dims, dims)

    def __call__(self, x):
        B, H, W, C = x.shape

        y = self.group_norm(x)

        queries = self.query_proj(y).reshape(B, H * W, C)
        keys = self.key_proj(y).reshape(B, H * W, C)
        values = self.value_proj(y).reshape(B, H * W, C)

        scale = 1 / math.sqrt(queries.shape[-1])
        scores = (queries * scale) @ keys.transpose(0, 2, 1)
        attn = mx.softmax(scores, axis=-1)
        y = (attn @ values).reshape(B, H, W, C)

        y = self.out_proj(y)
        x = x + y

        return x


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        groups: int = 32,
        temb_channels: Optional[int] = None,
    ):
        super().__init__()

        out_channels = out_channels or in_channels

        self.norm1 = nn.GroupNorm(groups, in_channels, pytorch_compatible=True)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels is not None:
            self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        if in_channels != out_channels:
            self.conv_shortcut = nn.Linear(in_channels, out_channels)

    def __call__(self, x, temb=None):
        dtype = x.dtype

        if temb is not None:
            temb = self.time_emb_proj(nn.silu(temb))

        y = self.norm1(x.astype(mx.float32)).astype(dtype)
        y = nn.silu(y)
        y = self.conv1(y)
        if temb is not None:
            y = y + temb[:, None, None, :]
        y = self.norm2(y.astype(mx.float32)).astype(dtype)
        y = nn.silu(y)
        y = self.conv2(y)

        x = y + (x if "conv_shortcut" not in self else self.conv_shortcut(x))

        return x


class EncoderDecoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        resnet_groups: int = 32,
        add_downsample=True,
        add_upsample=True,
    ):
        super().__init__()

        # Add the resnet blocks
        self.resnets = [
            ResnetBlock2D(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                groups=resnet_groups,
            )
            for i in range(num_layers)
        ]

        # Add an optional downsampling layer
        if add_downsample:
            self.downsample = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=2, padding=0
            )

        # or upsampling layer
        if add_upsample:
            self.upsample = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1
            )

    def __call__(self, x):
        for resnet in self.resnets:
            x = resnet(x)

        if "downsample" in self:
            x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])
            x = self.downsample(x)

        if "upsample" in self:
            x = self.upsample(upsample_nearest(x))

        return x


class Encoder(nn.Module):
    """Implements the encoder side of the Autoencoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block_out_channels: List[int] = [64],
        layers_per_block: int = 2,
        resnet_groups: int = 32,
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1
        )

        channels = [block_out_channels[0]] + list(block_out_channels)
        self.down_blocks = [
            EncoderDecoderBlock2D(
                in_channels,
                out_channels,
                num_layers=layers_per_block,
                resnet_groups=resnet_groups,
                add_downsample=i < len(block_out_channels) - 1,
                add_upsample=False,
            )
            for i, (in_channels, out_channels) in enumerate(zip(channels, channels[1:]))
        ]

        self.mid_blocks = [
            ResnetBlock2D(
                in_channels=block_out_channels[-1],
                out_channels=block_out_channels[-1],
                groups=resnet_groups,
            ),
            Attention(block_out_channels[-1], resnet_groups),
            ResnetBlock2D(
                in_channels=block_out_channels[-1],
                out_channels=block_out_channels[-1],
                groups=resnet_groups,
            ),
        ]

        self.conv_norm_out = nn.GroupNorm(
            resnet_groups, block_out_channels[-1], pytorch_compatible=True
        )
        self.conv_out = nn.Conv2d(block_out_channels[-1], out_channels, 3, padding=1)

    def __call__(self, x):
        x = self.conv_in(x)

        for l in self.down_blocks:
            x = l(x)

        x = self.mid_blocks[0](x)
        x = self.mid_blocks[1](x)
        x = self.mid_blocks[2](x)

        x = self.conv_norm_out(x)
        x = nn.silu(x)
        x = self.conv_out(x)

        return x


class Decoder(nn.Module):
    """Implements the decoder side of the Autoencoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block_out_channels: List[int] = [64],
        layers_per_block: int = 2,
        resnet_groups: int = 32,
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1
        )

        self.mid_blocks = [
            ResnetBlock2D(
                in_channels=block_out_channels[-1],
                out_channels=block_out_channels[-1],
                groups=resnet_groups,
            ),
            Attention(block_out_channels[-1], resnet_groups),
            ResnetBlock2D(
                in_channels=block_out_channels[-1],
                out_channels=block_out_channels[-1],
                groups=resnet_groups,
            ),
        ]

        channels = list(reversed(block_out_channels))
        channels = [channels[0]] + channels
        self.up_blocks = [
            EncoderDecoderBlock2D(
                in_channels,
                out_channels,
                num_layers=layers_per_block,
                resnet_groups=resnet_groups,
                add_downsample=False,
                add_upsample=i < len(block_out_channels) - 1,
            )
            for i, (in_channels, out_channels) in enumerate(zip(channels, channels[1:]))
        ]

        self.conv_norm_out = nn.GroupNorm(
            resnet_groups, block_out_channels[0], pytorch_compatible=True
        )
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

    def __call__(self, x):
        x = self.conv_in(x)

        x = self.mid_blocks[0](x)
        x = self.mid_blocks[1](x)
        x = self.mid_blocks[2](x)

        for l in self.up_blocks:
            x = l(x)

        x = self.conv_norm_out(x)
        x = nn.silu(x)
        x = self.conv_out(x)

        return x


class Autoencoder(nn.Module):
    """The autoencoder that allows us to perform diffusion in the latent space."""

    def __init__(self, config: AutoencoderConfig):
        super().__init__()

        self.latent_channels = config.latent_channels_in
        self.scaling_factor = config.scaling_factor
        self.encoder = Encoder(
            config.in_channels,
            config.latent_channels_out,
            config.block_out_channels,
            config.layers_per_block,
            resnet_groups=config.norm_num_groups,
        )
        self.decoder = Decoder(
            config.latent_channels_in,
            config.out_channels,
            config.block_out_channels,
            config.layers_per_block + 1,
            resnet_groups=config.norm_num_groups,
        )

        self.quant_proj = nn.Linear(
            config.latent_channels_out, config.latent_channels_out
        )
        self.post_quant_proj = nn.Linear(
            config.latent_channels_in, config.latent_channels_in
        )

    def decode(self, z):
        z = z / self.scaling_factor
        return self.decoder(self.post_quant_proj(z))

    def encode(self, x):
        x = self.encoder(x)
        x = self.quant_proj(x)
        mean, logvar = x.split(2, axis=-1)
        mean = mean * self.scaling_factor
        logvar = logvar + 2 * math.log(self.scaling_factor)

        return mean, logvar

    def __call__(self, x, key=None):
        mean, logvar = self.encode(x)
        z = mx.random.normal(mean.shape, key=key) * mx.exp(0.5 * logvar) + mean
        x_hat = self.decode(z)

        return dict(x_hat=x_hat, z=z, mean=mean, logvar=logvar)


class VAEDecoder(nn.Module):
    """Implements the decoder side of the Autoencoder for SD3"""

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 3,
        block_out_channels: List[int] = [128, 256, 512, 512],
        layers_per_block: int = 3,
        resnet_groups: int = 32,
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1
        )

        self.mid_blocks = [
            ResnetBlock2D(
                in_channels=block_out_channels[-1],
                out_channels=block_out_channels[-1],
                groups=resnet_groups,
            ),
            Attention(block_out_channels[-1], resnet_groups),
            ResnetBlock2D(
                in_channels=block_out_channels[-1],
                out_channels=block_out_channels[-1],
                groups=resnet_groups,
            ),
        ]

        channels = list(reversed(block_out_channels))
        channels = [channels[0]] + channels
        self.up_blocks = []
        for i, (in_c, out_c) in enumerate(zip(channels, channels[1:])):
            up = EncoderDecoderBlock2D(
                in_c,
                out_c,
                num_layers=layers_per_block,
                resnet_groups=resnet_groups,
                add_downsample=False,
                add_upsample=i < len(block_out_channels) - 1,
            )
            self.up_blocks.insert(0, up)

        self.conv_norm_out = nn.GroupNorm(
            resnet_groups, block_out_channels[0], pytorch_compatible=True
        )
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

    def __call__(self, x):
        t = x.dtype
        x = self.conv_in(x)

        x = self.mid_blocks[0](x)
        if mx.isnan(x).any():
            raise ValueError("NaN detected in VAE Decoder after mid_blocks[0]")
        x = x.astype(mx.float32)
        x = self.mid_blocks[1](x)
        if mx.isnan(x).any():
            raise ValueError("NaN detected in VAE Decoder after mid_blocks[1]")
        x = x.astype(t)
        x = self.mid_blocks[2](x)
        if mx.isnan(x).any():
            raise ValueError("NaN detected in VAE Decoder after mid_blocks[2]")

        for l in reversed(self.up_blocks):
            x = l(x)
            mx.eval(x)

        if mx.isnan(x).any():
            raise ValueError("NaN detected in VAE Decoder after up_blocks")

        x = x.astype(mx.float32)
        x = self.conv_norm_out(x)
        if mx.isnan(x).any():
            raise ValueError("NaN detected in VAE Decoder after conv_norm_out")
        x = x.astype(t)
        x = nn.silu(x)
        x = self.conv_out(x)
        if mx.isnan(x).any():
            raise ValueError("NaN detected in VAE Decoder after conv_out")

        return x


class VAEEncoder(nn.Module):
    """Implements the encoder side of the Autoencoder."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 32,
        block_out_channels: List[int] = [128, 256, 512, 512],
        layers_per_block: int = 2,
        resnet_groups: int = 32,
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1
        )

        channels = [block_out_channels[0]] + list(block_out_channels)
        self.down_blocks = [
            EncoderDecoderBlock2D(
                in_channels,
                out_channels,
                num_layers=layers_per_block,
                resnet_groups=resnet_groups,
                add_downsample=i < len(block_out_channels) - 1,
                add_upsample=False,
            )
            for i, (in_channels, out_channels) in enumerate(zip(channels, channels[1:]))
        ]

        self.mid_blocks = [
            ResnetBlock2D(
                in_channels=block_out_channels[-1],
                out_channels=block_out_channels[-1],
                groups=resnet_groups,
            ),
            Attention(block_out_channels[-1], resnet_groups),
            ResnetBlock2D(
                in_channels=block_out_channels[-1],
                out_channels=block_out_channels[-1],
                groups=resnet_groups,
            ),
        ]

        self.conv_norm_out = nn.GroupNorm(
            resnet_groups, block_out_channels[-1], pytorch_compatible=True
        )
        self.conv_out = nn.Conv2d(block_out_channels[-1], out_channels, 3, padding=1)

    def __call__(self, x):
        x = self.conv_in(x)

        for l in self.down_blocks:
            x = l(x)

        x = self.mid_blocks[0](x)
        x = self.mid_blocks[1](x)
        x = self.mid_blocks[2](x)

        x = self.conv_norm_out(x)
        x = nn.silu(x)
        x = self.conv_out(x)

        return x
