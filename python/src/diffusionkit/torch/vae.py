from dataclasses import dataclass, field
from functools import partial
from typing import List

import torch.nn as nn
import torch.nn.functional as F
from argmaxtools._sdpa import Cat as sdpa


@dataclass
class VAEDecoderConfig:
    resolution: int
    in_channels: int = 16
    out_channels: int = 3
    base_channels: int = 128
    channel_multipliers: List[int] = field(default_factory=lambda: [1, 2, 4, 4])
    num_res_blocks: int = 2


SD3GroupNorm = partial(nn.GroupNorm, num_groups=32, eps=1e-6)


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = SD3GroupNorm(num_channels=in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = SD3GroupNorm(num_channels=out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return self.nin_shortcut(residual) + x


class SingleHeadAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = SD3GroupNorm(num_channels=in_channels)
        self.q_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sdpa = partial(
            sdpa(embed_dim=in_channels, n_heads=1).sdpa,
            key_padding_mask=None,
            causal=False,
        )

    def forward(self, x):
        residual = x
        x = self.norm(x)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        b, c, h, w = q.shape
        q, k, v = map(lambda x: x.view(b, c, 1, h * w), (q, k, v))
        attn = self.sdpa(q, k, v)
        attn = self.out_proj(attn.view(b, c, h, w))
        return residual + attn


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x


class VAEDecoder(nn.Module):
    def __init__(self, config: VAEDecoderConfig):
        super().__init__()
        self.num_resolutions = len(config.channel_multipliers)
        self.num_res_blocks = config.num_res_blocks
        block_in = (
            config.base_channels * config.channel_multipliers[self.num_resolutions - 1]
        )

        self.conv_in = nn.Conv2d(config.in_channels, block_in, kernel_size=3, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = SingleHeadAttentionBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = config.base_channels * config.channel_multipliers[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out

            up = nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample(block_in)
            self.up.insert(0, up)

        self.norm_out = SD3GroupNorm(num_channels=block_in)
        self.conv_out = nn.Conv2d(
            block_in, config.out_channels, kernel_size=3, padding=1
        )

    def forward(self, z):
        # Mid-blocks with self-attention
        x = self.mid.block_2(self.mid.attn_1(self.mid.block_1(self.conv_in(z))))

        # Upsampling blocks
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                x = self.up[i_level].block[i_block](x)
            if i_level != 0:
                x = self.up[i_level].upsample(x)

        return self.conv_out(F.silu(self.norm_out(x)))
