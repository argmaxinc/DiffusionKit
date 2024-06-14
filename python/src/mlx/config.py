# reference: https://github.com/ml-explore/mlx-examples/tree/main/stable_diffusion

#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Argmax, Inc. All Rights Reserved.
#
from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx


@dataclass
class MMDiTConfig:
    """Multi-modal Diffusion Transformer Configuration"""

    # Transformer spec
    depth: int = 24  # 24, 38
    mlp_ratio: int = 4
    vae_latent_dim: int = 16  # = in_channels = out_channels
    layer_norm_eps: float = 1e-6

    @property
    def hidden_size(self) -> int:
        return 64 * self.depth

    # x: Latent image input spec
    max_latent_resolution: int = 192
    patch_size: int = 2

    # y: Text input spec
    pooled_text_embed_dim: int = 2048  # 768 (CLIP-L/14) + 1280 (CLIP-G/14) = 2048
    token_level_text_embed_dim: int = (
        4096  # 4096 (T5-XXL) = 768 (CLIP-L/14) + 1280 (CLIP-G/14) + 2048 (zero padding)
    )

    # t: Timestep input spec
    frequency_embed_dim: int = 256
    max_period: int = 10000

    # latent dims
    latent_height: int = 64  # img height // 8
    latent_width: int = 64

    dtype: mx.Dtype = mx.float16


SD3_8b = MMDiTConfig(depth=38)
SD3_2b = MMDiTConfig(depth=24)


@dataclass
class AutoencoderConfig:
    in_channels: int = 3
    out_channels: int = 3
    latent_channels_out: int = 8
    latent_channels_in: int = 4
    block_out_channels: Tuple[int] = (128, 256, 512, 512)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    scaling_factor: float = 0.18215


@dataclass
class VAEDecoderConfig:
    in_channels: int = 16
    out_channels: int = 3
    block_out_channels: Tuple[int] = (128, 256, 512, 512)
    layers_per_block: int = 3
    resnet_groups: int = 32


@dataclass
class VAEEncoderConfig:
    in_channels: int = 3
    out_channels: int = 32
    block_out_channels: Tuple[int] = (128, 256, 512, 512)
    layers_per_block: int = 2
    resnet_groups: int = 32


@dataclass
class CLIPTextModelConfig:
    num_layers: int = 23
    model_dims: int = 1024
    num_heads: int = 16
    max_length: int = 77
    vocab_size: int = 49408
    projection_dim: Optional[int] = None
    hidden_act: str = "quick_gelu"
