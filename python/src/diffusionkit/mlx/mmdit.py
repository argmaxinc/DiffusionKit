#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Argmax, Inc. All Rights Reserved.
#

from functools import partial

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from argmaxtools.utils import get_logger
from beartype.typing import Tuple
from jaxtyping import Float

from .config import MMDiTConfig

logger = get_logger(__name__)


class MMDiT(nn.Module):
    """Multi-modal Diffusion Transformer Architecture
    as described in https://arxiv.org/abs/2403.03206
    """

    def __init__(self, config: MMDiTConfig):
        super().__init__()
        self.config = config
        self.input_config: dict = {
            "latent_image_embeddings": (
                2,
                self.config.latent_height,
                self.config.latent_width,
                16,
            ),
            "token_level_text_embeddings": (2, 154, 1, 4096),
            "pooled_text_embeddings": (2, 1, 1, 2048),
            "timestep": [2],
        }

        # Input adapters and embeddings
        self.x_embedder = LatentImageAdapter(config)
        self.x_pos_embedder = LatentImagePositionalEmbedding(config)
        self.y_embedder = PooledTextEmbeddingAdater(config)
        self.t_embedder = TimestepAdapter(config)
        self.context_embedder = nn.Linear(
            config.token_level_text_embed_dim,
            config.hidden_size,
        )

        self.multimodal_transformer_blocks = [
            MultiModalTransformerBlock(
                config, skip_text_post_sdpa=i == config.depth - 1
            )
            for i in range(config.depth)
        ]

        self.final_layer = FinalLayer(config)

    def __call__(
        self,
        latent_image_embeddings: mx.array,
        token_level_text_embeddings: mx.array,
        pooled_text_embeddings: mx.array,
        timestep: mx.array,
    ) -> mx.array:
        t = latent_image_embeddings.dtype

        batch, latent_height, latent_width, _ = latent_image_embeddings.shape

        # Prepare input embeddings
        modulation_inputs = self.y_embedder(pooled_text_embeddings) + self.t_embedder(
            timestep
        )
        token_level_text_embeddings = self.context_embedder(token_level_text_embeddings)
        latent_image_embeddings = self.x_embedder(
            latent_image_embeddings
        ) + self.x_pos_embedder(latent_image_embeddings)
        latent_image_embeddings = latent_image_embeddings.reshape(
            batch, -1, 1, self.config.hidden_size
        )

        # Transformer layers
        count = 0
        for block in self.multimodal_transformer_blocks:
            # Convert to float32 at block 35 to prevent NaNs and infs
            if count == 35:
                if t != mx.float32:
                    logger.debug(
                        "Converting activations at block 35 to float32 to prevent NaNs and infs."
                    )
                latent_image_embeddings = latent_image_embeddings.astype(mx.float32)
                token_level_text_embeddings = token_level_text_embeddings.astype(
                    mx.float32
                )
                modulation_inputs = modulation_inputs.astype(mx.float32)
            latent_image_embeddings, token_level_text_embeddings = block(
                latent_image_embeddings,
                token_level_text_embeddings,
                modulation_inputs,
            )
            mx.eval(latent_image_embeddings)
            mx.eval(token_level_text_embeddings)
            if mx.isnan(latent_image_embeddings).any():
                raise ValueError(
                    f"NaN detected in latent_image_embeddings at block {count}"
                )
            if (
                token_level_text_embeddings is not None
                and mx.isnan(token_level_text_embeddings).any()
            ):
                raise ValueError(
                    f"NaN detected in token_level_text_embeddings at block {count}"
                )
            if count == 35:
                latent_image_embeddings = latent_image_embeddings.astype(t)
                if token_level_text_embeddings is not None:
                    token_level_text_embeddings = token_level_text_embeddings.astype(t)
                modulation_inputs = modulation_inputs.astype(t)
            count += 1

        # Final layer
        latent_image_embeddings = self.final_layer(
            latent_image_embeddings, modulation_inputs
        )
        return unpatchify(
            latent_image_embeddings,
            patch_size=self.config.patch_size,
            target_height=latent_height,
            target_width=latent_width,
            vae_latent_dim=self.config.vae_latent_dim,
        )


class LatentImageAdapter(nn.Module):
    """Adapts the latent image input by:
    - Patchifying to reduce sequence length by `config.patch_size ** 2`
    - Projecting to `hidden_size`
    """

    def __init__(self, config: MMDiTConfig):
        super().__init__()
        self.proj = nn.Conv2d(
            config.vae_latent_dim,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(x)


class LatentImagePositionalEmbedding(nn.Module):
    def __init__(self, config: MMDiTConfig):
        super().__init__()
        self.pos_embed = nn.Embedding(
            num_embeddings=config.max_latent_resolution**2, dims=config.hidden_size
        )
        self.max_hw = config.max_latent_resolution
        self.patch_size = config.patch_size
        self.weight_shape = (1, self.max_hw, self.max_hw, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        b, h, w, _ = x.shape
        assert h <= self.max_hw and w <= self.max_hw

        h = h // self.patch_size
        w = w // self.patch_size

        # Center crop the positional embedding to match the input resolution
        y0 = (self.max_hw - h) // 2
        y1 = y0 + h
        x0 = (self.max_hw - w) // 2
        x1 = x0 + w

        w = self.pos_embed.weight.reshape(*self.weight_shape)
        w = w[:, y0:y1, x0:x1, :]
        return mx.repeat(w, repeats=b, axis=0)


class PooledTextEmbeddingAdater(nn.Module):
    def __init__(self, config: MMDiTConfig):
        super().__init__()

        d1, d2 = config.pooled_text_embed_dim, config.hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(d1, d2),
            nn.SiLU(),
            nn.Linear(d2, d2),
        )

    def __call__(self, y: mx.array) -> mx.array:
        return self.mlp(y)


class TimestepAdapter(nn.Module):
    def __init__(self, config: MMDiTConfig):
        super().__init__()

        d1, d2 = config.frequency_embed_dim, config.hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(d1, d2),
            nn.SiLU(),
            nn.Linear(d2, d2),
        )
        self.config = config

    def timestep_embedding(self, t: mx.array) -> mx.array:
        half = self.config.frequency_embed_dim // 2

        frequencies = mx.exp(
            -mx.log(mx.array(self.config.max_period))
            * mx.arange(start=0, stop=half, dtype=self.config.dtype)
            / half
        ).astype(self.config.dtype)

        args = t[:, None].astype(self.config.dtype) * frequencies[None]
        return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)

    def __call__(self, t: mx.array) -> mx.array:
        return self.mlp(self.timestep_embedding(t)[:, None, None, :])


class TransformerBlock(nn.Module):
    def __init__(self, config: MMDiTConfig, skip_post_sdpa: bool = False):
        super().__init__()
        self.norm1 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = Attention(config.hidden_size, config.depth)
        self.norm2 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.skip_post_sdpa = skip_post_sdpa
        if skip_post_sdpa:
            self.attn.o_proj = nn.Identity()
        else:
            self.mlp = FFN(
                embed_dim=config.hidden_size,
                expansion_factor=config.mlp_ratio,
                activation_fn=nn.GELU(),
            )

        self.num_modulation_params = 6 if not skip_post_sdpa else 2
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                config.hidden_size, self.num_modulation_params * config.hidden_size
            ),
        )

    def pre_sdpa(self, tensor: mx.array, modulation_inputs: mx.array):
        # Project Adaptive LayerNorm modulation parameters
        modulation_params = self.adaLN_modulation(modulation_inputs)
        modulation_params = mx.split(
            modulation_params, self.num_modulation_params, axis=-1
        )

        post_norm1_shift = modulation_params[0]
        post_norm1_residual_scale = modulation_params[1]

        # LayerNorm and modulate before SDPA
        pre_attn = affine_transform(
            self.norm1(tensor),
            shift=post_norm1_shift,
            residual_scale=post_norm1_residual_scale,
        )

        results = {
            "q": self.attn.q_proj(pre_attn),
            "k": self.attn.k_proj(pre_attn),
            "v": self.attn.v_proj(pre_attn),
        }

        if len(modulation_params) > 2:
            results.update(
                {
                    "post_attn_scale": modulation_params[2],
                    "post_norm2_shift": modulation_params[3],
                    "post_norm2_residual_scale": modulation_params[4],
                    "post_mlp_scale": modulation_params[5],
                }
            )

        return results

    def post_sdpa(
        self,
        residual: mx.array,
        sdpa_output: mx.array,
        post_attn_scale: mx.array,
        post_norm2_shift: mx.array,
        post_norm2_residual_scale: mx.array,
        post_mlp_scale: mx.array,
        **kwargs,
    ):
        residual = residual + self.attn.o_proj(sdpa_output) * post_attn_scale
        return residual + post_mlp_scale * self.mlp(
            affine_transform(
                self.norm2(residual),
                shift=post_norm2_shift,
                residual_scale=post_norm2_residual_scale,
            )
        )

    def __call__(self):
        raise NotImplementedError("This module is not intended to be used directly")


class MultiModalTransformerBlock(nn.Module):
    def __init__(self, config: MMDiTConfig, skip_text_post_sdpa: bool = False):
        super().__init__()
        self.image_transformer_block = TransformerBlock(config)
        self.text_transformer_block = TransformerBlock(
            config, skip_post_sdpa=skip_text_post_sdpa
        )

        sdpa_impl = mx.fast.scaled_dot_product_attention
        self.sdpa = partial(sdpa_impl)

        self.config = config
        self.per_head_dim = config.hidden_size // config.depth

    def __call__(
        self,
        latent_image_embeddings: mx.array,  # latent image embeddings
        token_level_text_embeddings: mx.array,  # token-level text embeddings
        modulation_inputs: mx.array,  # pooled text embeddings + timestep embeddings
    ):
        # Prepare multi-modal SDPA inputs
        image_intermediates = self.image_transformer_block.pre_sdpa(
            latent_image_embeddings, modulation_inputs=modulation_inputs
        )

        text_intermediates = self.text_transformer_block.pre_sdpa(
            token_level_text_embeddings, modulation_inputs=modulation_inputs
        )

        batch = latent_image_embeddings.shape[0]

        def rearrange_for_sdpa(t):
            # Target data layout: (batch, head, seq_len, channel)
            return t.reshape(batch, -1, self.config.depth, self.per_head_dim).transpose(
                0, 2, 1, 3
            )

        multimodal_sdpa_inputs = {
            "q": rearrange_for_sdpa(
                mx.concatenate(
                    [image_intermediates["q"], text_intermediates["q"]], axis=1
                )
            ),
            "k": rearrange_for_sdpa(
                mx.concatenate(
                    [image_intermediates["k"], text_intermediates["k"]], axis=1
                )
            ),
            "v": rearrange_for_sdpa(
                mx.concatenate(
                    [image_intermediates["v"], text_intermediates["v"]], axis=1
                )
            ),
            "scale": 1.0 / np.sqrt(self.per_head_dim),
        }

        # Compute multi-modal SDPA
        sdpa_outputs = (
            self.sdpa(**multimodal_sdpa_inputs)
            .transpose(0, 2, 1, 3)
            .reshape(batch, -1, 1, self.config.hidden_size)
        )

        # Split into image-text sequences for post-SDPA layers
        img_seq_len = latent_image_embeddings.shape[1]
        txt_seq_len = token_level_text_embeddings.shape[1]

        image_sdpa_output = sdpa_outputs[:, :img_seq_len, :, :]
        text_sdpa_output = sdpa_outputs[:, -txt_seq_len:, :, :]

        # Post-SDPA layers
        latent_image_embeddings = self.image_transformer_block.post_sdpa(
            residual=latent_image_embeddings,
            sdpa_output=image_sdpa_output,
            **image_intermediates,
        )
        if self.text_transformer_block.skip_post_sdpa:
            # Text token related outputs from the final layer do not impact the model output
            token_level_text_embeddings = None
        else:
            token_level_text_embeddings = self.text_transformer_block.post_sdpa(
                residual=token_level_text_embeddings,
                sdpa_output=text_sdpa_output,
                **text_intermediates,
            )

        return latent_image_embeddings, token_level_text_embeddings


class FinalLayer(nn.Module):
    def __init__(self, config: MMDiTConfig):
        super().__init__()
        self.norm_final = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.linear = nn.Linear(
            config.hidden_size,
            (config.patch_size**2) * config.vae_latent_dim,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 2 * config.hidden_size),
        )

    def __call__(
        self, latent_image_embeddings: mx.array, modulation_inputs: mx.array
    ) -> mx.array:
        shift, residual_scale = mx.split(
            self.adaLN_modulation(modulation_inputs), 2, axis=-1
        )

        latent_image_embeddings = affine_transform(
            self.norm_final(latent_image_embeddings),
            shift=shift,
            residual_scale=residual_scale,
        )
        return self.linear(latent_image_embeddings)


class Attention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
    ):
        super().__init__()

        # Configure dimensions for SDPA
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        assert (
            self.embed_dim % self.n_heads == 0
        ), "Embedding dimension must be divisible by number of heads"

        self._sdpa_implementation = mx.fast.scaled_dot_product_attention

        # Initialize layers
        self.per_head_dim = self.embed_dim // self.n_heads
        self.kv_proj_embed_dim = self.per_head_dim * n_heads

        # Note: key bias is redundant due to softmax invariance
        self.k_proj = nn.Linear(embed_dim, self.kv_proj_embed_dim, bias=False)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, self.kv_proj_embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)


class FFN(nn.Module):
    def __init__(self, embed_dim, expansion_factor, activation_fn):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim * expansion_factor)
        self.act_fn = activation_fn
        self.fc2 = nn.Linear(embed_dim * expansion_factor, embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.act_fn(self.fc1(x)))


class LayerNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps

    def __call__(self, inputs: mx.array) -> mx.array:
        input_rank = len(inputs.shape)
        if input_rank != 4:
            raise ValueError(f"Input tensor must have rank 4, got {input_rank}")

        return mx.fast.layer_norm(inputs, weight=None, bias=None, eps=self.eps)


def affine_transform(
    x: mx.array, shift: mx.array, residual_scale: mx.array
) -> mx.array:
    """Affine transformation (Used for Adaptive LayerNorm Modulation)"""
    return x * (1.0 + residual_scale) + shift


def unpatchify(
    x: mx.array,
    patch_size: int,
    target_height: int,
    target_width: int,
    vae_latent_dim: int,
) -> mx.array:
    """Unpatchify to restore VAE latent space compatible data format"""
    h, w = target_height // patch_size, target_width // patch_size
    x = x.reshape(x.shape[0], h, w, patch_size, patch_size, vae_latent_dim)
    x = x.transpose(0, 5, 1, 3, 2, 4)  # x = mx.einsum("bhwpqc->bchpwq", x)
    return x.reshape(x.shape[0], vae_latent_dim, target_height, target_width).transpose(
        0, 2, 3, 1
    )
