from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
from argmaxtools import nn as agx
from argmaxtools._sdpa import Cat as sdpa
from beartype.typing import Tuple
from jaxtyping import Float
from torch import Tensor


@dataclass
class MMDiTConfig:
    """Multi-modal Diffusion Transformer Configuration"""

    # Transformer spec
    depth: int  # 15, 30, 38
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
    text_seq_len: int = 77 + 77  # Concatenation of CLIP-L/14 and CLIP-G/14 tokens

    # t: Timestep input spec
    frequency_embed_dim: int = 256
    max_period: int = 10000


SD3_8b = MMDiTConfig(depth=38)
SD3_2b = MMDiTConfig(depth=24)


class MMDiT(nn.Module):
    """Multi-modal Diffusion Transformer Architecture
    as described in https://arxiv.org/abs/2403.03206
    """

    def __init__(self, config: MMDiTConfig):
        super().__init__()
        self.config = config
        self.dtype = torch.float32  # FIXME(atiorh)

        # Input adapters and embeddings
        self.x_embedder = LatentImageAdapter(config)
        self.x_pos_embedder = LatentImagePositionalEmbedding(config)
        self.y_embedder = PooledTextEmbeddingAdapter(config)
        self.t_embedder = TimestepAdapter(config)
        self.context_embedder = nn.Conv2d(
            config.token_level_text_embed_dim, config.hidden_size, kernel_size=1
        )

        # Transformer layers
        self.multimodal_transformer_blocks = nn.ModuleList(
            [
                MultiModalTransformerBlock(
                    config, skip_text_post_sdpa=i == config.depth - 1
                )
                for i in range(config.depth)
            ]
        )

        self.final_layer = FinalLayer(config)

        # self._register_load_state_dict_pre_hook(mmdit_state_dict_adjustments)

    def forward(
        self,
        latent_image_embeddings: Float[
            Tensor, "batch vae_latent_dim img_height img_width"
        ],
        token_level_text_embeddings: Float[
            Tensor, "batch token_level_text_embed_dim 1 txt_seq_len"
        ],
        pooled_text_embeddings: Float[Tensor, "batch pooled_text_embed_dim 1 1"],
        timestep: Float[Tensor, "batch 1 1 1"],
    ) -> Float[Tensor, "batch vae_latent_dim 1 img_height img_width"]:
        batch, _, latent_height, latent_width = latent_image_embeddings.shape

        # Prepare input embeddings
        modulation_inputs = self.y_embedder(pooled_text_embeddings) + self.t_embedder(
            timestep
        )
        token_level_text_embeddings = self.context_embedder(token_level_text_embeddings)
        latent_image_embeddings = self.x_embedder(
            latent_image_embeddings
        ) + self.x_pos_embedder(latent_image_embeddings)
        latent_image_embeddings = latent_image_embeddings.view(
            batch, self.config.hidden_size, 1, -1
        )

        # Transformer layers
        for block in self.multimodal_transformer_blocks:
            latent_image_embeddings, token_level_text_embeddings = block(
                latent_image_embeddings,
                token_level_text_embeddings,
                modulation_inputs,
            )

        # Final layer
        latent_image_embeddings = self.final_layer(
            latent_image_embeddings, modulation_inputs
        )
        latent_image_embeddings = unpatchify(
            latent_image_embeddings,
            patch_size=self.config.patch_size,
            target_height=latent_height,
            target_width=latent_width,
            vae_latent_dim=self.config.vae_latent_dim,
        )

        return (latent_image_embeddings,)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class LatentImagePositionalEmbedding(nn.Module):
    def __init__(self, config: MMDiTConfig):
        super().__init__()
        self.pos_embed = nn.Embedding(
            num_embeddings=config.max_latent_resolution**2,
            embedding_dim=config.hidden_size,
        )
        self.max_hw = config.max_latent_resolution
        self.patch_size = config.patch_size
        self.weight_shape = (1, self.max_hw, self.max_hw, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        assert h <= self.max_hw and w <= self.max_hw

        h = h // self.patch_size
        w = w // self.patch_size

        # Center crop the positional embedding to match the input resolution
        y0 = (self.max_hw - h) // 2
        y1 = y0 + h
        x0 = (self.max_hw - w) // 2
        x1 = x0 + w

        w = self.pos_embed.weight.view(*self.weight_shape)
        w = w[:, y0:y1, x0:x1, :].permute(0, 3, 1, 2)
        return w.expand(b, -1, -1, -1)


class PooledTextEmbeddingAdapter(nn.Module):
    def __init__(self, config: MMDiTConfig):
        super().__init__()

        d1, d2 = config.pooled_text_embed_dim, config.hidden_size
        self.mlp = nn.Sequential(
            nn.Conv2d(d1, d2, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(d2, d2, kernel_size=1),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.mlp(y)


class TimestepAdapter(nn.Module):
    def __init__(self, config: MMDiTConfig):
        super().__init__()
        d1, d2 = config.frequency_embed_dim, config.hidden_size
        self.mlp = nn.Sequential(
            nn.Conv2d(d1, d2, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(d2, d2, kernel_size=1),
        )
        self.config = config

    def timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half = self.config.frequency_embed_dim // 2

        frequencies = torch.exp(
            -torch.log(torch.tensor(self.config.max_period).to(t.dtype))
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)

        args = t[:, None].float() * frequencies[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t)[:, :, None, None])


class TransformerBlock(nn.Module):
    def __init__(self, config: MMDiTConfig, skip_post_sdpa: bool = False):
        super().__init__()
        self.norm1 = agx.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False
        )

        self.attn = agx.Attention(
            embed_dim=config.hidden_size,
            n_heads=config.depth,
            attention_type=agx.AttentionType.SelfAttention,
        )

        self.norm2 = agx.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False
        )

        self.skip_post_sdpa = skip_post_sdpa
        if skip_post_sdpa:
            delattr(self.attn, "o_proj")
        else:
            self.mlp = agx.FFN(
                embed_dim=config.hidden_size,
                expansion_factor=config.mlp_ratio,
                activation_fn=nn.GELU(approximate="tanh"),
            )

        # modulation_params[0:2]: shift, residual_scale post self.norm1
        # modulation_params[2]:   scale post self.attn.out_proj
        # modulation_params[3:5]: shift, residual_scale post self.norm2
        # modulation_params[5]:   scale post self.mlp
        self.num_modulation_params = 6 if not skip_post_sdpa else 2
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(
                config.hidden_size, self.num_modulation_params * config.hidden_size, 1
            ),
        )

    def pre_sdpa(
        self,
        tensor: Float[Tensor, "batch hidden_size 1 seq_len"],
        modulation_inputs: Float[Tensor, "batch hidden_size 1 1"],
    ):
        # Project Adaptive LayerNorm Modulation parameters
        modulation_params = self.adaLN_modulation(modulation_inputs)
        modulation_params = modulation_params.chunk(self.num_modulation_params, dim=1)

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
        residual: Float[Tensor, "batch hidden_size 1 seq_len"],
        sdpa_output: Float[Tensor, "batch hidden_size 1 seq_len"],
        post_attn_scale: Float[Tensor, "batch hidden_size 1 1"],
        post_norm2_shift: Float[Tensor, "batch hidden_size 1 1"],
        post_norm2_residual_scale: Float[Tensor, "batch hidden_size 1 1"],
        post_mlp_scale: Float[Tensor, "batch hidden_size 1 1"],
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

    def forward(self):
        raise NotImplementedError("This module is not intended to be used directly")


class MultiModalTransformerBlock(nn.Module):
    def __init__(self, config: MMDiTConfig, skip_text_post_sdpa: bool = False):
        super().__init__()
        self.image_transformer_block = TransformerBlock(config)
        self.text_transformer_block = TransformerBlock(
            config, skip_post_sdpa=skip_text_post_sdpa
        )

        sdpa_impl = sdpa(embed_dim=config.hidden_size, n_heads=config.depth)
        self.sdpa = partial(sdpa_impl.sdpa, causal=False, key_padding_mask=None)

    def forward(
        self,
        latent_image_embeddings: Float[Tensor, "batch hidden_size 1 img_seq_len"],
        token_level_text_embeddings: Float[Tensor, "batch hidden_size 1 txt_seq_len"],
        modulation_inputs: Float[Tensor, "batch hidden_size 1 1"],
    ) -> Tuple[
        Float[Tensor, "batch hidden_size 1 img_seq_len"],
        Float[Tensor, "batch hidden_size 1 txt_seq_len"],
    ]:
        # Prepare multi-modal SDPA inputs
        image_intermediates = self.image_transformer_block.pre_sdpa(
            latent_image_embeddings, modulation_inputs=modulation_inputs
        )

        text_intermediates = self.text_transformer_block.pre_sdpa(
            token_level_text_embeddings, modulation_inputs=modulation_inputs
        )

        multimodal_sdpa_inputs = {
            "query": torch.cat(
                [image_intermediates["q"], text_intermediates["q"]], dim=-1
            ),
            "key": torch.cat(
                [image_intermediates["k"], text_intermediates["k"]], dim=-1
            ),
            "value": torch.cat(
                [image_intermediates["v"], text_intermediates["v"]], dim=-1
            ),
        }

        # Compute multi-modal SDPA
        sdpa_outputs = self.sdpa(**multimodal_sdpa_inputs)

        # Split into image-text sequences for post-SDPA layers
        img_seq_len = latent_image_embeddings.shape[-1]
        txt_seq_len = token_level_text_embeddings.shape[-1]

        image_sdpa_output = sdpa_outputs[:, :, :, :img_seq_len]
        text_sdpa_output = sdpa_outputs[:, :, :, -txt_seq_len:]

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
        self.norm_final = agx.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False
        )
        self.linear = nn.Conv2d(
            config.hidden_size,
            (config.patch_size**2) * config.vae_latent_dim,
            kernel_size=1,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Conv2d(config.hidden_size, 2 * config.hidden_size, 1)
        )

    def forward(
        self,
        latent_image_embeddings: Float[Tensor, "batch hidden_size 1 img_seq_len"],
        modulation_inputs: Float[Tensor, "batch hidden_size 1 1"],
    ) -> Float[Tensor, "batch hidden_size 1 img_seq_len"]:
        shift, residual_scale = self.adaLN_modulation(modulation_inputs).chunk(2, dim=1)

        latent_image_embeddings = affine_transform(
            self.norm_final(latent_image_embeddings),
            shift=shift,
            residual_scale=residual_scale,
        )

        return self.linear(latent_image_embeddings)


# FIXME(atiorh): Switch to checkpoint conversion, away from load_state_dict_pre_hook
# def mmdit_state_dict_adjustments(state_dict, prefix, local_metadata,
#                                  strict, missing_keys, unexpected_keys, error_msgs):
def mmdit_state_dict_adjustments(state_dict, prefix=""):
    # Unsqueeze nn.Linear -> nn.Conv2d
    state_dict = {
        k: v[:, :, None, None] if "mlp" in k and "weight" in k else v
        for k, v in state_dict.items()
    }
    state_dict = {
        k: v[:, :, None, None] if "adaLN_modulation" in k and "weight" in k else v
        for k, v in state_dict.items()
    }
    state_dict[prefix + "final_layer.linear.weight"] = state_dict[
        prefix + "final_layer.linear.weight"
    ][:, :, None, None]
    state_dict[prefix + "context_embedder.weight"] = state_dict[
        prefix + "context_embedder.weight"
    ][:, :, None, None]

    # Split qkv proj and rename:
    # *transformer_block.attn.qkv.{weigth/bias}  -> transformer_block.attn.{q/k/v}_proj.{weigth/bias}
    # *transformer_block.attn.proj.{weigth/bias} -> transformer_block.attn.o_proj.{weight/bias}
    keys_to_pop = []
    state_dict_update = {}
    for k in state_dict:
        if "attn.qkv" in k:
            keys_to_pop.append(k)
            for name, weight in zip(["q", "k", "v"], state_dict[k].chunk(3)):
                state_dict_update[k.replace("attn.qkv", f"attn.{name}_proj")] = (
                    weight[:, :, None, None] if "weight" in k else weight
                )

    [state_dict.pop(k) for k in keys_to_pop]
    state_dict.update(state_dict_update)

    state_dict = {
        k.replace("attn.proj", "attn.o_proj"): (
            v[:, :, None, None] if "attn.proj" in k and "weight" in k else v
        )
        for k, v in state_dict.items()
    }

    # Rename joint_blocks -> multimodal_transformer_blocks
    state_dict = {
        k.replace("joint_blocks", "multimodal_transformer_blocks"): v
        for k, v in state_dict.items()
    }

    # Remap pos_embed buffer -> nn.Embedding
    state_dict[prefix + "x_pos_embedder.pos_embed.weight"] = state_dict.pop(
        prefix + "pos_embed"
    )[0]

    # Remap context_block -> text_block
    state_dict = {
        k.replace("context_block", "text_transformer_block"): v
        for k, v in state_dict.items()
    }

    # Remap x_block -> image_block
    state_dict = {
        k.replace("x_block", "image_transformer_block"): v
        for k, v in state_dict.items()
    }

    # Filter out VAE Decoder related tensors
    state_dict = {k: v for k, v in state_dict.items() if "decoder." not in k}

    pre_sdpa_norm_params = [k for k in state_dict if "ln_q" in k or "ln_k" in k]
    if len(pre_sdpa_norm_params) != 0:
        raise KeyError("Unexpected ln_q or ln_k key:", pre_sdpa_norm_params)

    # Remove k_proj bias
    keys_to_pop = []
    for k in state_dict:
        if "k_proj.bias" in k:
            keys_to_pop.append(k)
    [state_dict.pop(k) for k in keys_to_pop]

    return state_dict


def affine_transform(
    x: Float[Tensor, "batch embed_dim 1 seq_len"],
    shift: Float[Tensor, "batch embed_dim 1 1"],
    residual_scale: Float[Tensor, "batch embed_dim 1 1"],
) -> Float[Tensor, "batch embed_dim 1 seq_len"]:
    """Affine transformation (Used for Adaptive LayerNorm Modulation)"""
    return x * (1.0 + residual_scale) + shift


def unpatchify(
    x: Float[
        Tensor, "batch vae_latent_dim*patch_size**2 1 height*width/(patch_size**2)"
    ],
    patch_size: int,
    target_height: int,
    target_width: int,
    vae_latent_dim: int,
) -> Float[Tensor, "batch vae_latent_dim height width"]:
    """Unpatchify to restore VAE latent space compatible data format"""
    h, w = target_height // patch_size, target_width // patch_size
    x = x.transpose(1, 3).view(x.shape[0], h, w, patch_size, patch_size, vae_latent_dim)
    x = torch.einsum("bhwpqc->bchpwq", x)
    return x.reshape(x.shape[0], vae_latent_dim, target_height, target_width)
