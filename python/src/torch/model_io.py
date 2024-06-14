import numpy as np
import torch
import torch.nn as nn
from argmaxtools.utils import get_logger
from PIL import Image
from safetensors import safe_open

logger = get_logger(__name__)


def _load_vae_decoder_weights(module: nn.Module, path: str) -> None:
    """Load weights from a checkpoint file (safetensors or pt)"""
    total_params_in_module = sum(p.numel() for p in module.parameters())
    logger.info(
        f"Loading state_dict into nn.Module with  {len([n for n,p in module.named_parameters()])} "
        f"parameter tensors totaling {total_params_in_module} "
        f"parameters from {path}"
    )

    if path.endswith(".pt"):
        state_dict = torch.load(path, map_location="cpu")
        module.load_state_dict(state_dict)
    elif path.endswith(".safetensors"):
        state_dict = {}

        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
            state_dict = vae_decoder_state_dict_adjustments(state_dict)
    else:
        raise ValueError(f"Unsupported file format: {path}")

    total_params_in_state_dict = sum(np.prod(v.shape) for v in state_dict.values())
    logger.info(
        f"Loaded state dict with {len(state_dict)} tensors totaling "
        f"{total_params_in_state_dict} parameters"
    )

    if total_params_in_module != total_params_in_state_dict:
        raise ValueError(
            f"Total number of parameters in state_dict ({total_params_in_state_dict}) "
            f"does not match the number of parameters in the module ({total_params_in_module})"
        )

    module.load_state_dict(state_dict)


def _load_mmdit_weights(module: nn.Module, path: str) -> None:
    """Load weights from a checkpoint file (safetensors or pt)"""
    total_params_in_module = sum(p.numel() for p in module.parameters())
    logger.info(
        f"Loading state_dict into nn.Module with  {len([n for n,p in module.named_parameters()])} "
        f"parameter tensors totaling {total_params_in_module} "
        f"parameters from {path}"
    )

    if path.endswith(".pt"):
        state_dict = torch.load(path, map_location="cpu")
        module.load_state_dict(state_dict)
    elif path.endswith(".safetensors"):
        state_dict = {}

        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
            # Adjust the state dict to match the MMDiT model for Argmax implementation
            state_dict = {
                ".".join(k.rsplit(".")[2:]): f.get_tensor(k)
                for k in f.keys()
                if all(s not in k for s in ["encoder", "decoder"])
            }
            state_dict = mmdit_state_dict_adjustments(state_dict)
    else:
        raise ValueError(f"Unsupported file format: {path}")

    total_params_in_state_dict = sum(np.prod(v.shape) for v in state_dict.values())
    logger.info(
        f"Loaded state dict with {len(state_dict)} tensors totaling "
        f"{total_params_in_state_dict} parameters"
    )

    if total_params_in_module != total_params_in_state_dict:
        raise ValueError(
            f"Total number of parameters in state_dict ({total_params_in_state_dict}) "
            f"does not match the number of parameters in the module ({total_params_in_module})"
        )

    module.load_state_dict(state_dict)


def vae_decoder_state_dict_adjustments(state_dict, prefix=""):
    state_dict = {k: v.cpu() for k, v in state_dict.items() if "decoder." in k}
    state_dict = {
        k.replace("first_stage_model.decoder.", ""): v for k, v in state_dict.items()
    }

    state_dict = {
        k.replace(".attn_1.k", ".attn_1.k_proj") if ".attn_1.k" in k else k: v
        for k, v in state_dict.items()
    }
    state_dict = {
        k.replace(".attn_1.v", ".attn_1.v_proj") if ".attn_1.v" in k else k: v
        for k, v in state_dict.items()
    }
    state_dict = {
        k.replace(".attn_1.q", ".attn_1.q_proj") if ".attn_1.q" in k else k: v
        for k, v in state_dict.items()
    }
    state_dict = {
        (
            k.replace(".attn_1.proj_out", ".attn_1.out_proj")
            if ".attn_1.proj_out" in k
            else k
        ): v
        for k, v in state_dict.items()
    }

    return state_dict


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
