import numpy as np
import torch
import torch.nn as nn
from argmaxtools.utils import get_logger
from PIL import Image
from safetensors import safe_open

logger = get_logger(__name__)


def _load_weights(module: nn.Module, path: str) -> None:
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


def bytes2gigabytes(n: int) -> int:
    """Convert bytes to gigabytes"""
    return n / 1024**3


def image_psnr(reference: Image, proxy: Image) -> float:
    """Peak-Signal-to-Noise-Ratio in dB between a reference
    and a proxy PIL.Image
    """
    reference = np.asarray(reference)
    proxy = np.asarray(proxy)

    assert (
        reference.squeeze().shape == proxy.squeeze().shape
    ), f"{reference.shape} is incompatible with {proxy.shape}!"
    reference = reference.flatten()
    proxy = proxy.flatten()

    peak_signal = np.abs(reference).max()
    mse = np.sqrt(np.mean((reference - proxy) ** 2))
    return 20 * np.log10((peak_signal + 1e-5) / (mse + 1e-10))


def compute_psnr(reference: np.array, proxy: np.array) -> float:
    """Peak-Signal-to-Noise-Ratio in dB between a reference
    and a proxy np.array
    """
    assert (
        reference.squeeze().shape == proxy.squeeze().shape
    ), f"{reference.shape} is incompatible with {proxy.shape}!"
    reference = reference.flatten()
    proxy = proxy.flatten()

    peak_signal = np.abs(reference).max()
    mse = np.sqrt(np.mean((reference - proxy) ** 2))
    return 20 * np.log10((peak_signal + 1e-5) / (mse + 1e-10))
