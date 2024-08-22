import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
from huggingface_hub import hf_hub_download, HfApi
import os
import sys
from pathlib import Path

from sympy import false
from tqdm import tqdm


current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Now try to import using both relative and absolute imports
try:
    from .config import FLUX_DEV, FLUX_SCHNELL, MMDiTConfig, PositionalEncoding
    from .mmdit import MMDiT
    from .model_io import flux_state_dict_adjustments
except ImportError:
    from diffusionkit.mlx.config import FLUX_DEV, FLUX_SCHNELL, MMDiTConfig, PositionalEncoding
    from diffusionkit.mlx.mmdit import MMDiT
    from diffusionkit.mlx.model_io import flux_state_dict_adjustments


def load_flux_weights(model_key="flux-dev"):
    config = FLUX_DEV if model_key == "flux-dev" else FLUX_SCHNELL
    repo_id = "black-forest-labs/FLUX.1-dev" if model_key == "flux-dev" else "black-forest-labs/FLUX.1-schnell"
    file_name = "flux1-dev.safetensors" if model_key == "flux-dev" else "flux1-schnell.safetensors"

    # Set custom HF_HOME location
    custom_hf_home = "/Volumes/USB/huggingface/hub"
    os.environ["HF_HOME"] = custom_hf_home

    # Use the custom HF_HOME location or fall back to the default
    hf_home = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    # Check if the file already exists in the custom location
    local_file = os.path.join(hf_home, "hub", repo_id.split("/")[-1], file_name)
    # Download the file if it doesn't exist

    if not os.path.exists(local_file):
        print(f"Downloading {file_name} to {hf_home}")
        local_file = hf_hub_download(
            repo_id,
            file_name,
            cache_dir=hf_home,
            force_download=False,
            resume_download=True,
        )
    else:
        print(f"Using existing file: {local_file}")

    # Load the weights
    weights = mx.load(local_file)
    return weights, config

def verify_conversion(weights, config):
    # Initialize the model
    model = MMDiT(config)
    mlx_model = tree_flatten(model)
    mlx_dict = {m[0]: m[1] for m in mlx_model if isinstance(m[1], mx.array)}

    # Adjust the weights
    adjusted_weights = flux_state_dict_adjustments(
        weights, prefix="", hidden_size=config.hidden_size, mlp_ratio=config.mlp_ratio
    )

    # Verify the conversion
    weights_set = set(adjusted_weights.keys())
    mlx_dict_set = set(mlx_dict.keys())

    print("Keys in weights but not in model:")
    for k in weights_set - mlx_dict_set:
        print(k)
    print(f"Count: {len(weights_set - mlx_dict_set)}")

    print("\nKeys in model but not in weights:")
    for k in mlx_dict_set - weights_set:
        print(k)
    print(f"Count: {len(mlx_dict_set - weights_set)}")

    print("\nShape mismatches:")
    count = 0
    for k in weights_set & mlx_dict_set:
        if adjusted_weights[k].shape != mlx_dict[k].shape:
            print(f"{k}: weights {adjusted_weights[k].shape}, model {mlx_dict[k].shape}")
            count += 1
    print(f"Total mismatches: {count}")


def save_modified_weights(weights, output_file):
    print(f"Saving modified weights to {output_file}")

    # Convert the dictionary of arrays to a flat dictionary
    flat_weights = {}
    for key, value in weights.items():
        if not isinstance(value, mx.array):
            flat_weights[key] = mx.array(value)
        else:
            flat_weights[key] = value

    # Save the flat dictionary
    mx.save(output_file, flat_weights)
    print("Weights saved successfully!")

def upload_to_hub(file_path, repo_id, token):
    print(f"Uploading {file_path} to {repo_id}")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=os.path.basename(file_path),
        repo_id=repo_id,
        token=token
    )
    print("Upload completed successfully!")

def main():
    # Load the weights and config
    weights, config = load_flux_weights("flux-dev")  # or "flux-schnell"

    # Verify the conversion
    verify_conversion(weights, config)

    output_file = "flux1-dev-mlx.safetensors"
    save_modified_weights(weights, output_file)
    repo_id = "raoulritter/flux-dev-mlx"
    token = os.getenv("HF_TOKEN")  # Make sure to set this environment variable
    upload_to_hub(output_file, repo_id, token)

if __name__ == "__main__":
    main()