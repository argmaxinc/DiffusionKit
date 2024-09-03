import unittest
import mlx.core as mx
import os
from pathlib import Path
import sys

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

try:
    from .config import FLUX_DEV, FLUX_SCHNELL, MMDiTConfig
    from .mmdit import MMDiT
    from .model_io import flux_state_dict_adjustments
except ImportError:
    from diffusionkit.mlx.config import FLUX_DEV, FLUX_SCHNELL, MMDiTConfig
    from diffusionkit.mlx.mmdit import MMDiT
    from diffusionkit.mlx.model_io import flux_state_dict_adjustments

class TestFluxConversion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.custom_hf_home = "/Volumes/USB/huggingface/hub"
        os.environ["HF_HOME"] = cls.custom_hf_home

    def load_flux_weights(self, model_key="flux-dev"):
        config = FLUX_DEV if model_key == "flux-dev" else FLUX_SCHNELL
        repo_id = "black-forest-labs/FLUX.1-dev" if model_key == "flux-dev" else "black-forest-labs/FLUX.1-schnell"
        file_name = "flux1-dev.safetensors" if model_key == "flux-dev" else "flux1-schnell.safetensors"

        hf_home = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        local_file = os.path.join(hf_home, "hub", repo_id.split("/")[-1], file_name)

        if not os.path.exists(local_file):
            self.fail(f"Test file not found: {local_file}. Please download it before running the test.")

        weights = mx.load(local_file)
        return weights, config

    def test_flux_conversion(self):
        weights, config = self.load_flux_weights("flux-dev")
        
        model = MMDiT(config)
        mlx_model = mx.tree_flatten(model)
        mlx_dict = {m[0]: m[1] for m in mlx_model if isinstance(m[1], mx.array)}

        adjusted_weights = flux_state_dict_adjustments(
            weights, prefix="", hidden_size=config.hidden_size, mlp_ratio=config.mlp_ratio
        )

        weights_set = set(adjusted_weights.keys())
        mlx_dict_set = set(mlx_dict.keys())

        self.assertEqual(len(weights_set - mlx_dict_set), 0, "There are keys in weights but not in model")
        self.assertEqual(len(mlx_dict_set - weights_set), 0, "There are keys in model but not in weights")

        mismatches = 0
        for k in weights_set & mlx_dict_set:
            if adjusted_weights[k].shape != mlx_dict[k].shape:
                mismatches += 1

        self.assertEqual(mismatches, 0, f"Found {mismatches} shape mismatches between weights and model")

if __name__ == "__main__":
    unittest.main()