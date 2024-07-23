#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Argmax, Inc. All Rights Reserved.
#

import os
import unittest
from typing import Dict

import coremltools as ct
import torch
from argmaxtools import test_utils as argmaxtools_test_utils
from argmaxtools.utils import get_fastest_device, get_logger
from diffusionkit.torch import mmdit
from diffusionkit.torch.model_io import _load_mmdit_weights
from huggingface_hub import hf_hub_download

torch.set_grad_enabled(False)
logger = get_logger(__name__)

TEST_SD3_CKPT_PATH = os.getenv("TEST_SD3_CKPT_PATH", None) or None
TEST_CKPT_FILE_NAME = os.getenv("TEST_CKPT_FILE_NAME", None) or None
TEST_SD3_HF_REPO = os.getenv("TEST_SD3_HF_REPO", None) or None
TEST_CACHE_DIR = os.getenv("TEST_CACHE_DIR", None) or "/tmp"
TEST_DEV = os.getenv("TEST_DEV", None) or get_fastest_device()
TEST_TORCH_DTYPE = torch.float32
TEST_PSNR_THR = 35
TEST_LATENT_SIZE = 64  # 64 latent -> 512 image, 128 latent -> 1024 image
TEST_LATENT_HEIGHT = TEST_LATENT_SIZE
TEST_LATENT_WIDTH = TEST_LATENT_SIZE

TEST_MODELS = {
    "2b": mmdit.SD3_2b,
    "8b": mmdit.SD3_8b,
}


def setup_test_config(
    min_speedup_vs_cpu=3.0,
    compute_precision=ct.precision.FLOAT32,
    compute_unit=ct.ComputeUnit.CPU_AND_GPU,
    compression_min_speedup=0.2,
    default_nbits=None,
    skip_speed_tests=True,
    compile_coreml=False,
):
    argmaxtools_test_utils.TEST_MIN_SPEEDUP_VS_CPU = min_speedup_vs_cpu
    argmaxtools_test_utils.TEST_COREML_PRECISION = compute_precision
    argmaxtools_test_utils.TEST_COMPUTE_UNIT = compute_unit
    argmaxtools_test_utils.TEST_COMPRESSION_MIN_SPEEDUP = compression_min_speedup
    argmaxtools_test_utils.TEST_DEFAULT_NBITS = default_nbits
    argmaxtools_test_utils.TEST_SKIP_SPEED_TESTS = skip_speed_tests
    argmaxtools_test_utils.TEST_COMPILE_COREML = compile_coreml


class TestSD3MMDiT(argmaxtools_test_utils.CoreMLTestsMixin, unittest.TestCase):
    """Unit tests for stable_duffusion_3.mmdit.MMDiT module"""

    model_version = "2b"

    @classmethod
    def setUpClass(cls):
        global TEST_SD3_CKPT_PATH
        cls.model_name = "MultiModalDiffusionTransformer"
        cls.test_output_names = ["denoiser_output"]
        cls.test_cache_dir = TEST_CACHE_DIR

        # Base test model
        logger.info("Initializing SD3 model")
        cls.test_torch_model = (
            mmdit.MMDiT(TEST_MODELS[cls.model_version])
            .to(TEST_DEV)
            .to(TEST_TORCH_DTYPE)
            .eval()
        )
        logger.info("Initialized.")
        TEST_SD3_CKPT_PATH = TEST_SD3_CKPT_PATH or hf_hub_download(
            TEST_SD3_HF_REPO, "sd3_medium.safetensors"
        )
        if TEST_SD3_CKPT_PATH is not None:
            logger.info(f"Loading SD3 model checkpoint from {TEST_SD3_CKPT_PATH}")
            _load_mmdit_weights(cls.test_torch_model, TEST_SD3_CKPT_PATH)
            logger.info("Loaded.")
        else:
            logger.info(
                "No TEST_SD3_CKPT_PATH (--sd3-ckpt-path) provided, exporting random weights"
            )

        # Sample inputs
        # TODO(atiorh): CLI configurable model version
        cls.test_torch_inputs = get_test_inputs(TEST_MODELS[cls.model_version])

        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        cls.test_torch_model = None
        cls.test_torch_inputs = None
        super().tearDownClass()


def get_test_inputs(cfg: mmdit.MMDiTConfig) -> Dict[str, torch.Tensor]:
    """Generate random inputs for the SD3 MMDiT model"""
    batch_size = 2  # classifier-free guidance
    assert TEST_LATENT_HEIGHT <= cfg.max_latent_resolution
    assert TEST_LATENT_WIDTH <= cfg.max_latent_resolution

    latent_image_embeddings_dims = (
        batch_size,
        cfg.vae_latent_dim,
        TEST_LATENT_HEIGHT,
        TEST_LATENT_WIDTH,
    )
    pooled_text_embeddings_dims = (batch_size, cfg.pooled_text_embed_dim, 1, 1)
    token_level_text_embeddings_dims = (
        batch_size,
        cfg.token_level_text_embed_dim,
        1,
        cfg.text_seq_len,
    )
    timestep_dims = (2,)

    torch_test_inputs = {
        "latent_image_embeddings": torch.randn(*latent_image_embeddings_dims),
        "token_level_text_embeddings": torch.randn(*token_level_text_embeddings_dims),
        "pooled_text_embeddings": torch.randn(pooled_text_embeddings_dims),
        "timestep": torch.randn(*timestep_dims),
    }

    return {
        k: v.to(TEST_DEV).to(TEST_TORCH_DTYPE) for k, v in torch_test_inputs.items()
    }


def convert_mmdit_to_mlpackage(
    model_version: str,
    latent_h: int,
    latent_w: int,
    output_dir: str = None,
    **test_config_kwargs,
) -> str:
    """Converts a MMDiT model to a CoreML package.

    Returns:
        `str`: path to the converted model.
    """
    global TEST_SD3_CKPT_PATH, TEST_SD3_HF_REPO, TEST_LATENT_WIDTH, TEST_LATENT_HEIGHT, TEST_CACHE_DIR

    # Convert to CoreML
    TEST_SD3_HF_REPO = model_version
    TEST_LATENT_HEIGHT = latent_h or TEST_LATENT_SIZE
    TEST_LATENT_WIDTH = latent_w or TEST_LATENT_SIZE

    setup_test_config(compile_coreml=False, **test_config_kwargs)

    with argmaxtools_test_utils._get_test_cache_dir(
        persistent_cache_dir=output_dir
    ) as TEST_CACHE_DIR:
        suite = unittest.TestSuite()
        suite.addTest(TestSD3MMDiT("test_torch2coreml_correctness_and_speedup"))

        if os.getenv("DEBUG", False):
            suite.debug()
        else:
            runner = unittest.TextTestRunner()
            runner.run(suite)

    return os.path.join(TEST_CACHE_DIR, f"{TestSD3MMDiT.model_name}.mlpackage")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sd3-ckpt-path", default=TEST_SD3_CKPT_PATH, type=str)
    parser.add_argument("--ckpt-file-name", default="sd3_medium.safetensors", type=str)
    parser.add_argument(
        "--model-version",
        required=True,
        default="2b",
        choices=TEST_MODELS.keys(),
        type=str,
    )
    parser.add_argument("-o", default=TEST_CACHE_DIR, type=str)
    parser.add_argument("--latent-size", default=TEST_LATENT_SIZE, type=int)
    args = parser.parse_args()

    TEST_SD3_CKPT_PATH = (
        args.sd3_ckpt_path if os.path.exists(args.sd3_ckpt_path) else None
    )
    TEST_SD3_HF_REPO = args.sd3_ckpt_path
    TEST_LATENT_SIZE = args.latent_size
    TEST_CKPT_FILE_NAME = args.ckpt_file_name

    setup_test_config()

    with argmaxtools_test_utils._get_test_cache_dir(args.o) as TEST_CACHE_DIR:
        suite = unittest.TestSuite()
        suite.addTest(TestSD3MMDiT("test_torch2coreml_correctness_and_speedup"))

        if os.getenv("DEBUG", False):
            suite.debug()
        else:
            runner = unittest.TextTestRunner()
            runner.run(suite)
