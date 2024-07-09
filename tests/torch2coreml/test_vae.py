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
from huggingface_hub import hf_hub_download

from python.src.diffusionkit.torch import vae
from python.src.diffusionkit.torch.model_io import _load_vae_decoder_weights

torch.set_grad_enabled(False)
logger = get_logger(__name__)

TEST_SD3_CKPT_PATH = os.getenv("TEST_SD3_CKPT_PATH", None) or None
TEST_SD3_HF_REPO = os.getenv("TEST_SD3_HF_REPO", None) or None
TEST_CACHE_DIR = os.getenv("TEST_CACHE_DIR", None) or "/tmp"
TEST_DEV = os.getenv("TEST_DEV", None) or get_fastest_device()
TEST_TORCH_DTYPE = torch.float32
TEST_PSNR_THR = 35
TEST_LATENT_SIZE = 64  # 64 latent -> 512 image, 128 latent -> 1024 image

# Test configuration
# argmaxtools_test_utils.TEST_DEFAULT_NBITS = 8
argmaxtools_test_utils.TEST_MIN_SPEEDUP_VS_CPU = 3.0
argmaxtools_test_utils.TEST_COREML_PRECISION = ct.precision.FLOAT16
argmaxtools_test_utils.TEST_COMPRESSION_MIN_SPEEDUP = 0.5
argmaxtools_test_utils.TEST_COMPUTE_UNIT = ct.ComputeUnit.CPU_AND_GPU
argmaxtools_test_utils.TEST_SKIP_SPEED_TESTS = True


SD3_8b = vae.VAEDecoderConfig(resolution=1024)
SD3_2b = vae.VAEDecoderConfig(resolution=512)


class TestSD3VAEDecoder(argmaxtools_test_utils.CoreMLTestsMixin, unittest.TestCase):
    """Unit tests for stable_duffusion_3.vae.VAEDecoder module"""

    @classmethod
    def setUpClass(cls):
        global TEST_SD3_CKPT_PATH
        cls.model_name = "VAEDecoder"
        cls.test_output_names = ["image"]
        cls.test_cache_dir = TEST_CACHE_DIR

        # Base test model
        logger.info("Initializing SD3 VAEDecoder model")
        cls.test_torch_model = (
            vae.VAEDecoder(SD3_2b).to(TEST_DEV).to(TEST_TORCH_DTYPE).eval()
        )
        logger.info("Initialized.")

        TEST_SD3_CKPT_PATH = TEST_SD3_CKPT_PATH or hf_hub_download(
            TEST_SD3_HF_REPO, "sd3_medium.safetensors"
        )
        if TEST_SD3_CKPT_PATH is not None:
            logger.info(f"Loading SD3 model checkpoint from {TEST_SD3_CKPT_PATH}")
            _load_vae_decoder_weights(cls.test_torch_model, TEST_SD3_CKPT_PATH)
            logger.info("Loaded.")
        else:
            logger.info(
                "No TEST_SD3_CKPT_PATH (--sd3-ckpt-path) provided, exporting random weights"
            )

        # Sample inputs
        # TODO(atiorh): CLI configurable model version
        cls.test_torch_inputs = get_test_inputs(SD3_2b)

        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        cls.test_torch_model = None
        cls.test_torch_inputs = None
        super().tearDownClass()


def get_test_inputs(config: vae.VAEDecoderConfig) -> Dict[str, torch.Tensor]:
    """Generate random inputs for the SD3 MMDiT model"""
    config_expected_latent_resolution = (
        config.resolution // 2 ** len(config.channel_multipliers) - 1
    )
    if TEST_LATENT_SIZE != config_expected_latent_resolution:
        logger.warning(
            f"TEST_LATENT_SIZE ({TEST_LATENT_SIZE}) does not match the implied "
            "latent resolution from the model config "
        )

    z_dims = (1, config.in_channels, TEST_LATENT_SIZE, TEST_LATENT_SIZE)
    return {"z": torch.randn(*z_dims).to(TEST_DEV).to(TEST_TORCH_DTYPE)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sd3-ckpt-path", default=TEST_SD3_CKPT_PATH, type=str)
    parser.add_argument("-o", default=TEST_CACHE_DIR, type=str)
    parser.add_argument("--latent-size", default=TEST_LATENT_SIZE, type=int)
    args = parser.parse_args()

    TEST_SD3_CKPT_PATH = (
        args.sd3_ckpt_path if os.path.exists(args.sd3_ckpt_path) else None
    )
    TEST_SD3_HF_REPO = args.sd3_ckpt_path
    TEST_LATENT_SIZE = args.latent_size

    with argmaxtools_test_utils._get_test_cache_dir(args.o) as TEST_CACHE_DIR:
        suite = unittest.TestSuite()
        suite.addTest(TestSD3VAEDecoder("test_torch2coreml_correctness_and_speedup"))

        if os.getenv("DEBUG", False):
            suite.debug()
        else:
            runner = unittest.TextTestRunner()
            runner.run(suite)
