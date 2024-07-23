#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Argmax, Inc. All Rights Reserved.
#

import json
import os
import unittest

import mlx.core as mx
import numpy as np
from argmaxtools.utils import get_logger
from diffusionkit.mlx import DiffusionPipeline
from diffusionkit.utils import image_psnr
from huggingface_hub import hf_hub_download
from PIL import Image

logger = get_logger(__name__)

W16 = False
A16 = False
TEST_PSNR_THRESHOLD = 20
TEST_MIN_SPEEDUP = 0.95
SD3_TEST_IMAGES_REPO = "argmaxinc/sd-test-images"
TEST_CACHE_DIR = ".cache"
CACHE_SUBFOLDER = None

LOW_MEMORY_MODE = True
SAVE_IMAGES = True
MODEL_SIZE = "2b"
USE_T5 = False
SKIP_CORRECTNESS = False


class TestSD3Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sd_test_images_metadata = hf_hub_download(
            SD3_TEST_IMAGES_REPO, "metadata.json", repo_type="dataset"
        )

    @classmethod
    def tearDownClass(cls):
        del cls.sd_test_images_metadata
        cls.sd_test_images_metadata = None

        super().tearDownClass()

    def test_sd3_pipeline_correctness(self):
        with open(self.sd_test_images_metadata, "r") as f:
            metadata = json.load(f)

        # Group metadata by model size
        model_examples = {"2b": [], "8b": []}
        for data in metadata:
            model_examples[data["model_size"]].append(data)

        for model_size, examples in model_examples.items():
            sd3 = DiffusionPipeline(
                model_size=model_size, w16=W16, low_memory_mode=LOW_MEMORY_MODE, a16=A16
            )
            if not LOW_MEMORY_MODE:
                sd3.ensure_models_are_loaded()
            for example in examples:
                image_path = example["image"]
                sd3.use_t5 = example["use_t5"]

                f = hf_hub_download(
                    SD3_TEST_IMAGES_REPO, image_path, repo_type="dataset"
                )
                image = Image.open(f)

                generated_image, _ = sd3.generate_image(
                    text=example["prompt"],
                    num_steps=example["steps"],
                    cfg_weight=example["cfg"],
                    negative_text=example["neg_prompt"],
                    latent_size=(example["height"] // 8, example["width"] // 8),
                    seed=example["seed"],
                )

                if SAVE_IMAGES:
                    img_cache_dir = os.path.join(TEST_CACHE_DIR, "img")
                    out_path = os.path.join(img_cache_dir, image_path)
                    if not os.path.exists(img_cache_dir):
                        os.makedirs(img_cache_dir, exist_ok=True)
                    generated_image.save(out_path)
                    logger.info(f"Saved the image to {out_path}")

                psnr = image_psnr(image, generated_image)
                logger.info(f"Image: {image_path} | PSNR: {psnr} dB")
                self.assertGreaterEqual(psnr, TEST_PSNR_THRESHOLD)
                if LOW_MEMORY_MODE:
                    del sd3
                    sd3 = DiffusionPipeline(
                        model_size=model_size,
                        w16=W16,
                        low_memory_mode=LOW_MEMORY_MODE,
                        a16=A16,
                    )
            del sd3

    def test_memory_usage(self):
        with open(self.sd_test_images_metadata, "r") as f:
            metadata = json.load(f)

        # Group metadata by model size
        model_examples = {"2b": [], "8b": []}
        for data in metadata:
            model_examples[data["model_size"]].append(data)

        sd3 = DiffusionPipeline(
            model_size=MODEL_SIZE, w16=W16, low_memory_mode=LOW_MEMORY_MODE, a16=A16
        )
        if not LOW_MEMORY_MODE:
            sd3.ensure_models_are_loaded()

        log = None
        for example in model_examples[MODEL_SIZE]:
            sd3.use_t5 = USE_T5
            logger.info(
                f"Testing memory usage... USE_T5 = {USE_T5} | MODEL_SIZE = {MODEL_SIZE}"
            )
            _, log = sd3.generate_image(
                text=example["prompt"],
                num_steps=3,
                cfg_weight=example["cfg"],
                negative_text=example["neg_prompt"],
                latent_size=(example["height"] // 8, example["width"] // 8),
                seed=example["seed"],
            )
            break

        out_folder = os.path.join(TEST_CACHE_DIR, CACHE_SUBFOLDER)
        out_path = os.path.join(out_folder, f"{MODEL_SIZE}_log.json")
        if not os.path.exists(out_folder):
            os.makedirs(out_folder, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(log, f, indent=2)
        logger.info(f"Saved the memory log to {out_path}")
        self.assertIsNotNone(log)


def main(args):
    global LOW_MEMORY_MODE, SAVE_IMAGES, SKIP_CORRECTNESS, MODEL_SIZE, W16, A16, CACHE_SUBFOLDER, USE_T5

    LOW_MEMORY_MODE = args.low_memory_mode
    SAVE_IMAGES = args.save_images
    SKIP_CORRECTNESS = args.skip_correctness
    MODEL_SIZE = args.model_size
    W16 = args.w16
    A16 = args.a16
    CACHE_SUBFOLDER = args.subfolder
    USE_T5 = args.use_t5

    suite = unittest.TestSuite()
    if not SKIP_CORRECTNESS:
        suite.addTest(TestSD3Pipeline("test_sd3_pipeline_correctness"))

    suite.addTest(TestSD3Pipeline("test_memory_usage"))
    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-low-memory-mode",
        action="store_false",
        dest="low_memory_mode",
        help="Disable low memory mode: models remains loaded in memory after forward pass.",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Saves generated images to .cache/img/ folder.",
    )
    parser.add_argument(
        "--skip-correctness", action="store_true", help="Skip the correctness test."
    )
    parser.add_argument(
        "--model-size", type=str, default="2b", help="Model size to use for the test."
    )
    parser.add_argument(
        "--w16", action="store_true", help="Loads the models in float16."
    )
    parser.add_argument(
        "--a16", action="store_true", help="Use float16 for the model activations."
    )
    parser.add_argument(
        "--subfolder",
        default="default",
        type=str,
        help=f"If specified, this string will be appended to the cache directory name.",
    )
    parser.add_argument(
        "--use-t5", action="store_true", help="Use T5 model for text generation."
    )
    args = parser.parse_args()

    main(args)
