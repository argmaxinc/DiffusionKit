# DiffusionKit

[![Latest Python Version](https://img.shields.io/pypi/v/diffusionkit)](https://pypi.org/project/diffusionkit)

Run Diffusion Models on Apple Silicon with Core ML and MLX

This repository comprises
- `diffusionkit`, a Python package for converting PyTorch models to Core ML format and performing image generation with [MLX](https://github.com/ml-explore/mlx) in Python
- `DiffusionKit`, a Swift package for on-device inference of diffusion models using Core ML and MLX

<div align="center">
<img src="assets/diffusionkit.png" width=256>
</div>


## Installation

The following installation steps are required for:
- MLX inference
- PyTorch to Core ML model conversion

### Python Environment Setup

```bash
conda create -n diffusionkit python=3.11 -y
conda activate diffusionkit
cd /path/to/diffusionkit/repo
pip install -e .
```

### Hugging Face Hub Credentials

<details>
  <summary> Click to expand </summary>


[Stable Diffusion 3](https://huggingface.co/stabilityai/stable-diffusion-3-medium) requires users to accept the terms before downloading the checkpoint.

[FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) also requires users to accept the terms before downloading the checkpoint.

Once you accept the terms, sign in with your Hugging Face hub READ token as below:
> [!IMPORTANT]
> If using a fine-grained token, it is also necessary to [edit permissions](https://huggingface.co/settings/tokens) to allow `Read access to contents of all public gated repos you can access`

```bash
huggingface-cli login --token YOUR_HF_HUB_TOKEN
```

</details>


## <a name="converting-models-to-coreml"></a> Converting Models from PyTorch to Core ML

<details>
  <summary> Click to expand </summary>

**Step 1:** Follow the installation steps from the previous section

**Step 2:** Verify you've accepted the [StabilityAI license terms](https://huggingface.co/stabilityai/stable-diffusion-3-medium) and have allowed gated access on your [HuggingFace token](https://huggingface.co/settings/tokens)

**Step 3:** Prepare the denoise model (MMDiT) Core ML model files (`.mlpackage`)

```shell
python -m python.src.diffusionkit.tests.torch2coreml.test_mmdit --sd3-ckpt-path stabilityai/stable-diffusion-3-medium --model-version 2b -o <output-mlpackages-directory> --latent-size {64, 128}
```

**Step 4:** Prepare the VAE Decoder Core ML model files (`.mlpackage`)

```shell
python -m python.src.diffusionkit.tests.torch2coreml.test_vae --sd3-ckpt-path stabilityai/stable-diffusion-3-medium -o <output-mlpackages-directory> --latent-size {64, 128}
```

Note:
- `--sd3-ckpt-path` can be a path any HuggingFace repo (e.g. `stabilityai/stable-diffusion-3-medium`) OR a path to a local `sd3_medium.safetensors` file
</details>

## <a name="image-generation-with-python-mlx"></a> Image Generation with Python MLX

<details>
  <summary> Click to expand </summary>

### CLI ###

Most simple:
```shell
diffusionkit-cli --prompt "a photo of a cat" --output-path </path/to/output/image.png>
```

Some notable optional arguments for:
- Reproduciblity of results, use `--seed`
- image-to-image, use `--image-path` (path to input image) and `--denoise` (value between 0. and 1.)
- Enabling T5 encoder in SD3, use `--t5` (FLUX must use T5 regardless of this argument)
- Different resolutions, use `--height` and `--width`
- Using a local checkpoint, use `--local-ckpt </path/to/ckpt.safetensors>` (e.g. `~/models/stable-diffusion-3-medium/sd3_medium.safetensors`).

Please refer to the help menu for all available arguments: `diffusionkit-cli -h`.

Note: When using `FLUX.1-dev`, verify you've accepted the [FLUX.1-dev licence](https://huggingface.co/black-forest-labs/FLUX.1-dev) and have allowed gated access on your [HuggingFace token](https://huggingface.co/settings/tokens)

### Code ###

For Stable Diffusion 3:
```python
from diffusionkit.mlx import DiffusionPipeline
pipeline = DiffusionPipeline(
  shift=3.0,
  use_t5=False,
  model_version="argmaxinc/mlx-stable-diffusion-3-medium",
  low_memory_mode=True,
  a16=True,
  w16=True,
)
```

For FLUX:
```python
from diffusionkit.mlx import FluxPipeline
pipeline = FluxPipeline(
  shift=1.0,
  model_version="argmaxinc/mlx-FLUX.1-schnell", # model_version="argmaxinc/mlx-FLUX.1-dev" for FLUX.1-dev
  low_memory_mode=True,
  a16=True,
  w16=True,
)
```

Finally, to generate the image, use the `generate_image()` function:
```python
HEIGHT = 512
WIDTH = 512
NUM_STEPS = 4  #  4 for FLUX.1-schnell, 50 for SD3 and FLUX.1-dev
CFG_WEIGHT = 0. # for FLUX.1-schnell, 5. for SD3

image, _ = pipeline.generate_image(
  "a photo of a cat",
  cfg_weight=CFG_WEIGHT,
  num_steps=NUM_STEPS,
  latent_size=(HEIGHT // 8, WIDTH // 8),
)
```
Some notable optional arguments:
- For image-to-image, use `image_path` (path to input image) and `denoise` (value between 0. and 1.) input variables.
- For seed, use `seed` input variable.
- For negative prompt, use `negative_text` input variable.

The generated `image` can be saved with:
```python
image.save("path/to/save.png")
```

</details>

## Image Generation with Swift

<details>
  <summary> Click to expand </summary>

### Core ML Swift

[Apple Core ML Stable Diffusion](https://github.com/apple/ml-stable-diffusion) is the initial Core ML backend for DiffusionKit. Stable Diffusion 3 support is upstreamed to that repository while we build the holistic Swift inference package.

### MLX Swift
🚧

</details>

## License

DiffusionKit is released under the MIT License. See [LICENSE](LICENSE) for more details.

## Citation

If you use DiffusionKit for something cool or just find it useful, please drop us a note at [info@takeargmax.com](mailto:info@takeargmax.com)!

If you use DiffusionKit for academic work, here is the BibTeX:

```bibtex
@misc{diffusionkit-argmax,
   title = {DiffusionKit},
   author = {Argmax, Inc.},
   year = {2024},
   URL = {https://github.com/argmaxinc/DiffusionKit}
}
```
