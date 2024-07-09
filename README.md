# DiffusionKit

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

[Stable Diffusion 3](https://huggingface.co/stabilityai/stable-diffusion-3-medium) requires users to accept the terms before downloading the checkpoint. Once you accept the terms, sign in with your Hugging Face hub READ token as below:
> [!IMPORTANT]
> If using a fine-grained token, it is also necessary to [edit permissions](https://huggingface.co/settings/tokens) to allow `Read access to contents of all public gated repos you can access`

```bash
huggingface-cli login --token YOUR_HF_HUB_TOKEN
```

## <a name="converting-models-to-coreml"></a> Converting Models from PyTorch to Core ML

<details>
  <summary> Click to expand </summary>

**Step 1:** Follow the installation steps from the previous section

**Step 2:** Verify you've accepted the [StabilityAI license terms](https://huggingface.co/stabilityai/stable-diffusion-3-medium) and have allowed gated access on your [HuggingFace token](https://huggingface.co/settings/tokens)

**Step 3:** Prepare the denoise model (MMDiT) Core ML model files (`.mlpackage`)

```shell
python -m tests.torch2coreml.test_mmdit --sd3-ckpt-path stabilityai/stable-diffusion-3-medium --model-version 2b -o <output-mlpackages-directory> --latent-size {64, 128}
```

**Step 4:** Prepare the VAE Decoder Core ML model files (`.mlpackage`)

```shell
python -m tests.torch2coreml.test_vae --sd3-ckpt-path stabilityai/stable-diffusion-3-medium -o <output-mlpackages-directory> --latent-size {64, 128}
```

Note:
- `--sd3-ckpt-path` can be a path any HuggingFace repo (e.g. `stabilityai/stable-diffusion-3-medium`) OR a path to a local `sd3_medium.safetensors` file
</details>

## <a name="image-generation-with-python-mlx"></a> Image Generation with Python MLX

<details>
  <summary> Click to expand </summary>

### CLI ###
For simple text-to-image in float16 precision:
```shell
diffusionkit-cli --prompt "a photo of a cat" --output-path </path/to/output/image.png> --seed 0 --w16 --a16
```

Some notable optional arguments:
- For image-to-image, use `--image-path` (path to input image) and `--denoise` (value between 0. and 1.)
- T5 text embeddings, use `--t5`
- For different resolutions, use `--height` and `--width`
- For using a local checkpoint, use `--local-ckpt </path/to/ckpt.safetensors>` (e.g. `~/models/stable-diffusion-3-medium/sd3_medium.safetensors`).

Please refer to the help menu for all available arguments: `diffusionkit-cli -h`.

### Code ###
After installing the package, import it using:
```python
from diffusionkit.mlx import DiffusionPipeline
```

Then, initialize the pipeline object:
```python
pipeline = DiffusionPipeline(
  model="argmaxinc/stable-diffusion",
  w16=True,
  shift=3.0,
  use_t5=False,
  model_size="2b",
  low_memory_mode=False,
  a16=True,
)
```

Some notable optional arguments:
- For T5 text embeddings, set `use_t5=True`
- For using a local checkpoint, set `local_ckpt=</path/to/ckpt.safetensors>` (e.g. `~/models/stable-diffusion-3-medium/sd3_medium.safetensors`).
- If you want to use the `pipeline` object more than once, set `low_memory_mode=False`.
- For loading weights in FP32, set `w16=False`
- For FP32 activations, set `a16=False`

Note: Only `2b` model size is available for this pipeline.

Finally, to generate the image, use the `generate_image()` function:
```python
HEIGHT = 512
WIDTH = 512

image, _ = pipeline.generate_image(
  "a photo of a cat holding a sign that says 'Hello!'",
  cfg_weight=5.0,
  num_steps=50,
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
