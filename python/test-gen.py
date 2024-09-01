import os
from pathlib import Path
from diffusionkit.mlx import FluxPipeline
from huggingface_hub import HfFolder, HfApi
from PIL import Image

# Define cache paths
usb_cache_path = "/Volumes/USB/huggingface/cache"
local_cache_path = os.path.expanduser("~/.cache/huggingface")


# Function to set and verify cache directory
def set_hf_cache():
    if os.path.exists("/Volumes/USB"):
        os.environ["HF_HOME"] = usb_cache_path
        Path(usb_cache_path).mkdir(parents=True, exist_ok=True)
        print(f"Using USB cache: {usb_cache_path}")
    else:
        os.environ["HF_HOME"] = local_cache_path
        print(f"USB not found. Using local cache: {local_cache_path}")

    print(f"HF_HOME is set to: {os.environ['HF_HOME']}")
    HfFolder.save_token(HfFolder.get_token())


# Set cache before initializing the pipeline
set_hf_cache()

# Initialize the pipeline
pipeline = FluxPipeline(
    shift=1.0,
    model_version="FLUX.1-dev",
    low_memory_mode=True,
    a16=True,
    w16=True,
)

# Load LoRA weights
# pipeline.load_lora_weights("XLabs-AI/flux-RealismLora")

# Define image generation parameters
HEIGHT = 512
WIDTH = 512
NUM_STEPS = 10 # 4 for FLUX.1-schnell, 50 for SD3
CFG_WEIGHT = 0.  # for FLUX.1-schnell, 5. for SD3
# LORA_SCALE = 0.8  # LoRA strength

# Define the prompt
prompt = "A photo realistic cat holding a sign that says hello world in the style of a snapchat from 2015"

# Generate the image
image, _ = pipeline.generate_image(
    prompt,
    cfg_weight=CFG_WEIGHT,
    num_steps=NUM_STEPS,
    latent_size=(HEIGHT // 8, WIDTH // 8),
    # lora_scale=LORA_SCALE,
)

# Save the generated image
output_format = "png"
output_quality = 100
image.save(f"flux_image.{output_format}", format=output_format, quality=output_quality)

print(f"Image generation complete. Saved image in {output_format} format.")