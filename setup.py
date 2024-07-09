from setuptools import find_packages, setup

VERSION = "0.1.0"

with open("README.md") as f:
    readme = f.read()

setup(
    name="diffusionkit",
    version=VERSION,
    url="https://github.com/argmaxinc/DiffusionKit",
    description="Argmax Model Optimization Toolkit for Diffusion Models.",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Argmax, Inc.",
    install_requires=[
        "argmaxtools",
        "torch",
        "safetensors",
        "mlx",
        "jaxtyping",
        "transformers",
        "pillow",
        "sentencepiece",
    ],
    packages=["diffusionkit"],
    package_dir={"": "python/src",
                 "diffusionkit": "python/src/diffusionkit"},
    entry_points={
        "console_scripts": [
            "diffusionkit-cli=python.src.mlx.scripts.generate_images:cli",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
)
