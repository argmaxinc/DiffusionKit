import os

from setuptools import find_packages, setup
from setuptools.command.install import install

VERSION = "0.2.16"


class VersionInstallCommand(install):
    def run(self):
        install.run(self)
        version_file = os.path.join(self.install_lib, "diffusionkit", "version.py")
        with open(version_file, "w") as f:
            f.write(f"__version__ = '{VERSION}'\n")


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
        "argmaxtools>=0.1.13",
        "torch",
        "safetensors",
        "mlx",
        "jaxtyping",
        "transformers",
        "pillow",
        "sentencepiece",
    ],
    packages=find_packages(where="python/src"),
    package_dir={"": "python/src"},
    entry_points={
        "console_scripts": [
            "diffusionkit-cli=diffusionkit.mlx.scripts.generate_images:cli",
        ],
    },
    cmdclass={
        "install": VersionInstallCommand,
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
