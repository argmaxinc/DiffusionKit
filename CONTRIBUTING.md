# Contributing to DiffusionKit

## Overview

We welcome and encourage contributions to DiffusionKit! Whether you're fixing bugs, improving documentation, or adding new features from the roadmap, your help is appreciated. This guide will help you get started with contributing to DiffusionKit.

## Getting Started

1. **Fork the Repository**: Start by [forking](https://github.com/argmaxinc/DiffusionKit/fork) the DiffusionKit repository on GitHub to your personal account.

2. **Clone Your Fork**: Clone your fork to your local machine to start making changes.

   ```bash
   git clone https://github.com/[your-username]/DiffusionKit.git
   cd DiffusionKit
   ```

## Setting Up Your Development Environment

1. **Install Dependencies**: Use the provided `Makefile` to set up your environment. Run `make setup` to install necessary dependencies.

   ```bash
   make setup
   ```

   This will add any necessary CLI tools, as well as the pre-commit hooks for code formatting.

2. **Download Models**: Run to download the required models to run and test locally.

   You can specify the model version as follows:

    For python and mlx:

    ```bash
    make download-model MODEL=stabilityai/stable-diffusion-3-medium
    ```

    For Swift with Core ML:

    ```bash
    make download-model MODEL=argmaxinc/coreml-stable-diffusion-3-medium
    ```

## Making Changes

1. **Create a Branch**: Create a new branch for your changes.

   ```bash
   git checkout -b my-descriptive-branch-name
   ```

2. **Make Your Changes**: Implement your changes, add new features, or fix bugs. Ensure you adhere to the existing coding style. If you're adding new features, make sure to update or add any documentation or tests as needed.

## Submitting Your Changes

1. **Commit Your Changes**: Once you're satisfied with your changes, commit them with a clear and concise commit message.

   ```bash
   git commit -am "Add a new feature"
   ```

2. **Push to Your Fork**: Push your changes to your fork on GitHub.

   ```bash
   git push origin my-descriptive-branch-name
   ```

3. **Create a Pull Request**: Go to the DiffusionKit repository on GitHub and create a new pull request from your fork. Ensure your pull request has a clear title and description.

4. **Code Review**: Wait for the maintainers to review your pull request. Be responsive to feedback and make any necessary changes.

## Guidelines

- **Code Style**: Follow the existing code style in the project.
- **Commit Messages**: Write meaningful commit messages that clearly describe the changes.
- **Documentation**: Update documentation if you're adding new features or making changes that affect how users interact with DiffusionKit.
- **Tests**: Add or update tests for new features or bug fixes.

## Final Steps

After your pull request has been reviewed and approved, a maintainer will merge it into the main branch. Congratulations, you've successfully contributed to DiffusionKit!

Thank you for making DiffusionKit better for everyone! ❤️‍🔥
