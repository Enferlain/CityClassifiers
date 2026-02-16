# CityClassifiers

**A flexible framework for building high-performance image classifiers and aesthetic predictors.**

Whether you want to sort anime images, detect photographic defects, or build a custom aesthetic scorer, CityClassifiers provides the tools to do it efficiently using state-of-the-art vision models (like SigLIP, DINOv2, and AIMv2).

## 🚀 Getting Started

New to the project? We've written a step-by-step guide to get you up and running in minutes.

👉 **[Read the Quickstart Guide](docs/quickstart.md)**

This guide covers:
1.  Setting up your environment.
2.  Preparing your image dataset.
3.  Generating features (the "Entrypoint" to our backend).
4.  Training a model.
5.  Running predictions on new images.

## 💡 How It Works

Traditional deep learning can be resource-intensive. CityClassifiers optimizes this by splitting the process into two stages:

1.  **Feature Extraction:** We use a powerful, pre-trained "vision backbone" to look at your images and extract their essential features into compact mathematical representations (embeddings).
2.  **Training:** We train a smaller, specialized "Head" model on these features. This is incredibly fast and allows you to experiment with different architectures without re-processing your images.

*We also support end-to-end training if you need to fine-tune the vision backbone itself.*

## 📚 Documentation

For more detailed information, explore our documentation:

*   **[Quickstart Guide](docs/quickstart.md)** - Start here!
*   **[Developer Notes](docs/developer_notes.md)** - Known issues and dependency info.
*   **[Configs](docs/topics/configs.md)** - Understanding the YAML configuration system.
*   **[Training](docs/topics/training.md)** - Deep dive into training modes and parameters.
*   **[Inference](docs/topics/inference.md)** - How to use your trained models.
*   **[Models](docs/topics/models.md)** - Supported backbones and head architectures.

## ✨ Key Features

*   **Modern Backbones:** Support for SigLIP, DINOv2, and AIMv2.
*   **Advanced Heads:** Configurable models with Self-Attention, ResBlocks, and more.
*   **Flexible Training:** Train on embeddings, feature sequences, or raw images.
*   **Easy Inference:** Built-in scripts for batch processing folders and interactive Gradio demos.
*   **Experiment Tracking:** Integrated with Weights & Biases.

## 📂 Project Structure

*   `config/`: YAML configuration files.
*   `cityclassifiers/`: The core Python package.
*   `generate_embeddings.py`: Script to create single-vector embeddings.
*   `generate_feature_sequences.py`: Script to create rich feature sequences.
*   `demo_folder.py`: CLI tool for batch inference.
*   `demo_class_gradio.py`: Web UI for testing classifiers.

## Pre-trained Models

Check out our pre-trained models on Hugging Face:

*   **[CityAesthetics](https://huggingface.co/city96/CityAesthetics)**: Anime aesthetic predictor.
*   **[AnimeClassifiers](https://huggingface.co/city96/AnimeClassifiers)**: Detectors for chromatic aberration and compression artifacts.

---

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Enferlain/CityClassifiers)
