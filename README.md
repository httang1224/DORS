<div align="center">

# DORS: Dynamic Attention Routing for Diffusion-based Object Removal in Dense Scenes

**ACM Multimedia 2026**

[Haitong Tang](mailto:httang1224@gmail.com) · [Haipeng Liu](mailto:hpliu_hfut@hotmail.com) · [Yang Wang](mailto:yangwang@hfut.edu.cn)  
Hefei University of Technology

[![Project Page](https://img.shields.io/badge/Project-Page-1F6FEB.svg)](https://httang1224.github.io/DORS/)
[![GitHub stars](https://img.shields.io/github/stars/httang1224/DORS?style=flat&logo=github)](https://github.com/httang1224/DORS)

![Paper](https://img.shields.io/badge/Paper-Coming_Soon-6E7781.svg?logo=adobeacrobatreader&logoColor=white)
![arXiv](https://img.shields.io/badge/arXiv-Coming_Soon-6E7781.svg?logo=arxiv&logoColor=white)
![DOR-Bench](https://img.shields.io/badge/DOR--Bench-Coming_Soon-6E7781.svg?logo=huggingface&logoColor=white)

</div>

## 🔍 Introduction

**DORS** is a training-free, plug-and-play framework for diffusion-based object removal in dense scenes with multiple visually similar instances.

Object removal is particularly challenging in dense scenes, where visually similar instances outside the removal mask can interfere with reconstruction. Existing diffusion-based methods may therefore leave target remnants, duplicate nearby objects, or introduce semantic artifacts. DORS addresses this problem by dynamically controlling information flow in self-attention during denoising.

For a detailed method overview and additional experimental results, visit our [project page](https://httang1224.github.io/DORS/).

## ✨ Key Features

- **Dense-scene object removal.** DORS mitigates incomplete removal and duplicate artifacts caused by interference from nearby similar instances.
- **Dynamic Attention Routing.** Our mechanism combines Instance-Filtered Attention (IFA) and Context-Guided Routing (CGR) to suppress misleading semantics while preserving useful contextual information.
- **Training-free and plug-and-play.** DORS operates entirely at inference time and requires no additional training or fine-tuning.
- **DOR-Bench.** We introduce a dedicated benchmark for evaluating object removal in challenging dense scenes. The benchmark will be released separately.

## 📢 News
- **2026-07-18:** The DORS code is publicly available.
- **2026-07-19:** The DORS [project page](https://httang1224.github.io/DORS/) is publicly available.
- **Coming soon:** Paper, arXiv preprint, and DOR-Bench.

## 🧠 Method Overview

DORS introduces **Dynamic Attention Routing (DAR)** into the diffusion denoising process. DAR consists of two complementary components:

- **Instance-Filtered Attention (IFA)** uses the removal target as a prompt for SAM3 to identify target-similar instances. It then prevents masked queries from aggregating misleading semantic information from these regions.
- **Context-Guided Routing (CGR)** adaptively fuses instance-filtered and full-context attention pathways, balancing semantic suppression with the structural context required for coherent reconstruction.

## 🛠️ Getting Started

### 📥 1. Clone this repository

```bash
git clone https://github.com/httang1224/DORS.git
cd DORS
```

### 📦 2. Install the dependencies

We recommend using a clean Python environment. Please ensure that your PyTorch build is compatible with the CUDA version installed on your system.

```bash
pip install -r requirements.txt
```

> **Note:** DORS inference requires a CUDA-compatible GPU.

### 🧩 3. Prepare the SAM3 source code

Clone the official [SAM3 repository](https://github.com/facebookresearch/sam3) and copy the `sam3` package to the root of this project:

```bash
git clone https://github.com/facebookresearch/sam3.git sam3_official
cp -r sam3_official/sam3 ./sam3
```

The copied `sam3/` directory is intentionally excluded from version control through `.gitignore`.

## 🧱 Model Weights

By default, the inference script downloads and loads the following models from Hugging Face:

```python
SDXL_MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
SAM3_MODEL_ID = "facebook/sam3"
```

To use locally downloaded weights, update the corresponding values in `run_inference.py`:

```python
SDXL_MODEL_ID = "/path/to/stable-diffusion-xl-1.0-inpainting-0.1"
SAM3_MODEL_ID = "/path/to/sam3.pt"
```

## 🚀 Quick Start

### 🖼️ 1. Set the input paths

Edit the following values in `run_inference.py`:

```python
IMAGE_PATH = "/path/to/source_image.png"
MASK_PATH = "/path/to/source_mask.png"
```

The image and mask must be spatially aligned. In the input mask:

- white pixels specify the target region to remove;
- black pixels specify the region to preserve.

The current inference configuration resizes the image and mask to `512 × 512`.

### ▶️ 2. Run inference

```bash
python run_inference.py
```

The script will:

- load the SDXL inpainting pipeline and register Dynamic Attention Routing;
- use SAM3 to identify instances similar to the removal target;
- perform object removal at the configured resolution;
- blend the reconstructed region with the source image;
- save the final result to `outputs/<mask-filename>`.

## 📊 Results

DORS produces cleaner object removal results in dense scenes, with fewer target remnants and duplicate structures caused by nearby similar instances. Qualitative examples and quantitative comparisons on DOR-Bench are available on the [project page](https://httang1224.github.io/DORS/).

## 📝 Citation

If you find this work useful for your research, please consider citing:

```bibtex
@inproceedings{tang2026dors,
  title     = {{DORS}: Dynamic Attention Routing for Diffusion-based
               Object Removal in Dense Scenes},
  author    = {Tang, Haitong and Liu, Haipeng and Wang, Yang},
  booktitle = {Proceedings of the 34th ACM International Conference
               on Multimedia},
  year      = {2026}
}
```

## 🙏 Acknowledgements

This implementation builds upon [Hugging Face Diffusers](https://github.com/huggingface/diffusers), [Stable Diffusion XL Inpainting](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1), [SAM3](https://github.com/facebookresearch/sam3), and [AttentiveEraser](https://github.com/Alibaba-VELLDEPTH/AttentiveEraser). We thank the authors and maintainers of these projects for making their work publicly available.

## 📬 Contact

For questions about the code or paper, please contact [Haitong Tang](mailto:httang1224@gmail.com).
