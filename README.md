<div align="center">

# DORS: Dynamic Attention Routing for Diffusion-based Object Removal in Dense Scenes

**ACM Multimedia 2026**

[Haitong Tang](https://scholar.google.com/citations?user=oG7-rK4AAAAJ&hl=en) · [Haipeng Liu](https://scholar.google.com/citations?user=Xw0l6x8AAAAJ&hl=en) · [Yang Wang](https://scholar.google.com/citations?user=uzljmU8AAAAJ&hl=en)<br>
Hefei University of Technology

[![Project Page](https://img.shields.io/badge/Project-Page-1F6FEB.svg)](https://httang1224.github.io/DORS/)&nbsp;&nbsp;[![arXiv](https://img.shields.io/badge/arXiv-2607.16656-B31B1B.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2607.16656)&nbsp;&nbsp;[![Paper](https://img.shields.io/badge/Paper-PDF-B31B1B.svg?logo=adobeacrobatreader&logoColor=white)](https://arxiv.org/pdf/2607.16656)&nbsp;&nbsp;[![DOR-Bench](https://img.shields.io/badge/Hugging_Face-DOR--Bench-D4A900.svg?logo=huggingface&logoColor=FFD21E)](https://huggingface.co/datasets/qc1752/DOR-Bench)&nbsp;&nbsp;[![GitHub stars](https://img.shields.io/github/stars/httang1224/DORS?style=flat&logo=github)](https://github.com/httang1224/DORS)

</div>

## 🔍 Introduction

**DORS** is a training-free, plug-and-play framework for diffusion-based object removal in dense scenes with multiple visually similar instances.

Object removal is particularly challenging in dense scenes, where visually similar instances outside the removal mask can interfere with reconstruction. Existing diffusion-based methods may therefore leave target remnants, duplicate nearby objects, or introduce semantic artifacts. DORS addresses this problem by dynamically controlling information flow in self-attention during denoising.

For a detailed method overview and additional experimental results, visit our [project page](https://httang1224.github.io/DORS/).

## ✨ Key Features

- **Dense-scene object removal.** DORS mitigates incomplete removal and duplicate artifacts caused by interference from nearby similar instances.
- **Dynamic Attention Routing.** Our mechanism combines Instance-Filtered Attention (IFA) and Context-Guided Routing (CGR) to suppress misleading semantics while preserving useful contextual information.
- **Training-free and plug-and-play.** DORS operates entirely at inference time and requires no additional training or fine-tuning.
- **DOR-Bench.** We introduce a dedicated benchmark for evaluating object removal in challenging dense scenes, publicly available on [Hugging Face](https://huggingface.co/datasets/qc1752/DOR-Bench).

## 📢 News
- **2026-07-22:** Ready-to-run inference examples are now available.
- **2026-07-21:** The DORS [arXiv preprint](https://arxiv.org/abs/2607.16656) is now available.
- **2026-07-20:** [DOR-Bench](https://huggingface.co/datasets/qc1752/DOR-Bench) is now available on Hugging Face.
- **2026-07-19:** The DORS [project page](https://httang1224.github.io/DORS/) is now live.
- **2026-07-18:** The official DORS implementation has been released.

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

### 🖼️ 1. Try a bundled example

The repository includes three image-mask pairs under `examples/`. By default,
`run_inference.py` uses `sample_01.png`:

```python
IMAGE_PATH = os.path.join(SCRIPT_DIR, "examples", "images", "sample_01.png")
MASK_PATH = os.path.join(SCRIPT_DIR, "examples", "masks", "sample_01.png")
```

You can switch both filenames to `sample_02.png` or `sample_03.png` to run the
other examples. To process your own image, replace `IMAGE_PATH` and `MASK_PATH`
with the corresponding file paths. The image and mask must be spatially aligned.
In the input mask:

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
- save the result to `outputs/<image-filename>`.

## 📊 Results

DORS produces cleaner object removal results in dense scenes, with fewer target remnants and duplicate structures caused by nearby similar instances. Qualitative examples and quantitative comparisons on DOR-Bench are available on the [project page](https://httang1224.github.io/DORS/).

## 📬 Contact

For questions about the code or paper, please contact [Haitong Tang](mailto:httang1224@gmail.com).

## 📝 Citation

If you find this work useful for your research, please consider citing:

```bibtex
@article{tang2026dors,
  title     = {{DORS}: Dynamic Attention Routing for Diffusion-based
               Object Removal in Dense Scenes},
  author    = {Tang, Haitong and Liu, Haipeng and Wang, Yang},
  journal   = {arXiv preprint arXiv:2607.16656},
  year      = {2026}
}
```

## 🙏 Acknowledgements

This implementation builds upon [Hugging Face Diffusers](https://github.com/huggingface/diffusers), [Stable Diffusion XL Inpainting](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1), [SAM3](https://github.com/facebookresearch/sam3), and [AttentiveEraser](https://github.com/Alibaba-VELLDEPTH/AttentiveEraser). We thank the authors and maintainers of these projects for making their work publicly available.
