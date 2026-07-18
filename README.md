# DORS

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备 SAM3 源码

请从 Meta 官方仓库下载 SAM3，并将 `sam3/` 包复制到项目根目录：

```bash
git clone https://github.com/facebookresearch/sam3.git sam3_official
cp -r sam3_official/sam3 ./sam3
```

完成后的目录结构：

```text
DORS_github/
├── sam3/
├── run_inference.py
├── requirements.txt
└── ...
```

### 3. 准备模型权重

默认配置会在首次运行时从 Hugging Face 下载模型：

```python
SDXL_MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
SAM3_MODEL_ID = "facebook/sam3"
```

也可以自行下载权重，并在 `run_inference.py` 中填写本地路径：

```python
SDXL_MODEL_ID = "/path/to/stable-diffusion-xl-1.0-inpainting-0.1"
SAM3_MODEL_ID = "/path/to/sam3.pt"
```

### 4. 设置输入

在 `run_inference.py` 中填写图像和 mask 路径：

```python
IMAGE_PATH = "/path/to/source_image.png"
MASK_PATH = "/path/to/source_mask.png"
```

> Mask 中的白色区域表示需要移除的目标，黑色区域表示保留区域。

### 5. 运行推理

```bash
python run_inference.py
```

生成结果保存在：

```text
outputs/<mask 文件名>
```
