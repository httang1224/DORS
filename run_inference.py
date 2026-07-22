# -*- encoding: utf-8 -*-
# Author  : HaiTong
# Time    : 2026/7/17 15:30
# File    : run_inference.py
# Software: PyCharm


import os
import numpy as np
import torch
from diffusers.utils import load_image
from PIL import Image

from src.dar_processor import (
    DynamicAttentionRouting,
    register_attention_editor_diffusers,
)
from src.pipeline_dors import DORSPipeline
from src.sam_processor import SAM3
from src.utils import (
    MaskRegionBuilder,
    blended_func,
    build_weight_inner_v2,
    dilate_by_area,
    set_seed,
)

GREEN = "\033[32m"
RESET = "\033[0m"

SDXL_MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
SAM3_MODEL_ID = "facebook/sam3"


GPU_ID = 0
DEVICE = torch.device(f"cuda:{GPU_ID}")
DTYPE = torch.float16
RESOLUTION = (512, 512)
SEED = 12345
BLEND = False
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(SCRIPT_DIR, "examples", "images", "sample_01.png")
MASK_PATH = os.path.join(SCRIPT_DIR, "examples", "masks", "sample_01.png")
OUT_IMG_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUT_IMG_DIR, exist_ok=True)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("DORS inference requires a CUDA-compatible GPU.")
    torch.cuda.set_device(GPU_ID)

    pipe = DORSPipeline.from_pretrained(
        SDXL_MODEL_ID,
        torch_dtype=DTYPE,
        variant="fp16",
    ).to(DEVICE)

    # DORS is applied to the up-block transformer layers.
    editor_common = DynamicAttentionRouting(start_layer=34, end_layer=70)

    register_attention_editor_diffusers(pipe.unet, editor_common)

    sam3 = SAM3(SAM3_MODEL_ID)

    if not os.path.isfile(IMAGE_PATH):
        raise FileNotFoundError(f"Image does not exist: {IMAGE_PATH}")
    if not os.path.isfile(MASK_PATH):
        raise FileNotFoundError(f"Mask does not exist: {MASK_PATH}")

    output_path = os.path.join(OUT_IMG_DIR, os.path.basename(IMAGE_PATH))

    source_image_pil = load_image(IMAGE_PATH).resize(RESOLUTION)
    source_mask_pil = (
        load_image(MASK_PATH)
        .convert("L")
        .resize(RESOLUTION, Image.Resampling.NEAREST)
    )

    dilate_mask_pil = dilate_by_area(source_mask_pil, 1.2)
    sam3_ss_mask_pil = sam3.run(
        IMAGE_PATH,
        MASK_PATH,
        cur_resolutions=RESOLUTION,
    ).convert("L")

    mask_builder_now = MaskRegionBuilder(dilate_mask_pil, visualize=False)
    editor_common.reset_context()

    dilate_mask_np = (
        np.array(dilate_mask_pil.convert("L").resize(RESOLUTION)) > 0
    ).astype(np.float32)
    dilate_mask_tensor = (
        torch.from_numpy(dilate_mask_np).unsqueeze(0).unsqueeze(0)
    )
    sam3_ss_mask_np = (np.array(sam3_ss_mask_pil) > 0).astype(np.float32)
    sam3_ss_mask_tensor = (
        torch.from_numpy(sam3_ss_mask_np).unsqueeze(0).unsqueeze(0)
    )

    inner_weight_np, _ = build_weight_inner_v2(
        source_mask_pil,
        sam3_ss_mask_pil,
    )
    inner_weight_tensor = (
        torch.from_numpy(inner_weight_np).unsqueeze(0).unsqueeze(0)
    )

    editor_common.set_context(
        mask_pil=dilate_mask_pil,
        mask_builder=mask_builder_now,
        visualize=False,
        ss_mask_tensor=sam3_ss_mask_tensor,
        dilate_mask_tensor=dilate_mask_tensor,
        inner_weight_tensor=inner_weight_tensor,
        cur_resize=RESOLUTION[0],
    )

    set_seed(SEED)
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    generated_image = pipe(
        prompt="",
        image=source_image_pil,
        mask_image=dilate_mask_pil,
        guidance_scale=8.0,
        num_inference_steps=20,
        strength=1.0,
        generator=generator,
        height=RESOLUTION[0],
        width=RESOLUTION[1],
    ).images[0]

    output_image = generated_image
    if BLEND:
        output_image = blended_func(
            source_mask_pil,
            source_image_pil,
            generated_image,
        )
    output_image.save(output_path)

    print(
        f"{GREEN}Saving as: {output_path}, "
        f"{np.array(output_image).shape}{RESET}"
    )
