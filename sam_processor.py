# -*- encoding: utf-8 -*-
# Author  : HaiTong
# Time    : 2026/1/4 08:37
# File    : sam_processor.py
# Software: PyCharm


import os

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from scipy import ndimage
from scipy.ndimage import binary_fill_holes

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.visualization_utils import (
    normalize_bbox,
)


def mask_path_to_bbox_xywh(mask_path, res, pad=10):
    mask = Image.open(mask_path).convert("L").resize(res, Image.Resampling.NEAREST)
    mask_np = np.array(mask)

    mask_np = (mask_np > 1).astype(np.uint8) * 255

    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0:
        raise ValueError("Mask is empty, cannot extract bbox.")

    h_img, w_img = mask_np.shape

    x_min = max(0, xs.min() - pad)
    y_min = max(0, ys.min() - pad)
    x_max = min(w_img - 1, xs.max() + pad)
    y_max = min(h_img - 1, ys.max() + pad)

    w = x_max - x_min
    h = y_max - y_min

    return torch.tensor([[float(x_min), float(y_min), float(w), float(h)]], dtype=torch.float32)


def remove_small_black_holes(mask, max_hole_area_percentage=1.2):
    mask_bool = mask.astype(bool)

    black = ~mask_bool

    labeled, num = ndimage.label(black)

    sizes = ndimage.sum(black, labeled, range(1, num + 1))

    cleaned = mask_bool.copy()
    max_hole_area = mask_bool.size * max_hole_area_percentage / 1000
    for i, area in enumerate(sizes):
        if area < max_hole_area:
            cleaned[labeled == (i + 1)] = True

    return cleaned.astype(np.uint8)


def fill_mask_holes(mask_bool):
    return binary_fill_holes(mask_bool)


class SAM3:
    def __init__(self, model_id_or_path="facebook/sam3"):

        if os.path.isfile(model_id_or_path):
            checkpoint_path = model_id_or_path
        else:
            checkpoint_path = hf_hub_download(
                repo_id=model_id_or_path,
                filename="sam3.pt",
            )
        model = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            load_from_HF=False,
        )
        self.processor = Sam3Processor(model)

    def run(self, image_path, mask_path, pad=0, refined_mask_path=None, cur_resolutions=(1024, 1024)):

        image = Image.open(image_path).convert("RGB").resize(cur_resolutions)
        width, height = image.size
        state = self.processor.set_image(image)

        box_xywh = mask_path_to_bbox_xywh(mask_path, pad=pad, res=cur_resolutions)
        box_cxcywh = box_xywh_to_cxcywh(box_xywh)
        norm_box = normalize_bbox(box_cxcywh, width, height).flatten().tolist()

        self.processor.reset_all_prompts(state)
        state = self.processor.add_geometric_prompt(
            state=state,
            box=norm_box,
            label=True,
        )

        sam_masks = state["masks"]  # [N, 1, H, W] or [N, H, W]
        sam_masks = sam_masks.squeeze(1) if sam_masks.dim() == 4 else sam_masks

        if sam_masks.shape[0] == 0:
            return Image.new("L", cur_resolutions, 0)

        orig_mask = Image.open(mask_path).convert("L").resize(cur_resolutions, Image.Resampling.NEAREST)
        orig_mask_np = np.array(orig_mask) > 0  # bool
        orig_mask_t = torch.from_numpy(orig_mask_np).to(sam_masks.device)

        def compute_iou(m1, m2):
            inter = (m1 & m2).sum().float()
            union = (m1 | m2).sum().float()
            return inter / (union + 1e-6)

        ious = []
        for i in range(sam_masks.shape[0]):
            sam_i = sam_masks[i] > 0
            iou = compute_iou(sam_i, orig_mask_t)
            ious.append(iou.item())

        ious = np.array(ious)
        best_idx = int(np.argmax(ious))

        extra_masks = [sam_masks[i].cpu().numpy().astype(bool) for i in range(sam_masks.shape[0]) if i != best_idx]

        extra_masks = [fill_mask_holes(m) for m in extra_masks]

        if refined_mask_path is not None:
            refined_mask_tensor = sam_masks[best_idx].detach().cpu()
            refined_mask_pil = Image.fromarray(refined_mask_tensor.numpy().astype(np.uint8) * 255, mode="L")
            refined_mask_pil.save(refined_mask_path)

        if len(extra_masks) > 0:
            extra_mask = np.logical_or.reduce(extra_masks)
        else:
            extra_mask = np.zeros_like(orig_mask_np)

        cleaned_mask = remove_small_black_holes(
            mask=extra_mask,
            max_hole_area_percentage=1.2,
        )

        mask_uint8 = cleaned_mask.astype(np.uint8) * 255

        mask_uint8_pil = Image.fromarray(mask_uint8, mode="L")

        return mask_uint8_pil

    def run_show(self, image_path, mask_path, pad=0, refined_mask_path=None, cur_resolutions=(1024, 1024)):

        image = Image.open(image_path).convert("RGB").resize(cur_resolutions)
        width, height = image.size
        state = self.processor.set_image(image)

        box_xywh = mask_path_to_bbox_xywh(mask_path, pad=pad, res=cur_resolutions)
        box_cxcywh = box_xywh_to_cxcywh(box_xywh)
        norm_box = normalize_bbox(box_cxcywh, width, height).flatten().tolist()

        self.processor.reset_all_prompts(state)
        state = self.processor.add_geometric_prompt(
            state=state,
            box=norm_box,
            label=True,
        )

        sam_masks = state["masks"]  # [N, 1, H, W] or [N, H, W]
        sam_masks = sam_masks.squeeze(1) if sam_masks.dim() == 4 else sam_masks

        if sam_masks.shape[0] == 0:
            return Image.new("L", cur_resolutions, 0), []

        orig_mask = Image.open(mask_path).convert("L").resize(cur_resolutions, Image.Resampling.NEAREST)
        orig_mask_np = np.array(orig_mask) > 0  # bool
        orig_mask_t = torch.from_numpy(orig_mask_np).to(sam_masks.device)

        def compute_iou(m1, m2):
            inter = (m1 & m2).sum().float()
            union = (m1 | m2).sum().float()
            return inter / (union + 1e-6)

        ious = []
        for i in range(sam_masks.shape[0]):
            sam_i = sam_masks[i] > 0
            iou = compute_iou(sam_i, orig_mask_t)
            ious.append(iou.item())

        ious = np.array(ious)
        best_idx = int(np.argmax(ious))

        extra_masks = [sam_masks[i].cpu().numpy().astype(bool) for i in range(sam_masks.shape[0]) if i != best_idx]

        extra_masks = [fill_mask_holes(m) for m in extra_masks]

        if refined_mask_path is not None:
            refined_mask_tensor = sam_masks[best_idx].detach().cpu()
            refined_mask_pil = Image.fromarray(refined_mask_tensor.numpy().astype(np.uint8) * 255, mode="L")
            refined_mask_pil.save(refined_mask_path)

        if len(extra_masks) > 0:
            extra_mask = np.logical_or.reduce(extra_masks)
        else:
            extra_mask = np.zeros_like(orig_mask_np)

        cleaned_mask = extra_mask

        mask_uint8 = cleaned_mask.astype(np.uint8) * 255

        mask_uint8_pil = Image.fromarray(mask_uint8, mode="L")

        return mask_uint8_pil, extra_masks
