# -*- encoding: utf-8 -*-
# Author  : HaiTong
# Time    : 2026/7/17 16:27
# File    : dors_utils.py
# Software: PyCharm

import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt


def _load_mask(mask):
    """Convert a PIL mask to a binary uint8 array."""
    mask = np.array(mask.convert("L"), dtype=np.float32)
    return (mask > 127).astype(np.uint8)


class MaskRegionBuilder:
    """Build the mask pyramid consumed by the DORS attention processor."""

    def __init__(
        self,
        mask,
        visualize=False,
    ):

        self.visualize = visualize
        self.mask = _load_mask(mask)
        self.MR = self.mask.astype(np.float32)
        self.mask_pyramid = {}
        self._build_mask_pyramid()

    def get_regions(self):
        return {"MR": self.MR}

    def _build_mask_pyramid(self):
        for name, mask in self.get_regions().items():
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
            self.mask_pyramid[name] = {
                str(size): F.interpolate(
                    mask_tensor,
                    size=(size, size),
                    mode="nearest",
                )
                .round()
                .squeeze()
                for size in (64, 32, 16, 8)
            }


def set_seed(seed: int):
    """Seed PyTorch, NumPy, and Python random generators."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)


def blended_func(mask_img_pil, src_image_pil, tgt_image_pil):
    """Blend the generated region into the source image with a soft boundary."""
    mask_np = (np.array(mask_img_pil.convert("L")) > 128).astype(np.uint8)[:, :, np.newaxis]
    target_np = np.array(tgt_image_pil)
    source_np = np.array(src_image_pil)

    mask_blurred = cv2.GaussianBlur(mask_np * 255, (21, 21), 0) / 255
    mask_blurred = mask_blurred[:, :, np.newaxis]
    blend_mask = 1 - (1 - mask_np) * (1 - mask_blurred)

    blended = source_np * (1 - blend_mask) + target_np * blend_mask
    return Image.fromarray(blended.astype(target_np.dtype))


def pil_to_bin_np(pil_img, thresh=128):
    """Convert a PIL mask to a binary NumPy array."""
    return (np.array(pil_img.convert("L")) > thresh).astype(np.uint8)


def contact_filter(ss_mask, mr_mask, r_contact=1):
    """Discard a semantic region that does not touch the removal mask."""
    kernel = np.ones((2 * r_contact + 1, 2 * r_contact + 1), np.uint8)
    ss_dilate = cv2.dilate(ss_mask, kernel)
    return ss_mask if (ss_dilate & mr_mask).sum() > 0 else None


class PiecewiseFunction:
    """Distance-decay function used to construct the inner DORS weight."""

    def __init__(self, d, trunc):
        self.d = d
        self.trunc = trunc

    def calculate(self, x_vals):
        x_vals = np.asarray(x_vals)
        y_vals = np.zeros_like(x_vals, dtype=float)

        mask_near = x_vals <= self.d / 3
        mask_transition = (x_vals > self.d / 3) & (x_vals < 2 * self.d / 3)
        y_max = 1.0 - self.trunc

        y_vals[mask_near] = y_max
        y_vals[mask_transition] = y_max - ((x_vals[mask_transition] - self.d / 3) * (y_max / (self.d / 3)))
        return y_vals


def dilate_by_area(mask, target_ratio, max_iter=100):
    mask_np = np.array(mask)
    initial_area = np.sum(mask_np > 128)
    target_area = int(initial_area * target_ratio)

    if target_area <= initial_area:
        return Image.fromarray(mask_np)

    current_mask = mask_np.copy()
    current_area = initial_area
    iteration = 0
    while iteration < max_iter and current_area != target_area:
        area_diff = abs(current_area - target_area)
        kernel_size = max(3, int(np.sqrt(area_diff) / 10))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        current_mask = cv2.dilate(current_mask, kernel, iterations=1)
        current_area = np.sum(current_mask > 128)
        if current_area >= target_area:
            break
        iteration += 1

    return Image.fromarray(current_mask)


def build_weight_inner_v2(
    mr_mask_pil,
    ss_mask_pil,
    r_contact=1,
    sigma=None,
    w_max=None,
    alpha_=0.5,
):
    """Build the spatial inner-attention weight used by DORS."""
    mr = pil_to_bin_np(mr_mask_pil)
    ss = pil_to_bin_np(ss_mask_pil)
    height, width = mr.shape

    num_regions, labels = cv2.connectedComponents(ss)
    combined_weight = np.zeros((height, width), np.float32)

    if sigma is None:
        boundary_distance = distance_transform_edt(mr)
        sigma = np.quantile(boundary_distance[mr > 0], 0.90)

    if w_max is None:
        unmask = 1 - mr
        scene_density = np.sum(ss) / (np.sum(unmask) + 1e-8)
        w_min = max(alpha_ * scene_density, 0.1)
        w_max = 1 - w_min
    else:
        w_min = 1 - w_max

    piecewise_func = PiecewiseFunction(d=sigma * 2.0, trunc=w_min)

    intersection_count = 0
    for region_index in range(1, num_regions):
        semantic_region = (labels == region_index).astype(np.uint8)
        semantic_region = contact_filter(semantic_region, mr, r_contact)
        if semantic_region is None:
            continue

        intersection_count += 1
        semantic_distance = distance_transform_edt(1 - semantic_region)
        region_weight = piecewise_func.calculate(semantic_distance) * mr

        combined_weight = np.maximum(combined_weight, region_weight)

    outer_weight = np.clip(combined_weight, 0, 1)
    inner_weight = np.clip((1 - outer_weight) * mr, 0, 1)
    return inner_weight, intersection_count
