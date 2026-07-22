# -*- encoding: utf-8 -*-
# Author  : HaiTong
# Time    : 2026/7/17 15:58
# File    : dar_processor.py
# Software: PyCharm


from einops import rearrange
from scipy.ndimage import distance_transform_edt
from collections import defaultdict
from typing import List, Optional
import torch
import torch.nn.functional as F
import numpy as np

from .utils import _load_mask


class AttentionBase:
    def __init__(self):
        self.num_att_layers = -1
        self.cur_step = 0
        self.cur_att_layer = 0

    def __call__(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, ip_dict, **kwargs):
        out = self.forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, ip_dict, **kwargs)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1

        return out

    def normal_forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = torch.einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=num_heads)
        return out


def register_attention_editor_diffusers(unet, editor: AttentionBase):
    """
    Register a attention editor to Diffuser Pipeline, refer from [Prompt-to-Prompt]
    """

    def ca_forward(self, place_in_unet):
        def forward(
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            ip_adapter_masks: Optional[torch.Tensor] = None,
        ):
            """
            The attention is similar to the original implementation of LDM CrossAttention class
            except adding some modifications on the attention
            """
            is_cross = encoder_hidden_states is not None

            if is_cross:
                if isinstance(encoder_hidden_states, tuple):
                    encoder_hidden_states, ip_hidden_states = encoder_hidden_states
                else:
                    encoder_hidden_states, ip_hidden_states = encoder_hidden_states, None
            else:
                encoder_hidden_states, ip_hidden_states = hidden_states, None

            query_ = self.to_q(hidden_states)  # [2 4096 640]
            key_ = self.to_k(encoder_hidden_states)  # [2 77 640]
            value_ = self.to_v(encoder_hidden_states)  # [2 77 640]

            query, key, value = (
                rearrange(t, "b n (h d) -> (b h) n d", h=self.heads) for t in (query_, key_, value_)
            )  # [20 4096 64]   [20 77 64] [20 77 64]
            sim = torch.einsum("b i d, b j d -> b i j", query, key) * self.scale
            attn = sim.softmax(dim=-1)

            if ip_hidden_states is not None:
                if ip_adapter_masks is not None:
                    if not isinstance(ip_adapter_masks, List):
                        ip_adapter_masks = list(ip_adapter_masks.unsqueeze(1))
                else:
                    ip_adapter_masks = [None] * len(self.processor.scale)

                ip_dict = {
                    "batch_size": encoder_hidden_states.shape[0],
                    "query": query_,
                    "ip_hidden_states": ip_hidden_states,
                    "ip_scale": self.processor.scale,
                    "to_k_ip": self.processor.to_k_ip,
                    "to_v_ip": self.processor.to_v_ip,
                    "ip_adapter_masks": ip_adapter_masks,
                }

            else:
                ip_dict = None

            hidden_states = editor(
                q=query,
                k=key,
                v=value,
                sim=sim,
                attn=attn,
                is_cross=is_cross,
                place_in_unet=place_in_unet,
                num_heads=self.heads,
                ip_dict=ip_dict,
                scale=self.scale,
            )

            hidden_states = self.to_out[0](hidden_states)  # MLP 640 640
            hidden_states = self.to_out[1](hidden_states)  # dropout
            hidden_states = hidden_states / self.rescale_output_factor

            return hidden_states

        return forward

    def register_editor(net, count, place_in_unet):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == "Attention":  # spatial Transformer layer
                net.forward = ca_forward(net, place_in_unet)
                return count + 1
            elif hasattr(net, "children"):
                count = register_editor(subnet, count, place_in_unet)
        return count

    cross_att_count = 0
    for net_name, net in unet.named_children():
        if "down" in net_name:
            cross_att_count += register_editor(net, 0, "down")
        elif "mid" in net_name:
            cross_att_count += register_editor(net, 0, "mid")
        elif "up" in net_name:
            cross_att_count += register_editor(net, 0, "up")
    editor.num_att_layers = cross_att_count  # 140


class DynamicAttentionRouting(AttentionBase):

    @staticmethod
    def get_empty_step_store(save_timesteps=None):
        d = defaultdict(list)
        for t in save_timesteps:
            d[t] = {}
        return d

    @staticmethod
    def get_empty_store():
        return {}

    def __init__(
        self,
        start_step=0,
        end_step=50,
        start_layer=34,
        end_layer=70,
        total_steps=50,
        attn_store=None,
    ):

        super().__init__()

        self.start_layer = start_layer
        self.end_layer = end_layer
        self.total_layers = 70

        self.layer_idx = list(range(start_layer, end_layer))
        self.step_idx = list(range(start_step, end_step))
        self.total_steps = total_steps
        self.attn_store = attn_store
        self.sam3_ss_mask_dict = None
        self.dilate_mask_dict = None
        self.inner_weight_dict = None
        self.mask_builder = None
        self.distance_pyramid_mask = None
        self.distance_pyramid_unmask_inversion = None
        self.attn_save_dir = None

    def reset_context(self):
        self.cur_step = 0
        self.cur_att_layer = 0
        self.sam3_ss_mask_dict = None
        self.dilate_mask_dict = None
        self.inner_weight_dict = None
        self.mask_builder = None
        self.distance_pyramid_mask = None
        self.distance_pyramid_unmask_inversion = None
        self.attn_save_dir = None

    def set_context(
        self,
        mask_pil,
        mask_builder=None,
        attn_save_dir=None,
        visualize=False,
        ss_mask_tensor=None,
        dilate_mask_tensor=None,
        inner_weight_tensor=None,
        cur_resize=1024,
    ):

        if ss_mask_tensor is not None:
            self.sam3_ss_mask_dict = {
                16: F.max_pool2d(ss_mask_tensor, (cur_resize // 16, cur_resize // 16)).round().squeeze().squeeze(),
                32: F.max_pool2d(ss_mask_tensor, (cur_resize // 32, cur_resize // 32)).round().squeeze().squeeze(),
                64: F.max_pool2d(ss_mask_tensor, (cur_resize // 64, cur_resize // 64)).round().squeeze().squeeze(),
                128: F.max_pool2d(ss_mask_tensor, (cur_resize // 128, cur_resize // 128)).round().squeeze().squeeze(),
            }
        if dilate_mask_tensor is not None:
            self.dilate_mask_dict = {
                16: F.max_pool2d(dilate_mask_tensor, (cur_resize // 16, cur_resize // 16)).round().squeeze().squeeze(),
                32: F.max_pool2d(dilate_mask_tensor, (cur_resize // 32, cur_resize // 32)).round().squeeze().squeeze(),
                64: F.max_pool2d(dilate_mask_tensor, (cur_resize // 64, cur_resize // 64)).round().squeeze().squeeze(),
                128: F.max_pool2d(dilate_mask_tensor, (cur_resize // 128, cur_resize // 128))
                .round()
                .squeeze()
                .squeeze(),
            }

        if inner_weight_tensor is not None:  # soft guidance
            self.inner_weight_dict = {
                16: F.interpolate(inner_weight_tensor, size=(16, 16), mode="bilinear", align_corners=False).squeeze(),
                32: F.interpolate(inner_weight_tensor, size=(32, 32), mode="bilinear", align_corners=False).squeeze(),
                64: F.interpolate(inner_weight_tensor, size=(64, 64), mode="bilinear", align_corners=False).squeeze(),
                128: F.interpolate(
                    inner_weight_tensor, size=(128, 128), mode="bilinear", align_corners=False
                ).squeeze(),
            }

        self.mask_builder = mask_builder

        self.distance_pyramid_mask = self._build_D_pyramid_mask(
            mask=_load_mask(mask_pil), vis=visualize, inversion=False, plateau=True
        )
        self.distance_pyramid_unmask_inversion = self._build_D_pyramid_unmask(
            mask=_load_mask(mask_pil),
            vis=visualize,
            inversion=True,
            soft_exponentiation=True,
        )

    def _build_D_pyramid_unmask(self, mask=None, vis=False, inversion=False, soft_exponentiation=False):

        mask_np = mask
        dist = distance_transform_edt(mask_np == 0)

        dist_norm = dist.copy()
        bg = mask_np == 0
        min_val = dist[bg].min()
        max_val = dist[bg].max()
        dist_norm[bg] = (dist[bg] - min_val) / (max_val - min_val + 1e-6)
        dist_norm[mask_np == 1] = 0

        import matplotlib.pyplot as plt

        if inversion:
            D_w = 1 - dist_norm
        else:
            D_w = dist_norm

        if soft_exponentiation:
            gamma = 0.5
            D_w = D_w**gamma

        D_w[mask_np == 1] = 0

        if vis:
            plt.figure(figsize=(6, 6))
            plt.imshow(D_w, cmap="jet")
            plt.colorbar()
            plt.title("Weight Heatmap")
            plt.axis("off")
            plt.show()

        D = torch.from_numpy(D_w).float().unsqueeze(0).unsqueeze(0)  # [1,1,512,512]

        # Downsample to the attention resolutions.
        D_64 = F.interpolate(D, size=(64, 64), mode="bilinear", align_corners=False).squeeze()
        D_32 = F.interpolate(D, size=(32, 32), mode="bilinear", align_corners=False).squeeze()
        D_16 = F.interpolate(D, size=(16, 16), mode="bilinear", align_corners=False).squeeze()
        D_8 = F.interpolate(D, size=(8, 8), mode="bilinear", align_corners=False).squeeze()

        return {"D_8": D_8, "D_16": D_16, "D_32": D_32, "D_64": D_64}

    def _build_D_pyramid_mask(self, mask=None, vis=False, inversion=False, plateau=False):

        mask_np = mask
        dist = distance_transform_edt(mask_np == 1)

        dist_norm = dist.copy()
        fg = mask_np == 1
        min_val = dist[fg].min()
        max_val = dist[fg].max()
        dist_norm[fg] = (dist[fg] - min_val) / (max_val - min_val + 1e-6)
        dist_norm[mask_np == 0] = 0
        import matplotlib.pyplot as plt

        if vis:
            plt.figure(figsize=(6, 6))
            plt.imshow(dist_norm, cmap="jet")
            plt.colorbar()
            plt.title("Distance Heatmap")
            plt.axis("off")
            plt.show()

        if plateau:
            inner_w = dist_norm.copy()
            low = 0.3
            high = 0.8
            # Outer-only region.
            inner_w[inner_w <= low] = 0.3
            # Inner-only region.
            inner_w[inner_w >= high] = 1.0
            # Apply linear interpolation in the transition region.
            mid = (inner_w > low) & (inner_w < high)
            inner_w[mid] = (inner_w[mid] - low) / (high - low) * (1.0 - low) + low  # 0-1
            dist_norm = inner_w * mask_np

            import matplotlib.pyplot as plt

            if vis:
                plt.figure(figsize=(6, 6))
                plt.imshow(dist_norm, cmap="jet")
                plt.colorbar()
                plt.title("Distance Heatmap")
                plt.axis("off")
                plt.show()

        if inversion:
            D_w = 1 - dist_norm
        else:
            D_w = dist_norm

        D_w[mask_np == 0] = 0

        if vis:
            plt.figure(figsize=(6, 6))
            plt.imshow(D_w, cmap="jet")
            plt.colorbar()
            plt.title("Weight Heatmap")
            plt.axis("off")
            plt.show()

        D = torch.from_numpy(D_w).float().unsqueeze(0).unsqueeze(0)  # [1,1,512,512]

        # Downsample to the attention resolutions.
        D_64 = F.interpolate(D, size=(64, 64), mode="bilinear", align_corners=False).squeeze()
        D_32 = F.interpolate(D, size=(32, 32), mode="bilinear", align_corners=False).squeeze()
        D_16 = F.interpolate(D, size=(16, 16), mode="bilinear", align_corners=False).squeeze()
        D_8 = F.interpolate(D, size=(8, 8), mode="bilinear", align_corners=False).squeeze()

        return {"D_8": D_8, "D_16": D_16, "D_32": D_32, "D_64": D_64}

    def attn_batch_normal(self, q, v, sim, num_heads, **kwargs):
        B = q.shape[0] // num_heads
        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
        return out

    def attn_batch_dors(
        self,
        q,
        k,
        v,
        sim,
        attn,
        is_cross,
        num_heads,
        mask_bool,
        dis_bool,
        ostu_bool,
        mask=None,
        dis_weight=None,
        dis_alpha=1,
        **kwargs,
    ):

        B = q.shape[0] // num_heads
        mask_flatten = mask.flatten(0).to(sim.device).to(sim.dtype)
        mask_flatten_bool = mask_flatten.bool()
        if dis_bool and dis_weight is not None:
            dis_weight_flatten = dis_weight.flatten(0).to(sim.device).to(sim.dtype)
            sim_mean = sim.mean(dim=0)
            max_sim, _ = sim_mean.max(dim=-1)
            max_sim_q = max_sim * mask_flatten_bool * dis_alpha
            guidance = max_sim_q[:, None] * dis_weight_flatten[None, :]  # [1024,1024]
            sim = sim + guidance

        if ostu_bool:

            def to_bool_flat(mask):
                return (mask.view(1, -1) > 0.5).bool().squeeze()

            cur_ostu_tensor = self.sam3_ss_mask_dict[int(np.sqrt(sim.shape[-1]))]

            ostu_mask_bool = to_bool_flat(cur_ostu_tensor)
            ostu_flatten = ostu_mask_bool.float().to(sim.device).to(sim.dtype)

            mask_penalty = torch.finfo(sim.dtype).min
            bias_ostu = ostu_flatten.masked_fill(ostu_flatten == 1, mask_penalty)
            zeros_like_bias_ostu = torch.zeros_like(bias_ostu)

            bias_ostu = torch.stack(
                [bias_ostu if is_mask else zeros_like_bias_ostu for is_mask in mask_flatten_bool], dim=0
            )  # shape: [Q, K]
            bias_matrix_ostu = bias_ostu.unsqueeze(0).expand(sim.shape[0], -1, -1)  # [B*H, Q, K]

            sim = sim + bias_matrix_ostu

        if mask_bool and mask is not None:
            unmask_penalty = torch.finfo(sim.dtype).min

            bias_row_mask = mask_flatten.masked_fill(mask_flatten == 1, unmask_penalty)
            bias_row_unmask = torch.zeros_like(bias_row_mask)

            bias_rows_mask = torch.stack(
                [bias_row_mask if is_mask else bias_row_unmask for is_mask in mask_flatten_bool], dim=0
            )  # shape: [Q, K]

            bias_matrix_mask = bias_rows_mask.unsqueeze(0).expand(sim.shape[0], -1, -1)  # [B*H, Q, K]

            sim = sim + bias_matrix_mask

        attn = sim.softmax(-1)

        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)

        return out, attn

    def forward_dors(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):

        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().normal_forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        B = q.shape[0] // num_heads
        H = int(np.sqrt(q.shape[1]))

        mr_mask = self.mask_builder.mask_pyramid["MR"][f"{H}"]
        mr_mask_flat = mr_mask.reshape(-1).float().to(sim.device).to(sim.dtype)

        inner_weight = self.inner_weight_dict[H]

        inner_weight_flat = inner_weight.reshape(-1).float().to(sim.device).to(sim.dtype)

        out_list = []
        if B == 2:
            q_w0, q_w1 = q.chunk(B)
            _, k_w1 = k.chunk(B)
            v_w0, v_w1 = v.chunk(B)
            sim_w0, sim_w1 = sim.chunk(B)
            _, attn_w1 = attn.chunk(B)
            batch0_out_normal = self.attn_batch_normal(q_w0, v_w0, sim_w0, num_heads)
            out_list.append(batch0_out_normal)
        else:
            q_w1 = q
            k_w1 = k
            v_w1 = v
            sim_w1 = sim
            attn_w1 = attn

        if -1 < self.cur_step < 10:

            batch1_inner, _ = self.attn_batch_dors(
                q_w1,
                k_w1,
                v_w1,
                sim_w1,
                attn_w1,
                is_cross,
                num_heads,
                dis_bool=False,
                ostu_bool=True,
                mask_bool=True,
                mask=mr_mask,
                dis_mask=None,
                **kwargs,
            )
            batch1_outer, _ = self.attn_batch_dors(
                q_w1,
                k_w1,
                v_w1,
                sim_w1,
                attn_w1,
                is_cross,
                num_heads,
                dis_bool=False,
                ostu_bool=False,
                mask_bool=True,
                mask=mr_mask,
                dis_weight=None,
                **kwargs,
            )

            inner_w = inner_weight_flat * mr_mask_flat
            inner_w = inner_w.view(1, -1, 1)  # [1, 1024, 1]

            batch1_out = (1.0 - inner_w) * batch1_outer + inner_w * batch1_inner

        else:
            batch1_out, attn_w1 = self.attn_batch_dors(
                q_w1,
                k_w1,
                v_w1,
                sim_w1,
                attn_w1,
                is_cross,
                num_heads,
                dis_bool=False,
                ostu_bool=False,
                mask_bool=False,
                mask=mr_mask,
                dis_mask=False,
                **kwargs,
            )

        out_list.append(batch1_out)

        out = torch.cat(out_list, dim=0)

        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, ip_dict, **kwargs):
        # Cross Attention
        if is_cross:
            out = super().normal_forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        # Self attention
        else:

            out = self.forward_dors(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)  #

        return out
