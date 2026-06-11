# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F

try:
    from pi3.models.pi3x import Pi3X
except ModuleNotFoundError:
    Pi3X = None

from vipe.utils.misc import unpack_optional

from ..base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType
from ..windowed import WindowedDepthAccumulator, window_starts


# Official Pi3 preprocessing pixel budget (pi3.utils.basic.load_images_as_tensor). Resolution
# is locked to this contract: feeding the model anything other than the official patch-aligned
# resolution is unsupported (issue #23). Memory is bounded only by the temporal window.
_PI3X_PIXEL_LIMIT = 255000


class Pi3XDepthModel(DepthEstimationModel):
    """
    Pi3X video-depth wrapper.

    This wrapper consumes the full `video_frame_list`, runs Pi3X in overlapping windows to
    keep memory bounded, and converts per-view local point-map z values into relative inverse depth.
    """

    def __init__(self, model_sub: str = "") -> None:
        super().__init__()
        del model_sub
        if Pi3X is None:
            raise RuntimeError("Pi3X is not found in the environment. Install the pi3x extras before using it.")
        if not torch.cuda.is_available():
            raise RuntimeError("Pi3XDepthModel requires CUDA")
        pretrained = _resolve_pi3x_pretrained()
        self.model = Pi3X.from_pretrained(pretrained, local_files_only=Path(pretrained).exists())
        self.model = self.model.cuda().eval()
        self.patch_size = _resolve_pi3x_patch_size(self.model)
        self.pixel_limit = max(_PI3X_PIXEL_LIMIT, self.patch_size**2)
        self.window_size = max(int(os.environ.get("VIPE_PI3X_WINDOW_SIZE", "64")), 1)
        self.window_overlap = max(int(os.environ.get("VIPE_PI3X_WINDOW_OVERLAP", "8")), 0)
        self.autocast_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    @property
    def depth_type(self) -> DepthType:
        return DepthType.AFFINE_DISP

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        frame_list = unpack_optional(src.video_frame_list)
        if not frame_list:
            raise ValueError("Pi3XDepthModel requires a non-empty video_frame_list")

        total_frames = len(frame_list)
        accumulator = WindowedDepthAccumulator(total_frames)

        for start in window_starts(total_frames, self.window_size, self.window_overlap):
            end = min(start + self.window_size, total_frames)
            window_frames = frame_list[start:end]
            inputs, original_size = self._frames_to_tensor(window_frames)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                outputs = self.model(inputs)

            local_points = outputs["local_points"]
            conf = outputs.get("conf")

            if local_points.dim() == 5 and local_points.shape[0] == 1:
                local_points = local_points.squeeze(0)
            if conf is not None and conf.dim() >= 4 and conf.shape[0] == 1:
                conf = conf.squeeze(0)

            local_depth = torch.clamp(local_points[..., 2], min=1e-4)
            relative_inv_depth = local_depth.reciprocal()
            if conf is None:
                confidence = torch.ones_like(relative_inv_depth)
            else:
                confidence = torch.sigmoid(conf.squeeze(-1) if conf.dim() == 4 else conf)
            relative_inv_depth = self._restore_map_size(relative_inv_depth, original_size)
            confidence = self._restore_map_size(confidence, original_size)
            accumulator.add(start, end, relative_inv_depth, confidence)

        relative_inv_depth, confidence = accumulator.result()
        return DepthEstimationResult(relative_inv_depth=relative_inv_depth, confidence=confidence)

    def _frames_to_tensor(self, frame_list: list[np.ndarray]) -> tuple[torch.Tensor, tuple[int, int]]:
        frames = torch.from_numpy(np.stack(frame_list, axis=0)).float().cuda()
        frames = frames.moveaxis(-1, 1)
        if frames.max() > 1.0:
            frames = frames / 255.0
        original_size = (int(frames.shape[-2]), int(frames.shape[-1]))
        target_size = self._aligned_image_size(*original_size)
        if target_size != original_size:
            frames = F.interpolate(frames, size=target_size, mode="bilinear", align_corners=False)
        return frames.unsqueeze(0), original_size

    def _aligned_image_size(self, height: int, width: int) -> tuple[int, int]:
        patch = max(int(self.patch_size), 1)
        if height <= 0 or width <= 0:
            return patch, patch

        scale = math.sqrt(self.pixel_limit / float(height * width)) if height * width > self.pixel_limit else 1.0
        target_width = width * scale
        target_height = height * scale
        width_steps = max(1, round(target_width / patch))
        height_steps = max(1, round(target_height / patch))

        while (width_steps * patch) * (height_steps * patch) > self.pixel_limit:
            if width_steps / max(height_steps, 1) > target_width / max(target_height, 1e-6):
                width_steps -= 1
            else:
                height_steps -= 1

        return max(height_steps, 1) * patch, max(width_steps, 1) * patch

    def _restore_map_size(self, tensor: torch.Tensor, original_size: tuple[int, int]) -> torch.Tensor:
        if tuple(tensor.shape[-2:]) == original_size:
            return tensor
        return F.interpolate(
            tensor.unsqueeze(1),
            size=original_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)


def _resolve_pi3x_pretrained() -> str:
    explicit_path = os.environ.get("VIPE_PI3X_MODEL_PATH")
    if explicit_path:
        return explicit_path

    root = os.environ.get("PI3X_LOCAL_MODEL_ROOT")
    if root and Path(root).exists():
        return root

    return "yyfz233/Pi3X"


def _resolve_pi3x_patch_size(model: torch.nn.Module) -> int:
    patch_embed = getattr(getattr(model, "encoder", None), "patch_embed", None)
    patch_size = getattr(patch_embed, "patch_size", 14)
    if isinstance(patch_size, tuple):
        return int(patch_size[0])
    return int(patch_size)
