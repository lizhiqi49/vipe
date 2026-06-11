# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
import torch
import torch.nn.functional as F

try:
    from vggt_omega.models import VGGTOmega
except ModuleNotFoundError:
    VGGTOmega = None

from vipe.utils.misc import unpack_optional

from ..base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType
from ..windowed import WindowedDepthAccumulator, window_starts


_VGGT_OMEGA_REPO_ID = "facebook/VGGT-Omega"
_VGGT_OMEGA_CHECKPOINT = "vggt_omega_1b_512.pt"


class VGGTOmegaDepthModel(DepthEstimationModel):
    """
    VGGT-Omega video-depth wrapper.

    The wrapper mirrors the Pi3X integration boundary: consume the full
    `video_frame_list`, run overlapping windows to keep memory bounded, and
    return relative inverse depth for the existing SANA-WM fusion processor.
    """

    def __init__(self, model_sub: str = "") -> None:
        super().__init__()
        del model_sub
        if VGGTOmega is None:
            raise RuntimeError(
                "VGGT-Omega is not found in the environment. Install the vggt extras before using it."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("VGGTOmegaDepthModel requires CUDA")

        checkpoint_path = _resolve_vggt_omega_checkpoint()
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.model = VGGTOmega().cuda().eval()
        self.model.load_state_dict(state_dict)

        # Resolution is locked to the official load_fn contract: balanced/max_size over a
        # patch-aligned image_resolution. There is no pixel_limit knob — the official
        # preprocessing has no such concept, and resizing outside this contract would feed
        # the model an unsupported resolution (issue #23). Memory is bounded only by the
        # temporal window below, never by shrinking resolution.
        self.patch_size = 16
        self.image_resolution = max(int(os.environ.get("VIPE_VGGT_OMEGA_IMAGE_RESOLUTION", "512")), self.patch_size)
        if self.image_resolution % self.patch_size != 0:
            raise ValueError(
                f"VIPE_VGGT_OMEGA_IMAGE_RESOLUTION must be a multiple of patch size {self.patch_size}, "
                f"got {self.image_resolution}"
            )
        self.mode = os.environ.get("VIPE_VGGT_OMEGA_MODE", "balanced")
        if self.mode not in ("balanced", "max_size"):
            raise ValueError(f"VIPE_VGGT_OMEGA_MODE must be 'balanced' or 'max_size', got {self.mode!r}")
        self.window_size = max(int(os.environ.get("VIPE_VGGT_OMEGA_WINDOW_SIZE", "64")), 1)
        self.window_overlap = max(int(os.environ.get("VIPE_VGGT_OMEGA_WINDOW_OVERLAP", "8")), 0)
        self.autocast_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    @property
    def depth_type(self) -> DepthType:
        return DepthType.AFFINE_DISP

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        frame_list = unpack_optional(src.video_frame_list)
        if not frame_list:
            raise ValueError("VGGTOmegaDepthModel requires a non-empty video_frame_list")

        total_frames = len(frame_list)
        accumulator = WindowedDepthAccumulator(total_frames)

        for start in window_starts(total_frames, self.window_size, self.window_overlap):
            end = min(start + self.window_size, total_frames)
            window_frames = frame_list[start:end]
            inputs, restore_spec = self._frames_to_tensor(window_frames)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                outputs = self.model(inputs)

            depth = outputs["depth"]
            conf = outputs.get("depth_conf")

            if depth.dim() == 5 and depth.shape[0] == 1:
                depth = depth.squeeze(0)
            if conf is not None and conf.dim() == 4 and conf.shape[0] == 1:
                conf = conf.squeeze(0)

            video_depth = torch.clamp(depth.squeeze(-1), min=1e-4)
            relative_inv_depth = video_depth.reciprocal()
            confidence = torch.ones_like(relative_inv_depth) if conf is None else conf

            relative_inv_depth = self._restore_map_size(relative_inv_depth, restore_spec)
            confidence = self._restore_map_size(confidence, restore_spec)
            accumulator.add(start, end, relative_inv_depth, confidence)

        relative_inv_depth, confidence = accumulator.result()
        return DepthEstimationResult(relative_inv_depth=relative_inv_depth, confidence=confidence)

    def _frames_to_tensor(self, frame_list: list[np.ndarray]) -> tuple[torch.Tensor, dict[str, tuple[int, int] | tuple[int, int, int, int]]]:
        restore_spec: dict[str, tuple[int, int] | tuple[int, int, int, int]] | None = None
        tensors: list[torch.Tensor] = []

        for frame in frame_list:
            image = _frame_to_pil(frame)
            original_size = (image.height, image.width)
            cropped_image, crop_box = _crop_to_supported_aspect_ratio(image)
            aspect_ratio = cropped_image.height / max(cropped_image.width, 1)
            if self.mode == "balanced":
                target_size = _balanced_target_shape(aspect_ratio, self.image_resolution, self.patch_size)
            else:
                target_size = _max_size_target_shape(aspect_ratio, self.image_resolution, self.patch_size)
            if target_size[0] % self.patch_size != 0 or target_size[1] % self.patch_size != 0:
                raise ValueError(f"VGGT-Omega target size {target_size} is not patch-aligned ({self.patch_size})")
            resized_image = cropped_image.resize((target_size[1], target_size[0]), Image.Resampling.BICUBIC)
            tensor = torch.from_numpy(np.array(resized_image, copy=True)).float() / 255.0
            tensors.append(tensor.permute(2, 0, 1))

            frame_restore_spec = {
                "original_size": original_size,
                "crop_box": crop_box,
            }
            if restore_spec is None:
                restore_spec = frame_restore_spec
            elif restore_spec != frame_restore_spec:
                raise ValueError("VGGTOmegaDepthModel expects all frames in a sequence to share size/aspect ratio")

        assert restore_spec is not None
        return torch.stack(tensors, dim=0).unsqueeze(0).cuda(), restore_spec

    def _restore_map_size(
        self,
        tensor: torch.Tensor,
        restore_spec: dict[str, tuple[int, int] | tuple[int, int, int, int]],
    ) -> torch.Tensor:
        original_size = tuple(restore_spec["original_size"])
        crop_box = tuple(restore_spec["crop_box"])
        if len(original_size) != 2 or len(crop_box) != 4:
            raise ValueError("Invalid VGGT-Omega restore specification")

        original_height, original_width = int(original_size[0]), int(original_size[1])
        left, top, right, bottom = (int(value) for value in crop_box)
        crop_height = bottom - top
        crop_width = right - left

        restored = tensor
        if tuple(restored.shape[-2:]) != (crop_height, crop_width):
            restored = F.interpolate(
                restored.unsqueeze(1),
                size=(crop_height, crop_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        if crop_height == original_height and crop_width == original_width:
            return restored

        return F.pad(
            restored.unsqueeze(1),
            (left, original_width - right, top, original_height - bottom),
            mode="replicate",
        ).squeeze(1)


def _resolve_vggt_omega_checkpoint() -> str:
    explicit_path = os.environ.get("VIPE_VGGT_OMEGA_MODEL_PATH")
    if explicit_path:
        return explicit_path

    root = os.environ.get("VGGT_OMEGA_LOCAL_MODEL_ROOT")
    if root:
        candidate = Path(root)
        if candidate.is_dir():
            checkpoint = candidate / _VGGT_OMEGA_CHECKPOINT
            if checkpoint.exists():
                return str(checkpoint)
        elif candidate.exists():
            return str(candidate)

    return hf_hub_download(repo_id=_VGGT_OMEGA_REPO_ID, filename=_VGGT_OMEGA_CHECKPOINT)


def _frame_to_pil(frame: np.ndarray) -> Image.Image:
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"Expected frame shape (H, W, 3), got {frame.shape}")
    if np.issubdtype(frame.dtype, np.floating):
        frame = np.clip(frame, 0.0, 1.0)
        frame = np.rint(frame * 255.0).astype(np.uint8)
    else:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return Image.fromarray(frame, mode="RGB")


def _crop_to_supported_aspect_ratio(
    image: Image.Image,
    min_aspect_ratio: float = 0.5,
    max_aspect_ratio: float = 2.0,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    width, height = image.size
    aspect_ratio = height / max(width, 1)

    if aspect_ratio < min_aspect_ratio:
        crop_width = min(width, max(1, int(round(height / min_aspect_ratio))))
        left = max((width - crop_width) // 2, 0)
        return image.crop((left, 0, left + crop_width, height)), (left, 0, left + crop_width, height)

    if aspect_ratio > max_aspect_ratio:
        crop_height = min(height, max(1, int(round(width * max_aspect_ratio))))
        top = max((height - crop_height) // 2, 0)
        return image.crop((0, top, width, top + crop_height)), (0, top, width, top + crop_height)

    return image, (0, 0, width, height)


def _balanced_target_shape(aspect_ratio: float, image_resolution: int, patch_size: int) -> tuple[int, int]:
    token_number = (image_resolution // patch_size) ** 2
    width_patches = np.sqrt(token_number / aspect_ratio)
    height_patches = token_number / width_patches
    width_patches = max(1, int(np.round(width_patches)))
    height_patches = max(1, int(np.round(height_patches)))
    return height_patches * patch_size, width_patches * patch_size


def _max_size_target_shape(aspect_ratio: float, image_resolution: int, patch_size: int) -> tuple[int, int]:
    # Official load_fn `max_size` mode: longest side becomes image_resolution, shorter side
    # is scaled to keep aspect ratio and rounded to a patch multiple.
    if aspect_ratio >= 1.0:
        height = image_resolution
        width = _round_to_patch_multiple(image_resolution / aspect_ratio, patch_size)
    else:
        width = image_resolution
        height = _round_to_patch_multiple(image_resolution * aspect_ratio, patch_size)
    return height, width


def _round_to_patch_multiple(value: float, patch_size: int) -> int:
    return max(patch_size, int(np.round(float(value) / patch_size)) * patch_size)
