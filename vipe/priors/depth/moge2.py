# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import torch

try:
    from moge.model.v2 import MoGeModel
except ModuleNotFoundError:
    MoGeModel = None

from vipe.utils.cameras import CameraType
from vipe.utils.misc import unpack_optional

from .base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType
from .moge import focal_length_to_fov_degrees


class Moge2Model(DepthEstimationModel):
    """MoGe-2 wrapper used by the SANA-WM custom-depth fork."""

    def __init__(self) -> None:
        super().__init__()
        if MoGeModel is None:
            raise RuntimeError(
                "moge v2 is not found in the environment. Install the fork extras before using moge2."
            )
        self.model = MoGeModel.from_pretrained(_resolve_moge2_pretrained())
        self.model = self.model.cuda().eval()

    @property
    def depth_type(self) -> DepthType:
        return DepthType.MODEL_METRIC_DEPTH

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        rgb: torch.Tensor = unpack_optional(src.rgb)
        assert rgb.dtype == torch.float32, "Input image should be float32"
        assert src.camera_type == CameraType.PINHOLE, "MoGe-2 only supports pinhole cameras"

        focal_length: float = unpack_optional(src.intrinsics)[0].item()
        if rgb.dim() == 3:
            rgb, batch_dim = rgb[None], False
        else:
            batch_dim = True

        width = rgb.shape[2]
        input_image = rgb.moveaxis(-1, 1)
        moge_input_dict = {"fov_x": focal_length_to_fov_degrees(focal_length, width)}

        with torch.no_grad():
            output = self.model.infer(input_image, **moge_input_dict)

        metric_depth = torch.nan_to_num(output["depth"], nan=1e4)
        metric_depth = torch.clamp(metric_depth, min=0, max=1e4)
        mask = output["mask"].float()
        metric_depth = metric_depth * mask
        confidence = mask

        if not batch_dim:
            metric_depth = metric_depth.squeeze(0)
            confidence = confidence.squeeze(0)

        return DepthEstimationResult(metric_depth=metric_depth, confidence=confidence)


def _resolve_moge2_pretrained() -> str:
    explicit_path = os.environ.get("VIPE_MOGE2_MODEL_PATH")
    if explicit_path:
        return explicit_path

    root = os.environ.get("MOGE2_LOCAL_MODEL_ROOT")
    if root:
        candidate = Path(root)
        if candidate.is_dir():
            vitl = candidate / "MoGe_2_vit_l_normal.pt"
            if vitl.exists():
                return str(vitl)
        if candidate.exists():
            return str(candidate)

    return "Ruicheng/moge-2-vitl-normal"
