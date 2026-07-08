# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from pathlib import Path
from typing import Literal

import torch

from vipe.utils.cameras import CameraType
from vipe.utils.misc import unpack_optional

from ..base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType
from .models.unidepthv2.unidepthv2 import Pinhole, UniDepthV2


def _load_unidepth_from_local_path(model_path: Path) -> UniDepthV2:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"UniDepth local model config not found: {config_path}")
    with config_path.open() as f:
        config = json.load(f)
    return UniDepthV2.from_pretrained(model_path, config=config)


class UniDepth2Model(DepthEstimationModel):
    def __init__(self, type: Literal["s", "b", "l"] = "l") -> None:
        super().__init__()
        local_model_path = os.environ.get("VIPE_UNIDEPTH_MODEL_PATH")
        if local_model_path:
            self.model = _load_unidepth_from_local_path(Path(local_model_path))
        else:
            self.model = UniDepthV2.from_pretrained(f"lpiccinelli/unidepth-v2-vit{type}14")
        self.model.interpolation_mode = "bilinear"
        self.model = self.model.cuda().eval()

    @property
    def depth_type(self) -> DepthType:
        return DepthType.MODEL_METRIC_DEPTH

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        rgb: torch.Tensor = unpack_optional(src.rgb)
        assert rgb.dtype == torch.float32, "Input image should be float32"

        assert src.camera_type == CameraType.PINHOLE, "UniDepth only supports pinhole cameras"
        focal_length: float = unpack_optional(src.intrinsics)[0].item()

        if rgb.dim() == 3:
            rgb, batch_dim = rgb[None], False
        else:
            batch_dim = True

        rgb = torch.clamp(rgb.moveaxis(-1, 1) * 255.0, max=255.0).byte()
        K = torch.tensor(
            [
                [focal_length, 0, rgb.shape[-1] / 2],
                [0, focal_length, rgb.shape[-2] / 2],
                [0, 0, 1],
            ],
            device=rgb.device,
        ).float()
        camera = Pinhole(K=K[None].repeat(rgb.shape[0], 1, 1))

        predictions = self.model.infer(rgb, camera)
        pred_depth = predictions["depth"].squeeze(1)
        confidence = predictions["confidence"].squeeze(1)

        if not batch_dim:
            pred_depth, confidence = pred_depth[0], confidence[0]

        return DepthEstimationResult(
            metric_depth=pred_depth,
            confidence=confidence,
        )
