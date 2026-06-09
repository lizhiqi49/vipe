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

from collections.abc import Callable

from .base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType


DepthModelFactory = Callable[[str], DepthEstimationModel]
_DEPTH_MODEL_REGISTRY: dict[str, DepthModelFactory] = {}


def register_depth_model(name: str) -> Callable[[DepthModelFactory], DepthModelFactory]:
    def decorator(factory: DepthModelFactory) -> DepthModelFactory:
        if name in _DEPTH_MODEL_REGISTRY:
            raise ValueError(f"Depth model already registered: {name}")
        _DEPTH_MODEL_REGISTRY[name] = factory
        return factory

    return decorator


def parse_depth_model_name(model: str) -> tuple[str, str]:
    if "-" not in model:
        return model, ""
    model_name, model_sub = model.split("-", 1)
    return model_name, model_sub


def make_depth_model(model: str) -> DepthEstimationModel:
    model_name, model_sub = parse_depth_model_name(model)
    try:
        factory = _DEPTH_MODEL_REGISTRY[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown depth model: {model}") from exc
    return factory(model_sub)


@register_depth_model("metric3d")
def _make_metric3d_depth_model(model_sub: str) -> DepthEstimationModel:
    from .metric3d import Metric3DDepthModel

    return Metric3DDepthModel(version=2, model=model_sub)


@register_depth_model("unidepth")
def _make_unidepth_depth_model(model_sub: str) -> DepthEstimationModel:
    from .unidepth import UniDepth2Model

    return UniDepth2Model(type=model_sub)


@register_depth_model("moge")
def _make_moge_depth_model(model_sub: str) -> DepthEstimationModel:
    del model_sub
    from .moge import MogeModel

    return MogeModel()


@register_depth_model("moge2")
def _make_moge2_depth_model(model_sub: str) -> DepthEstimationModel:
    del model_sub
    from .moge2 import Moge2Model

    return Moge2Model()


@register_depth_model("dav3")
def _make_dav3_depth_model(model_sub: str) -> DepthEstimationModel:
    del model_sub
    from .dav3 import DepthAnything3Model

    return DepthAnything3Model()


@register_depth_model("pi3x")
def _make_pi3x_depth_model(model_sub: str) -> DepthEstimationModel:
    from .pi3x import Pi3XDepthModel

    return Pi3XDepthModel(model_sub=model_sub)
