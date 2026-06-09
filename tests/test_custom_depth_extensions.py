import torch

from vipe.pipeline.default import make_post_depth_processor, register_post_processor_prefix
from vipe.priors.depth import make_depth_model, register_depth_model
from vipe.priors.depth.alignment import align_inv_depth_scale_only_weighted


class _FakeDepthModel:
    def __init__(self, model_sub: str) -> None:
        self.model_sub = model_sub


def test_depth_model_registry_preserves_name_suffix_split():
    name = "testfake"
    if name not in make_depth_model.__globals__["_DEPTH_MODEL_REGISTRY"]:
        @register_depth_model(name)
        def _make_fake_depth_model(model_sub: str):
            return _FakeDepthModel(model_sub)

    model = make_depth_model("testfake-vitl")
    assert isinstance(model, _FakeDepthModel)
    assert model.model_sub == "vitl"


def test_weighted_scale_only_alignment_matches_metric_target():
    source_inv_depth = torch.tensor([[0.5, 1.0], [1.5, 2.0]], dtype=torch.float32)
    expected_scale = torch.tensor(2.5)
    target_depth = torch.reciprocal(source_inv_depth * expected_scale)

    aligned_depth, scale = align_inv_depth_scale_only_weighted(source_inv_depth, target_depth)

    assert torch.isclose(scale, expected_scale, atol=1e-5)
    assert torch.allclose(aligned_depth, target_depth, atol=1e-5)


def test_post_processor_registry_dispatches_by_prefix():
    prefix = "unit_"
    registry = make_post_depth_processor.__globals__["_POST_PROCESSOR_FACTORIES"]
    if prefix not in registry:
        @register_post_processor_prefix(prefix)
        def _make_unit_processor(slam_output, view_idx: int, model: str):
            del slam_output
            return {"view_idx": view_idx, "model": model}

    processor = make_post_depth_processor(slam_output=None, view_idx=3, model="unit_recipe")  # type: ignore[arg-type]
    assert processor == {"view_idx": 3, "model": "unit_recipe"}
