import torch
import vipe.pipeline.processors as processors_module

from vipe.pipeline.default import make_post_depth_processor, register_post_processor_prefix
from vipe.pipeline.processors import SanaDepthProcessor
from vipe.priors.depth import make_depth_model, register_depth_model
from vipe.priors.depth.alignment import align_inv_depth_scale_only_weighted
from vipe.priors.depth.vggt_omega import (
    _balanced_target_shape,
    _crop_to_supported_aspect_ratio,
    _max_size_target_shape,
)
from vipe.priors.depth.windowed import WindowedDepthAccumulator, window_starts


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


def test_vggt_omega_depth_model_is_registered():
    assert "vggt_omega" in make_depth_model.__globals__["_DEPTH_MODEL_REGISTRY"]


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


def test_sana_depth_processor_accepts_vggt_recipe(monkeypatch):
    depth_models: list[str] = []

    def _fake_make_depth_model(name: str):
        depth_models.append(name)
        return {"name": name}

    monkeypatch.setattr(processors_module, "make_depth_model", _fake_make_depth_model)

    processor = SanaDepthProcessor(slam_output=None, view_idx=0, model="sana_vggt_moge2")  # type: ignore[arg-type]

    assert depth_models == ["moge2", "vggt_omega"]
    assert processor.video_model_label == "vggt"
    assert processor.use_affine_alignment is False


def test_sana_depth_processor_accepts_affine_pi3x_recipe(monkeypatch):
    depth_models: list[str] = []

    def _fake_make_depth_model(name: str):
        depth_models.append(name)
        return {"name": name}

    monkeypatch.setattr(processors_module, "make_depth_model", _fake_make_depth_model)

    processor = SanaDepthProcessor(slam_output=None, view_idx=0, model="sana_pi3x_moge2_affine")  # type: ignore[arg-type]

    assert depth_models == ["moge2", "pi3x"]
    assert processor.video_model_label == "pi3x"
    assert processor.use_affine_alignment is True


def test_window_starts_short_sequence_single_window():
    assert window_starts(40, window_size=64, window_overlap=8) == [0]


def test_window_starts_covers_last_frame_with_snapped_tail():
    starts = window_starts(150, window_size=64, window_overlap=8)
    # step = 64 - 8 = 56 -> 0, 56, then snap final start to 150 - 64 = 86
    assert starts[0] == 0
    assert starts[1] == 56
    assert starts[-1] == 150 - 64
    assert all(s + 64 <= 150 for s in starts)


def test_windowed_accumulator_equal_weight_average_on_overlap():
    accumulator = WindowedDepthAccumulator(total_frames=3)
    # frame 1 is shared by both windows; equal-weight average of 2.0 and 4.0 -> 3.0
    accumulator.add(0, 2, torch.full((2, 1, 1), 2.0), torch.full((2, 1, 1), 0.5))
    accumulator.add(1, 3, torch.full((2, 1, 1), 4.0), torch.full((2, 1, 1), 0.5))
    depth, conf = accumulator.result()
    assert torch.allclose(depth[:, 0, 0], torch.tensor([2.0, 3.0, 4.0]))
    assert torch.allclose(conf[:, 0, 0], torch.tensor([0.5, 0.5, 0.5]))


def test_vggt_balanced_target_shape_matches_official_load_fn():
    # 3:2 landscape (H/W = 2/3) at image_resolution=512, patch=16 -> 624x416 (README).
    assert _balanced_target_shape(2 / 3, 512, 16) == (416, 624)
    assert _balanced_target_shape(1.0, 512, 16) == (512, 512)


def test_vggt_max_size_target_shape_matches_official_load_fn():
    # Same 3:2 landscape in max_size mode -> 512x336 (README).
    assert _max_size_target_shape(2 / 3, 512, 16) == (336, 512)
    assert _max_size_target_shape(1.0, 512, 16) == (512, 512)


def test_vggt_target_shapes_are_patch_aligned():
    for ar in (0.5, 0.75, 1.0, 1.5, 2.0):
        h_b, w_b = _balanced_target_shape(ar, 512, 16)
        h_m, w_m = _max_size_target_shape(ar, 512, 16)
        assert h_b % 16 == 0 and w_b % 16 == 0
        assert h_m % 16 == 0 and w_m % 16 == 0


def test_vggt_crop_to_supported_aspect_ratio_clamps_extremes():
    from PIL import Image

    wide = Image.new("RGB", (1000, 200))  # aspect_ratio H/W = 0.2 < 0.5
    cropped, box = _crop_to_supported_aspect_ratio(wide)
    left, top, right, bottom = box
    cw, ch = cropped.size
    assert ch / cw <= 0.5 + 1e-6
    assert (right - left, bottom - top) == (cw, ch)

    in_range = Image.new("RGB", (640, 480))  # aspect_ratio 0.75, untouched
    cropped2, box2 = _crop_to_supported_aspect_ratio(in_range)
    assert cropped2.size == (640, 480)
    assert box2 == (0, 0, 640, 480)
