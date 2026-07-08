from __future__ import annotations

from types import SimpleNamespace

import torch

from vipe.priors.depth.base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType
from vipe.priors.depth.pi3x_moge2 import (
    apply_sim3_to_points,
    camera_z_from_world_points,
    compute_sim3_umeyama_masked,
    scale_free_aligned_poses,
)
from vipe.slam.components import backend as backend_module
from vipe.slam.components.backend import SLAMBackend
from vipe.slam.components.buffer import GraphBuffer
from vipe.utils.cameras import CameraType


class _DummySparseTracks:
    pass


def _make_buffer(frame_count: int = 3) -> GraphBuffer:
    buffer = GraphBuffer(
        height=16,
        width=16,
        n_views=1,
        buffer_size=8,
        init_disp=1.0,
        cross_view_idx=None,
        ba_config={},
        sparse_tracks=_DummySparseTracks(),
        camera_type=CameraType.PINHOLE,
        device=torch.device("cpu"),
    )
    buffer.n_frames = frame_count
    buffer.intrinsics[0] = torch.tensor([10.0, 10.0, 8.0, 8.0])
    buffer.tstamp[:frame_count] = torch.arange(frame_count) * 5
    for frame_idx in range(frame_count):
        buffer.images[frame_idx, 0] = float(frame_idx + 1) / 10.0
    return buffer


class _SingleFrameDepthModel(DepthEstimationModel):
    def __init__(self) -> None:
        self.calls: list[DepthEstimationInput] = []

    @property
    def depth_type(self) -> DepthType:
        return DepthType.MODEL_METRIC_DEPTH

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        self.calls.append(src)
        assert src.video_frame_list is None
        rgb = src.rgb
        assert rgb is not None
        return DepthEstimationResult(metric_depth=torch.full((rgb.shape[0], 16, 16), 2.0, device=rgb.device))


class _SequenceDepthModel(_SingleFrameDepthModel):
    def __init__(self) -> None:
        super().__init__()
        self.sequence_calls: list[DepthEstimationInput] = []

    def estimate_keyframe_sequence(self, src: DepthEstimationInput) -> DepthEstimationResult:
        self.sequence_calls.append(src)
        assert src.video_frame_list is not None
        depths = [
            torch.full((16, 16), float(frame_idx + 1), dtype=torch.float32)
            for frame_idx in range(len(src.video_frame_list))
        ]
        return DepthEstimationResult(metric_depth=torch.stack(depths, dim=0))


def test_single_frame_update_uses_official_depth_input_contract() -> None:
    buffer = _make_buffer(frame_count=1)
    model = _SingleFrameDepthModel()

    buffer.update_disps_sens(model, frame_idx=0)

    assert len(model.calls) == 1
    assert model.calls[0].video_frame_list is None
    assert torch.allclose(buffer.disps_sens[0, 0], torch.full((2, 2), 0.5))


def test_sequence_depth_update_uses_final_keyframe_list_once_and_invalidates_on_timestamps() -> None:
    buffer = _make_buffer(frame_count=3)
    model = _SequenceDepthModel()

    assert buffer.update_disps_sens_sequence(model)
    assert len(model.calls) == 0
    assert len(model.sequence_calls) == 1
    assert len(model.sequence_calls[0].video_frame_list or []) == 3
    assert torch.allclose(buffer.disps_sens[:3, 0, 0, 0], torch.tensor([1.0, 0.5, 1.0 / 3.0]))

    assert buffer.update_disps_sens_sequence(model)
    assert len(model.sequence_calls) == 1

    buffer.tstamp[2] = 99
    assert buffer.update_disps_sens_sequence(model)
    assert len(model.sequence_calls) == 2


def test_scale_free_aligned_pose_projects_sim3_points_to_scaled_camera_z() -> None:
    local_points = torch.tensor([[[[0.0, 0.0, 3.0]]]])
    camera_pose = torch.eye(4).reshape(1, 4, 4)
    sim3 = torch.eye(4)
    sim3[:3, :3] *= 2.0

    world_points = apply_sim3_to_points(local_points, sim3)
    raw_sim3_pose = sim3 @ camera_pose
    raw_camera_points = torch.einsum(
        "ij,thwj->thwi",
        torch.linalg.inv(raw_sim3_pose[0]),
        torch.cat([world_points, torch.ones_like(world_points[..., :1])], dim=-1),
    )
    raw_z = raw_camera_points[..., 2]

    aligned_pose = scale_free_aligned_poses(camera_pose, sim3)
    aligned_z = camera_z_from_world_points(world_points, aligned_pose)

    assert torch.allclose(raw_z, torch.tensor([[[3.0]]]))
    assert torch.allclose(aligned_z, torch.tensor([[[6.0]]]))


def test_masked_umeyama_sim3_aligns_source_points_to_target_points() -> None:
    y, x = torch.meshgrid(torch.arange(4.0), torch.arange(4.0), indexing="ij")
    src = torch.stack([x, y, torch.ones_like(x)], dim=-1)[None]
    tgt = 2.0 * src + torch.tensor([1.0, 2.0, 3.0])
    mask = torch.ones(src.shape[:-1], dtype=torch.bool)

    sim3 = compute_sim3_umeyama_masked(src, tgt, mask, mask)
    aligned = apply_sim3_to_points(src, sim3)

    assert torch.allclose(aligned, tgt, atol=1e-4)


def test_backend_update_depth_false_skips_sequence_depth_refresh(monkeypatch) -> None:
    class _FakeGraph:
        def __init__(self, *args, **kwargs) -> None:
            self.ii = torch.tensor([0])
            self.jj = torch.tensor([1])

        def add_proximity_factors(self, *args, **kwargs) -> None:
            return None

        def update_batch(self, *args, **kwargs) -> None:
            return None

    class _FakeVideo:
        n_frames = 2
        dirty = torch.zeros(2, dtype=torch.bool)

        def __init__(self) -> None:
            self.sequence_updates = 0

        def update_disps_sens_sequence(self, depth_model: DepthEstimationModel) -> bool:
            self.sequence_updates += 1
            return True

        def update_disps_sens(self, depth_model: DepthEstimationModel, frame_idx: int | None) -> None:
            raise AssertionError("single-frame depth refresh should not be used for sequence model")

    monkeypatch.setattr(backend_module, "FactorGraph", _FakeGraph)
    video = _FakeVideo()
    args = SimpleNamespace(
        backend_radius=2,
        backend_nms=1,
        backend_thresh=16.0,
        beta=0.3,
        cross_view=False,
        adaptive_cross_view=False,
        optimize_intrinsics=False,
        optimize_rig_rotation=False,
    )
    backend = SLAMBackend(net=object(), video=video, args=args, device=torch.device("cpu"))
    backend.depth_model = _SequenceDepthModel()

    backend.run(update_depth=False)
    assert video.sequence_updates == 0

    backend.run(update_depth=True)
    assert video.sequence_updates == 1


def test_backend_run_if_necessary_uses_sequence_depth_refresh(monkeypatch) -> None:
    class _FakeGraph:
        def __init__(self, *args, **kwargs) -> None:
            self.ii = torch.tensor([0])
            self.jj = torch.tensor([1])

        def add_proximity_factors(self, *args, **kwargs) -> None:
            return None

        def update_batch(self, *args, **kwargs) -> None:
            return None

    class _FakeVideo:
        n_frames = 2
        dirty = torch.zeros(2, dtype=torch.bool)

        def __init__(self) -> None:
            self.sequence_updates = 0
            self.single_frame_updates = 0

        def update_disps_sens_sequence(self, depth_model: DepthEstimationModel) -> bool:
            self.sequence_updates += 1
            return True

        def update_disps_sens(self, depth_model: DepthEstimationModel, frame_idx: int | None) -> None:
            raise AssertionError("run_if_necessary should use sequence depth when available")

    monkeypatch.setattr(backend_module, "FactorGraph", _FakeGraph)
    video = _FakeVideo()
    args = SimpleNamespace(
        backend_radius=2,
        backend_nms=1,
        backend_thresh=16.0,
        beta=0.3,
        cross_view=False,
        adaptive_cross_view=False,
        optimize_intrinsics=True,
        optimize_rig_rotation=False,
    )
    backend = SLAMBackend(net=object(), video=video, args=args, device=torch.device("cpu"))
    backend.depth_model = _SequenceDepthModel()

    backend.run_if_necessary()

    assert video.sequence_updates == 1
    assert video.single_frame_updates == 0
