# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from vipe.utils.cameras import CameraType
from vipe.utils.misc import unpack_optional

from .base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType

SingleFrameDepthProvider = Callable[[DepthEstimationInput], DepthEstimationResult]
SequenceDepthProvider = Callable[[DepthEstimationInput], DepthEstimationResult]


@dataclass(frozen=True)
class Pi3XResolutionPlan:
    slam_height: int
    slam_width: int
    pi3x_height: int
    pi3x_width: int


@dataclass(frozen=True)
class Pi3XMoge2SequenceEstimate:
    pi3x_depths: torch.Tensor
    moge_depths: torch.Tensor
    pi3x_conf: torch.Tensor | None = None
    moge_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class Pi3XMoge2FusionResult:
    fused_depths: torch.Tensor
    raw_scales: torch.Tensor
    smooth_scales: torch.Tensor
    valid_masks: torch.Tensor


Pi3XMoge2SequenceEstimator = Callable[[DepthEstimationInput], Pi3XMoge2SequenceEstimate]
logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _resolve_moge_checkpoint(path: str) -> str:
    local_path = Path(path)
    if local_path.is_dir() and (local_path / "model.pt").exists():
        return str(local_path / "model.pt")
    return path


def _load_pi3x_model(model_path: str, device: torch.device):
    from pi3.models.pi3x import Pi3X

    local_path = Path(model_path)
    if local_path.is_file():
        model = Pi3X(use_multimodal=True).eval()
        if local_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state_dict = load_file(str(local_path))
        else:
            state_dict = torch.load(local_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict, strict=False)
    else:
        model = Pi3X.from_pretrained(model_path).eval()
    return model.to(device)


def plan_pi3x_resolution(
    slam_height: int,
    slam_width: int,
    *,
    pixel_limit: int = 255_000,
    multiple: int = 14,
) -> Pi3XResolutionPlan:
    if slam_height <= 0 or slam_width <= 0:
        raise ValueError("SLAM resolution must be positive")
    if pixel_limit <= 0:
        raise ValueError("pixel_limit must be positive")
    if multiple <= 0:
        raise ValueError("multiple must be positive")

    scale = (pixel_limit / float(slam_height * slam_width)) ** 0.5
    target_height = slam_height * scale
    target_width = slam_width * scale
    rows = max(1, round(target_height / multiple))
    cols = max(1, round(target_width / multiple))

    while (rows * multiple) * (cols * multiple) > pixel_limit and (rows > 1 or cols > 1):
        if cols / rows > target_width / target_height and cols > 1:
            cols -= 1
        elif rows > 1:
            rows -= 1
        else:
            cols -= 1

    return Pi3XResolutionPlan(
        slam_height=slam_height,
        slam_width=slam_width,
        pi3x_height=rows * multiple,
        pi3x_width=cols * multiple,
    )


def vipe_pinhole_intrinsics_to_matrix(intrinsics: torch.Tensor) -> torch.Tensor:
    if intrinsics.shape[-1] != 4:
        raise ValueError("ViPE pinhole intrinsics must have shape (..., 4)")

    fx, fy, cx, cy = intrinsics.unbind(dim=-1)
    matrix = torch.zeros(intrinsics.shape[:-1] + (3, 3), dtype=intrinsics.dtype, device=intrinsics.device)
    matrix[..., 0, 0] = fx
    matrix[..., 1, 1] = fy
    matrix[..., 0, 2] = cx
    matrix[..., 1, 2] = cy
    matrix[..., 2, 2] = 1.0
    return matrix


def rescale_intrinsics_matrix(
    intrinsics: torch.Tensor,
    *,
    from_size: tuple[int, int],
    to_size: tuple[int, int],
) -> torch.Tensor:
    from_height, from_width = from_size
    to_height, to_width = to_size
    if min(from_height, from_width, to_height, to_width) <= 0:
        raise ValueError("image sizes must be positive")

    scaled = intrinsics.clone()
    scale_x = to_width / from_width
    scale_y = to_height / from_height
    scaled[..., 0, 0] *= scale_x
    scaled[..., 0, 2] *= scale_x
    scaled[..., 1, 1] *= scale_y
    scaled[..., 1, 2] *= scale_y
    return scaled


def resize_depth_like(depth: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if depth.dim() != 3:
        raise ValueError("depth must have shape (T, H, W)")
    return F.interpolate(depth[:, None].float(), size=size, mode="bilinear", align_corners=False)[:, 0]


def resize_mask_like(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if mask.dim() != 3:
        raise ValueError("mask must have shape (T, H, W)")
    return F.interpolate(mask[:, None].float(), size=size, mode="nearest")[:, 0].bool()


def compute_sim3_umeyama_masked(
    src_points: torch.Tensor,
    tgt_points: torch.Tensor,
    src_mask: torch.Tensor,
    tgt_mask: torch.Tensor,
    *,
    min_valid_points: int = 10,
    eps: float = 1e-6,
) -> torch.Tensor:
    if src_points.shape != tgt_points.shape:
        raise ValueError("source and target points must have the same shape")
    if src_points.shape[-1] != 3:
        raise ValueError("points must end with xyz dimension")
    if src_mask.shape != src_points.shape[:-1] or tgt_mask.shape != tgt_points.shape[:-1]:
        raise ValueError("masks must match point spatial dimensions")

    device = src_points.device
    src = src_points.reshape(-1, 3)
    tgt = tgt_points.reshape(-1, 3)
    mask = (src_mask.reshape(-1) & tgt_mask.reshape(-1)).float()[:, None]
    valid_count = mask.sum()
    if valid_count < min_valid_points:
        return torch.eye(4, dtype=src_points.dtype, device=device)

    src_mean = (src * mask).sum(dim=0, keepdim=True) / (valid_count + eps)
    tgt_mean = (tgt * mask).sum(dim=0, keepdim=True) / (valid_count + eps)
    src_centered = (src - src_mean) * mask
    tgt_centered = (tgt - tgt_mean) * mask

    covariance = src_centered.T @ tgt_centered
    u, singular_values, v = torch.svd(covariance)
    rotation = v @ u.T

    det = torch.det(rotation)
    diag = torch.ones(3, dtype=src_points.dtype, device=device)
    diag[2] = torch.sign(det)
    rotation = v @ torch.diag(diag) @ u.T

    src_var = (src_centered.square().sum(dim=1) * mask.squeeze(-1)).sum() / (valid_count + eps)
    singular_values = singular_values.clone()
    singular_values[2] *= diag[2]
    scale = singular_values.sum() / (src_var * valid_count + eps)
    translation = tgt_mean.squeeze(0) - scale * (rotation @ src_mean.squeeze(0))

    sim3 = torch.eye(4, dtype=src_points.dtype, device=device)
    sim3[:3, :3] = scale * rotation
    sim3[:3, 3] = translation
    return sim3


def apply_sim3_to_points(points: torch.Tensor, sim3: torch.Tensor) -> torch.Tensor:
    if points.shape[-1] != 3:
        raise ValueError("points must end with xyz dimension")
    flat_points = points.reshape(-1, 3)
    transformed = flat_points @ sim3[:3, :3].T + sim3[:3, 3]
    return transformed.reshape_as(points)


def scale_free_aligned_poses(camera_poses: torch.Tensor, sim3: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    if camera_poses.shape[-2:] != (4, 4):
        raise ValueError("camera_poses must have shape (..., 4, 4)")

    sim3_linear = sim3[:3, :3]
    sim3_scale = torch.linalg.norm(sim3_linear, dim=0).mean().clamp_min(eps)
    sim3_rotation = sim3_linear / sim3_scale

    aligned_poses = camera_poses.clone()
    aligned_poses[..., :3, :3] = sim3_rotation @ camera_poses[..., :3, :3]
    aligned_poses[..., :3, 3] = camera_poses[..., :3, 3] @ sim3_linear.T + sim3[:3, 3]
    return aligned_poses


def camera_z_from_world_points(points: torch.Tensor, camera_poses: torch.Tensor) -> torch.Tensor:
    if points.shape[-1] != 3:
        raise ValueError("points must end with xyz dimension")
    if camera_poses.shape[-2:] != (4, 4):
        raise ValueError("camera_poses must have shape (..., 4, 4)")
    if points.shape[0] != camera_poses.shape[0]:
        raise ValueError("points and poses must have matching frame counts")

    rotation = camera_poses[:, :3, :3]
    translation = camera_poses[:, :3, 3]
    camera_points = torch.einsum("tij,thwj->thwi", rotation.transpose(-1, -2), points - translation[:, None, None])
    return camera_points[..., 2]


def solve_weighted_scales(
    pi3x_depths: torch.Tensor,
    moge_depths: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
    min_valid_pixels: int = 64,
) -> torch.Tensor:
    if pi3x_depths.shape != moge_depths.shape:
        raise ValueError("Pi3X and MoGe depth tensors must have the same shape")
    if pi3x_depths.dim() != 3:
        raise ValueError("depth tensors must have shape (T, H, W)")

    mask = torch.isfinite(pi3x_depths) & torch.isfinite(moge_depths)
    mask &= (pi3x_depths > eps) & (moge_depths > eps)
    if valid_mask is not None:
        if valid_mask.shape != pi3x_depths.shape:
            raise ValueError("valid_mask must match depth tensor shape")
        mask &= valid_mask.bool()

    flat_mask = mask.flatten(1)
    valid_counts = flat_mask.sum(dim=1)
    if torch.any(valid_counts < min_valid_pixels):
        raise ValueError("not enough valid pixels to solve Pi3X/MoGe-2 scale")

    weights = torch.where(mask, moge_depths.clamp_min(eps).reciprocal(), torch.zeros_like(moge_depths))
    pi3x_valid = torch.where(mask, pi3x_depths, torch.zeros_like(pi3x_depths))
    moge_valid = torch.where(mask, moge_depths, torch.zeros_like(moge_depths))
    numerator = (weights * pi3x_valid * moge_valid).flatten(1).sum(dim=1)
    denominator = (weights * pi3x_valid.square()).flatten(1).sum(dim=1)
    if torch.any(denominator <= eps):
        raise ValueError("degenerate Pi3X/MoGe-2 scale denominator")
    return numerator / denominator


def smooth_scales_ema(raw_scales: torch.Tensor, *, momentum: float = 0.99) -> torch.Tensor:
    if raw_scales.dim() != 1:
        raise ValueError("raw_scales must have shape (T,)")
    if not 0.0 <= momentum < 1.0:
        raise ValueError("momentum must be in [0, 1)")
    if raw_scales.numel() == 0:
        raise ValueError("raw_scales must not be empty")

    smoothed = torch.empty_like(raw_scales)
    smoothed[0] = raw_scales[0]
    for idx in range(1, raw_scales.numel()):
        smoothed[idx] = momentum * smoothed[idx - 1] + (1.0 - momentum) * raw_scales[idx]
    return smoothed


def fuse_pi3x_moge2_depths(
    estimate: Pi3XMoge2SequenceEstimate,
    *,
    target_size: tuple[int, int],
    pi3x_conf_threshold: float = 0.1,
    ema_momentum: float = 0.99,
    eps: float = 1e-6,
    min_valid_pixels: int = 64,
) -> Pi3XMoge2FusionResult:
    pi3x_depths = estimate.pi3x_depths.float()
    moge_depths = estimate.moge_depths.float()

    if pi3x_depths.shape[-2:] != target_size:
        pi3x_depths = resize_depth_like(pi3x_depths, target_size)
    if moge_depths.shape[-2:] != target_size:
        moge_depths = resize_depth_like(moge_depths, target_size)

    valid_mask = torch.ones_like(pi3x_depths, dtype=torch.bool)
    if estimate.moge_mask is not None:
        moge_mask = estimate.moge_mask
        if moge_mask.shape[-2:] != target_size:
            moge_mask = resize_mask_like(moge_mask.bool(), target_size)
        valid_mask &= moge_mask.bool()

    if estimate.pi3x_conf is not None:
        pi3x_conf = estimate.pi3x_conf.float()
        if pi3x_conf.shape[-2:] != target_size:
            pi3x_conf = resize_depth_like(pi3x_conf, target_size)
        valid_mask &= pi3x_conf >= pi3x_conf_threshold

    finite_depth_mask = torch.isfinite(pi3x_depths) & torch.isfinite(moge_depths)
    finite_depth_mask &= (pi3x_depths > eps) & (moge_depths > eps)
    valid_mask &= finite_depth_mask

    raw_scales = solve_weighted_scales(
        pi3x_depths,
        moge_depths,
        valid_mask=valid_mask,
        eps=eps,
        min_valid_pixels=min_valid_pixels,
    )
    smooth_scales = smooth_scales_ema(raw_scales, momentum=ema_momentum)
    fused_depths = smooth_scales[:, None, None] * pi3x_depths
    fused_depths = torch.where(valid_mask, fused_depths, moge_depths)
    fused_depths = torch.where(
        torch.isfinite(fused_depths) & (fused_depths > eps),
        fused_depths,
        torch.full_like(fused_depths, eps),
    )
    return Pi3XMoge2FusionResult(
        fused_depths=fused_depths,
        raw_scales=raw_scales,
        smooth_scales=smooth_scales,
        valid_masks=valid_mask,
    )


class FusionPi3XMoge2SequenceProvider:
    def __init__(
        self,
        sequence_estimator: Pi3XMoge2SequenceEstimator,
        *,
        pi3x_conf_threshold: float = 0.1,
        ema_momentum: float = 0.99,
        min_valid_pixels: int = 64,
    ):
        self.sequence_estimator = sequence_estimator
        self.pi3x_conf_threshold = pi3x_conf_threshold
        self.ema_momentum = ema_momentum
        self.min_valid_pixels = min_valid_pixels

    def __call__(self, src: DepthEstimationInput) -> DepthEstimationResult:
        if src.video_frame_list is None:
            raise ValueError("Pi3X/MoGe-2 sequence fusion requires video_frame_list")
        if len(src.video_frame_list) == 0:
            raise ValueError("video_frame_list must not be empty")

        height, width = src.video_frame_list[0].shape[:2]
        start_time = time.perf_counter()
        estimate = self.sequence_estimator(src)
        estimate_time = time.perf_counter()
        fusion = fuse_pi3x_moge2_depths(
            estimate,
            target_size=(height, width),
            pi3x_conf_threshold=self.pi3x_conf_threshold,
            ema_momentum=self.ema_momentum,
            min_valid_pixels=self.min_valid_pixels,
        )
        fusion_time = time.perf_counter()
        logger.info(
            "Pi3X/MoGe-2 sequence fusion timing: frames=%d estimate=%.3fs fusion_cpu=%.3fs total=%.3fs",
            len(src.video_frame_list),
            estimate_time - start_time,
            fusion_time - estimate_time,
            fusion_time - start_time,
        )
        return DepthEstimationResult(metric_depth=fusion.fused_depths)


class RealPi3XMoge2Provider:
    def __init__(
        self,
        *,
        pi3x_model_path: str | None = None,
        moge_model_path: str | None = None,
        device: str | None = None,
        pi3x_pixel_limit: int | None = None,
        moge_num_tokens: int | None = None,
        moge_resolution_level: int = 9,
        use_fp16: bool | None = None,
    ):
        self.pi3x_model_path = (
            pi3x_model_path
            or os.environ.get("VIPE_PI3X_MODEL_PATH")
            or os.environ.get("PI3X_LOCAL_MODEL_ROOT")
            or "yyfz233/Pi3X"
        )
        self.moge_model_path = (
            moge_model_path
            or os.environ.get("VIPE_MOGE2_MODEL_PATH")
            or os.environ.get("MOGE2_LOCAL_MODEL_ROOT")
            or "Ruicheng/moge-2-vitl-normal"
        )
        self.device = torch.device(device or os.environ.get("VIPE_PI3X_MOGE2_DEVICE", "cuda"))
        self.pi3x_pixel_limit = int(os.environ.get("VIPE_PI3X_PIXEL_LIMIT", pi3x_pixel_limit or 255_000))
        self.moge_num_tokens = (
            int(os.environ["VIPE_MOGE2_NUM_TOKENS"]) if "VIPE_MOGE2_NUM_TOKENS" in os.environ else moge_num_tokens
        )
        self.moge_resolution_level = int(os.environ.get("VIPE_MOGE2_RESOLUTION_LEVEL", moge_resolution_level))
        self.pi3x_chunk_size = int(os.environ.get("VIPE_PI3X_CHUNK_SIZE", 16))
        self.pi3x_overlap = int(os.environ.get("VIPE_PI3X_OVERLAP", 6))
        self.pi3x_vo_conf_threshold = float(os.environ.get("VIPE_PI3X_VO_CONF_THRESHOLD", 0.05))
        if self.pi3x_chunk_size <= 0:
            raise ValueError("VIPE_PI3X_CHUNK_SIZE must be positive")
        if self.pi3x_overlap <= 0 or self.pi3x_overlap >= self.pi3x_chunk_size:
            raise ValueError("VIPE_PI3X_OVERLAP must be positive and smaller than chunk size")
        self.use_fp16 = _env_bool("VIPE_PI3X_MOGE2_FP16", use_fp16 if use_fp16 is not None else True)
        self.sequence_provider = FusionPi3XMoge2SequenceProvider(
            self._estimate_sequence,
            pi3x_conf_threshold=float(os.environ.get("VIPE_PI3X_CONF_THRESHOLD", 0.1)),
            ema_momentum=float(os.environ.get("VIPE_PI3X_MOGE2_EMA_MOMENTUM", 0.99)),
            min_valid_pixels=int(os.environ.get("VIPE_PI3X_MOGE2_MIN_VALID_PIXELS", 64)),
        )
        self._pi3x_model = None
        self._moge_model = None

    def _load_moge(self) -> None:
        if self._moge_model is not None:
            self._moge_model.to(self.device).eval()
            if self.use_fp16 and self.device.type == "cuda":
                self._moge_model.half()
            return

        from moge.model.v2 import MoGeModel

        moge_checkpoint = _resolve_moge_checkpoint(self.moge_model_path)
        self._moge_model = MoGeModel.from_pretrained(moge_checkpoint).to(self.device).eval()
        if self.use_fp16 and self.device.type == "cuda":
            self._moge_model.half()

    def _load_pi3x(self) -> None:
        if self._pi3x_model is not None:
            self._pi3x_model.to(self.device).eval()
            return
        self._pi3x_model = _load_pi3x_model(self.pi3x_model_path, self.device)

    def _empty_cuda_cache(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _offload_moge(self) -> None:
        if self._moge_model is not None and self.device.type == "cuda":
            self._moge_model.to("cpu")
            self._empty_cuda_cache()

    def _offload_pi3x(self) -> None:
        if self._pi3x_model is not None and self.device.type == "cuda":
            self._pi3x_model.to("cpu")
            self._empty_cuda_cache()

    def _fov_x(self, intrinsics: torch.Tensor | None, width: int) -> float | None:
        if intrinsics is None:
            return None
        intrinsics_matrix = vipe_pinhole_intrinsics_to_matrix(intrinsics.to(self.device).float())
        return torch.rad2deg(
            2.0 * torch.atan(torch.tensor(width, device=self.device) / (2.0 * intrinsics_matrix[0, 0]))
        ).item()

    def estimate_single(self, src: DepthEstimationInput) -> DepthEstimationResult:
        rgb: torch.Tensor = unpack_optional(src.rgb)
        if src.camera_type != CameraType.PINHOLE:
            raise ValueError("Pi3X/MoGe-2 keyframe depth only supports pinhole cameras")
        if rgb.dim() == 3:
            rgb = rgb[None]
            squeeze_batch = True
        elif rgb.dim() == 4:
            squeeze_batch = False
        else:
            raise ValueError("rgb must have shape (H, W, 3) or (B, H, W, 3)")

        self._load_moge()
        depths = []
        fov_x = self._fov_x(src.intrinsics, rgb.shape[2])
        with torch.no_grad():
            for frame in rgb:
                frame_chw = frame.to(self.device).float().permute(2, 0, 1).clamp(0.0, 1.0)
                output = self._moge_model.infer(
                    frame_chw,
                    fov_x=fov_x,
                    apply_mask=False,
                    resolution_level=self.moge_resolution_level,
                    num_tokens=self.moge_num_tokens,
                    use_fp16=self.use_fp16 and self.device.type == "cuda",
                )
                depth = torch.nan_to_num(output["depth"].float(), nan=1e4).clamp(min=0.0, max=1e4)
                if "mask" in output:
                    depth = depth * output["mask"].bool().float()
                depths.append(depth.to(device=rgb.device, dtype=rgb.dtype))

        metric_depth = torch.stack(depths, dim=0)
        if squeeze_batch:
            metric_depth = metric_depth[0]
        return DepthEstimationResult(metric_depth=metric_depth)

    def _estimate_sequence(self, src: DepthEstimationInput) -> Pi3XMoge2SequenceEstimate:
        if src.video_frame_list is None:
            raise ValueError("Pi3X/MoGe-2 sequence estimator requires video_frame_list")
        if len(src.video_frame_list) == 0:
            raise ValueError("video_frame_list must not be empty")

        total_start = time.perf_counter()
        frames_cpu = torch.stack(
            [
                torch.as_tensor(frame, dtype=torch.float32).permute(2, 0, 1).clamp(0.0, 1.0)
                for frame in src.video_frame_list
            ],
            dim=0,
        )
        frame_count, _, slam_height, slam_width = frames_cpu.shape
        plan = plan_pi3x_resolution(slam_height, slam_width, pixel_limit=self.pi3x_pixel_limit)

        moge_load_start = time.perf_counter()
        self._load_moge()
        fov_x = self._fov_x(src.intrinsics, slam_width)
        moge_depths = []
        moge_masks = []
        moge_infer_start = time.perf_counter()
        with torch.no_grad():
            for frame in frames_cpu:
                output = self._moge_model.infer(
                    frame.to(self.device),
                    fov_x=fov_x,
                    apply_mask=False,
                    resolution_level=self.moge_resolution_level,
                    num_tokens=self.moge_num_tokens,
                    use_fp16=self.use_fp16 and self.device.type == "cuda",
                )
                moge_depths.append(output["depth"].float().cpu())
                moge_masks.append(output["mask"].bool().cpu())
        moge_offload_start = time.perf_counter()
        self._offload_moge()
        pi3x_load_start = time.perf_counter()

        self._load_pi3x()
        self._empty_cuda_cache()
        pi3x_preprocess_start = time.perf_counter()
        pi3x_frames = F.interpolate(
            frames_cpu,
            size=(plan.pi3x_height, plan.pi3x_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).to(self.device)

        pi3x_intrinsics = None
        if src.intrinsics is not None:
            intrinsics_matrix = vipe_pinhole_intrinsics_to_matrix(src.intrinsics.to(self.device).float())
            pi3x_intrinsics = rescale_intrinsics_matrix(
                intrinsics_matrix,
                from_size=(slam_height, slam_width),
                to_size=(plan.pi3x_height, plan.pi3x_width),
            )
            pi3x_intrinsics = pi3x_intrinsics[None, None].expand(1, frame_count, 3, 3).contiguous()

        autocast_enabled = self.use_fp16 and self.device.type == "cuda"
        autocast_dtype = (
            torch.bfloat16
            if self.device.type == "cuda" and torch.cuda.get_device_capability(self.device)[0] >= 8
            else torch.float16
        )
        pi3x_preprocess_done = time.perf_counter()

        from pi3.utils.geometry import depth_normal_edge

        chunk_depths = []
        chunk_confs = []
        prev_global_pts_overlap = None
        prev_global_mask_overlap = None
        stride = self.pi3x_chunk_size - self.pi3x_overlap
        chunk_count = 0
        pi3x_infer_total = 0.0
        pi3x_align_total = 0.0
        pi3x_to_cpu_total = 0.0

        for start_idx in range(0, frame_count, stride):
            end_idx = min(start_idx + self.pi3x_chunk_size, frame_count)
            current_len = end_idx - start_idx
            if current_len <= self.pi3x_overlap and start_idx > 0:
                break

            chunk_count += 1
            chunk_imgs = pi3x_frames[None, start_idx:end_idx]
            chunk_intrinsics = None
            if pi3x_intrinsics is not None:
                chunk_intrinsics = pi3x_intrinsics[:, start_idx:end_idx]

            infer_start = time.perf_counter()
            with torch.no_grad():
                with torch.amp.autocast(self.device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                    pi3x_output = self._pi3x_model(imgs=chunk_imgs, intrinsics=chunk_intrinsics)
            infer_done = time.perf_counter()
            pi3x_infer_total += infer_done - infer_start

            align_start = time.perf_counter()
            curr_points = pi3x_output["points"][0].float()
            curr_poses = pi3x_output["camera_poses"][0].float()
            curr_conf = torch.sigmoid(pi3x_output["conf"][0, ..., 0].float())
            curr_valid = curr_conf > self.pi3x_vo_conf_threshold
            edge = depth_normal_edge(pi3x_output["local_points"].float(), rtol=0.03, mask=curr_valid[None])
            curr_conf[edge[0]] = 0
            curr_mask = curr_conf > self.pi3x_vo_conf_threshold
            if curr_mask.sum() < 10:
                flat_conf = curr_conf.reshape(current_len, -1)
                k = max(1, int(flat_conf.shape[-1] * 0.1))
                topk_vals, _ = torch.topk(flat_conf, k, dim=-1)
                min_vals = topk_vals[:, -1].reshape(current_len, 1, 1)
                curr_mask = curr_conf >= min_vals

            if start_idx == 0:
                aligned_points = curr_points
                aligned_poses = curr_poses
            else:
                sim3 = compute_sim3_umeyama_masked(
                    curr_points[: self.pi3x_overlap],
                    prev_global_pts_overlap,
                    curr_mask[: self.pi3x_overlap],
                    prev_global_mask_overlap,
                )
                aligned_points = apply_sim3_to_points(curr_points, sim3)
                aligned_poses = scale_free_aligned_poses(curr_poses, sim3)

            aligned_depth = camera_z_from_world_points(aligned_points, aligned_poses)
            if start_idx == 0:
                keep = slice(None)
            else:
                keep = slice(self.pi3x_overlap, None)

            prev_global_pts_overlap = aligned_points[-self.pi3x_overlap :].detach()
            prev_global_mask_overlap = curr_mask[-self.pi3x_overlap :].detach()
            align_done = time.perf_counter()
            pi3x_align_total += align_done - align_start

            transfer_start = time.perf_counter()
            chunk_depths.append(aligned_depth[keep].float().cpu())
            chunk_confs.append(curr_conf[keep].float().cpu())
            transfer_done = time.perf_counter()
            pi3x_to_cpu_total += transfer_done - transfer_start

            del pi3x_output, curr_points, curr_poses, curr_conf, curr_valid, curr_mask, aligned_points, aligned_poses
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            if end_idx == frame_count:
                break

        pi3x_depths = torch.cat(chunk_depths, dim=0)
        pi3x_conf = torch.cat(chunk_confs, dim=0)
        if pi3x_depths.shape[0] != frame_count:
            raise ValueError("chunked Pi3X depth count must match sequence frame count")
        del pi3x_frames, pi3x_intrinsics
        pi3x_offload_start = time.perf_counter()
        self._offload_pi3x()
        done = time.perf_counter()

        logger.info(
            "Pi3X/MoGe-2 sequence estimator timing: frames=%d slam=%dx%d pi3x=%dx%d "
            "moge_load=%.3fs moge_infer=%.3fs moge_offload=%.3fs "
            "pi3x_load=%.3fs pi3x_preprocess=%.3fs chunks=%d chunk_size=%d overlap=%d "
            "pi3x_infer=%.3fs pi3x_align=%.3fs pi3x_to_cpu=%.3fs pi3x_offload=%.3fs total=%.3fs",
            frame_count,
            slam_height,
            slam_width,
            plan.pi3x_height,
            plan.pi3x_width,
            moge_infer_start - moge_load_start,
            moge_offload_start - moge_infer_start,
            pi3x_load_start - moge_offload_start,
            pi3x_preprocess_start - pi3x_load_start,
            pi3x_preprocess_done - pi3x_preprocess_start,
            chunk_count,
            self.pi3x_chunk_size,
            self.pi3x_overlap,
            pi3x_infer_total,
            pi3x_align_total,
            pi3x_to_cpu_total,
            done - pi3x_offload_start,
            done - total_start,
        )

        return Pi3XMoge2SequenceEstimate(
            pi3x_depths=pi3x_depths,
            moge_depths=torch.stack(moge_depths, dim=0),
            pi3x_conf=pi3x_conf,
            moge_mask=torch.stack(moge_masks, dim=0),
        )

    def estimate_sequence(self, src: DepthEstimationInput) -> DepthEstimationResult:
        return self.sequence_provider(src)


class Pi3XMoge2DepthModel(DepthEstimationModel):
    """
    Keyframe-depth model for SANA-WM-style ViPE.

    Online keyframe insertion uses MoGe-2 as a single-frame metric prior through
    estimate(). Backend keyframe refresh calls estimate_keyframe_sequence() to
    run Pi3X once on the finalized keyframe sequence and fuse it to MoGe-2.
    """

    def __init__(
        self,
        single_frame_provider: SingleFrameDepthProvider | None = None,
        sequence_provider: SequenceDepthProvider | None = None,
    ):
        if single_frame_provider is None or sequence_provider is None:
            real_provider = RealPi3XMoge2Provider()
            single_frame_provider = single_frame_provider or real_provider.estimate_single
            sequence_provider = sequence_provider or real_provider.estimate_sequence
        self.single_frame_provider = single_frame_provider
        self.sequence_provider = sequence_provider

    @property
    def depth_type(self) -> DepthType:
        return DepthType.MODEL_METRIC_DEPTH

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        result = self.single_frame_provider(src)
        if result.metric_depth is None:
            raise ValueError("pi3x_moge2 single-frame provider must return metric_depth")
        return result

    def estimate_keyframe_sequence(self, src: DepthEstimationInput) -> DepthEstimationResult:
        if src.video_frame_list is None:
            raise ValueError("pi3x_moge2 sequence fusion requires DepthEstimationInput.video_frame_list")
        result = self.sequence_provider(src)
        if result.metric_depth is None:
            raise ValueError("pi3x_moge2 sequence provider must return metric_depth")
        return result
