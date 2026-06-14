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
    from pi3.utils.geometry import depth_normal_edge
except ModuleNotFoundError:
    Pi3X = None
    depth_normal_edge = None

from vipe.utils.misc import unpack_optional

from ..base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType


# Official Pi3 preprocessing pixel budget (pi3.utils.basic.load_images_as_tensor). Resolution
# is locked to this contract: feeding the model anything other than the official patch-aligned
# resolution is unsupported (issue #23). Memory is bounded only by the temporal window.
_PI3X_PIXEL_LIMIT = 255000

# Pi3XVO confidence gate (pi3.pipe.pi3x_vo defaults). Views below this confidence are excluded
# from the Umeyama Sim3 fit and (via edge removal) reported as low confidence downstream.
_PI3X_CONF_THRE = 0.05

# Minimum overlap correspondences required for a well-posed Sim3 fit (mirrors Pi3XVO).
_PI3X_MIN_VALID = 10

# Positive-depth clamp shared by local depth and the synthesized global depth.
_PI3X_DEPTH_MIN = 1e-4

# Condition-injection conditions enabled by default (issue #24). Default is "none": the issue #24
# A/B (tinyset 3 videos, w16/o6) showed pose+depth injection worsened emit-seam temporal continuity
# on 2/3 videos, because the post-hoc Sim3 chaining already aligns chunks well. Injection stays
# available via VIPE_PI3X_INJECT_CONDITION (comma-separated: pose,depth,ray / intrinsic); set it to
# "none"/"" to keep it disabled. This matches the official example_vo.py default (no injection).
_PI3X_VALID_INJECT = {"pose", "depth", "ray", "intrinsic"}
_PI3X_DEFAULT_INJECT = "none"


class Pi3XDepthModel(DepthEstimationModel):
    """
    Pi3X video-depth wrapper with cross-chunk Sim3 scale alignment.

    This wrapper consumes the full `video_frame_list`, runs Pi3X in overlapping chunks to keep
    memory bounded, and reuses the official Pi3XVO long-sequence pipeline (inlined here so we do
    not depend on the vendored `pi3.pipe` module): each chunk's global point cloud is aligned to
    the previous chunk's overlap region with a masked Umeyama Sim3 (scale + R + t). The chunks are
    chained, so each chunk's Sim3 is already the cumulative chunk-local -> global transform and its
    isotropic scale `s_k` (s_0 = 1) needs no further multiplication.

    Per-frame globally-consistent depth is synthesized as `depth_global = local_depth * s_k`, NOT
    by back-projecting the aligned global points into their cameras. The latter would cancel the
    Sim3 exactly (M'^-1 P' = (S M)^-1 (S M p) = p), returning the original local depth and wasting
    the alignment (see docs/decisions/0007). The cross-chunk consistency lives entirely in the per-
    chunk scale scalar `s_k`.
    """

    def __init__(self, model_sub: str = "") -> None:
        super().__init__()
        del model_sub
        if Pi3X is None or depth_normal_edge is None:
            raise RuntimeError("Pi3X is not found in the environment. Install the pi3x extras before using it.")
        if not torch.cuda.is_available():
            raise RuntimeError("Pi3XDepthModel requires CUDA")
        pretrained = _resolve_pi3x_pretrained()
        self.model = Pi3X.from_pretrained(pretrained, local_files_only=Path(pretrained).exists())
        self.model = self.model.cuda().eval()
        self.patch_size = _resolve_pi3x_patch_size(self.model)
        self.pixel_limit = max(_PI3X_PIXEL_LIMIT, self.patch_size**2)
        # Defaults match the official Pi3XVO pipeline (chunk 16 / overlap 6); env knobs override.
        self.chunk_size = max(int(os.environ.get("VIPE_PI3X_WINDOW_SIZE", "16")), 1)
        # overlap > 0 is required: the cross-chunk Sim3 fit needs an overlap region of
        # correspondences. overlap=0 would silently degrade to per-chunk identity (scale=1),
        # i.e. no alignment and hard seams — worse than the equal-weight baseline. Fail loud.
        self.overlap = int(os.environ.get("VIPE_PI3X_WINDOW_OVERLAP", "6"))
        if self.overlap <= 0:
            raise ValueError(
                f"VIPE_PI3X_WINDOW_OVERLAP must be > 0 for cross-chunk Sim3 alignment, got {self.overlap}"
            )
        self.inject_condition = _resolve_inject_condition()
        self.autocast_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    @property
    def depth_type(self) -> DepthType:
        return DepthType.AFFINE_DISP

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        frame_list = unpack_optional(src.video_frame_list)
        if not frame_list:
            raise ValueError("Pi3XDepthModel requires a non-empty video_frame_list")

        total_frames = len(frame_list)
        chunk_size = self.chunk_size
        overlap = min(self.overlap, max(chunk_size - 1, 0))
        step = max(chunk_size - overlap, 1)

        # Per-chunk outputs, already deduplicated against overlap and restored to source size,
        # concatenated in frame order to a contiguous [0, total_frames) sequence on CPU.
        inv_depth_chunks: list[torch.Tensor] = []
        conf_chunks: list[torch.Tensor] = []
        pose_chunks: list[torch.Tensor] = []
        emitted_frames = 0

        # Chained-alignment carry: previous chunk's overlap region, already in the global frame.
        prev_global_pts_overlap: torch.Tensor | None = None
        prev_global_mask_overlap: torch.Tensor | None = None
        # Condition-injection carry (overlap region of the previous chunk). Poses are in the global
        # frame (post-Sim3); local depth/conf/rays are chunk-local model outputs, which is what the
        # model's prior branches expect for the shared overlap frames.
        prev_aligned_poses_overlap: torch.Tensor | None = None
        prev_local_depth_overlap: torch.Tensor | None = None
        prev_local_conf_overlap: torch.Tensor | None = None
        prev_rays_overlap: torch.Tensor | None = None

        for start in range(0, total_frames, step):
            end = min(start + chunk_size, total_frames)
            current_len = end - start

            # Trailing chunk whose frames are all already covered by the previous chunk's overlap
            # region (mirrors Pi3XVO). These frames carry no new information; skipping them drops
            # nothing because [start, start+current_len) is a subset of the previous coverage.
            if current_len <= overlap and start > 0:
                break

            window_frames = frame_list[start:end]
            inputs, original_size = self._frames_to_tensor(window_frames)
            model_h, model_w = int(inputs.shape[-2]), int(inputs.shape[-1])

            # Condition injection (issue #24): feed the previous chunk's overlap-region priors
            # (pose / local depth / rays) into the multimodal branches so the new chunk actively
            # reconstructs toward the known global geometry, reducing drift and seams. Chunk 0 has
            # no prior; `with_prior=False` keeps the model in its plain feed-forward mode.
            model_kwargs: dict = {"with_prior": False}
            if start > 0:
                model_kwargs.update(
                    self._build_injection_kwargs(
                        current_len,
                        overlap,
                        model_h,
                        model_w,
                        inputs.device,
                        prev_aligned_poses_overlap,
                        prev_local_depth_overlap,
                        prev_local_conf_overlap,
                        prev_rays_overlap,
                    )
                )

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                outputs = self.model(inputs, **model_kwargs)

            # (B, N, H, W, 3) global points and local points; (B, N, H, W, 1) conf. B == 1 here.
            global_points = outputs["points"]
            local_points = outputs["local_points"]
            conf_raw = outputs.get("conf")
            camera_poses = outputs["camera_poses"]
            rays = outputs["rays"]

            local_depth = torch.clamp(local_points[..., 2], min=_PI3X_DEPTH_MIN)

            if conf_raw is None:
                conf = torch.ones_like(local_depth)
            else:
                conf = torch.sigmoid(conf_raw)[..., 0]
                # Drop depth/normal discontinuities so geometry breaks neither corrupt the Sim3
                # fit nor get reported as confident depth (Pi3XVO policy).
                valid = conf > _PI3X_CONF_THRE
                edge = depth_normal_edge(local_points, rtol=0.03, mask=valid)
                conf = conf.clone()
                conf[edge] = 0.0

            chunk_mask = self._confidence_mask(conf)

            if start == 0:
                # Chunk 0 defines the global frame: identity Sim3, s_0 = 1.
                aligned_pts = global_points
                aligned_poses = camera_poses
                scale = 1.0
            else:
                assert prev_global_pts_overlap is not None and prev_global_mask_overlap is not None
                src_pts = global_points[:, :overlap]
                src_mask = chunk_mask[:, :overlap]
                sim3, scale_tensor = self._compute_sim3_umeyama_masked(
                    src_pts, prev_global_pts_overlap, src_mask, prev_global_mask_overlap
                )
                # Chain the alignment forward: the next chunk fits against this chunk's overlap
                # expressed in the (now cumulative) global frame.
                aligned_pts = self._apply_sim3_to_points(global_points, sim3)
                aligned_poses = self._apply_sim3_to_poses(camera_poses, sim3)
                scale = float(scale_tensor.reshape(-1)[0].item())

            # Frames to emit: chunk 0 emits all; later chunks drop the overlap prefix (those
            # frames were already emitted by the previous chunk's prediction). local_depth and
            # scale are taken from the SAME chunk's SAME prediction, so the frame -> s_k mapping
            # cannot drift.
            emit_slice = slice(0, current_len) if start == 0 else slice(overlap, current_len)
            depth_global = local_depth[:, emit_slice] * scale
            inv_depth = torch.clamp(depth_global, min=_PI3X_DEPTH_MIN).reciprocal()
            conf_emit = conf[:, emit_slice]

            inv_depth = self._restore_map_size(inv_depth.squeeze(0), original_size)
            conf_emit = self._restore_map_size(conf_emit.squeeze(0), original_size)
            inv_depth_chunks.append(inv_depth.to(device="cpu", dtype=torch.float32))
            conf_chunks.append(conf_emit.to(device="cpu", dtype=torch.float32))
            pose_chunks.append(aligned_poses[0, emit_slice].to(device="cpu", dtype=torch.float32))
            emitted_frames += inv_depth.shape[0]

            if overlap > 0:
                prev_global_pts_overlap = aligned_pts[:, -overlap:]
                prev_global_mask_overlap = chunk_mask[:, -overlap:]
                prev_aligned_poses_overlap = aligned_poses[:, -overlap:]
                prev_local_depth_overlap = local_depth[:, -overlap:]
                prev_local_conf_overlap = conf[:, -overlap:]
                prev_rays_overlap = rays[:, -overlap:]
            else:
                prev_global_pts_overlap = aligned_pts[:, 0:0]
                prev_global_mask_overlap = chunk_mask[:, 0:0]

            if end == total_frames:
                break

        relative_inv_depth = torch.cat(inv_depth_chunks, dim=0)
        confidence = torch.cat(conf_chunks, dim=0)
        camera_poses_out = torch.cat(pose_chunks, dim=0)
        assert relative_inv_depth.shape[0] == total_frames, (
            f"Pi3X synthesized {relative_inv_depth.shape[0]} frames, expected {total_frames}"
        )
        return DepthEstimationResult(
            relative_inv_depth=relative_inv_depth,
            confidence=confidence,
            camera_poses=camera_poses_out,
        )

    def _build_injection_kwargs(
        self,
        current_len: int,
        overlap: int,
        model_h: int,
        model_w: int,
        device: torch.device,
        prev_aligned_poses_overlap: torch.Tensor | None,
        prev_local_depth_overlap: torch.Tensor | None,
        prev_local_conf_overlap: torch.Tensor | None,
        prev_rays_overlap: torch.Tensor | None,
    ) -> dict:
        """
        Assemble the multimodal prior kwargs for a non-first chunk (mirrors Pi3XVO injection).

        The previous chunk's overlap region is supplied as a prior on the new chunk's first
        `overlap` frames via per-frame boolean masks. Poses are the global (Sim3-aligned) overlap
        poses; depth/rays are the previous chunk's local model outputs. Priors live at the model
        resolution, which is constant across chunks under the issue #23 resolution contract.
        """
        B = 1
        kwargs: dict = {}
        conditions = self.inject_condition

        if "pose" in conditions and prev_aligned_poses_overlap is not None:
            prior_poses = torch.eye(4, device=device).repeat(B, current_len, 1, 1)
            prior_poses[:, :overlap] = prev_aligned_poses_overlap
            mask_pose = torch.zeros((B, current_len), dtype=torch.bool, device=device)
            mask_pose[:, :overlap] = True
            kwargs["poses"] = prior_poses
            kwargs["mask_add_pose"] = mask_pose
            kwargs["with_prior"] = True

        if "depth" in conditions and prev_local_depth_overlap is not None:
            prior_depths = torch.zeros((B, current_len, model_h, model_w), device=device)
            prior_depths[:, :overlap] = prev_local_depth_overlap
            if prev_local_conf_overlap is not None:
                valid = prev_local_conf_overlap > _PI3X_CONF_THRE
                prior_depths[:, :overlap][~valid] = 0
            mask_depth = torch.zeros((B, current_len), dtype=torch.bool, device=device)
            mask_depth[:, :overlap] = True
            kwargs["depths"] = prior_depths
            kwargs["mask_add_depth"] = mask_depth
            kwargs["with_prior"] = True

        if ("ray" in conditions or "intrinsic" in conditions) and prev_rays_overlap is not None:
            prior_rays = torch.zeros((B, current_len, model_h, model_w, 3), device=device)
            prior_rays[:, :overlap] = prev_rays_overlap
            mask_ray = torch.zeros((B, current_len), dtype=torch.bool, device=device)
            mask_ray[:, :overlap] = True
            kwargs["rays"] = prior_rays
            kwargs["mask_add_ray"] = mask_ray
            kwargs["with_prior"] = True

        return kwargs

    def _confidence_mask(self, conf: torch.Tensor) -> torch.Tensor:
        """
        Boolean validity mask for the Sim3 fit.

        Thresholds confidence at `_PI3X_CONF_THRE`; if a chunk is almost entirely masked out
        (fewer than `_PI3X_MIN_VALID` valid views total), falls back to the top-10% confident
        pixels so the fit still has correspondences (Pi3XVO policy).
        """
        B, N = conf.shape[0], conf.shape[1]
        mask = conf > _PI3X_CONF_THRE
        if mask.sum() >= _PI3X_MIN_VALID:
            return mask
        flat_conf = conf.reshape(B, N, -1)
        k = max(int(flat_conf.shape[-1] * 0.1), 1)
        topk_vals, _ = torch.topk(flat_conf, k, dim=-1)
        min_vals = topk_vals[..., -1].unsqueeze(-1).unsqueeze(-1)
        return conf >= min_vals

    def _compute_sim3_umeyama_masked(
        self,
        src_points: torch.Tensor,
        tgt_points: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Masked Umeyama Sim3 (scale + R + t) mapping src points onto tgt points.

        Returns both the (B, 4, 4) Sim3 matrix and the isotropic scale scalar (B, 1, 1). The
        scale is the cross-chunk consistency carrier the depth synthesis multiplies into local
        depth; it is returned directly rather than recovered via det(sim3[:3,:3])**(1/3).
        Degenerate fits (too few valid correspondences) fall back to identity / unit scale.
        """
        B = src_points.shape[0]
        device = src_points.device

        src = src_points.reshape(B, -1, 3).float()
        tgt = tgt_points.reshape(B, -1, 3).float()

        mask = (src_mask.reshape(B, -1) & tgt_mask.reshape(B, -1)).float().unsqueeze(-1)
        valid_cnt = mask.sum(dim=1).squeeze(-1)
        eps = 1e-6

        bad_mask = valid_cnt < _PI3X_MIN_VALID
        if bad_mask.all():
            sim3 = torch.eye(4, device=device).repeat(B, 1, 1)
            return sim3, torch.ones(B, 1, 1, device=device)

        src_mean = (src * mask).sum(dim=1, keepdim=True) / (valid_cnt.view(B, 1, 1) + eps)
        tgt_mean = (tgt * mask).sum(dim=1, keepdim=True) / (valid_cnt.view(B, 1, 1) + eps)

        src_centered = (src - src_mean) * mask
        tgt_centered = (tgt - tgt_mean) * mask

        cov = torch.bmm(src_centered.transpose(1, 2), tgt_centered)
        U, S, V = torch.svd(cov)

        R = torch.bmm(V, U.transpose(1, 2))
        det = torch.det(R)
        diag = torch.ones(B, 3, device=device)
        diag[:, 2] = torch.sign(det)
        R = torch.bmm(torch.bmm(V, torch.diag_embed(diag)), U.transpose(1, 2))

        src_var = (src_centered**2).sum(dim=2) * mask.squeeze(-1)
        src_var = src_var.sum(dim=1) / (valid_cnt + eps)

        corrected_S = S.clone()
        corrected_S[:, 2] *= diag[:, 2]
        trace_S = corrected_S.sum(dim=1)

        scale = trace_S / (src_var * valid_cnt + eps)
        scale = scale.view(B, 1, 1)

        t = tgt_mean.transpose(1, 2) - scale * torch.bmm(R, src_mean.transpose(1, 2))

        sim3 = torch.eye(4, device=device).repeat(B, 1, 1)
        sim3[:, :3, :3] = scale * R
        sim3[:, :3, 3] = t.squeeze(2)

        if bad_mask.any():
            identity = torch.eye(4, device=device).repeat(B, 1, 1)
            sim3[bad_mask] = identity[bad_mask]
            scale[bad_mask] = 1.0

        return sim3, scale

    def _apply_sim3_to_points(self, points: torch.Tensor, sim3: torch.Tensor) -> torch.Tensor:
        B, T, H, W, _ = points.shape
        flat_pts = points.reshape(B, -1, 3).float()
        R_s = sim3[:, :3, :3]
        t = sim3[:, :3, 3].unsqueeze(1)
        out_pts = torch.bmm(flat_pts, R_s.transpose(1, 2)) + t
        return out_pts.reshape(B, T, H, W, 3)

    def _apply_sim3_to_poses(self, poses: torch.Tensor, sim3: torch.Tensor) -> torch.Tensor:
        """
        Left-multiply camera-to-world poses by the chunk's cumulative Sim3 (mirrors Pi3XVO).

        The returned matrices carry the isotropic scale in their rotation block, so they are
        similarity transforms, not rigid SE(3); downstream SLAM use must factor out the scale.
        """
        sim3 = sim3.to(poses.dtype)
        return torch.matmul(sim3.unsqueeze(1), poses)

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


def _resolve_inject_condition() -> set[str]:
    raw = os.environ.get("VIPE_PI3X_INJECT_CONDITION", _PI3X_DEFAULT_INJECT)
    if raw.strip().lower() in ("none", "off", "false", ""):
        return set()
    conditions = {tok.strip().lower() for tok in raw.split(",") if tok.strip()}
    invalid = conditions - _PI3X_VALID_INJECT
    if invalid:
        raise ValueError(
            f"VIPE_PI3X_INJECT_CONDITION has unknown entries {sorted(invalid)}; "
            f"valid values are {sorted(_PI3X_VALID_INJECT)} or none"
        )
    return conditions


def _resolve_pi3x_patch_size(model: torch.nn.Module) -> int:
    patch_embed = getattr(getattr(model, "encoder", None), "patch_embed", None)
    patch_size = getattr(patch_embed, "patch_size", 14)
    if isinstance(patch_size, tuple):
        return int(patch_size[0])
    return int(patch_size)
