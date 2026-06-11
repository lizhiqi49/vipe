# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch


def window_starts(total_frames: int, window_size: int, window_overlap: int) -> list[int]:
    """
    Sliding-window start indices over a temporal sequence.

    Windows of length ``window_size`` step by ``window_size - window_overlap``; the final
    start is snapped to ``total_frames - window_size`` so the last frame is always covered.
    Shared verbatim by the Pi3X and VGGT-Omega multi-frame depth backends.
    """
    if total_frames <= window_size:
        return [0]
    step = max(window_size - window_overlap, 1)
    starts = list(range(0, total_frames - window_size + 1, step))
    if starts[-1] != total_frames - window_size:
        starts.append(total_frames - window_size)
    return starts


class WindowedDepthAccumulator:
    """
    Equal-weight overlap accumulator for windowed depth/confidence on CPU.

    Each window contributes its per-frame inverse depth and confidence into global CPU
    buffers; overlapping frames are averaged with equal weight (``sum / max(count, 1)``).
    Buffers are lazily allocated from the first window so backends need not know the
    spatial map size up front. The accumulator holds one full-sequence buffer on CPU while
    GPU memory only ever carries a single window — this is the memory-bounding contract the
    sliding window exists to satisfy.

    NOTE (issue #23): equal-weight averaging is intentionally preserved this round.
    Confidence/position-weighted fusion and cross-window scale alignment are deferred until
    GPU benchmark data is available.
    """

    def __init__(self, total_frames: int) -> None:
        self.total_frames = total_frames
        self._depth_sum: torch.Tensor | None = None
        self._conf_sum: torch.Tensor | None = None
        self._count: torch.Tensor | None = None

    def add(self, start: int, end: int, inv_depth: torch.Tensor, confidence: torch.Tensor) -> None:
        inv_depth = inv_depth.to(device="cpu", dtype=torch.float32)
        confidence = confidence.to(device="cpu", dtype=torch.float32)
        if self._depth_sum is None:
            shape = (self.total_frames,) + tuple(inv_depth.shape[1:])
            self._depth_sum = torch.zeros(shape, device="cpu", dtype=torch.float32)
            self._conf_sum = torch.zeros_like(self._depth_sum)
            self._count = torch.zeros_like(self._depth_sum)

        assert self._depth_sum is not None
        assert self._conf_sum is not None
        assert self._count is not None
        self._depth_sum[start:end] += inv_depth
        self._conf_sum[start:end] += confidence
        self._count[start:end] += 1

    def result(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._depth_sum is None:
            raise RuntimeError("WindowedDepthAccumulator received no windows")
        assert self._conf_sum is not None
        assert self._count is not None
        denom = torch.clamp(self._count, min=1)
        return self._depth_sum / denom, self._conf_sum / denom
