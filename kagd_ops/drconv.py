"""Dynamic Rectangular Convolution (DRConv).

The implementation follows Equations (4)--(7) and Algorithm 1 of the KAGD-Net
manuscript. It retains the fixed 3 x 5 lattice (K=15), predicts continuous
rectangular extents from axis-wise statistics, refines the lattice with bounded
offsets, and aggregates bilinearly sampled values with masked softmax weights.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class _ExtentBranch(nn.Sequential):
    """1 x 1 channel reduction -> SiLU -> 1 x 1 scalar projection."""

    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
        )


class DRConv(nn.Module):
    """Dynamic rectangular convolution with a fixed-cardinality sampling lattice.

    Args:
        channels: Number of input and output feature channels.
        hidden_channels: Hidden width of the extent branches. By default the
            manuscript setting ``max(8, floor(channels / 4))`` is used.
        height_max: Maximum predicted reference height in feature-grid pixels.
        width_max: Maximum predicted reference width in feature-grid pixels.
        offset_max: Maximum learned offset in feature-grid pixels.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int | None = None,
        height_max: float = 5.0,
        width_max: float = 9.0,
        offset_max: float = 0.5,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if height_max <= 0 or width_max <= 0:
            raise ValueError("height_max and width_max must be positive")
        if offset_max < 0:
            raise ValueError(f"offset_max must be non-negative, got {offset_max}")

        self.channels = int(channels)
        self.hidden_channels = int(
            hidden_channels if hidden_channels is not None else max(8, channels // 4)
        )
        if self.hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")

        self.height_max = float(height_max)
        self.width_max = float(width_max)
        self.offset_max = float(offset_max)

        eta = (-0.5, 0.0, 0.5)
        xi = (-0.5, -0.25, 0.0, 0.25, 0.5)
        self.register_buffer("eta", torch.tensor(eta, dtype=torch.float32))
        self.register_buffer("xi", torch.tensor(xi, dtype=torch.float32))
        self.sampling_points = len(eta) * len(xi)
        self.center_index = eta.index(0.0) * len(xi) + xi.index(0.0)
        if self.center_index != (self.sampling_points - 1) // 2:
            raise RuntimeError("the 3 x 5 lattice must have one central point")

        self.embedding = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(self.channels),
            nn.SiLU(inplace=True),
        )
        self.height_branch = _ExtentBranch(
            self.channels,
            self.hidden_channels,
        )
        self.width_branch = _ExtentBranch(
            self.channels,
            self.hidden_channels,
        )
        self.row_projection = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=1,
            bias=True,
        )
        self.column_projection = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=1,
            bias=True,
        )
        self.offset_head = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                self.channels,
                2 * self.sampling_points,
                kernel_size=1,
                bias=True,
            ),
        )
        self.weight_head = nn.Conv2d(
            self.channels,
            self.sampling_points,
            kernel_size=1,
            bias=True,
        )
        self.output_projection = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(self.channels),
            nn.SiLU(inplace=True),
        )
        self.reset_offsets()

    def reset_offsets(self) -> None:
        """Zero the final offset projection, as specified by Algorithm 1."""
        final_projection = self.offset_head[-1]
        if not isinstance(final_projection, nn.Conv2d):
            raise TypeError("DRConv final offset projection must be Conv2d")
        nn.init.zeros_(final_projection.weight)
        nn.init.zeros_(final_projection.bias)

    def _embed(self, x: Tensor) -> Tensor:
        return self.embedding(x)

    @staticmethod
    def _axis_profiles(feature: Tensor) -> tuple[Tensor, Tensor]:
        # B x C x H x 1 and B x C x 1 x W.
        row_profile = feature.mean(dim=3, keepdim=True)
        column_profile = feature.mean(dim=2, keepdim=True)
        return row_profile, column_profile

    def _window_extents(
        self,
        row_profile: Tensor,
        column_profile: Tensor,
        height: int,
        width: int,
    ) -> tuple[Tensor, Tensor]:
        pred_height = self.height_max * torch.sigmoid(
            self.height_branch(row_profile)
        )
        pred_width = self.width_max * torch.sigmoid(
            self.width_branch(column_profile)
        )
        return (
            pred_height.expand(-1, -1, height, width),
            pred_width.expand(-1, -1, height, width),
        )

    def _context(
        self,
        feature: Tensor,
        row_profile: Tensor,
        column_profile: Tensor,
    ) -> Tensor:
        return (
            feature
            + self.row_projection(row_profile)
            + self.column_projection(column_profile)
        )

    def _refined_offsets(self, context: Tensor) -> Tensor:
        """Return B x K x 2 x H x W offsets in feature-grid pixels.

        The last dimension stores (dy, dx). The central point is fixed at zero.
        """
        offsets = self.offset_head(context)
        batch, _, height, width = context.shape
        offsets = offsets.view(
            batch,
            self.sampling_points,
            2,
            height,
            width,
        )
        offsets = self.offset_max * torch.tanh(offsets)
        offsets = offsets.clone()
        offsets[:, self.center_index].zero_()
        return offsets

    def forward(self, x: Tensor) -> Tensor:
        """Apply DRConv while preserving the input spatial and channel dimensions."""
        if x.ndim != 4:
            raise ValueError(f"DRConv expects a 4-D tensor, got shape {tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(
                f"expected {self.channels} channels, got {x.shape[1]}"
            )
        if not x.is_floating_point():
            raise TypeError("DRConv expects a floating-point tensor")
        batch, _, height, width = x.shape
        if height < 1 or width < 1:
            raise ValueError("spatial dimensions must be positive")

        feature = self._embed(x)
        row_profile, column_profile = self._axis_profiles(feature)
        pred_height, pred_width = self._window_extents(
            row_profile,
            column_profile,
            height,
            width,
        )
        context = self._context(feature, row_profile, column_profile)
        refined_offsets = self._refined_offsets(context)

        eta = self.eta.to(dtype=x.dtype).repeat_interleave(self.xi.numel())
        xi = self.xi.to(dtype=x.dtype).repeat(self.eta.numel())
        reference_y = pred_height[:, 0].unsqueeze(1) * eta.view(
            1,
            -1,
            1,
            1,
        )
        reference_x = pred_width[:, 0].unsqueeze(1) * xi.view(
            1,
            -1,
            1,
            1,
        )

        base_y = torch.arange(height, device=x.device, dtype=x.dtype).view(
            1,
            1,
            height,
            1,
        )
        base_x = torch.arange(width, device=x.device, dtype=x.dtype).view(
            1,
            1,
            1,
            width,
        )
        source_y = base_y + reference_y + refined_offsets[:, :, 0]
        source_x = base_x + reference_x + refined_offsets[:, :, 1]

        valid_mask = (
            (source_y >= 0)
            & (source_y <= height - 1)
            & (source_x >= 0)
            & (source_x <= width - 1)
        )

        def to_normalized(source: Tensor, size: int) -> Tensor:
            # align_corners=False: pixel coordinate p maps to (2p + 1) / size - 1.
            return (2.0 * source + 1.0) / size - 1.0

        grid_y = to_normalized(source_y, height)
        grid_x = to_normalized(source_x, width)
        sampling_grid = torch.stack((grid_x, grid_y), dim=-1)
        sampling_grid = sampling_grid.reshape(
            batch,
            self.sampling_points * height,
            width,
            2,
        )

        sampled = F.grid_sample(
            feature,
            sampling_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.view(
            batch,
            self.channels,
            self.sampling_points,
            height,
            width,
        )
        sampled = sampled * valid_mask.unsqueeze(1).to(sampled.dtype)

        logits = self.weight_head(context)
        masked_logits = logits.masked_fill(
            ~valid_mask,
            torch.finfo(logits.dtype).min,
        )
        weights = torch.softmax(masked_logits, dim=1)
        weights = weights * valid_mask.to(weights.dtype)
        normalizer = weights.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).eps
        )
        weights = weights / normalizer

        aggregated = (weights.unsqueeze(1) * sampled).sum(dim=2)
        return self.output_projection(aggregated)
