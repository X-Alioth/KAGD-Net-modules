"""Adaptive Resampling Sampler (ARSample).

This implementation follows Section 3.4 of the KAGD-Net manuscript:

* 3 x 3 convolution -> SiLU -> 1 x 1 projection -> PixelShuffle(2)
* zero-initialized final offset projection
* offsets bounded to +/-0.5 input-feature pixels
* bilinear sampling with border-replication padding
* align_corners=False throughout

ARSample uses the same offset-based resampling principle as DySample. The code
implements the manuscript's integration, not a new sampling principle.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ARSample(nn.Module):
    """Two-times adaptive bilinear upsampling.

    Args:
        channels: Number of input and output feature channels.
        scale_factor: Spatial upsampling factor. KAGD-Net uses 2.
        max_offset: Maximum learned displacement in input-feature pixels.
    """

    def __init__(
        self,
        channels: int,
        scale_factor: int = 2,
        max_offset: float = 0.5,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if scale_factor <= 0:
            raise ValueError(
                f"scale_factor must be positive, got {scale_factor}"
            )
        if max_offset < 0:
            raise ValueError(f"max_offset must be non-negative, got {max_offset}")

        self.channels = int(channels)
        self.scale_factor = int(scale_factor)
        self.max_offset = float(max_offset)

        projection_channels = 2 * self.scale_factor * self.scale_factor
        self.coordinate_generator = nn.Sequential(
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
                projection_channels,
                kernel_size=1,
                bias=True,
            ),
        )
        self.pixel_shuffle = nn.PixelShuffle(self.scale_factor)
        self.reset_offsets()

    def reset_offsets(self) -> None:
        """Restore zero offsets, making the module initially a bilinear upsampler."""
        final_projection = self.coordinate_generator[-1]
        if not isinstance(final_projection, nn.Conv2d):
            raise TypeError("ARSample final coordinate projection must be Conv2d")
        nn.init.zeros_(final_projection.weight)
        nn.init.zeros_(final_projection.bias)

    def _base_grid(
        self,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Return the align_corners=False target lattice as normalized (x, y)."""
        y = (
            (torch.arange(height, device=device, dtype=dtype) + 0.5)
            * (2.0 / height)
            - 1.0
        )
        x = (
            (torch.arange(width, device=device, dtype=dtype) + 0.5)
            * (2.0 / width)
            - 1.0
        )
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=-1)

    def _predict_pixel_offsets(self, x: Tensor) -> Tensor:
        """Predict bounded offsets in input-feature pixels, shaped B x 2 x Hout x Wout."""
        raw = self.coordinate_generator(x)
        raw = self.pixel_shuffle(raw)
        return self.max_offset * torch.tanh(raw)

    def forward(self, x: Tensor) -> Tensor:
        """Upsample x while learning a bounded spatial displacement per output site.

        Args:
            x: Feature map with shape ``(B, C, H, W)``.

        Returns:
            Tensor with shape ``(B, C, scale_factor * H, scale_factor * W)``.
        """
        if x.ndim != 4:
            raise ValueError(f"ARSample expects a 4-D tensor, got shape {tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(
                f"expected {self.channels} channels, got {x.shape[1]}"
            )
        if not x.is_floating_point():
            raise TypeError("ARSample expects a floating-point tensor")

        batch, _, height, width = x.shape
        out_height = height * self.scale_factor
        out_width = width * self.scale_factor

        offset_pixels = self._predict_pixel_offsets(x)
        if offset_pixels.shape[-2:] != (out_height, out_width):
            raise RuntimeError("coordinate generator produced an unexpected shape")

        # Offsets are in input-feature pixels. grid_sample expects normalized
        # displacements: 2 / spatial_size per input pixel under align_corners=False.
        offset_norm = torch.stack(
            (
                offset_pixels[:, 0] * (2.0 / width),
                offset_pixels[:, 1] * (2.0 / height),
            ),
            dim=-1,
        )
        base_grid = self._base_grid(
            out_height,
            out_width,
            device=x.device,
            dtype=x.dtype,
        )
        sampling_grid = base_grid.unsqueeze(0) + offset_norm

        return F.grid_sample(
            x,
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
