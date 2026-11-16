from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from kagd_ops import ARSample, DRConv


@pytest.fixture()
def feature_map() -> torch.Tensor:
    generator = torch.Generator().manual_seed(7)
    return torch.randn(2, 8, 12, 10, generator=generator)


def test_arsample_output_shape(feature_map: torch.Tensor) -> None:
    model = ARSample(channels=8, scale_factor=2)
    output = model(feature_map)

    assert output.shape == (2, 8, 24, 20)
    assert output.dtype == feature_map.dtype


def test_arsample_zero_offsets_match_bilinear(feature_map: torch.Tensor) -> None:
    model = ARSample(channels=8, scale_factor=2)
    model.eval()

    actual = model(feature_map)
    expected = F.interpolate(
        feature_map,
        scale_factor=2,
        mode="bilinear",
        align_corners=False,
    )

    assert torch.allclose(actual, expected, atol=5e-6, rtol=5e-6)


def test_drconv_output_shape_and_gradients(feature_map: torch.Tensor) -> None:
    model = DRConv(channels=8)
    tensor = feature_map.detach().clone().requires_grad_(True)

    output = model(tensor)
    output.mean().backward()

    assert output.shape == feature_map.shape
    assert tensor.grad is not None
    assert torch.isfinite(output).all()
    assert torch.isfinite(tensor.grad).all()


@pytest.mark.parametrize("channels", [1, 3, 8])
def test_drconv_supports_reference_and_small_channel_counts(
    channels: int,
) -> None:
    x = torch.randn(1, channels, 6, 7)
    model = DRConv(channels=channels)

    assert model(x).shape == x.shape


def test_invalid_channel_count_is_rejected() -> None:
    with pytest.raises(ValueError):
        ARSample(channels=0)
    with pytest.raises(ValueError):
        DRConv(channels=0)


def test_forward_rejects_non_4d_input() -> None:
    x = torch.randn(2, 8, 10)
    with pytest.raises(ValueError):
        ARSample(channels=8)(x)
    with pytest.raises(ValueError):
        DRConv(channels=8)(x)
