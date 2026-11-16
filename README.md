# KAGD-Net Modules

[![Tests](https://github.com/X-Alioth/KAGD-Net-modules/actions/workflows/tests.yml/badge.svg)](https://github.com/X-Alioth/KAGD-Net-modules/actions/workflows/tests.yml)

Reference PyTorch implementations of two operators used in KAGD-Net:

| Module | Purpose | Output shape |
| --- | --- | --- |
| `ARSample` | Adaptive bilinear upsampling with learned, bounded sampling offsets | `(B, C, sH, sW)` |
| `DRConv` | Dynamic rectangular convolution with a fixed 3 x 5 sampling lattice | `(B, C, H, W)` |

The code is self-contained and does not depend on the rest of the KAGD-Net repository.

## Requirements

- Python 3.10 or newer
- PyTorch 2.0 or newer

## Installation

Install directly from this repository:

```bash
git clone https://github.com/X-Alioth/KAGD-Net-modules.git
cd KAGD-Net-modules
pip install -e .
```

For a local checkout, install the requirements without packaging:

```bash
pip install -r requirements.txt
```

## Quick start

```python
import torch
from kagd_ops import ARSample, DRConv

batch, channels, height, width = 2, 64, 64, 64
x = torch.randn(batch, channels, height, width)

upsample = ARSample(channels=channels, scale_factor=2)
dynamic_rect_conv = DRConv(channels=channels)

y = upsample(x)             # (2, 64, 128, 128)
z = dynamic_rect_conv(y)    # (2, 64, 128, 128)
```

## Modules

### `ARSample`

`ARSample` performs a two-times adaptive bilinear upsample:

1. A 3 x 3 convolution and SiLU activation extract local context.
2. A 1 x 1 projection predicts horizontal and vertical offsets.
3. `PixelShuffle(2)` rearranges the offsets to the target resolution.
4. The offsets are bounded with `tanh` and used with `grid_sample`.

The final offset projection is zero-initialized, so a freshly initialized module behaves as bilinear interpolation.

```python
from kagd_ops import ARSample

model = ARSample(
    channels=128,
    scale_factor=2,  # KAGD-Net uses 2
    max_offset=0.5,  # input-feature pixels
)
```

### `DRConv`

`DRConv` preserves the input spatial and channel dimensions. It predicts continuous rectangular extents from row/column feature statistics, refines a fixed 3 x 5 lattice with bounded offsets, and aggregates sampled features using masked softmax weights.

```python
from kagd_ops import DRConv

model = DRConv(
    channels=128,
    hidden_channels=None,  # max(8, channels // 4)
    height_max=5.0,
    width_max=9.0,
    offset_max=0.5,
)
```

The central sampling point is fixed at zero offset, and out-of-bounds samples are excluded before softmax normalization.

## Tests

The test suite covers output shapes, zero-offset initialization, differentiability, input validation, and CPU execution.

```bash
pip install pytest
python -m pytest -q
```

## Project structure

```text
KAGD-Net-modules/
├── kagd_ops/
│   ├── __init__.py
│   ├── arsample.py
│   └── drconv.py
├── tests/
│   └── test_modules.py
├── .gitignore
├── pyproject.toml
└── requirements.txt
```

## Provenance

- `DRConv` corresponds to Equations (4)-(7) and Algorithm 1 described in the KAGD-Net manuscript.
- `ARSample` follows Section 3.4 of the manuscript.
- `ARSample` shares the offset-based resampling principle of DySample; this repository provides the manuscript-specific integration rather than claiming a new sampling principle.

No dataset, pretrained weights, or additional KAGD-Net integration code are included.
