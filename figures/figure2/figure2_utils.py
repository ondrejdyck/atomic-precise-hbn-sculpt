"""Shared utilities for Figure 2 figure-generation scripts."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def configure_matplotlib(outdir: Path, show: bool = False) -> None:
    """Configure matplotlib for headless or interactive execution."""
    outdir.mkdir(parents=True, exist_ok=True)
    mplconfig = outdir / ".mplconfig"
    mplconfig.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mplconfig))
    if not show:
        import matplotlib

        matplotlib.use("Agg", force=True)


def square_roi_fractional(
    unitcell: np.ndarray,
    target_fraction: float = 0.68,
    x_shift: float = 0.05,
) -> list[float]:
    """Return a square ROI in fractional coordinates.

    The side length is chosen to preserve the approximate area of the original
    375/512-by-375/512 crop used in the notebook, while ensuring the real-space
    sampling is square. `x_shift` nudges the crop horizontally in fractional
    coordinates so the interesting features can be centered a little better.
    The returned ROI is [y0, x0, y1, x1].
    """
    x_size, y_size = map(float, unitcell[:2])
    side = float(target_fraction) * float(np.sqrt(x_size * y_size))
    span_y = side / x_size
    span_x = side / y_size
    y0 = 0.5 * (1.0 - span_y)
    x0 = 0.5 * (1.0 - span_x) + float(x_shift)
    x0 = min(max(x0, 0.0), 1.0 - span_x)
    return [y0, x0, y0 + span_y, x0 + span_x]


def interp_image(pyms_module, im, tiling, interp_shape=(512, 512)):
    """Fourier-interpolate and tile a simulated image for display."""
    return pyms_module.utils.fourier_interpolate(np.tile(im, tiling), interp_shape)
