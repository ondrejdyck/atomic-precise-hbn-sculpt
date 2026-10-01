"""Estimate an effective STEM probe width from the figure-2 multislice setup.

This script is a notebook-to-script conversion for the twisted h-BN simulation
workflow. It does *not* perform a fit to experimental data. Instead, it
constructs the coherent STEM probe used by py_multislice, applies a chosen
Gaussian broadening in real space, and measures a line-profile HWHM from the
resulting probe intensity.

The broadening parameter is meant as a phenomenological probe-width proxy.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


DEFAULT_STRUCTURE = "hbn_bilayer_8deg_v2.p1"
DEFAULT_GRID = 1024
DEFAULT_ELECTRON_VOLTAGE = 100_000.0
DEFAULT_APERTURE_MRAD = 30.0
DEFAULT_DEFLECT = 0.0
DEFAULT_SMOOTHING_NM = 0.246  # physical width corresponding to the earlier image blur estimate


def square_window_from_unitcell(unitcell_nm: np.ndarray) -> tuple[np.ndarray, float]:
    """Return a square real-space window centered on a rectangular unit cell.

    The returned ROI is expressed in nanometers as [x0, y0, x1, y1] and uses
    the smaller unit-cell side length so that the sampled field of view is
    square in real-space units.
    """
    side = float(min(unitcell_nm[0], unitcell_nm[1]))
    x0 = 0.5 * (float(unitcell_nm[0]) - side)
    y0 = 0.5 * (float(unitcell_nm[1]) - side)
    roi_nm = np.array([x0, y0, x0 + side, y0 + side], dtype=float)
    return roi_nm, side


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--structure",
        type=Path,
        default=Path(__file__).resolve().parent / "inputs" / DEFAULT_STRUCTURE,
        help="Path to the twisted h-BN .p1 structure file.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for saved plots.",
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=DEFAULT_GRID,
        help="Probe grid size in pixels for both axes (default: 1024).",
    )
    parser.add_argument(
        "--voltage",
        type=float,
        default=DEFAULT_ELECTRON_VOLTAGE,
        help="Accelerating voltage in eV (default: 100000).",
    )
    parser.add_argument(
        "--aperture",
        type=float,
        default=DEFAULT_APERTURE_MRAD,
        help="Probe-forming aperture in mrad (default: 30).",
    )
    parser.add_argument(
        "--defocus",
        type=float,
        default=DEFAULT_DEFLECT,
        help="Probe defocus in Angstrom (default: 0).",
    )
    parser.add_argument(
        "--sigma-nm",
        type=float,
        default=DEFAULT_SMOOTHING_NM,
        help=(
            "Phenomenological Gaussian broadening width in nm. This is the physical "
            "width to apply to the probe intensity before measuring the HWHM."
        ),
    )
    show_group = parser.add_mutually_exclusive_group()
    show_group.add_argument(
        "--show",
        dest="show",
        action="store_true",
        help="Display the figure interactively.",
    )
    show_group.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="Do not display the figure interactively (default).",
    )
    parser.set_defaults(show=False)
    return parser.parse_args()


def hwhm_from_profile(x_nm: np.ndarray, y: np.ndarray) -> float:
    """Return the half-width at half-maximum from a 1D profile."""
    x_nm = np.asarray(x_nm)
    y = np.asarray(y)
    peak_idx = int(np.argmax(y))
    peak = float(y[peak_idx])
    half = peak / 2.0

    # Search to the right of the peak.
    right = None
    for i in range(peak_idx, len(y) - 1):
        if y[i + 1] <= half <= y[i]:
            x0, x1 = x_nm[i], x_nm[i + 1]
            y0, y1 = y[i], y[i + 1]
            frac = 0.0 if y1 == y0 else (half - y0) / (y1 - y0)
            right = float(x0 + frac * (x1 - x0))
            break

    # Search to the left of the peak.
    left = None
    for i in range(peak_idx, 0, -1):
        if y[i - 1] <= half <= y[i]:
            x0, x1 = x_nm[i - 1], x_nm[i]
            y0, y1 = y[i - 1], y[i]
            frac = 0.0 if y1 == y0 else (half - y0) / (y1 - y0)
            left = float(x0 + frac * (x1 - x0))
            break

    if left is None or right is None:
        raise RuntimeError("Could not determine HWHM from the profile.")

    return 0.5 * (right - left)


def main() -> int:
    args = parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.outdir / ".mplconfig"))
    (args.outdir / ".mplconfig").mkdir(parents=True, exist_ok=True)

    # Import matplotlib only after MPLCONFIGDIR is set.
    if not args.show:
        import matplotlib

        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    import pyms
    from pyms.Probe import focused_probe

    structure = pyms.structure.fromfile(
        args.structure, temperature_factor_units="urms", atomic_coordinates="fractional"
    )

    # The probe grid used in the notebook.
    gridshape = (args.grid, args.grid)
    real_size_angstrom = np.asarray(structure.unitcell[:2], dtype=float)
    real_size_nm = real_size_angstrom / 10.0
    square_roi_nm, square_side_nm = square_window_from_unitcell(real_size_nm)
    pixel_size_nm = np.array([square_side_nm, square_side_nm], dtype=float) / np.asarray(gridshape, dtype=float)

    # Construct the coherent probe wavefunction.
    probe = focused_probe(
        gridshape=gridshape,
        rsize=real_size_angstrom,
        eV=args.voltage,
        app=args.aperture,
        df=args.defocus,
    )
    intensity = np.abs(probe) ** 2

    # Move the probe from FFT corner ordering into centered real-space ordering
    # before applying any additional blur, otherwise the Gaussian kernel will
    # wrap across the periodic array boundaries and create quadrant artifacts.
    intensity_centered = np.fft.fftshift(intensity)

    # Apply a physical Gaussian broadening to the centered probe intensity.
    sigma_nm = float(args.sigma_nm)
    sigma_px_y = sigma_nm / pixel_size_nm[0]
    sigma_px_x = sigma_nm / pixel_size_nm[1]
    broadened_centered = gaussian_filter(
        intensity_centered,
        sigma=(sigma_px_y, sigma_px_x),
        mode="reflect",
    )

    center = (gridshape[0] // 2, gridshape[1] // 2)

    # Real-space axes centered on the probe origin.
    y_nm = (np.arange(gridshape[0]) - center[0]) * pixel_size_nm[0]
    x_nm = (np.arange(gridshape[1]) - center[1]) * pixel_size_nm[1]

    # Center line profiles.
    y_profile = broadened_centered[:, center[1]]
    x_profile = broadened_centered[center[0], :]

    hwhm_x = hwhm_from_profile(x_nm, x_profile)
    hwhm_y = hwhm_from_profile(y_nm, y_profile)
    hwhm_mean = 0.5 * (hwhm_x + hwhm_y)

    # Print a compact summary for quick inspection.
    print(f"structure: {args.structure}")
    print(f"unitcell (A): {real_size_angstrom[0]:.4f} x {real_size_angstrom[1]:.4f}")
    print(f"square ROI (nm): x0={square_roi_nm[0]:.4f}, y0={square_roi_nm[1]:.4f}, x1={square_roi_nm[2]:.4f}, y1={square_roi_nm[3]:.4f}")
    print(f"square side (nm): {square_side_nm:.4f}")
    print(f"probe grid: {gridshape[0]} x {gridshape[1]}")
    print(
        f"pixel size (nm): y={pixel_size_nm[0]:.6f}, x={pixel_size_nm[1]:.6f}"
    )
    print(f"applied Gaussian sigma (nm): {sigma_nm:.6f}")
    print(
        f"applied Gaussian sigma (px): y={sigma_px_y:.2f}, x={sigma_px_x:.2f}"
    )
    print(f"measured HWHM from broadened probe (nm): x={hwhm_x:.6f}, y={hwhm_y:.6f}")
    print(f"mean HWHM (nm): {hwhm_mean:.6f}")

    # Plot the centered probe and the line profiles.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    im0 = axes[0].imshow(
        intensity_centered,
        cmap="magma",
        origin="lower",
        extent=[x_nm[0], x_nm[-1], y_nm[0], y_nm[-1]],
    )
    axes[0].set_title("Coherent probe intensity")
    axes[0].set_xlabel("x (nm)")
    axes[0].set_ylabel("y (nm)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(
        broadened_centered,
        cmap="magma",
        origin="lower",
        extent=[x_nm[0], x_nm[-1], y_nm[0], y_nm[-1]],
    )
    axes[1].set_title("Broadened probe intensity")
    axes[1].set_xlabel("x (nm)")
    axes[1].set_ylabel("y (nm)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].plot(x_nm, x_profile / x_profile.max(), label="x profile")
    axes[2].plot(y_nm, y_profile / y_profile.max(), label="y profile")
    axes[2].axhline(0.5, color="k", ls="--", lw=1)
    axes[2].axvline(hwhm_x, color="C0", ls=":", lw=1)
    axes[2].axvline(-hwhm_x, color="C0", ls=":", lw=1)
    axes[2].axvline(hwhm_y, color="C1", ls=":", lw=1)
    axes[2].axvline(-hwhm_y, color="C1", ls=":", lw=1)
    axes[2].set_title("Center line profiles")
    axes[2].set_xlabel("Distance from peak (nm)")
    axes[2].set_ylabel("Normalized intensity")
    axes[2].legend(frameon=False)
    axes[2].set_ylim(0, 1.05)

    fig.suptitle(
        f"Effective probe broadening: sigma={sigma_nm:.3f} nm, mean HWHM={hwhm_mean:.3f} nm",
        fontsize=12,
    )
    fig.tight_layout()

    png = args.outdir / "probe_profile_estimate.png"
    svg = args.outdir / "probe_profile_estimate.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
