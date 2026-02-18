"""Generate AA and AA' stacking schematic intensity profiles.

This script is a notebook-to-script conversion of:
  notebooks/figure2/Stacking illustration.ipynb

It produces two SVG figures:
  - hBN_AA_prime_stacking.svg
  - hBN_AA_stacking.svg
"""

from __future__ import annotations

import argparse
from pathlib import Path


def gaussian(x, amplitude, mu, sigma=0.5):
    """Simple Gaussian used for schematic 1D profiles."""
    import numpy as np

    return amplitude * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Output directory for saved SVGs (default: figures/figure2/outputs).",
    )
    p.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Gaussian sigma used in schematic plots (default: 1.0).",
    )
    p.add_argument(
        "--x-min",
        type=float,
        default=-5.0,
        help="Minimum x for profiles (default: -5).",
    )
    p.add_argument(
        "--x-max",
        type=float,
        default=5.0,
        help="Maximum x for profiles (default: 5).",
    )
    p.add_argument(
        "--npts",
        type=int,
        default=500,
        help="Number of points in x grid (default: 500).",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display figures interactively.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
    except ModuleNotFoundError as e:  # pragma: no cover
        raise SystemExit(
            "Missing dependency 'matplotlib'. Install it (e.g. pip/conda) and rerun."
        ) from e

    # Use a non-interactive backend when running headless.
    if args.no_show:
        matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as plt
    import numpy as np

    x = np.linspace(args.x_min, args.x_max, args.npts)
    sigma = float(args.sigma)

    # Parameters from the original notebook
    B_amplitude = 5**2  # Z_B^2 = 25
    N_amplitude = 7**2  # Z_N^2 = 49
    offset = 2
    line_width = 4
    label_fontsize = 25
    tick_fontsize = 20

    # --- Plot 1: AA' stacking (B on N, N on B) ---
    B1 = gaussian(x, B_amplitude, 2, sigma)
    B2 = gaussian(x, B_amplitude, -2, sigma)
    N1 = gaussian(x, N_amplitude, 2, sigma)
    N2 = gaussian(x, N_amplitude, -2, sigma)
    summed_intensity_AA_prime = B1 + N1 + B2 + N2

    plt.figure(figsize=(6, 6))
    plt.plot(x, B1, "b-", label="B", linewidth=line_width)
    plt.plot(x, B2, "b--", label="B", linewidth=line_width)
    plt.plot(x, N1, "r-", label="N", linewidth=line_width)
    plt.plot(x, N2, "r--", label="N", linewidth=line_width)
    plt.plot(
        x,
        summed_intensity_AA_prime,
        "k-",
        label="Summed Intensity",
        linewidth=line_width,
    )
    plt.title("AA' Stacking", fontsize=18)
    plt.ylabel("Intensity", fontsize=label_fontsize)
    plt.grid(False)
    plt.legend(fontsize=12)
    plt.xticks([])
    plt.yticks(fontsize=tick_fontsize)
    plt.ylim(0, 100)
    plt.savefig(outdir / "hBN_AA_prime_stacking.svg", bbox_inches="tight")
    if not args.no_show:
        plt.show()
    plt.close()

    # --- Plot 2: AA stacking (B on B, N on N) ---
    B1 = gaussian(x, B_amplitude, 2, sigma)
    B2 = gaussian(x, B_amplitude, 2, sigma)
    N1 = gaussian(x, N_amplitude, -2, sigma)
    N2 = gaussian(x, N_amplitude, -2, sigma)
    B2_offset = B2 + offset
    N2_offset = N2 + offset
    summed_intensity_AA = B1 + B2 + N1 + N2

    plt.figure(figsize=(6, 6))
    plt.plot(x, B1, "b-", label="B", linewidth=line_width)
    plt.plot(x, B2_offset, "b--", label="B (offset)", linewidth=line_width)
    plt.plot(x, N1, "r-", label="N", linewidth=line_width)
    plt.plot(x, N2_offset, "r--", label="N (offset)", linewidth=line_width)
    plt.plot(
        x, summed_intensity_AA, "k-", label="Summed Intensity", linewidth=line_width
    )
    plt.title("AA Stacking", fontsize=18)
    plt.ylabel("Intensity", fontsize=label_fontsize)
    plt.grid(False)
    plt.legend(fontsize=12)
    plt.xticks([])
    plt.yticks(fontsize=tick_fontsize)
    plt.ylim(0, 100)
    plt.savefig(outdir / "hBN_AA_stacking.svg", bbox_inches="tight")
    if not args.no_show:
        plt.show()
    plt.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
