"""Multislice simulations of h-BN.

This script is a standalone conversion of the notebook used to generate the
Figure 2 multislice results. It reproduces the core stacking contrast, the
defocus series, the simplified image comparison, and the optional frozen-phonon
convergence test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from figure2_utils import configure_matplotlib, interp_image

from time import perf_counter


def log(msg: str, start: float | None = None) -> float:
    now = perf_counter()
    if start is None:
        print(f"[figure2] {msg}", flush=True)
    else:
        print(f"[figure2] {msg} (+{now - start:.1f}s)", flush=True)
    return now



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        default=Path(__file__).resolve().parent / "inputs",
        help="Directory containing the h-BN .p1 files.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for saved output figures.",
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=512,
        help="Simulation grid size in pixels for the multislice STEM calculations.",
    )
    parser.add_argument(
        "--nfph",
        type=int,
        default=1,
        help=(
            "Number of frozen-phonon passes for the main multislice calculations "
            "(default: 1 for fast development; use 40 for the manuscript figures)."
        ),
    )
    parser.add_argument(
        "--skip-convergence",
        action="store_true",
        help="Skip the expensive frozen-phonon convergence test.",
    )
    parser.add_argument(
        "--skip-defocus-series",
        action="store_true",
        help="Skip the defocus series section.",
    )
    show_group = parser.add_mutually_exclusive_group()
    show_group.add_argument(
        "--show",
        dest="show",
        action="store_true",
        help="Display figures interactively.",
    )
    show_group.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="Do not display figures interactively (default).",
    )
    parser.set_defaults(show=False)
    return parser


def save_figure(fig, outdir: Path, stem: str) -> None:
    fig.savefig(outdir / f"{stem}.png", bbox_inches="tight", dpi=300)
    fig.savefig(outdir / f"{stem}.svg", bbox_inches="tight")


def main() -> int:
    args = build_parser().parse_args()
    configure_matplotlib(args.outdir, show=args.show)

    import matplotlib.pyplot as plt
    import pyms
    import torch

    t0 = log("starting script")

    inputs = args.inputs
    bilayer_path = inputs / "h-BN.p1"
    monolayer_path = inputs / "monolayer_h-BN.p1"

    log("loading structures", t0)
    structure = pyms.structure.fromfile(
        bilayer_path, temperature_factor_units="urms", atomic_coordinates="fractional"
    )
    structure2 = pyms.structure.fromfile(
        monolayer_path, temperature_factor_units="urms", atomic_coordinates="fractional"
    )

    # Quick potential sanity check for the bilayer structure.
    t0 = log("building projected potential", t0)
    tiling = [5, 3]
    pixels = [args.grid, args.grid]
    potential = structure.make_potential(pixels, tiling=tiling, displacements=True)
    fig, ax = plt.subplots()
    ax.imshow(potential[0].cpu())
    ax.set_title("Projected potential sanity check")
    save_figure(fig, args.outdir, "projected_potential_bilayer")
    if args.show:
        plt.show()
    plt.close(fig)

    # Visualize the slicing planes for both structures.
    t0 = log("building slicing figures", t0)
    slices = np.asarray([1.8, 5.1, structure.unitcell[2]]) / structure.unitcell[2]
    fig1 = structure.generate_slicing_figure(slices, show=args.show)
    if fig1 is not None:
        save_figure(fig1, args.outdir, "slicing_bilayer")
        if not args.show:
            plt.close(fig1)

    slices2 = np.asarray([1.75, structure2.unitcell[2]]) / structure2.unitcell[2]
    fig2 = structure2.generate_slicing_figure(slices2, show=args.show)
    if fig2 is not None:
        save_figure(fig2, args.outdir, "slicing_monolayer")
        if not args.show:
            plt.close(fig2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log(f"using device: {device}", t0)
    eV = 100 * 1000
    app = 30
    thicknesses = [[7]]
    detectors = [[80, 150]]
    df = np.linspace(-40, 40, 11)
    nfph = args.nfph

    # Use the notebook-style ROI definition directly for this workflow.
    rsize_y, rsize_x = structure.unitcell[:2]
    square_side = min(rsize_y, rsize_x)
    roi = [
        0.5 * (1 - square_side / rsize_y),
        0.0,
        0.5 * (1 + square_side / rsize_y),
        1.0,
    ]

    # Core STEM multislice runs.
    log("running bilayer multislice", t0)
    result = pyms.STEM_multislice(
        structure,
        eV,
        app,
        thicknesses,
        nfph=nfph,
        detector_ranges=detectors,
        showProgress=False,
        tiling=tiling,
        gridshape=pixels,
        subslices=[1.0],
        df=df,
        ROI=roi,
        device_type=device,
    )

    thicknesses2 = [[4]]
    log("running monolayer multislice", t0)
    result2 = pyms.STEM_multislice(
        structure2,
        eV,
        app,
        thicknesses2,
        nfph=nfph,
        detector_ranges=detectors,
        showProgress=False,
        tiling=tiling,
        gridshape=pixels,
        subslices=[1.0],
        df=df,
        ROI=roi,
        device_type=device,
    )

    # Defocus series.
    prof = []
    x = 175
    y = 100
    if not args.skip_defocus_series:
        log("starting defocus series", t0)
        ndef, ny, nx = np.shape(result["STEM images"])
        ndef2, ny2, nx2 = np.shape(result2["STEM images"])
        fig, ax = plt.subplots(figsize=(20, 5), ncols=ndef, nrows=2)
        for i, axi in enumerate(np.ravel(ax)):
            if i < 11:
                im = interp_image(pyms, result["STEM images"][i], tiling=tiling)
                prof.append(im[y, :x])
                axi.imshow(im)
                axi.set_title(f'Def. {df[i]} $\\AA$')
                axi.set_axis_off()
            else:
                im = interp_image(pyms, result2["STEM images"][i % 11], tiling=tiling)
                prof.append(im[y, :x])
                axi.imshow(im)
                axi.set_title(f'Def. {df[i % 11]} $\\AA$')
                axi.set_axis_off()
        save_figure(fig, args.outdir, "Defocus_series")
        if args.show:
            plt.show()
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(15, 5), ncols=2, nrows=1)
        for ln, d in zip(prof[:11], df):
            ax[0].plot(ln, label=d)
            ax[0].set_title("Bilayer")
            ax[0].legend()
        for ln, d in zip(prof[11:], df):
            ax[1].plot(ln, label=d)
            ax[1].set_title("Single Layer")
            ax[1].legend()
        save_figure(fig, args.outdir, "Intensity_profiles")
        if args.show:
            plt.show()
        plt.close(fig)

        height_differences = []
        for i, ln in enumerate(prof):
            peaks, properties = find_peaks(ln, prominence=0.00005)
            if len(peaks) >= 2:
                peak_heights = ln[peaks]
                peak_heights = np.sort(peak_heights)[-2:]
                height_diff = abs(peak_heights[1] - peak_heights[0])
                height_differences.append(height_diff)
            else:
                height_differences.append(np.nan)
                print(f"Warning: Fewer than 2 peaks found in profile {i} (prominence=0.00005)")

        fig = plt.figure(figsize=(8, 5))
        plt.plot(df, height_differences[:11], marker="o", linestyle="-", label="Bilayer")
        plt.plot(df, height_differences[11:], marker="o", linestyle="-", label="Single Layer")
        plt.xlabel("Defocus [Å]")
        plt.ylabel("Peak Height Difference")
        plt.title("Difference in Peak Heights")
        plt.legend()
        plt.grid(True)
        save_figure(fig, args.outdir, "Peak Difference")
        if args.show:
            plt.show()
        plt.close(fig)

    # Simplified presentation.
    fig, ax = plt.subplots(figsize=(20, 20), ncols=2, nrows=1)
    single_layer_im = interp_image(pyms, result2["STEM images"][6], tiling=tiling)
    bilayer_im = interp_image(pyms, result["STEM images"][6], tiling=tiling)
    ax[0].imshow(single_layer_im)
    ax[1].imshow(bilayer_im)
    ax[0].set_title("Single Layer", fontsize=20)
    ax[1].set_title("Bilayer", fontsize=20)
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    fig.savefig(args.outdir / "Single-Bilayer image comparison.svg", bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), ncols=1, nrows=1)
    ax.plot(single_layer_im[y, :x], label="Single Layer", linestyle="--")
    ax.plot(bilayer_im[y, :x], label="Bilayer")
    ax.set_xlabel("Pixels", fontsize=20)
    ax.set_ylabel("Scattering Probability", fontsize=20)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(fontsize=15)
    fig.savefig(args.outdir / "Intensity profile comparision.svg", bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close(fig)

    log("done", t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
