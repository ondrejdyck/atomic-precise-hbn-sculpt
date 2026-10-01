"""Twisted bilayer h-BN STEM multislice comparison.

This script reproduces the notebook workflow in a standalone, rerunnable form.
It simulates HAADF-STEM images for two bilayer h-BN stackings, saves the raw
comparisons, and saves lightly blurred comparisons for qualitative inspection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from figure2_utils import configure_matplotlib, interp_image, square_roi_fractional

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
        help="Directory containing the twisted-bilayer .p1 files.",
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
        default=1024,
        help="Simulation grid size in pixels for the multislice STEM calculation.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=8.0,
        help="Gaussian blur sigma in pixels for the display-space comparison.",
    )
    parser.add_argument(
        "--nfph",
        type=int,
        default=1,
        help="Number of frozen phonon configurations for the development run.",
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
    t0 = log("starting script")

    import matplotlib.pyplot as plt
    import pyms
    import torch

    inputs = args.inputs
    structure_path = inputs / "hbn_bilayer_8deg_v2.p1"
    structure2_path = inputs / "hbn_bilayer_188deg_v2.p1"

    structure = pyms.structure.fromfile(
        structure_path, temperature_factor_units="urms", atomic_coordinates="fractional"
    )
    structure2 = pyms.structure.fromfile(
        structure2_path, temperature_factor_units="urms", atomic_coordinates="fractional"
    )

    tiling = [1, 1]
    pixels = [args.grid, args.grid]

    # Quick projected-potential sanity check.
    t0 = log("building projected potential", t0)
    potential = structure.make_potential([512, 512], tiling=tiling, displacements=True)
    t0 = log("projected potential built", t0)
    fig, ax = plt.subplots()
    ax.imshow(potential[0].cpu())
    ax.set_title("Projected potential sanity check")
    save_figure(fig, args.outdir, "twisted_potential")
    if args.show:
        plt.show()
    plt.close(fig)

    # Visualize the chosen slicing planes.
    t0 = log("building slicing figures", t0)
    slices1 = np.asarray([1.8, 5.1, structure.unitcell[2]]) / structure.unitcell[2]
    fig1 = structure.generate_slicing_figure(slices1, show=args.show)
    if fig1 is not None:
        save_figure(fig1, args.outdir, "twisted_slices_structure1")
        if not args.show:
            plt.close(fig1)

    slices2 = np.asarray([1.8, 5.1, structure2.unitcell[2]]) / structure2.unitcell[2]
    fig2 = structure2.generate_slicing_figure(slices2, show=args.show)
    if fig2 is not None:
        save_figure(fig2, args.outdir, "twisted_slices_structure2")
        if not args.show:
            plt.close(fig2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log(f"using device: {device}", t0)
    eV = 100 * 1000
    app = 30
    thicknesses = [[7]]
    detectors = [[80, 150]]
    df = 0
    nfph = args.nfph

    # Square real-space ROI from the rectangular unit cell.
    roi = square_roi_fractional(structure.unitcell[:2])

    log("running multislice for structure 1", t0)
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
    log("running multislice for structure 2", t0)
    result2 = pyms.STEM_multislice(
        structure2,
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

    log("interpolating and saving comparison figures", t0)
    im = interp_image(pyms, result["STEM images"], tiling=tiling)
    im2 = interp_image(pyms, result2["STEM images"], tiling=tiling)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
    ax[0].imshow(im)
    ax[0].set_title("AA Stacked")
    ax[0].set_axis_off()
    ax[1].imshow(im2)
    ax[1].set_title("AA' Stacked")
    ax[1].set_axis_off()
    save_figure(fig, args.outdir, "twisted comparison")
    if args.show:
        plt.show()
    plt.close(fig)

    log("applying gaussian blur", t0)
    blurred_image = gaussian_filter(im, sigma=float(args.sigma), mode="reflect")
    blurred_image2 = gaussian_filter(im2, sigma=float(args.sigma), mode="reflect")

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
    ax[0].imshow(blurred_image)
    ax[0].set_title("AA Stacked")
    ax[0].set_axis_off()
    ax[1].imshow(blurred_image2)
    ax[1].set_title("AA' Stacked")
    ax[1].set_axis_off()
    save_figure(fig, args.outdir, "twisted comparison blurred")
    if args.show:
        plt.show()
    plt.close(fig)

    log("done", t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
