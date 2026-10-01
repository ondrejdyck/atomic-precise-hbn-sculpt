"""Basic multislice comparison for Figure 2.

This script reproduces the core bilayer-versus-monolayer comparison and the
line-profile figure from the multislice notebook without the defocus series or
the frozen-phonon convergence study.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

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

    tiling = [5, 3]
    pixels = [args.grid, args.grid]

    # Quick potential sanity check for the bilayer structure.
    t0 = log("building projected potential", t0)
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
    thicknesses2 = [[4]]
    detectors = [[80, 150]]
    df = np.array([0])
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

    log("interpolating and saving comparison figures", t0)
    single_layer_im = interp_image(pyms, result2["STEM images"], tiling=tiling)
    bilayer_im = interp_image(pyms, result["STEM images"], tiling=tiling)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
    ax[0].imshow(single_layer_im)
    ax[0].set_title("Single Layer")
    ax[0].set_axis_off()
    ax[1].imshow(bilayer_im)
    ax[1].set_title("Bilayer")
    ax[1].set_axis_off()
    save_figure(fig, args.outdir, "Single-Bilayer image comparison")
    if args.show:
        plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), ncols=1, nrows=1)
    x = 175
    y = 100
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
