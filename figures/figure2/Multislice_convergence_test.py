"""Frozen-phonon convergence test for the Figure 2 multislice simulations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from figure2_utils import configure_matplotlib

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
        "--phonons",
        type=int,
        nargs='*',
        default=[1, 2, 5, 10, 20, 30, 40, 50],
        help="Frozen-phonon pass counts to compare (default: 1 2 5 10 20 30 40 50).",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=15,
        help="Number of repeat runs for each frozen-phonon count.",
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    eV = 100 * 1000
    app = 30
    thicknesses = [[7]]
    detectors = [[80, 150]]
    rsize_y, rsize_x = structure.unitcell[:2]
    square_side = min(rsize_y, rsize_x)
    roi = [
        0.5 * (1 - square_side / rsize_y),
        0.0,
        0.5 * (1 + square_side / rsize_y),
        1.0,
    ]

    phonons = list(args.phonons)
    num_runs = args.num_runs
    res = []

    log("starting convergence loop", t0)
    for nphon in phonons:
        run_results = []
        for run in range(num_runs):
            result = pyms.STEM_multislice(
                structure,
                eV,
                app,
                thicknesses,
                nfph=nphon,
                detector_ranges=detectors,
                showProgress=False,
                tiling=tiling,
                gridshape=pixels,
                subslices=[1.0],
                df=0,
                ROI=roi,
                device_type=device,
            )
            run_results.append(result)
        res.append(run_results)

    mad_per_run = []
    for run in range(num_runs):
        images = [res[i][run]["STEM images"][0] for i in range(len(phonons))]
        differences = []
        for i in range(len(phonons) - 1):
            mad = np.mean(np.abs(images[i] - images[i + 1]))
            differences.append(mad)
        mad_per_run.append(differences)

    mad_per_run = np.array(mad_per_run)
    mean_mad = np.mean(mad_per_run, axis=0)
    std_mad = np.std(mad_per_run, axis=0, ddof=1)

    fig = plt.figure(figsize=(8, 5))
    plt.errorbar(phonons[1:], mean_mad, yerr=std_mad, marker="o", capsize=5, linestyle="-")
    plt.xlabel("Number of Frozen Phonon Passes")
    plt.ylabel("Mean Absolute Difference (MAD)")
    plt.title("Convergence of Multislice Simulation with Variability")
    plt.grid(True)
    fig.savefig(args.outdir / "Multislice Convergence Test.png", bbox_inches="tight", dpi=200)
    fig.savefig(args.outdir / "Multislice Convergence Test.svg", bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close(fig)

    log("done", t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
