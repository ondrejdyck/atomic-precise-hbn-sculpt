"""Ensemble version of the parallel vs. sequential milling model.

Re-implements the model in ``Atomistic Simulation.ipynb`` (Steps 1-8) with
vectorized neighbor counting so that it can be run over many random seeds and
parameter sets. The lattice, beam, dose convolution, ejection rule, and
stopping criteria follow the notebook exactly; ``--validate`` replays the
notebook's single seed-42 realization in the notebook's geometry and should
reproduce its step counts (parallel: 12, sequential: 64) and its per-step
ejection sequences.

Two deliberate departures from the notebook in the default (ensemble) geometry:
the target box is shifted by half an atomic-row period (``box_dy``) so that no
atomic row sits on the box edge, and when the box step does not divide the
target length the last sequential position is clamped to the target end.

Randomness in the ensemble uses per-atom "toughness" (common random numbers):
each atom draws T ~ Exp(1) once and is ejected when its accumulated hazard
H = sum over scans of -ln(1 - P) reaches T. For a single atom this is
distributionally identical to the notebook's fresh uniform draw each scan
(survival = prod(1 - P) = exp(-H)), so each protocol's statistics are unchanged;
but parallel and sequential can then mill the *same* virtual sample (same T),
making per-seed comparisons genuinely paired. ``--validate`` uses the notebook's
per-scan draws.

Usage:
    uv run python figures/figure6_7/ensemble_model.py --validate
    uv run python figures/figure6_7/ensemble_model.py --n-seeds 500
"""

import argparse
import json
from dataclasses import dataclass, replace, asdict
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import convolve2d
from scipy.spatial import cKDTree

OUT = Path(__file__).parent / "outputs"

# Fixed lattice / beam constants from the notebook
A = 0.145  # B-N bond length (nm)
NX, NY = 30, 10
THRESHOLD = 0.16  # neighbor cutoff (nm)
GRID_STEP = 0.05  # dose grid resolution (nm)
PHI_0 = (18e-12 * 36e-6) / 1.602e-19  # electrons per dwell
MAX_STEPS = 1000


@dataclass(frozen=True)
class Params:
    sigma: float = 0.1  # Gaussian beam width (nm)
    box_width: float = 4.0  # target region length (nm)
    box_height: float = 0.5  # target region height = sequential box side (nm)
    step_size: float = 0.5  # sequential box translation (nm)
    k_eject: float = 5e13
    bulk_B: float = 50e-22
    bulk_N: float = 30e-22
    edge_B: float = 70e-22
    edge_N: float = 50e-22
    # False reproduces the notebook: a new box position always receives one scan
    # before it is checked. True skips positions already cleared by beam tails.
    skip_cleared: bool = False
    # Sequential protocol: "fixed" = box steps by step_size once its box is empty
    # (notebook; used only for validation); "tracking" = edge-following: the box
    # slides forward in track_step increments so that its rear edge sits a vacuum
    # margin behind the rearmost surviving target atom (a stubborn atom holds the
    # box: no skipping).
    seq_mode: str = "tracking"
    # vacuum margin behind the cut edge, in beam diameters (FWHM = 2.3548 sigma), so
    # that box width can change without changing the exposed vacuum
    margin_fwhm: float = 1.0
    track_step: float = 0.05  # one dose-grid pixel (nm)
    # tracking box length along the cut (nm); spans the full target height. Keep it a
    # multiple of the 0.05 nm grid so every position covers the same grid columns.
    track_width: float = 0.75
    # Vertical shift of the target box from the lattice centre (nm). The notebook
    # uses 0, which puts two atomic rows 0.004 nm outside the box edges; -0.10875
    # (half the 0.2175 nm row period) leaves >= 0.04 nm between each edge and the
    # nearest row. The dose kernel stays centred on the grid either way.
    box_dy: float = -0.10875


def build_lattice():
    a1 = np.array([A * np.sqrt(3), 0])
    a2 = np.array([A * np.sqrt(3) / 2, A * 3 / 2])
    xy, is_N = [], []
    for i in range(NX):
        for j in range(NY):
            t = i * a1 + j * a2
            xy += [t, t + [0, A]]
            is_N += [False, True]
    xy = np.array(xy)
    is_N = np.array(is_N)
    # Layer 2: same xy, B/N swapped (AA' stacking)
    xy = np.vstack([xy, xy])
    is_N = np.concatenate([is_N, ~is_N])
    layer = np.repeat([1, 2], len(xy) // 2)
    # Same-layer neighbor pairs
    n = len(xy)
    rows, cols = [], []
    for L in (1, 2):
        idx = np.where(layer == L)[0]
        tree = cKDTree(xy[idx])
        pairs = tree.query_pairs(THRESHOLD, output_type="ndarray")
        rows += [idx[pairs[:, 0]], idx[pairs[:, 1]]]
        cols += [idx[pairs[:, 1]], idx[pairs[:, 0]]]
    from scipy.sparse import csr_matrix

    rows, cols = np.concatenate(rows), np.concatenate(cols)
    adj = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return xy, is_N, layer, adj


XY, IS_N, LAYER, ADJ = build_lattice()


class Geometry:
    """Deterministic dose fields for one parameter set (grid + at-atom values)."""

    def __init__(self, p: Params):
        a1x = A * np.sqrt(3)
        a2 = np.array([A * np.sqrt(3) / 2, A * 3 / 2])
        x_max = (NX - 1) * a1x + (NY - 1) * a2[0]
        y_max = (NY - 1) * a2[1] + A
        cx, cy0 = x_max / 2, y_max / 2  # kernel centre (must not move with the box)
        cy = cy0 + p.box_dy
        self.xb = (cx - p.box_width / 2, cx + p.box_width / 2)
        self.yb = (cy - p.box_height / 2, cy + p.box_height / 2)
        self.x_grid = np.arange(0, XY[:, 0].max() + GRID_STEP, GRID_STEP)
        self.y_grid = np.arange(0, XY[:, 1].max() + GRID_STEP, GRID_STEP)
        X, Y = np.meshgrid(self.x_grid, self.y_grid)
        self.X, self.Y = X, Y
        gauss = (PHI_0 / (2 * np.pi * p.sigma**2)) * np.exp(
            -((X - cx) ** 2 + (Y - cy0) ** 2) / (2 * p.sigma**2)
        )

        def field(x0, x1, y0, y1):
            box = np.zeros_like(X)
            box[(X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)] = 1.0
            g = convolve2d(box, gauss, mode="same")
            interp = RegularGridInterpolator(
                (self.y_grid, self.x_grid), g, bounds_error=False, fill_value=0
            )
            return g, interp((XY[:, 1], XY[:, 0]))

        self._field = field
        self._track_cache = {}
        self.par_grid, self.par_atoms = field(*self.xb, *self.yb)
        self.target = (
            (XY[:, 0] >= self.xb[0]) & (XY[:, 0] <= self.xb[1])
            & (XY[:, 1] >= self.yb[0]) & (XY[:, 1] <= self.yb[1])
        )
        s = p.box_height
        def centres(lo, hi):
            # box centres from lo to hi; a shorter final step lands the last box on the end
            c = np.arange(lo, hi + 1e-9, p.step_size)
            return np.append(c, hi) if hi - c[-1] > 1e-9 else c

        xs = centres(self.xb[0] + s / 2, self.xb[1] - s / 2)
        ys = centres(self.yb[0] + s / 2, self.yb[1] - s / 2)
        self.positions = [(x, y) for x in xs for y in ys]
        self.seq_grid, self.seq_atoms, self.seq_mask = [], [], []
        for bx, by in self.positions:
            g, d = field(bx - s / 2, bx + s / 2, by - s / 2, by + s / 2)
            self.seq_grid.append(g)
            self.seq_atoms.append(d)
            self.seq_mask.append(
                (XY[:, 0] >= bx - s / 2) & (XY[:, 0] <= bx + s / 2)
                & (XY[:, 1] >= by - s / 2) & (XY[:, 1] <= by + s / 2)
            )
        # Time cost of one sequential scan relative to one parallel scan
        self.seq_time = (s * s) / (p.box_width * p.box_height)
        self.offtarget_grid = ~(
            (X >= self.xb[0]) & (X <= self.xb[1]) & (Y >= self.yb[0]) & (Y <= self.yb[1])
        )


def p_eject(alive, p: Params):
    nb = ADJ @ alive.astype(float)
    edge = nb < 3
    return np.where(
        IS_N,
        np.where(edge, p.edge_N, p.bulk_N),
        np.where(edge, p.edge_B, p.bulk_B),
    )


def eject(alive, dose_atoms, p, rng):
    P = np.minimum(p_eject(alive, p) * dose_atoms * p.k_eject, 1.0)
    r = rng.random(alive.sum())
    new = np.zeros_like(alive)
    new[alive] = P[alive] > r
    return alive & ~new


def eject_tough(alive, dose_atoms, p, state):
    """Toughness-based ejection: state = (T, H); H accumulates -ln(1 - P)."""
    T, H = state
    P = np.minimum(p_eject(alive, p) * dose_atoms * p.k_eject, 1.0)
    with np.errstate(divide="ignore"):
        h = np.where(P >= 1.0, np.inf, -np.log1p(-P))
    H[alive] += h[alive]
    return alive & ~(H >= T)


def _ejector(p, rng, toughness):
    """Per-scan ejection step: notebook-style fresh draws, or shared toughness."""
    if toughness is None:
        return lambda alive, dose: eject(alive, dose, p, rng)
    state = (toughness, np.zeros(len(XY)))
    return lambda alive, dose: eject_tough(alive, dose, p, state)


def run_parallel(g: Geometry, p: Params, rng=None, trace=None, toughness=None):
    step = _ejector(p, rng, toughness)
    alive = np.ones(len(XY), bool)
    steps = 0
    while alive[g.target].any() and steps < MAX_STEPS - 1:
        steps += 1
        before = alive[g.target].sum()
        alive = step(alive, g.par_atoms)
        if trace is not None:
            trace.append(int(before - alive[g.target].sum()))
    dose = g.par_grid * steps
    return steps, alive, dose, steps * 1.0


def run_sequential(g: Geometry, p: Params, rng=None, trace=None, toughness=None):
    step = _ejector(p, rng, toughness)
    alive = np.ones(len(XY), bool)
    steps, pos = 0, 0
    dose = np.zeros_like(g.X)
    while alive[g.target].any() and steps < MAX_STEPS - 1:
        if p.skip_cleared:
            while not alive[g.seq_mask[pos]].any():
                pos = (pos + 1) % len(g.positions)
        steps += 1
        before = alive[g.target].sum()
        alive = step(alive, g.seq_atoms[pos])
        if trace is not None:
            trace.append(int(before - alive[g.target].sum()))
        dose += g.seq_grid[pos]
        if not alive[g.seq_mask[pos]].any():
            pos = (pos + 1) % len(g.positions)
    return steps, alive, dose, steps * g.seq_time


def _track_field(g: Geometry, p: Params, k):
    """Dose field for a tracking box whose rear edge is at xb0 + k * track_step.
    Positions stay on that lattice, so every box covers the same grid columns."""
    if k not in g._track_cache:
        r = g.xb[0] + k * p.track_step
        g._track_cache[k] = g._field(r, r + p.track_width, *g.yb)
    return g._track_cache[k]


FWHM_PER_SIGMA = 2 * np.sqrt(2 * np.log(2))
LATTICE_CONST = A * np.sqrt(3)  # 0.251 nm: practical minimum margin for seeing the edge


def margin_nm(p: Params):
    return p.margin_fwhm * FWHM_PER_SIGMA * p.sigma


def run_tracking(g: Geometry, p: Params, rng=None, toughness=None):
    step = _ejector(p, rng, toughness)
    s, d = p.track_width, p.track_step
    k_max = int(round((g.xb[1] - s - g.xb[0]) / d))  # box front at the target end
    alive = np.ones(len(XY), bool)
    steps, k = 0, 0  # start with the box at the target start
    dose = np.zeros_like(g.X)
    while alive[g.target].any() and steps < MAX_STEPS - 1:
        # place the box: rear edge one vacuum margin behind the rearmost surviving atom
        x_rear = XY[alive & g.target, 0].min()
        k_want = int(np.floor((x_rear - margin_nm(p) - g.xb[0]) / d + 1e-9))
        k = min(max(k, k_want), k_max)  # forward only, never past the target end
        grid, atoms = _track_field(g, p, k)
        steps += 1
        alive = step(alive, atoms)
        dose += grid
    return steps, alive, dose, steps * p.track_width / p.box_width  # scan time ∝ box area


def metrics(g: Geometry, alive, dose, steps, time):
    off = g.offtarget_grid
    d_off = dose[off]
    single_scan = g.par_grid[~off].max()  # peak in-box dose of one scan
    cell = GRID_STEP**2
    ejected_off = (~alive) & (~g.target)
    return {
        "steps": steps,
        "time_parallel_scan_units": time,
        "total_dose": dose.sum() * cell,
        "offtarget_dose": d_off.sum() * cell,
        "offtarget_max": d_off.max(),
        # area outside the target receiving more than N single-scan peak doses
        "offtarget_area_gt1": (d_off > 1 * single_scan).sum() * cell,
        "offtarget_area_gt5": (d_off > 5 * single_scan).sum() * cell,
        "offtarget_ejected": int(ejected_off.sum()),
    }


def ensemble(p: Params, seeds):
    g = Geometry(p)
    out = {"parallel": [], "sequential": []}
    for s in seeds:
        # one virtual sample per seed: the same atom toughness for both protocols
        T = np.random.default_rng(s).exponential(size=len(XY))
        st, al, d, t = run_parallel(g, p, toughness=T)
        out["parallel"].append(metrics(g, al, d, st, t))
        run_seq = run_tracking if p.seq_mode == "tracking" else run_sequential
        st, al, d, t = run_seq(g, p, toughness=T)
        out["sequential"].append(metrics(g, al, d, st, t))
    return {k: {m: np.array([r[m] for r in v]) for m in v[0]} for k, v in out.items()}


def validate():
    """Replay the notebook's legacy-RNG seed-42 sequence (Steps 5, 6, 7) in the
    notebook's geometry and compare step counts and per-step in-target ejections
    with the outputs saved in the notebook."""
    import re

    nb = json.loads((Path(__file__).parent / "Atomistic Simulation.ipynb").read_text())
    saved = []
    for c in nb["cells"]:
        for o in c.get("outputs", []):
            txt = "".join(o.get("text", ""))
            m = re.search(r"^Ejections per step: (\[.*?\])", txt, re.M)
            if m:
                saved.append(json.loads(m.group(1)))
    nb_par, nb_seq = saved[:2]
    p = replace(Params(), box_dy=0.0)
    rs = np.random.RandomState(42)

    class R:  # adapter so eject() can use the legacy global-state stream
        def random(self, n):
            return rs.random_sample(n)

    rng = R()
    # Step 5: stationary beam at (4, 1), 40 steps, k_eject = 5e15
    dose_pt = (PHI_0 / (2 * np.pi * 0.1**2)) * np.exp(
        -((XY[:, 0] - 4) ** 2 + (XY[:, 1] - 1) ** 2) / (2 * 0.1**2)
    )
    alive = np.ones(len(XY), bool)
    for _ in range(40):
        alive = eject(alive, dose_pt, replace(p, k_eject=5e15), rng)
    g = Geometry(p)
    tr_p, tr_s = [], []
    st_p, *_ = run_parallel(g, p, rng, tr_p)
    st_s, *_ = run_sequential(g, p, rng, tr_s)
    ok = (st_p, st_s) == (12, 64) and tr_p == nb_par and tr_s == nb_seq
    print(f"validation: parallel {st_p} steps (notebook 12), sequential {st_s} steps (notebook 64); "
          f"per-step ejections match notebook: {tr_p == nb_par and tr_s == nb_seq}")
    return ok


def summarize(res):
    rows = {}
    for mode, d in res.items():
        rows[mode] = {m: (float(v.mean()), float(v.std()), float(np.median(v))) for m, v in d.items()}
    # paired comparison: fraction of samples (seeds) on which sequential < parallel;
    # both protocols mill the same virtual sample (shared atom toughness)
    rows["frac_seq_lower"] = {
        m: float((res["sequential"][m] < res["parallel"][m]).mean()) for m in res["parallel"]
    }
    # effect size: percent reduction of the mean, 1 - mean(S)/mean(P), with a
    # paired bootstrap 95% CI over samples
    rng = np.random.default_rng(0)
    n = len(res["parallel"]["steps"])
    boot = rng.integers(0, n, size=(2000, n))
    rows["reduction_pct"] = {}
    for m in res["parallel"]:
        P, S = res["parallel"][m], res["sequential"][m]
        if P.mean() == 0:  # metric is zero in both modes (e.g. no area above threshold)
            rows["reduction_pct"][m] = None
            continue
        red = 100 * (1 - S.mean() / P.mean())
        Pb = P[boot].mean(1)
        if (Pb == 0).any():  # too rare for a ratio CI: some resamples have a zero mean
            rows["reduction_pct"][m] = (float(red), None, None)
            continue
        b = 100 * (1 - S[boot].mean(1) / Pb)
        rows["reduction_pct"][m] = (float(red), *map(float, np.percentile(b, [2.5, 97.5])))
    return rows


SWEEPS = {
    "baseline": [{}],
    # beam width with the vacuum margin held at its baseline value in nm (0.235 nm),
    # so the sweep isolates probe sharpness from the edge-visibility margin
    "sigma": [{"sigma": s, "margin_fwhm": 0.1 / s} for s in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3)],
    # multipliers on the default probabilities, so x1 is the baseline model.
    # Edge x0.5 and below make edge atoms harder to eject than bulk atoms
    # (default edge/bulk: 1.4 for B, 1.67 for N), mimicking reconstruction-stabilized edges.
    "edge_mult": [{"edge_B": 70e-22 * m, "edge_N": 50e-22 * m} for m in (0.25, 0.5, 1, 2, 4)],
    # boron (bulk and edge) relative to nitrogen; default bulk B/N = 1.67
    "boron_mult": [{"bulk_B": 50e-22 * m, "edge_B": 70e-22 * m} for m in (0.5, 0.75, 1, 1.5, 2)],
    "k_eject": [{"k_eject": 5e13 * f} for f in (0.25, 0.5, 1, 2, 4)],
    # tracking protocol: vacuum margin (beam diameters) x box width (nm); the
    # margin must be smaller than the box. Width 4.0 nm = the whole target, which
    # must reduce to parallel milling.
    "margin_width": [{"margin_fwhm": m, "track_width": w}
                     for m in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
                     for w in (0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
                     if m * FWHM_PER_SIGMA * 0.1 < w],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--n-seeds", type=int, default=500)
    ap.add_argument("--sweeps", nargs="*", help="run only these sweeps (merged into the summary)")
    args = ap.parse_args()
    if args.validate:
        raise SystemExit(0 if validate() else 1)
    OUT.mkdir(exist_ok=True)
    seeds = range(args.n_seeds)
    summary_path = OUT / "ensemble_summary.json"
    results = {}
    if args.sweeps:
        old = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        results = {k: v for k, v in old.items() if k.split(":")[0] not in args.sweeps}
    for name, variants in SWEEPS.items():
        if args.sweeps and name not in args.sweeps:
            continue
        for v in variants:
            p = replace(Params(), **v)
            key = f"{name}:" + ",".join(f"{k}={val:.3g}" if not isinstance(val, str) else f"{k}={val}"
                                        for k, val in v.items())
            res = ensemble(p, seeds)
            results[key] = {"params": asdict(p), "summary": summarize(res)}
            np.savez(OUT / f"ensemble_{key.replace(':', '_').replace(',', '_')}.npz",
                     **{f"{m}__{k}": a for m, d in res.items() for k, a in d.items()})
            s = results[key]["summary"]
            print(
                f"{key:45s} off-target dose P/S = {s['parallel']['offtarget_dose'][0]:.3g}/"
                f"{s['sequential']['offtarget_dose'][0]:.3g}  "
                f"off-target ejected P/S = {s['parallel']['offtarget_ejected'][0]:.1f}/"
                f"{s['sequential']['offtarget_ejected'][0]:.1f}  "
                f"frac seq lower (ejected) = {s['frac_seq_lower']['offtarget_ejected']:.2f}",
                flush=True,
            )
    summary_path.write_text(json.dumps(results, indent=1, allow_nan=False))


if __name__ == "__main__":
    main()
