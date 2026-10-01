"""Export scan-by-scan milling histories for the interactive run viewer.

For a handful of samples (shared atom toughness), records when every atom is
ejected under parallel milling and under edge-following ("tracking") milling at
several box widths, plus the box position at every scan. The viewer
recomputes dose maps from the box positions with the model's separable
Gaussian sum, so only positions are stored.

Usage:
    uv run python figures/figure6_7/export_viewer_data.py
"""

import json
from dataclasses import replace

import numpy as np

import ensemble_model as em

MARGIN_FWHM = 1.0
WIDTHS = (0.35, 0.75, 1.5, 3.0)
OUT = em.OUT / "viewer_data.json"


def history_parallel(g, p, T):
    alive, H = np.ones(len(em.XY), bool), np.zeros(len(em.XY))
    eject_at = np.zeros(len(em.XY), int)  # scan number at which ejected; 0 = survived
    n = 0
    while alive[g.target].any():
        n += 1
        new = em.eject_tough(alive, g.par_atoms, p, (T, H))
        eject_at[alive & ~new] = n
        alive = new
    return eject_at, [float(g.xb[0])] * n


def history_tracking(g, p, T):
    alive, H = np.ones(len(em.XY), bool), np.zeros(len(em.XY))
    eject_at = np.zeros(len(em.XY), int)
    s, d = p.track_width, p.track_step
    k_max = int(round((g.xb[1] - s - g.xb[0]) / d))
    k, rear = 0, []
    while alive[g.target].any():
        x_rear = em.XY[alive & g.target, 0].min()
        k_want = int(np.floor((x_rear - em.margin_nm(p) - g.xb[0]) / d + 1e-9))
        k = min(max(k, k_want), k_max)
        rear.append(float(g.xb[0] + k * d))
        new = em.eject_tough(alive, em._track_field(g, p, k)[1], p, (T, H))
        eject_at[alive & ~new] = len(rear)
        alive = new
    return eject_at, rear


def pick_seeds():
    """Typical, best, closest-call and highest-peak samples at the 0.75 nm baseline."""
    key = f"margin_width:margin_fwhm={MARGIN_FWHM:.3g},track_width=0.75"
    d = np.load(em.OUT / f"ensemble_{key.replace(':', '_').replace(',', '_')}.npz")
    adv = 1 - d["sequential__offtarget_ejected"] / d["parallel__offtarget_ejected"]
    peak = d["sequential__offtarget_max"] / d["parallel__offtarget_max"]
    order = np.argsort(adv)
    return [
        ("typical", int(order[len(order) // 2]), "median advantage"),
        ("best case", int(order[-1]), "largest advantage"),
        ("closest call", int(order[0]), "smallest advantage"),
        ("highest peak", int(np.argmax(peak)), "largest sequential peak-dose penalty"),
    ]


def main():
    base = em.Params()
    g0 = em.Geometry(base)
    rows, cols = em.ADJ.nonzero()
    keep = rows < cols
    data = {
        "atoms": {
            "x": np.round(em.XY[:, 0], 4).tolist(),
            "y": np.round(em.XY[:, 1], 4).tolist(),
            "isN": em.IS_N.astype(int).tolist(),
            "layer": em.LAYER.tolist(),
            "target": g0.target.astype(int).tolist(),
        },
        "pairs": np.stack([rows[keep], cols[keep]], 1).tolist(),
        "target_box": [*map(float, g0.xb), *map(float, g0.yb)],
        "grid": {"x": np.round(g0.x_grid, 4).tolist(), "y": np.round(g0.y_grid, 4).tolist()},
        "sigma": base.sigma,
        "dose_per_pixel": float(em.PHI_0 / (2 * np.pi * base.sigma**2)),  # e-/nm^2 per box pixel at r = 0
        "margin_nm": float(em.margin_nm(replace(base, margin_fwhm=MARGIN_FWHM))),
        "margin_fwhm": MARGIN_FWHM,
        "target_length": base.box_width,
        "samples": [],
    }
    for label, seed, why in pick_seeds():
        T = np.random.default_rng(seed).exponential(size=len(em.XY))
        runs = {}
        ej, rear = history_parallel(g0, base, T)
        runs["parallel"] = {"eject_at": ej.tolist(), "rear": rear, "width": base.box_width}
        for w in WIDTHS:
            p = replace(base, margin_fwhm=MARGIN_FWHM, track_width=w)
            g = em.Geometry(p)
            ej, rear = history_tracking(g, p, T)
            # cross-check against the ensemble code path
            st, *_ = em.run_tracking(g, p, toughness=T)
            assert st == len(rear), (seed, w, st, len(rear))
            runs[f"{w:g}"] = {"eject_at": ej.tolist(), "rear": rear, "width": w}
        data["samples"].append({"label": label, "seed": seed, "why": why,
                                "toughness": np.round(T, 3).tolist(), "runs": runs})
        print(label, seed, {k: len(v["rear"]) for k, v in runs.items()})
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print("wrote", OUT, f"{OUT.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
