"""Supplementary figure for the parallel vs. sequential milling ensemble.

Reads the ``ensemble_*.npz`` files written by ``ensemble_model.py`` and
recomputes the baseline dose profiles for panel (e).

Usage:
    uv run python figures/figure6_7/ensemble_model.py --n-seeds 500
    uv run python figures/figure6_7/ensemble_figure.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import FancyBboxPatch
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde

import ensemble_model as em
from ensemble_model import GRID_STEP

OUT = em.OUT
PAR, SEQ = "#1f77b4", "#ff7f0e"  # parallel, sequential (Tableau 10)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LATTICE = "#e34948"  # atom positions on the dose maps
# Colour roles (one job each):
#   Tableau blue / orange      parallel / sequential protocol
#   ColorBrewer RdYlGn         heat maps: sequential worse (red) ... better (green)
#   grey                       metric markers in (d) (filled o = atoms, open D = dose)
#   Tableau purple / cyan      heat-map slices: outline in (h, i), matching tint band in (d)
M_ATOMS = M_DOSE = "#555555"
SLICE = {"Box width (nm)": ("#9467bd", "#ece4f4"), "Vacuum margin (FWHM)": ("#17becf", "#dcf4f7")}
N_PROFILE_SEEDS = 500

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.5, "legend.frameon": False, "savefig.dpi": 300,
})


LABEL_SIZE = 13  # side of the square panel-label box (points)


def panel_label(ax, letter, x=0.02, y=0.97, xycoords="axes fraction", box_alignment=(0, 1)):
    """Fixed-size, rounded square panel label in the style of the existing manuscript figures."""
    da = DrawingArea(LABEL_SIZE, LABEL_SIZE, 0, 0)
    da.add_artist(FancyBboxPatch((0, 0), LABEL_SIZE, LABEL_SIZE, fc="#ffffff", ec=INK, lw=0.8,
                                 boxstyle="round,pad=0,rounding_size=2.5"))
    da.add_artist(Text(LABEL_SIZE / 2, LABEL_SIZE / 2, letter, ha="center", va="center_baseline",
                       fontsize=10, color=INK))
    ax.add_artist(AnnotationBbox(da, (x, y), xycoords=xycoords, box_alignment=box_alignment,
                                 frameon=False, pad=0, zorder=10))


def load(key):
    d = np.load(OUT / f"ensemble_{key.replace(':', '_').replace(',', '_')}.npz")
    return {k: d[k] for k in d.files}


def paired(ax, d, metric, scale, title, letter):
    """Paired scatter (one dot per seed) with marginal KDEs on identical axes."""
    x, y = d[f"parallel__{metric}"] / scale, d[f"sequential__{metric}"] / scale
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    pad = 0.05 * (hi - lo)
    lim = (lo - pad, hi + pad)
    ticks = [t for t in MaxNLocator(4).tick_values(*lim) if lim[0] <= t <= lim[1]]
    ax.plot(lim, lim, color=MUTED, lw=1, ls="--", zorder=1)
    ax.scatter(x, y, s=6, color=INK, alpha=0.35, linewidths=0, zorder=2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_aspect("equal")
    # description line (smaller) above the protocol name
    ax.set_xlabel(title, fontsize=6.5, color=MUTED)
    ax.annotate("Parallel", xy=(0.5, 0), xycoords=ax.xaxis.label, xytext=(0, -1),
                textcoords="offset points", ha="center", va="top", fontsize=8, color=INK)
    ax.set_ylabel("Sequential")
    t = lim[0] + 0.8 * (lim[1] - lim[0])
    ax.annotate("y = x", (t, t), xytext=(-3, 3), textcoords="offset points", rotation=45,
                ha="right", va="bottom", color=MUTED, fontsize=7, rotation_mode="anchor")

    grid = np.linspace(*lim, 300)
    top = ax.inset_axes([0, 1.04, 1, 0.28], sharex=ax)
    right = ax.inset_axes([1.04, 0, 0.28, 1], sharey=ax)
    for m_ax, v, color, horiz in ((top, x, PAR, True), (right, y, SEQ, False)):
        k = gaussian_kde(v)(grid)
        if horiz:
            m_ax.fill_between(grid, k, color=color, alpha=0.25, linewidth=0)
            m_ax.plot(grid, k, color=color, lw=1.2)
            m_ax.set_ylim(0, None)
        else:
            m_ax.fill_betweenx(grid, k, color=color, alpha=0.25, linewidth=0)
            m_ax.plot(k, grid, color=color, lw=1.2)
            m_ax.set_xlim(0, None)
        m_ax.axis("off")
    return top


X_MIN = -29  # panel (d) x-axis floor; values below it are drawn as arrows


def sensitivity(ax):
    """Percent reduction of the mean (sequential vs parallel) with 95% bootstrap CI."""
    summary = json.loads((OUT / "ensemble_summary.json").read_text())
    in_sweep = lambda name: (lambda k, p: k.startswith(name + ":"))
    base_margin = em.Params().margin_fwhm
    groups = [
        ("Box width (nm)", lambda k, p: k.startswith("margin_width:") and p["margin_fwhm"] == base_margin
         and p["track_width"] < em.Params().box_width, lambda p: f"{p['track_width']:g}"),
        ("Vacuum margin (FWHM)", lambda k, p: k.startswith("margin_width:")
         and p["track_width"] == em.Params().track_width, lambda p: f"{p['margin_fwhm']:g}"),
        ("Beam width σ (nm)", in_sweep("sigma"), lambda p: f"{p['sigma']:g}"),
        ("Edge ejection (× default)", in_sweep("edge_mult"), lambda p: f"×{p['edge_B'] / em.Params().edge_B:g}"),
        ("B ejection (× default)", in_sweep("boron_mult"), lambda p: f"×{p['bulk_B'] / em.Params().bulk_B:g}"),
        ("Ejection-rate scale k", in_sweep("k_eject"), lambda p: f"×{p['k_eject'] / 5e13:g}"),
    ]
    base = em.Params()
    y, ticks, labels, heads = 0, [], [], []
    series = (("offtarget_ejected", M_ATOMS, "o", -0.14), ("offtarget_dose", M_DOSE, "D", 0.14))
    for title, select, fmt in groups:
        keys = [k for k in summary if select(k, summary[k]["params"])]
        if title.startswith("Vacuum margin"):
            keys.sort(key=lambda k: summary[k]["params"]["margin_fwhm"])
        heads.append((y, title))
        y_head = y
        y += 1
        for k in keys:
            red, p = summary[k]["summary"]["reduction_pct"], summary[k]["params"]
            is_base = all(p[f] == getattr(base, f) for f in p)
            for metric, color, marker, dy in series:
                m, lo, hi = red[metric]
                if m < X_MIN:  # off-scale: arrow at the axis edge with the value printed
                    ax.annotate("", xy=(X_MIN + 0.3, y + dy), xytext=(X_MIN + 3.2, y + dy),
                                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3), zorder=3)
                    # label right of the row's other markers so it cannot sit on them
                    others = [red[mm][0] for mm, *_ in series if red[mm][0] >= X_MIN]
                    x_txt = max([X_MIN + 4.0, *(o + 3.0 for o in others)])
                    ax.text(x_txt, y + dy, f"{m:.0f}%", color=color, fontsize=6.5, va="center")
                    continue
                ax.plot([lo, hi], [y + dy] * 2, color=color, lw=1.5, solid_capstyle="round", zorder=2)
                hollow = marker == "D"
                ax.scatter(m, y + dy, s=18, marker=marker, zorder=3, linewidths=1.1 if hollow else 0.8,
                           facecolors="#fcfcfb" if hollow else color, edgecolors=color if hollow else "#fcfcfb")
            ticks.append(y)
            labels.append(fmt(p) + ("  (default)" if is_base else ""))
            y += 1
        if title in SLICE:  # tint band linking this sweep to its outlined slice in (h, i)
            ax.axhspan(y_head - 0.5, y - 0.5, color=SLICE[title][1], zorder=0, lw=0)
        y += 0.4
    ax.axvline(0, color=MUTED, lw=1, ls="--", zorder=1)
    header_texts = []
    for yy, t in heads:
        header_texts.append(ax.text(-0.02, yy, t, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontweight="bold", color=INK, fontsize=7))
    ax.set_yticks(ticks, labels)
    ax.tick_params(axis="y", length=0, labelsize=7)
    ax.set_ylim(y - 0.4, -2.0)
    ax.set_xlim(X_MIN, 45)
    ax.set_xlabel("Off-target reduction with sequential milling (%)")
    ax.text(1, -1.3, "sequential better →", color=MUTED, va="center", ha="left", fontsize=6.5)
    ax.text(-1, -1.3, "← parallel better", color=MUTED, va="center", ha="right", fontsize=6.5)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.scatter([], [], color=M_ATOMS, marker="o", label="atoms\nejected")
    ax.scatter([], [], facecolors="#fcfcfb", edgecolors=M_DOSE, linewidths=1.1, marker="D",
               label="dose")
    leg = ax.legend(loc="lower left", handletextpad=0.3, fontsize=6.5, framealpha=1, facecolor="#ffffff",
                    edgecolor=MUTED, frameon=True, fancybox=False, borderaxespad=0.4, title="off-target",
                    title_fontsize=6.5, alignment="left")
    leg.get_frame().set_linewidth(0.6)
    ax.set_title("Off-target reduction vs. model parameters", loc="left", color=INK)
    return header_texts


def profiles(ax, map_axes, letters=("e", "f", "g")):
    """Mean cumulative dose maps (map_axes) and the profile along a line 0.1 nm
    outside the long edge of the target box (ax)."""
    p = em.Params()
    g = em.Geometry(p)
    y_line = g.yb[1] + 0.1
    iy = np.argmin(np.abs(g.y_grid - y_line))
    runs = {"parallel": [], "sequential": []}
    maps = {"parallel": 0, "sequential": 0}
    for s in range(N_PROFILE_SEEDS):
        T = np.random.default_rng(s).exponential(size=len(em.XY))  # same samples as the ensemble
        for name, fn in (("parallel", em.run_parallel), ("sequential", em.run_tracking)):
            _, _, d, _ = fn(g, p, thresholds=T)
            runs[name].append(d[iy])
            maps[name] = maps[name] + d / N_PROFILE_SEEDS
    x = g.x_grid
    single = 1e6  # plot in 10^6 e-/nm^2 (model units)
    xlim = (g.xb[0] - 0.6, g.xb[1] + 0.6)

    # Dose maps: viridis, matching the dose maps of main-text Figure 7
    vmax = max(m.max() for m in maps.values()) / single
    h = GRID_STEP / 2
    extent = (x[0] - h, x[-1] + h, g.y_grid[0] - h, g.y_grid[-1] + h)
    for m_ax, (name, color) in zip(map_axes, (("parallel", PAR), ("sequential", SEQ))):
        im = m_ax.imshow(maps[name] / single, origin="lower", extent=extent, cmap="viridis",
                         vmin=0, vmax=vmax, aspect="auto", interpolation="nearest")
        m_ax.add_patch(plt.Rectangle((g.xb[0], g.yb[0]), g.xb[1] - g.xb[0], g.yb[1] - g.yb[0],
                                     fill=False, ec="#fcfcfb", lw=0.8, ls="--"))
        m_ax.scatter(em.XY[em.LAYER == 1, 0], em.XY[em.LAYER == 1, 1], s=3.5, color=LATTICE,
                     alpha=0.75, linewidths=0, zorder=2)
        m_ax.axhline(g.y_grid[iy], color=color, lw=1.5, zorder=3)  # the grid row sampled in (g)
        m_ax.set_xlim(xlim)
        fig = m_ax.figure
        pos = m_ax.get_position()
        w_in, h_in = pos.width * fig.get_figwidth(), pos.height * fig.get_figheight()
        y_span = (xlim[1] - xlim[0]) * h_in / w_in  # equal aspect without changing axes width
        yc = np.mean(g.yb)
        m_ax.set_ylim(yc - y_span / 2, yc + y_span / 2)
        m_ax.set_ylabel("y (nm)")
        m_ax.tick_params(labelbottom=False)
        shown = "parallel" if name == "parallel" else f"sequential, {p.track_width:g} nm box"
        m_ax.text(0.99, 0.94, f"{shown}, mean of {N_PROFILE_SEEDS} runs", transform=m_ax.transAxes,
                  ha="right", va="top", color=INK, fontsize=7,
                  bbox=dict(fc="#fcfcfb", ec="none", pad=1.2, alpha=0.85))
        for sp in m_ax.spines.values():
            sp.set_visible(True)
    cax = map_axes[0].inset_axes([1.02, -1.08, 0.03, 2.08])
    cb = plt.colorbar(im, cax=cax)
    cb.set_label("Cumulative dose\n(10⁶ e⁻ nm⁻²)", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_visible(False)
    map_axes[0].set_title("Cumulative dose around the cut (baseline)", loc="left", color=INK)
    for m_ax, letter in zip((*map_axes, ax), letters):
        panel_label(m_ax, letter, y=0.94 if m_ax is not ax else 0.97)

    for name, color in (("parallel", PAR), ("sequential", SEQ)):
        r = np.array(runs[name]) / single
        ax.fill_between(x, np.percentile(r, 10, 0), np.percentile(r, 90, 0), color=color,
                        alpha=0.18, linewidth=0)
        ax.plot(x, r.mean(0), color=color, label=f"{name} (mean, 10–90% band)")
    ax.axvspan(*g.xb, color=GRID, alpha=0.5, zorder=0, linewidth=0)
    ax.text(np.mean(g.xb), 0.03, "extent of target box", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", color=MUTED, fontsize=7)
    ax.set_xlim(xlim)
    ax.set_ylim(0, None)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel(f"Dose along y = {g.y_grid[iy]:.2f} nm\n(10⁶ e⁻ nm⁻²)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.08), fontsize=7, framealpha=0.9,
              facecolor="#fcfcfb", edgecolor="none")
    ax.set_title("")


# diverging: blue = parallel better, neutral grey = no difference, orange = sequential better
CMAP_PS = plt.get_cmap("RdYlGn")


def margin_width_grid(summary, metric):
    rows = [(v["params"]["margin_fwhm"], v["params"]["track_width"], v["summary"]["reduction_pct"][metric][0])
            for k, v in summary.items() if k.startswith("margin_width:")]
    ms, ws = sorted({r[0] for r in rows}), sorted({r[1] for r in rows})
    z = np.full((len(ms), len(ws)), np.nan)
    for m, w, val in rows:
        z[ms.index(m), ws.index(w)] = val
    return ms, ws, z


def heatmap(ax, summary, metric, title, letter, lim, colorbar=True):
    """Reduction with sequential vs vacuum margin (rows) and box width (columns)."""
    ms, ws, z = margin_width_grid(summary, metric)
    im = ax.imshow(np.clip(z, -lim, lim), cmap=CMAP_PS, norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim),
                   origin="lower", aspect="auto")
    zb = np.where(np.isnan(z), -np.inf, z)
    zb[:, np.array(ws) >= em.Params().box_width] = -np.inf  # whole-target box = parallel
    best = np.argmax(zb, axis=1)
    for i in range(len(ms)):
        for j in range(len(ws)):
            if np.isnan(z[i, j]):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fc="#ffffff", ec=GRID, hatch="////", lw=0))
                continue
            ax.text(j, i, f"{z[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                    color="#ffffff" if abs(z[i, j]) > 0.6 * lim else INK,
                    fontweight="bold" if j == best[i] else "normal")
    sig = em.Params().sigma
    ax.set_xticks(range(len(ws)), [f"{w:g}" for w in ws])
    ax.set_yticks(range(len(ms)), [f"{m:g} ({m * em.FWHM_PER_SIGMA * sig:.2f})" for m in ms])
    ax.tick_params(labelsize=7)
    ax.set_xlabel("Box width along the cut (nm)")
    ax.set_ylabel("Vacuum margin, FWHM (nm)")
    # outline the slices plotted in (d): box-width sweep = default-margin row,
    # margin sweep = default-width column (both excluding the whole-target box)
    base = em.Params()
    i0 = ms.index(base.margin_fwhm)
    j_in = [j for j, w in enumerate(ws) if w < base.box_width]
    ax.add_patch(plt.Rectangle((j_in[0] - 0.5, i0 - 0.5), len(j_in), 1, fill=False,
                               ec=SLICE["Box width (nm)"][0], lw=2.2, zorder=4))
    j0 = ws.index(base.track_width)
    i_in = [i for i in range(len(ms)) if not np.isnan(z[i, j0])]
    ax.add_patch(plt.Rectangle((j0 - 0.5, i_in[0] - 0.5), 1, len(i_in), fill=False,
                               ec=SLICE["Vacuum margin (FWHM)"][0], lw=2.2, zorder=4))
    ax.set_title(title, loc="left", color=INK)
    if colorbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.06, pad=0.03, extend="both")
        cb.set_label("reduction with sequential (%)", fontsize=7)
        cb.ax.tick_params(labelsize=7)
        cb.outline.set_visible(False)


def main():
    d = load("baseline:")
    fig = plt.figure(figsize=(7.2, 10.95), facecolor="#fcfcfb")
    # rows (a-c) | (d-g) | (h, i): tight gap above (d), room above (h, i) for the (d)/(g) axis labels
    outer = fig.add_gridspec(2, 1, height_ratios=[0.72, 2.99 * 1.06], hspace=0.1)
    gs_top = outer[0].subgridspec(1, 3, wspace=1.45)
    gs_low = outer[1].subgridspec(2, 1, height_ratios=[2.35, 0.64], hspace=0.26)
    scatter_panels = []
    for i, (metric, scale, title, letter) in enumerate((
            ("offtarget_ejected", 1, "Atoms ejected outside target", "a"),
            ("offtarget_dose", 1e6, "Total off-target dose (10⁶ e⁻)", "b"),
            ("offtarget_max", 1e6, "Peak off-target dose (10⁶ e⁻ nm⁻²)", "c"))):
        ax = fig.add_subplot(gs_top[0, i])
        scatter_panels.append((ax, paired(ax, d, metric, scale, title, letter), letter))
    sub = gs_low[0].subgridspec(1, 2, width_ratios=[1, 1.25], wspace=0.35)
    ax_d = fig.add_subplot(sub[0, 0])
    headers = sensitivity(ax_d)
    col = sub[0, 1].subgridspec(3, 1, height_ratios=[0.9, 0.9, 1.85], hspace=0.12)
    m1 = fig.add_subplot(col[0])
    m2 = fig.add_subplot(col[1], sharex=m1)
    profiles(fig.add_subplot(col[2], sharex=m1), (m1, m2))
    summary = json.loads((OUT / "ensemble_summary.json").read_text())
    heat_axes = []
    hrow = gs_low[1].subgridspec(1, 2, wspace=0.62)
    for j, (metric, title, letter, lim) in enumerate((
            ("offtarget_ejected", "Off-target atoms ejected", "h", 35),
            ("offtarget_dose", "Total off-target dose", "i", 50))):
        hax = fig.add_subplot(hrow[0, j])
        heatmap(hax, summary, metric, title, letter, lim)
        heat_axes.append((hax, letter))
    # (d) label outside the plot: centred on the row-header column, level with the title.
    # Anchored in ax_d's axes coordinates so the tight-bbox crop at save time cannot shift it.
    fig.canvas.draw()
    inv = ax_d.transAxes.inverted()
    xs = [inv.transform(t.get_window_extent().get_points()) for t in headers]
    x_mid = (min(p[0, 0] for p in xs) + max(p[1, 0] for p in xs)) / 2
    tb = inv.transform(ax_d._left_title.get_window_extent().get_points())
    panel_label(ax_d, "d", x=x_mid, y=tb[:, 1].mean(), box_alignment=(0.5, 0.5))
    # (a)-(c): centred on the y-label + tick-label column, level with the middle of the top KDE
    for ax, top, letter in scatter_panels:
        inv = ax.transAxes.inverted()
        ext = [ax.yaxis.label.get_window_extent(), *[t.get_window_extent() for t in ax.get_yticklabels()]]
        pts = [inv.transform(e.get_points()) for e in ext]
        x_mid = (min(p[0, 0] for p in pts) + max(p[1, 0] for p in pts)) / 2
        y_mid = inv.transform(top.get_window_extent().get_points())[:, 1].mean()
        panel_label(ax, letter, x=x_mid, y=y_mid, box_alignment=(0.5, 0.5))
    # (h), (i): outside the plot, centred on the y-tick column, level with the title
    for hax, letter in heat_axes:
        inv = hax.transAxes.inverted()
        pts = [inv.transform(t.get_window_extent().get_points()) for t in hax.get_yticklabels()]
        x_mid = (min(p[0, 0] for p in pts) + max(p[1, 0] for p in pts)) / 2
        tb = inv.transform(hax._left_title.get_window_extent().get_points())
        panel_label(hax, letter, x=x_mid, y=tb[:, 1].mean(), box_alignment=(0.5, 0.5))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"ensemble_figure.{ext}", bbox_inches="tight", facecolor=fig.get_facecolor())
    print("wrote", OUT / "ensemble_figure.png")


if __name__ == "__main__":
    main()
