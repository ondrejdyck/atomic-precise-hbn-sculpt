"""Figure 4 (model explanation): how the milling model works.

(a) sample and target, (b) random ejection thresholds, (c) the two protocols
mid-run, (d) one scan of each protocol (dose footprint), (e) one sample milled
both ways at equal beam times, (f) progress traces for that sample.

Usage:
    uv run python figures/figure6_7/model_figure.py
"""

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

import ensemble_figure as ef
import ensemble_model as em
from export_viewer_data import pick_seeds

B_COL, N_COL = "#b9b9b9", "#555555"
EDGE, OFF, GONE = "#9467bd", "#d62728", "#cfcfcf"
P = em.Params()
G = em.Geometry(P)
XB0, XB1 = G.xb
YB0, YB1 = G.yb
VIEW = (XB0 - 0.55, XB1 + 0.55, YB0 - 0.42, YB1 + 0.42)
LAT = (XB0 - 0.55, XB1 + 0.55, YB0 - 0.42, YB1 + 0.42)  # region of lattice drawn in (a)-(d)
VIEW_A = (XB0 - 0.55, XB1 + 0.95, YB0 - 0.42, YB1 + 0.62)
VIEW_C = (XB0 - 0.55, XB1 + 0.55, YB0 - 0.68, YB1 + 0.68)


def in_lat(mask=True):
    x, y = em.XY[:, 0], em.XY[:, 1]
    return mask & (x > LAT[0]) & (x < LAT[1]) & (y > LAT[2]) & (y < LAT[3])
TOP = em.LAYER == 1


def record_parallel(T, track=None):
    alive, H = np.ones(len(em.XY), bool), np.zeros(len(em.XY))
    eject_at, h = np.zeros(len(em.XY), int), []
    n = 0
    while alive[G.target].any():
        n += 1
        new = em.eject_threshold(alive, G.par_atoms, P, (T, H))
        eject_at[alive & ~new] = n
        alive = new
        if track is not None:
            h.append(H[track])
    return eject_at, [XB0] * n, np.array(h), 1.0


def record_tracking(T, track=None):
    alive, H = np.ones(len(em.XY), bool), np.zeros(len(em.XY))
    eject_at, rear, h = np.zeros(len(em.XY), int), [], []
    s, d = P.track_width, P.track_step
    k_max = int(round((XB1 - s - XB0) / d))
    k = 0
    while alive[G.target].any():
        x_rear = em.XY[alive & G.target, 0].min()
        k_want = int(np.floor((x_rear - em.margin_nm(P) - XB0) / d + 1e-9))
        k = min(max(k, k_want), k_max)
        rear.append(XB0 + k * d)
        new = em.eject_threshold(alive, em._track_field(G, P, k)[1], P, (T, H))
        eject_at[alive & ~new] = len(rear)
        alive = new
        if track is not None:
            h.append(H[track])
    return eject_at, rear, np.array(h), P.track_width / P.box_width


def setup_lattice_ax(ax, view=VIEW):
    ax.set_xlim(view[0], view[1])
    ax.set_ylim(view[2], view[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def draw_state(ax, eject_at, k, box=None, color=None, s=9):
    """Top-layer atoms after k scans; box = (rear, width) of the box being scanned."""
    alive = (eject_at == 0) | (eject_at > k)
    nb = em.ADJ @ alive.astype(float)
    sel = in_lat(TOP)
    x, y = em.XY[:, 0], em.XY[:, 1]
    gone_t = sel & ~alive & G.target
    gone_o = sel & ~alive & ~G.target
    bulk = sel & alive & (nb >= 3)
    edge = sel & alive & (nb < 3)
    ax.scatter(x[gone_t], y[gone_t], s=s, facecolors="none", edgecolors=GONE, linewidths=0.6)
    ax.scatter(x[gone_o], y[gone_o], s=s, facecolors="none", edgecolors=OFF, linewidths=1.0)
    ax.scatter(x[bulk], y[bulk], s=s, color=em_col(bulk), linewidths=0)
    ax.scatter(x[edge], y[edge], s=s, color=EDGE, linewidths=0)
    ax.add_patch(Rectangle((XB0, YB0), XB1 - XB0, YB1 - YB0, fill=False, ec=ef.MUTED, lw=0.8, ls="--"))
    if box is not None:
        r, w = box
        ax.add_patch(Rectangle((r, YB0), w, YB1 - YB0, fc=color, alpha=0.18, ec=color, lw=1.4))
    return alive


def em_col(mask):
    return np.where(em.IS_N[mask], N_COL, B_COL)


def panel_a(ax):
    keep = in_lat(TOP)
    ax.scatter(em.XY[keep, 0], em.XY[keep, 1], s=10, c=np.where(em.IS_N[keep], N_COL, B_COL), linewidths=0)
    ax.add_patch(Rectangle((XB0, YB0), XB1 - XB0, YB1 - YB0, fill=False, ec=ef.INK, lw=1.0, ls="--"))
    setup_lattice_ax(ax, VIEW_A)
    arrow = dict(arrowstyle="<->", color=ef.INK, lw=0.8, shrinkA=0, shrinkB=0)
    ax.annotate("", (XB0, YB1 + 0.5), (XB1, YB1 + 0.5), arrowprops=arrow)
    ax.text((XB0 + XB1) / 2, YB1 + 0.53, f"target {P.box_width:g} nm", ha="center", va="bottom", fontsize=7)
    xa = LAT[1] + 0.12
    ax.annotate("", (xa, YB0), (xa, YB1), arrowprops=arrow)
    for yy in (YB0, YB1):
        ax.plot([LAT[1] - 0.05, xa + 0.05], [yy, yy], color=ef.INK, lw=0.5, ls=":")
    ax.text(xa + 0.05, (YB0 + YB1) / 2, f"{P.box_height:g}\nnm", ha="left", va="center", fontsize=7)
    ax.set_title("AA′ bilayer h-BN (top layer)", loc="center", color=ef.INK)


def side_label(ax, text, color):
    ax.text(-0.02, 0.5, text, transform=ax.transAxes, rotation=90, ha="right", va="center", color=color,
            fontweight="bold", fontsize=7.5)


def panel_c(ax_par, ax_seq, ej_par, ej_seq, rear_seq):
    draw_state(ax_par, ej_par, 3, box=(XB0, P.box_width), color=ef.PAR)
    setup_lattice_ax(ax_par, VIEW_C)
    side_label(ax_par, "parallel", ef.PAR)
    ax_par.set_title("Protocols mid-run", loc="center", color=ef.INK)
    k = len(rear_seq) // 2
    alive = draw_state(ax_seq, ej_seq, k, box=(rear_seq[k], P.track_width), color=ef.SEQ)
    setup_lattice_ax(ax_seq, VIEW_C)
    side_label(ax_seq, "edge-following", ef.SEQ)
    r = rear_seq[k]
    x_rear = em.XY[alive & G.target, 0].min()
    arrow = dict(arrowstyle="<->", color=ef.INK, lw=0.8, shrinkA=0, shrinkB=0)
    y_top, y_bot = LAT[3] + 0.1, LAT[2] - 0.1
    # w above the lattice, m below it, with guide lines down to the box
    ax_seq.annotate("", (r, y_top), (r + P.track_width, y_top), arrowprops=arrow)
    ax_seq.text(r + P.track_width / 2, y_top + 0.03, "w", ha="center", va="bottom", fontsize=7.5, style="italic")
    ax_seq.annotate("", (r, y_bot), (x_rear, y_bot), arrowprops=arrow)
    ax_seq.text((r + x_rear) / 2, y_bot - 0.03, "m", ha="center", va="top", fontsize=7.5, style="italic")
    for xx, y0, y1 in ((r, y_bot, y_top), (r + P.track_width, YB1, y_top), (x_rear, y_bot, YB1)):
        ax_seq.plot([xx, xx], [y0, y1], color=ef.INK, lw=0.5, ls=":")
    ax_seq.annotate("", (r + P.track_width + 0.55, (YB0 + YB1) / 2), (r + P.track_width + 0.1, (YB0 + YB1) / 2),
                    arrowprops=dict(arrowstyle="-|>", color=ef.SEQ, lw=1.4))


def panel_threshold(ax, T):
    """Random ejection threshold: one Exp(1) value per atom, drawn once per sample."""
    x0, x1, y0, y1 = VIEW_A
    keep = in_lat(TOP)
    sc = ax.scatter(em.XY[keep, 0], em.XY[keep, 1], s=10, c=T[keep], cmap="Greys", vmin=0, vmax=3,
                    linewidths=0.2, edgecolors="#9a9a9a")
    ax.add_patch(Rectangle((XB0, YB0), XB1 - XB0, YB1 - YB0, fill=False, ec=ef.INK, lw=1.0, ls="--"))
    setup_lattice_ax(ax, VIEW_A)
    ax.set_title("Random ejection threshold", loc="center", color=ef.INK)
    cax = ax.inset_axes([1.02, 0.1, 0.025, 0.8])
    cb = plt.colorbar(sc, cax=cax)
    cb.set_ticks([0, 1, 2, 3])
    cb.set_label("T", fontsize=6.5, labelpad=1)
    cb.ax.tick_params(labelsize=6, length=2)
    cb.outline.set_visible(False)


def panel_dose(ax_par, ax_seq, rear):
    """One scan of each protocol: dose footprint on a shared scale."""
    k = int(round((rear - XB0) / P.track_step))
    fields = ((ax_par, G.par_grid, (XB0, P.box_width), 1.0), (ax_seq, em._track_field(G, P, k)[0],
              (rear, P.track_width), P.track_width / P.box_width))
    vmax = max(f.max() for _, f, _, _ in fields) / 1e6
    h = em.GRID_STEP / 2
    ext = (G.x_grid[0] - h, G.x_grid[-1] + h, G.y_grid[0] - h, G.y_grid[-1] + h)
    for ax, field, (r, w), cost in fields:
        im = ax.imshow(field / 1e6, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=vmax,
                       interpolation="nearest")
        # show the dose only over the drawn lattice, so (d) lines up with (c)
        im.set_clip_path(Rectangle((LAT[0], LAT[2]), LAT[1] - LAT[0], LAT[3] - LAT[2], transform=ax.transData))
        keep = in_lat(TOP)
        ax.scatter(em.XY[keep, 0], em.XY[keep, 1], s=2.5, color=ef.LATTICE, alpha=0.6, linewidths=0)
        ax.add_patch(Rectangle((r, YB0), w, YB1 - YB0, fill=False, ec="#ffffff", lw=0.9, ls="--"))
        setup_lattice_ax(ax, VIEW_C)
        ax.text(LAT[1] - 0.06, LAT[2] + 0.06, f"scan time {cost:.2f}", ha="right", va="bottom", fontsize=6.5,
                color="#ffffff")
    ax_par.set_title("One scan dose footprint", loc="center", color=ef.INK)
    cax = ax_seq.inset_axes([1.03, 0.0, 0.035, 2.06])
    cb = plt.colorbar(im, cax=cax)
    cb.set_label("dose per scan\n(10⁶ e⁻ nm⁻²)", fontsize=6.5, labelpad=2)
    cb.ax.tick_params(labelsize=6, length=2)
    cb.outline.set_visible(False)


def panel_e(axes, ej_par, ej_seq, rear_seq, c_par, c_seq, times):
    for j, t in enumerate(times):
        kp = min(len([1]) * 0 + int(np.floor(t / c_par + 1e-9)), int(ej_par.max()))
        ks = min(int(np.floor(t / c_seq + 1e-9)), len(rear_seq))
        boxp = (XB0, P.box_width) if kp < ej_par.max() else None
        boxs = (rear_seq[ks], P.track_width) if ks < len(rear_seq) else None
        draw_state(axes[0][j], ej_par, kp, box=boxp, color=ef.PAR, s=5)
        draw_state(axes[1][j], ej_seq, ks, box=boxs, color=ef.SEQ, s=5)
        for row, k, ej in ((0, kp, ej_par), (1, ks, ej_seq)):
            ax = axes[row][j]
            setup_lattice_ax(ax)
            off = int(((ej > 0) & (ej <= k) & ~G.target).sum())
            ax.text(0.5, -0.04, f"{off} off-target lost", transform=ax.transAxes, ha="center", va="top",
                    fontsize=6.5, color=OFF)
        axes[0][j].set_title(f"beam time t = {t:.1f}", loc="center", color=ef.INK, fontsize=7.5, pad=14)
    for r, (lab, col) in enumerate((("parallel", ef.PAR), ("edge-following", ef.SEQ))):
        axes[r][0].text(0.0, 1.03, lab, transform=axes[r][0].transAxes, ha="left", va="bottom", color=col,
                        fontweight="bold", fontsize=7)


def panel_f(ax_left, ax_off, ax_box, ej_par, ej_seq, rear_seq, c_par, c_seq):
    for ej, c, col, lab in ((ej_par, c_par, ef.PAR, "parallel"), (ej_seq, c_seq, ef.SEQ, "edge-following")):
        n = int(ej.max())
        ks = np.arange(n + 1)
        left = [int((G.target & ((ej == 0) | (ej > k))).sum()) for k in ks]
        off = [int(((ej > 0) & (ej <= k) & ~G.target).sum()) for k in ks]
        ax_left.step(ks * c, left, where="post", color=col, lw=1.6, label=lab)
        ax_off.step(ks * c, off, where="post", color=col, lw=1.6, label=lab)
    for ax, lab in ((ax_left, "target atoms left"), (ax_off, "off-target lost")):
        ax.set_xlabel("beam time")
        ax.set_ylabel(lab, fontsize=7)
        ax.grid(color=ef.GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
    ax_left.legend(loc="upper right", fontsize=6.5)
    ax_box.step(np.arange(len(rear_seq)) * c_seq, np.array(rear_seq) - XB0, where="post", color=ef.SEQ, lw=1.6)
    ax_box.set_xlabel("beam time")
    ax_box.set_ylabel("box position (nm)", fontsize=7)
    ax_box.grid(color=ef.GRID, lw=0.6)
    ax_box.set_axisbelow(True)
    ax_box.set_xlim(0, None)


def main():
    seed = dict((lab, s) for lab, s, _ in pick_seeds())["typical"]
    T = np.random.default_rng(seed).exponential(size=len(em.XY))
    ej_par, _, _, c_par = record_parallel(T)
    ej_seq, rear_seq, _, c_seq = record_tracking(T)
    # sanity: same scan counts as the ensemble code paths
    assert int(ej_par.max()) == em.run_parallel(G, P, thresholds=T)[0]
    assert len(rear_seq) == em.run_tracking(G, P, thresholds=T)[0]

    fig = plt.figure(figsize=(7.2, 8.4), facecolor="#fcfcfb")
    gs = fig.add_gridspec(4, 2, height_ratios=[0.5, 1.15, 0.8, 0.5], hspace=0.42, wspace=0.14)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    panel_a(ax_a)
    panel_threshold(ax_b, T)
    csub = gs[1, 0].subgridspec(2, 1, hspace=0.06)
    ax_cp, ax_cs = fig.add_subplot(csub[0]), fig.add_subplot(csub[1])
    panel_c(ax_cp, ax_cs, ej_par, ej_seq, rear_seq)
    dsub = gs[1, 1].subgridspec(2, 1, hspace=0.06)
    ax_dp, ax_ds = fig.add_subplot(dsub[0]), fig.add_subplot(dsub[1])
    panel_dose(ax_dp, ax_ds, rear_seq[len(rear_seq) // 2])
    t_end = max(int(ej_par.max()) * c_par, len(rear_seq) * c_seq)
    times = np.round(np.array([0.2, 0.45, 0.7, 1.0]) * t_end, 1)
    esub = gs[2, :].subgridspec(2, len(times), hspace=0.38, wspace=0.06)
    axes_e = [[fig.add_subplot(esub[r, c]) for c in range(len(times))] for r in range(2)]
    panel_e(axes_e, ej_par, ej_seq, rear_seq, c_par, c_seq, times)
    fsub = gs[3, :].subgridspec(1, 3, wspace=0.45)
    ax_fl, ax_fo, ax_fb = (fig.add_subplot(fsub[i]) for i in range(3))
    panel_f(ax_fl, ax_fo, ax_fb, ej_par, ej_seq, rear_seq, c_par, c_seq)
    for ax, letter in ((ax_a, "a"), (ax_b, "b"), (ax_cp, "c"), (ax_dp, "d"), (axes_e[0][0], "e"), (ax_fl, "f")):
        ef.panel_label(ax, letter, x=-0.02, y=1.0, box_alignment=(1, 0))
    leg = [Line2D([], [], ls="", marker="o", ms=4, color=B_COL, label="B (bulk)"),
           Line2D([], [], ls="", marker="o", ms=4, color=N_COL, label="N (bulk)"),
           Line2D([], [], ls="", marker="o", ms=4, color=EDGE, label="edge atom (< 3 neighbours)"),
           Line2D([], [], ls="", marker="o", ms=4, mfc="none", mec=GONE, label="target atom ejected"),
           Line2D([], [], ls="", marker="o", ms=4, mfc="none", mec=OFF, label="off-target atom ejected")]
    pos = axes_e[1][0].get_position()
    fig.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, pos.y0 - 0.022), ncol=5, fontsize=6.5,
               frameon=False, handletextpad=0.2, columnspacing=1.0)
    out = ef.OUT / "model_figure.png"
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("wrote", out, "seed", seed)


if __name__ == "__main__":
    main()
