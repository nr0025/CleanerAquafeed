"""Figures of the paper, drawn from the result tables.

Every figure is drawn at the size at which it is printed (landscape A4, text width 9.7 in), so the font sizes
below are the printed sizes: 12 pt for tick labels, legends and annotations, 13 pt for axis labels and 14 pt
bold for panel letters. Text is set in Times New Roman; the figures have no grid lines.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .data import FISHMEAL, IMPACT_LABELS, IMPACTS  # noqa: E402
from .evaluation import rank_robust  # noqa: E402

# colour palette
NAVY, TEAL, RED, SKY, SALMON, LAVENDER, MINT, TAUPE = (
    "#3C5488", "#00A087", "#DC6A50", "#5DA9C9", "#F2A385", "#8491B4", "#9CD2C4", "#B09C85")
INK, INK2, LIGHT, OTHERS, DARK = "#1A1A1A", "#4D4D4D", "#BFBFBF", "#D9D9D9", "#6E6E6E"
DIET = {"FM35": TEAL, "FM45": NAVY, "FM55": RED}          # reference diets
STACK = [NAVY, SKY, TEAL, MINT, RED, SALMON, LAVENDER, TAUPE]
SEQ = ["#F7F9FC", "#D7E0EE", "#AEBFDA", "#7D96C0", "#53709F", "#3C5488", "#26365A"]
DISPLAY = {FISHMEAL: "Fishmeal (65% CP)", "Meat and bone": "Meat and bone meal", "Soy concentrate": "Soy protein concentrate"}

TICK, LABEL, PANEL = 12, 13, 14

# Times New Roman; where it is not installed, Liberation Serif or TeX Gyre Termes (same character widths)
SERIF = ["Times New Roman", "Liberation Serif", "TeX Gyre Termes"]
_installed = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((n for n in SERIF if n in _installed), "DejaVu Serif")

plt.rcParams.update({
    "font.family": "serif", "font.serif": SERIF + ["DejaVu Serif"], "mathtext.fontset": "custom", "mathtext.rm": FONT,
    "mathtext.it": f"{FONT}:italic", "mathtext.bf": f"{FONT}:bold", "mathtext.sf": FONT, "mathtext.cal": FONT,
    "font.size": TICK, "axes.titlesize": LABEL, "axes.labelsize": LABEL, "xtick.labelsize": TICK,
    "ytick.labelsize": TICK, "legend.fontsize": TICK, "legend.title_fontsize": TICK,
    "axes.edgecolor": INK, "axes.linewidth": 0.8, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "axes.axisbelow": True,
    "savefig.dpi": 600, "svg.fonttype": "none", "legend.frameon": False, "lines.solid_capstyle": "butt",
})


def _panel(ax, letter, x=-0.02, y=1.02):
    """Bold panel letter above the top-left corner of the axes (in axes coordinates of the figure)."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=PANEL, fontweight="bold", ha="right", va="bottom")


def _save(fig, out, name):
    d = out / "figures"
    d.mkdir(exist_ok=True)
    for old in d.glob(f"{name}.*"):
        old.unlink()
    # the tight box of matplotlib ignores the horizontal extent of axis labels; add them so that none is cut off
    extra = [lab for ax in fig.axes for lab in (ax.xaxis.label, ax.yaxis.label) if lab.get_text()]
    extra += [a for ax in fig.axes for a in ax.texts]
    extra += [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
    kw = dict(bbox_inches="tight", bbox_extra_artists=extra, pad_inches=0.06, facecolor="white")
    fig.savefig(d / f"{name}.png", **kw)
    fig.savefig(d / f"{name}.svg", **kw)
    plt.close(fig)
    # SVG keeps text as text; name Times New Roman first so that it is used wherever it is installed
    svg = d / f"{name}.svg"
    txt = svg.read_text(encoding="utf-8")
    family = ", ".join(f"'{n}'" for n in SERIF) + ", serif"
    # Microsoft Word does not read the CSS shorthand 'font: 700 12px ...' written by recent matplotlib and then
    # falls back to Arial; write the longhand properties instead
    def longhand(m):
        out = []
        for tok in (m.group(1) or "").split():
            out.append(f"font-weight: {'bold' if tok in ('700', 'bold') else tok}" if tok[0].isdigit() or tok in ("bold", "normal")
                       else f"font-style: {tok}")
        out += [f"font-size: {m.group(2)}px", f"font-family: {family}"]
        return "; ".join(out)
    txt = re.sub(r"font: ((?:[\w]+ )*?)?([\d.]+)px [^;\"]*", longhand, txt)
    txt = re.sub(r"font-family: [^;\"]*", f"font-family: {family}", txt)
    svg.write_text(txt, encoding="utf-8")


def _thousands(ax, axis="y"):
    fmt = mticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def fig1_reference(run):
    ev = run.load("reference_diets_evaluation.csv").set_index("diet")
    b = run.cfg["reference"]["benchmark"]
    metrics = [("cost_ratio", "Cost")] + [("ge", "GE")] + [(f"{k}_ratio", IMPACT_LABELS[k]) for k in IMPACTS]
    ev["ge"] = ev["ge_mj_kg"] / ev.loc[b, "ge_mj_kg"]
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    x = np.arange(len(metrics)); w = 0.26
    for i, diet in enumerate(ev.index):
        v = [ev.loc[diet, m] for m, _ in metrics]
        ax.bar(x + (i - 1) * w, v, w * 0.9, color=DIET.get(diet, LIGHT), label=diet, linewidth=0)
    ax.axhline(1.0, color=INK2, lw=0.8, ls="--", zorder=0)
    ax.set_xticks(x, [lab for _, lab in metrics])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel(f"Ratio to {b}")
    top = max(ev.loc[d, m] for d in ev.index for m, _ in metrics)
    ax.set_ylim(0, np.ceil(top * 1.18 * 5) / 5)
    ax.legend(ncol=3, loc="upper left", handlelength=1.2, columnspacing=1.4)
    _save(fig, run.out, "fig1_reference_diets")


def fig2_reference(run):
    g = run.load("growth_summary.csv").set_index("formula_id")
    ref = run.cfg["reference"]["labels"]
    gr = g.loc[ref]
    ts = run.load("growth_timeseries.csv")
    fig = plt.figure(figsize=(9.6, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.35], wspace=0.42)
    a1, a2, a3 = (fig.add_subplot(gs[0, i]) for i in range(3))
    cols = [DIET.get(d, LIGHT) for d in ref]
    for ax, col, fmt, lab in ((a1, "feed_unit_cost_cny_kg", "{:.2f}", "Feed unit cost (CNY kg$^{-1}$)"),
                              (a2, "feed_cost_population_cny", "{:,.0f}", "Total feed cost (CNY)")):
        ax.bar(range(len(ref)), gr[col], 0.62, color=cols, linewidth=0)
        for i, v in enumerate(gr[col]):
            ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=TICK, color=INK)
        ax.set_xticks(range(len(ref)), ref)
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, gr[col].max() * 1.15)
        ax.set_ylabel(lab)
    _thousands(a2)
    tmax = 0.0
    for d in ref:
        t = ts[ts.formula_id == d]
        a3.plot(t.day, t.W_g, color=DIET.get(d, LIGHT), lw=1.8)
        tmax = max(tmax, t.day.iloc[-1])
    a3.axhline(run.cfg["growth"].get("w_target_g", 50.0), color=INK2, lw=0.8, ls=":", zorder=0)
    xmax = 10 * np.ceil(tmax / 10)
    for k, d in enumerate(ref):
        days = ts[ts.formula_id == d].day.iloc[-1]
        a3.text(0.97, 0.05 + 0.09 * (len(ref) - 1 - k), f"{d}: {days:.1f} d", transform=a3.transAxes, ha="right", va="bottom", fontsize=TICK, color=DIET.get(d, INK))
    a3.set_xlim(0, xmax); a3.set_xticks(np.arange(0, xmax + 1, 10))
    a3.set_xlabel("Day"); a3.set_ylabel("Predicted body weight (g fish$^{-1}$)")
    _panel(a1, "a", x=-0.30); _panel(a2, "b", x=-0.30); _panel(a3, "c", x=-0.20)
    _save(fig, run.out, "fig2_reference_cost_growth")


def _spread_labels(items, gap):
    """Vertical positions for end-of-line labels, at least `gap` apart, kept in the order of the
    values and centred on them, so that no label drifts further from its line than needed."""
    items = sorted(items)
    pos = []
    for y, lab in items:
        y2 = y if not pos or y - pos[-1][0] >= gap else pos[-1][0] + gap
        pos.append([y2, lab])
    shift = np.mean([y for y, _ in items]) - np.mean([y for y, _ in pos])
    return [(y + shift, lab) for y, lab in pos]


def fig3_space(run):
    ev = run.load("candidates_evaluation.csv")
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    groups = [("Exceeds cost tolerance", ev["class"] == "cost", RED, 0.55),
              ("Other, within cost tolerance", ev["class"].isin(["partial", "limited", "low_gwp_burden"]), LIGHT, 0.8),
              ("Robust cleaner candidate", ev["robust"], NAVY, 0.7)]
    for lab, m, c, al in groups:
        ax.scatter(ev.cost_change_pct[m], ev.mean_env_change_pct[m], s=11, color=c, alpha=al, lw=0,
                   label=f"{lab} (n = {int(m.sum())})")
    front = run.load("benchmark_garcia_launay_cp_cap.csv").drop_duplicates(["C_r", "E_m"]).sort_values("C_r")
    front = front[front.alpha > 0]          # alpha = 0 is cost-only; its impacts are arbitrary among cost ties
    front = front[front.E_m <= front.E_m.cummin() + 1e-12]
    ax.plot(100 * (front.C_r - 1), 100 * (front.E_m - 1), color=INK, lw=1.2, marker="o", ms=3.5,
            label="Weighted-sum method")
    best = rank_robust(ev).iloc[0]
    ax.scatter([best.cost_change_pct], [best.mean_env_change_pct], s=80, facecolor="none", edgecolor=INK, lw=1.3, zorder=5)
    ax.annotate(best.formula_id, (best.cost_change_pct, best.mean_env_change_pct), xytext=(-9, -9), textcoords="offset points",
                ha="right", va="top", fontsize=TICK)
    ax.axvline(0, color=INK2, lw=0.8, ls="--", zorder=0); ax.axhline(0, color=INK2, lw=0.8, ls="--", zorder=0)
    ax.axvline(100 * run.rules.cost_tol, color=INK2, lw=0.8, ls=":", zorder=0)
    ax.set_xlabel("Cost change relative to FM45 (%)")
    ax.set_ylabel("Mean change across six indicators (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, columnspacing=1.6, handletextpad=0.4,
              markerscale=1.8)
    _save(fig, run.out, "fig3_candidate_space")


def fig4_heatmap(run):
    ev = run.load("candidates_evaluation.csv")
    top = rank_robust(ev).head(20)
    M = -top[[f"{k}_change_pct" for k in IMPACTS]].to_numpy()
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    vmax = max(90, np.ceil(M.max() / 10) * 10)
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(IMPACTS)), [IMPACT_LABELS[k] for k in IMPACTS])
    ax.set_yticks(range(len(top)), top.formula_id)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=TICK,
                    color="white" if M[i, j] > 0.55 * vmax else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("Reduction relative to FM45 (%)")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    _save(fig, run.out, "fig4_top20_impacts")


def fig5_cost(run):
    """(a) unit cost and (b) total feed cost of FM45 and the top 20 (dashed line: FM45). The cost under price
    uncertainty is shown in Figure 8a."""
    ev = run.load("candidates_evaluation.csv")
    g = run.load("growth_summary.csv").set_index("formula_id")
    top = rank_robust(ev).head(20)
    ids = list(top.formula_id)
    b = run.cfg["reference"]["benchmark"]
    rows = [b] + ids
    y = np.arange(len(rows))[::-1]
    col = [INK] + [NAVY] * len(ids)
    fig = plt.figure(figsize=(7.0, 5.0))
    gs = fig.add_gridspec(1, 2, wspace=0.12)
    a1 = fig.add_subplot(gs[0, 0]); a2 = fig.add_subplot(gs[0, 1], sharey=a1)
    for ax, c, lab in ((a1, "feed_unit_cost_cny_kg", "Feed unit cost (CNY kg$^{-1}$)"),
                       (a2, "feed_cost_population_cny", "Total feed cost (CNY)")):
        v = g.loc[rows, c].to_numpy(float)
        ax.barh(y, v, 0.66, color=col, linewidth=0)
        ax.axvline(v[0], color=INK, lw=0.9, ls="--")
        ax.set_xlabel(lab)
        ax.set_xlim(0, v.max() * 1.08)
    a1.xaxis.set_major_locator(mticker.MultipleLocator(2))
    a2.xaxis.set_major_locator(mticker.MultipleLocator(300)); _thousands(a2, "x")
    a1.set_yticks(y, rows)
    a1.set_ylim(-0.7, len(rows) - 0.3)
    a2.tick_params(axis="y", labelleft=False)
    for ax in (a1, a2):
        ax.tick_params(axis="y", length=0)
    _panel(a1, "a", x=-0.16); _panel(a2, "b", x=-0.03)
    _save(fig, run.out, "fig5_top20_cost")


def fig8_uncertainty(run):
    """Uncertainty analyses (Section 3.6): (a) cost ratio to FM45 of the top 20 under ingredient-price uncertainty
    (median and 5th-95th percentile; dashed line: cost tolerance of R1) with P_risk; (b) P_robust of the robust candidates
    and of the top 20 under LCA-coefficient uncertainty."""
    from .uncertainty import sample_prices
    ev = run.load("candidates_evaluation.csv")
    top = rank_robust(ev).head(20)
    ids = list(top.formula_id)
    b = run.cfg["reference"]["benchmark"]
    tol = run.rules.cost_tol if hasattr(run.rules, "cost_tol") else 0.10

    fig = plt.figure(figsize=(8.6, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.40)
    aa = fig.add_subplot(gs[0]); ab = fig.add_subplot(gs[1])

    # (a) price uncertainty
    rng = np.random.default_rng(run.cfg["uncertainty"]["price_seed"])
    P = sample_prices(run.db, run.cfg["uncertainty"]["price_samples"], 0.0, rng)
    cand = run.load("candidates_composition.csv").set_index("formula_id")
    X = cand.loc[ids, run.db.names].to_numpy(float)
    ratio = (X @ P.T) / (P @ run.x_bench)
    q = np.percentile(ratio, [5, 50, 95], axis=1)
    pr = (ratio > 1.0 + tol).mean(axis=1)
    y = np.arange(len(ids))[::-1]
    aa.hlines(y, q[0], q[2], color=NAVY, lw=1.8)
    aa.scatter(q[1], y, s=20, color=NAVY, zorder=3)
    aa.axvline(1.0 + tol, color=RED, lw=0.9, ls="--")
    aa.axvline(1.0, color=INK, lw=0.7, ls=":")
    aa.set_yticks(y, ids); aa.tick_params(axis="y", length=0)
    aa.set_ylim(-0.7, len(ids) - 0.3)
    aa.set_xlabel("Cost ratio to FM45")
    aa.xaxis.set_major_locator(mticker.MultipleLocator(0.05))
    x_txt = q[2].max() + 0.012
    for yi, p in zip(y, pr):
        aa.text(x_txt, yi, f"{p:.2f}", va="center", fontsize=TICK - 2, color=INK2)
    aa.text(x_txt, len(ids) - 0.2, "$P_{risk}$", va="bottom", fontsize=TICK - 1, color=INK2)
    aa.set_xlim(q[0].min() - 0.02, x_txt + 0.05)
    _panel(aa, "a", x=-0.20)

    # (b) LCA-coefficient uncertainty
    r = ev[ev.robust]
    cases = [("p_robust_30_indep", "±30%\nindep."), ("p_robust_30_shared", "±30%\nshared"),
             ("p_robust_50_indep", "±50%\nindep."), ("p_robust_50_shared", "±50%\nshared")]
    data = [r[c].to_numpy(float) for c, _ in cases]
    bp = ab.boxplot(data, widths=0.55, patch_artist=True, showfliers=False, whis=(5, 95),
                    medianprops=dict(color=INK, lw=1.2), whiskerprops=dict(color=INK2, lw=0.8),
                    capprops=dict(color=INK2, lw=0.8), boxprops=dict(lw=0.8, edgecolor=INK2))
    for patch in bp["boxes"]:
        patch.set_facecolor(LAVENDER); patch.set_alpha(0.55)
    jit = np.random.default_rng(1)
    for i, (c, _) in enumerate(cases, start=1):
        v = r[c].to_numpy(float)
        ab.scatter(i + jit.uniform(-0.18, 0.18, len(v)), v, s=4, color=OTHERS, zorder=2, linewidths=0)
        t = ev.set_index("formula_id").loc[ids, c].to_numpy(float)
        ab.scatter(i + jit.uniform(-0.12, 0.12, len(t)), t, s=12, color=NAVY, zorder=4, linewidths=0)
    ab.axhline(0.95, color=RED, lw=0.9, ls="--"); ab.axhline(0.5, color=INK, lw=0.7, ls=":")
    from matplotlib.lines import Line2D
    hb = [Line2D([], [], ls="", marker="o", ms=3.5, color=OTHERS, label="Robust candidates"),
          Line2D([], [], ls="", marker="o", ms=5, color=NAVY, label="Top 20")]
    ab.legend(handles=hb, loc="lower left", fontsize=TICK - 1, handletextpad=0.2, borderaxespad=0.2)
    ab.set_xticks(range(1, len(cases) + 1), [l for _, l in cases])
    ab.tick_params(axis="x", length=0, labelsize=TICK - 1)
    ab.set_ylim(0, 1.03); ab.set_ylabel("$P_{robust}$")
    _panel(ab, "b", x=-0.22)

    _save(fig, run.out, "fig8_uncertainty")


def fig9_sensitivity(run):
    """Sensitivity analyses (Section 3.6): (a) number of robust candidates when one eligibility threshold is changed;
    (b) optimal cleaner score of the direct LP when one inclusion limit is changed; (c) changes in cost and impacts of
    the least-cost FM35 diet relative to FM45 under three limits on total plant meals."""
    fig = plt.figure(figsize=(9.7, 8.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.85], hspace=0.32, wspace=0.42)
    ac = fig.add_subplot(gs[0, 0]); ad = fig.add_subplot(gs[0, 1]); ae = fig.add_subplot(gs[1, :])

    # (a) eligibility thresholds, one at a time
    th = run.load("threshold_one_at_a_time.csv")
    lab = {"cost tolerance": "Cost tolerance", "min. reduced indicators": "Min. reduced indicators",
           "burden-shifting tolerance": "Burden-shifting tol.", "GWP rule removed": "GWP rule (R2)"}
    rows, vals, base = [], [], []
    for par in ["cost tolerance", "min. reduced indicators", "burden-shifting tolerance", "GWP rule removed"]:
        t = th[th.parameter == par]
        for _, rr in t.iterrows():
            if par == "GWP rule removed":
                rows.append("R2 removed"); base.append(False)
            elif par == "min. reduced indicators":
                rows.append(f"R3: ≥ {int(rr.value)}"); base.append(int(rr.value) == 4)
            else:
                pre = "R1" if par == "cost tolerance" else "R4"
                rows.append(f"{pre}: {rr.value * 100:.0f}%"); base.append(abs(rr.value - 0.10) < 1e-9)
            vals.append(rr.n_robust)
    yy = np.arange(len(rows))[::-1]
    ac.barh(yy, vals, 0.66, color=[NAVY if bb else SKY for bb in base], linewidth=0)
    for yi, v in zip(yy, vals):
        ac.text(v + 8, yi, f"{v}", va="center", fontsize=TICK - 2, color=INK2)
    ac.set_yticks(yy, rows); ac.tick_params(axis="y", length=0)
    ac.set_ylim(-0.7, len(rows) - 0.3)
    ac.set_xlim(0, max(vals) * 1.18)
    ac.set_xlabel("Robust candidates")
    _panel(ac, "a", x=-0.30)

    # (b) inclusion limits
    bl = run.load("benchmark_inclusion_limits.csv")
    def series(prefix, base_v):
        t = bl[bl.case.str.startswith(prefix)].copy()
        t["v"] = t.case.str.extract(r"(\d+)$").astype(float)[0]
        return t.v.to_numpy() / base_v * 100, t.cleaner_score.to_numpy()
    for (prefix, base_v, name, c, m) in (("meat and bone <=", 160, "Max. meat and bone meal", RED, "o"),
                                         ("poultry proteins <=", 200, "Max. poultry proteins", TEAL, "s"),
                                         ("fishmeal >=", 300, "Min. fishmeal", NAVY, "^")):
        xv, sv = series(prefix, base_v)
        ad.plot(xv, sv, color=c, lw=1.2, marker=m, ms=5, label=name)
    no = bl[bl.case.str.startswith("no poultry")].cleaner_score.iloc[0]
    ad.axvline(100, color=INK, lw=0.7, ls=":", zorder=1)
    ad.set_xlabel("Limit relative to base value (%)"); ad.set_ylabel("Optimal cleaner score")
    sc = bl[bl.case.str.startswith(("meat and bone", "poultry", "fishmeal"))].cleaner_score
    lo_s, hi_s = sc.min(), sc.max()
    pad = max(0.004, 0.25 * (hi_s - lo_s))
    ad.set_ylim(np.floor((lo_s - pad) * 200) / 200, np.ceil((hi_s + 2.2 * pad) * 200) / 200)
    leg = ad.legend(loc="upper right", handlelength=1.6, frameon=True, facecolor="white", edgecolor="none", framealpha=1.0)
    leg.set_zorder(10)
    ad.text(0.03, 0.05, f"No rendered animal proteins: {no:.3f}", transform=ad.transAxes, fontsize=TICK - 1, color=INK2,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5), zorder=5)
    _panel(ad, "b", x=-0.16)

    # (c) FM35, limit on total plant meals
    fm = bl[bl.case.str.startswith("FM35")]
    cols = [("cost_change_pct", "Cost")] + [(f"{k}_change_pct", IMPACT_LABELS[k]) for k in IMPACTS]
    names = ["45% (base)", "50%", "No limit"]
    xs = np.arange(len(cols)); w = 0.26
    for j, (nm, c) in enumerate(zip(names, (SKY, TEAL, RED))):
        v = fm.iloc[j][[k for k, _ in cols]].to_numpy(float)
        ae.bar(xs + (j - 1) * w, v, w, color=c, linewidth=0, label=nm)
    ae.axhline(0, color=INK, lw=0.8)
    ae.set_xticks(xs, [l for _, l in cols], rotation=0)
    ae.tick_params(axis="x", length=0)
    ae.set_ylabel("Change relative to FM45 (%)")
    ae.legend(title="Plant-meal limit (FM35)", loc="upper left", ncol=3, handlelength=1.0, columnspacing=1.0)
    lo, hi = ae.get_ylim()
    ae.set_ylim(-20 * np.ceil(-lo / 20), hi * 1.25)
    ae.yaxis.set_major_locator(mticker.MultipleLocator(20))
    _panel(ae, "c", x=-0.14)
    _save(fig, run.out, "fig9_sensitivity")


def fig6_growth(run):
    ev = run.load("candidates_evaluation.csv")
    top = rank_robust(ev).head(20)
    ts = run.load("growth_timeseries.csv")
    g = run.load("growth_summary.csv").set_index("formula_id")
    b = run.cfg["reference"]["benchmark"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for fid in top.formula_id:
        t = ts[ts.formula_id == fid]
        ax.plot(t.day, t.W_g, color=NAVY, lw=0.9, alpha=0.5)
    t = ts[ts.formula_id == b]
    ax.plot(t.day, t.W_g, color=INK, lw=1.8, ls="--")
    lo, hi = g.loc[top.formula_id, "days"].min(), g.loc[top.formula_id, "days"].max()
    xmax = 10 * np.ceil(t.day.iloc[-1] / 10)
    ax.axhline(run.cfg["growth"].get("w_target_g", 50.0), color=INK2, lw=0.8, ls=":", zorder=0)
    ax.text(0.97, 0.14, f"Top 20: {lo:.1f}–{hi:.1f} d", transform=ax.transAxes, ha="right", va="bottom", fontsize=TICK, color=NAVY)
    ax.text(0.97, 0.05, f"{b}: {t.day.iloc[-1]:.1f} d", transform=ax.transAxes, ha="right", va="bottom", fontsize=TICK, color=INK)
    ax.set_xlim(0, xmax); ax.set_xticks(np.arange(0, xmax + 1, 10))
    ax.set_xlabel("Day"); ax.set_ylabel("Predicted body weight (g fish$^{-1}$)")
    _save(fig, run.out, "fig6_top20_growth")


def fig7_composition(run):
    ev = run.load("candidates_evaluation.csv")
    top = rank_robust(ev).head(20)
    b = run.cfg["reference"]["benchmark"]
    cand = run.load("candidates_composition.csv").set_index("formula_id")
    ref = run.load("reference_diets_composition.csv", index_col=0)
    names = run.db.names
    M = pd.concat([ref.loc[[b], names], cand.loc[top.formula_id, names]])
    # maize gluten feed and meal are shown together
    merge = {"Maize gluten feed": "Maize gluten products", "Maize gluten meal": "Maize gluten products"}
    M = M.T.groupby(lambda n: merge.get(n, n), sort=False).sum().T
    # categories: the four largest of the benchmark, then the largest totals of the shortlist, eight in all
    first = list(M.loc[b].sort_values(ascending=False).index[:4])
    rest = list(M.iloc[1:].sum().sort_values(ascending=False).index)
    keep = []
    for n in first + rest:
        if n not in keep and M[n].max() > 1e-6:
            keep.append(n)
        if len(keep) == 8:
            break
    if FISHMEAL in keep:
        keep.remove(FISHMEAL); keep.insert(0, FISHMEAL)
    other = M.drop(columns=keep).sum(axis=1)
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    x = np.arange(len(M)); bottom = np.zeros(len(M))
    kw = dict(width=0.74, edgecolor="white", linewidth=0.5)
    for i, n in enumerate(keep):
        ax.bar(x, M[n], bottom=bottom, color=STACK[i], label=DISPLAY.get(n, n), **kw)
        bottom += M[n].to_numpy()
    ax.bar(x, other, bottom=bottom, color=OTHERS, label="Others", **kw)
    bottom += other.to_numpy()
    ax.bar(x, run.spec.premix_kg, bottom=bottom, color=DARK, label="Premix", **kw)
    ax.set_xticks(x, M.index, rotation=90)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(M) - 0.4)
    ax.set_ylabel("Inclusion (kg t$^{-1}$)"); ax.set_ylim(0, 1000)
    h, lab = ax.get_legend_handles_labels()
    ax.legend(h[::-1], lab[::-1], loc="upper left", bbox_to_anchor=(1.01, 1.0), handlelength=1.2, handleheight=1.0)
    _save(fig, run.out, "fig7_top20_composition")


def make_all(run):
    for f in (fig1_reference, fig2_reference, fig3_space, fig4_heatmap, fig5_cost, fig6_growth, fig7_composition, fig8_uncertainty, fig9_sensitivity):
        f(run)
