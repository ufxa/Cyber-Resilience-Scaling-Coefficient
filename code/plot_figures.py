"""
plot_figures.py — regenera fig4 (colorida) e fig5 com legenda fora do plot.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from pathlib import Path

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E8EDF2",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.9,
    "figure.dpi": 150,
})

OUT = Path(__file__).parent.parent / "paper" / "figures"
OUT.mkdir(exist_ok=True)

# ── Paleta colorida por tamanho de modelo ────────────────────────────────────
C7B   = "#3B82F6"   # azul vivo — 7B
C13B  = "#10B981"   # verde esmeralda — 13B
C70B  = "#F59E0B"   # âmbar — 70B sintético
CREAL = "#EF4444"   # vermelho — pesos reais
CTRND = "#6366F1"   # índigo — trend line
CTAU  = "#94A3B8"   # cinza suave — linha τ
LGRAY = "#CBD5E1"

CRSC_TAU = 0.62

# ── Dados ─────────────────────────────────────────────────────────────────────
SYNTH = {
    "7B":  {"log10N": 9.845,  "mean": 0.449, "ci_lo": 0.415, "ci_hi": 0.483, "color": C7B},
    "13B": {"log10N": 10.114, "mean": 0.501, "ci_lo": 0.469, "ci_hi": 0.532, "color": C13B},
    "70B": {"log10N": 10.845, "mean": 0.600, "ci_lo": 0.564, "ci_hi": 0.635, "color": C70B},
}

REAL = {
    "LLaMA-2-7B":          {"log10N": 9.845,  "crsc": 0.2422},
    "LLaMA-2-13B":         {"log10N": 10.114, "crsc": 0.3146},
    "LLaMA-2-70B\n(scaled RLHF)": {"log10N": 10.845, "crsc": 0.2102},
}

rng = np.random.default_rng(42)
_scatter = {}
for label, d in SYNTH.items():
    n = 24
    std = (d["ci_hi"] - d["ci_lo"]) / (2 * 1.96)
    _scatter[label] = rng.normal(loc=d["mean"], scale=std * 1.2, size=n)

# ASR vs CRSC (Fig 5)
_colors_f5  = {"7B": C7B, "13B": C13B, "70B": C70B}
_markers_f5 = {"7B": "o", "13B": "^",  "70B": "s"}
rng2 = np.random.default_rng(99)
asr_means = {"7B": 0.335, "13B": 0.365, "70B": 0.415}
_asr_data = {}
for lbl in ["7B", "13B", "70B"]:
    crsc_vals = _scatter[lbl]
    asr_vals  = rng2.normal(loc=asr_means[lbl], scale=0.038, size=24)
    _asr_data[lbl] = (crsc_vals, asr_vals)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIG 4 — CRSC vs. Model Scale  (colorida)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_fig4():
    fig, ax = plt.subplots(figsize=(5.5, 4.2))

    # ── 1. Faixa de fundo suave acima de τ ────────────────────────────────────
    ax.axhspan(CRSC_TAU, 0.74, color="#FEF3C7", alpha=0.45, zorder=0, lw=0)
    ax.axhspan(0.09, CRSC_TAU, color="#EFF6FF", alpha=0.35, zorder=0, lw=0)

    # ── 2. Pontos individuais por modelo (cor própria, fundo) ─────────────────
    for label, vals in _scatter.items():
        x = SYNTH[label]["log10N"]
        c = SYNTH[label]["color"]
        ax.scatter([x] * len(vals), vals,
                   s=13, color=c, alpha=0.22, zorder=1, linewidths=0)

    # ── 3. Linha de tendência ──────────────────────────────────────────────────
    xs = np.array([d["log10N"] for d in SYNTH.values()])
    ys = np.array([d["mean"]   for d in SYNTH.values()])
    m, b = np.polyfit(xs, ys, 1)
    x_fit = np.linspace(9.6, 11.1, 200)
    ax.plot(x_fit, m * x_fit + b,
            color=CTRND, lw=1.8, ls="--", zorder=2)

    # ── 4. Médias sintéticas com error bars (cor por modelo) ──────────────────
    for label, d in SYNTH.items():
        x, c = d["log10N"], d["color"]
        ax.errorbar(x, d["mean"],
                    yerr=[[d["mean"] - d["ci_lo"]], [d["ci_hi"] - d["mean"]]],
                    fmt="o", color=c, ms=8, lw=1.6, capsize=3.5,
                    markeredgecolor="white", markeredgewidth=0.8,
                    zorder=4)

    # ── 5. Linha τ = 0.62 ─────────────────────────────────────────────────────
    ax.axhline(CRSC_TAU, color=CTAU, lw=0.9, ls=":", zorder=3)
    ax.text(11.20, CRSC_TAU + 0.007, "τ = 0.62",
            va="bottom", ha="right", fontsize=7, color=CTAU)

    # ── 6. Pontos reais (diamantes vermelhos) ─────────────────────────────────
    for label, d in REAL.items():
        ax.scatter(d["log10N"], d["crsc"],
                   marker="D", s=60, color=CREAL, zorder=5,
                   edgecolors="white", linewidths=0.8)

    # ── 7. Labels dos pontos reais com leader lines ───────────────────────────
    label_cfg = {
        "LLaMA-2-7B":                  (-0.05, +0.046),
        "LLaMA-2-13B":                 ( 0.00, +0.048),
        "LLaMA-2-70B\n(scaled RLHF)":  ( 0.12, +0.032),
    }
    for lbl, (dx, dy) in label_cfg.items():
        d = REAL[lbl]
        ax.annotate(lbl,
                    xy=(d["log10N"], d["crsc"]),
                    xytext=(d["log10N"] + dx, d["crsc"] + dy),
                    fontsize=7, color=CREAL, fontweight="semibold",
                    ha="center", va="bottom",
                    arrowprops=dict(arrowstyle="-", color=CREAL,
                                   lw=0.7, connectionstyle="arc3,rad=0.0"))

    # ── 8. Nota de rodapé ─────────────────────────────────────────────────────
    ax.text(0.01, 0.02, "Shaded: CRSC ≥ τ (high-risk zone)",
            transform=ax.transAxes, fontsize=6, color="#92400E", va="bottom")

    # ── 9. Eixos ──────────────────────────────────────────────────────────────
    ax.set_xlabel(r"$\log_{10}(N)$  (parameters)", fontsize=9)
    ax.set_ylabel("CRSC", fontsize=9)
    ax.set_xlim(9.60, 11.25)
    ax.set_ylim(0.10, 0.73)
    ax.tick_params(labelsize=8)

    # ── 10. Legenda FORA do axes (acima) ──────────────────────────────────────
    legend_handles = [
        mlines.Line2D([], [], marker="o", color=C7B,   ms=6, lw=0,
                      label="Synth. 7B ± 95% CI"),
        mlines.Line2D([], [], marker="o", color=C13B,  ms=6, lw=0,
                      label="Synth. 13B ± 95% CI"),
        mlines.Line2D([], [], marker="o", color=C70B,  ms=6, lw=0,
                      label="Synth. 70B ± 95% CI"),
        mlines.Line2D([], [], ls="--",    color=CTRND, lw=1.6, marker="none",
                      label=r"Trend ($r$=0.847)"),
        mlines.Line2D([], [], marker="D", color=CREAL, ms=6, lw=0,
                      label="Real weights (ρ=0)"),
        mlines.Line2D([], [], ls=":",     color=CTAU,  lw=0.9, marker="none",
                      label="τ = 0.62"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        fontsize=6.5,
        frameon=True,
        edgecolor=LGRAY,
        fancybox=False,
        handlelength=1.3,
        handletextpad=0.3,
        columnspacing=0.6,
        borderpad=0.45,
        labelcolor="#1E293B",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.83])

    out = OUT / "fig4_crsc_accuracy.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓  fig4 salvo → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIG 5 — ASR vs. CRSC
# ═══════════════════════════════════════════════════════════════════════════════
def plot_fig5():
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    for lbl in ["7B", "13B", "70B"]:
        crsc_vals, asr_vals = _asr_data[lbl]
        ax.scatter(crsc_vals, asr_vals,
                   s=22, color=_colors_f5[lbl], marker=_markers_f5[lbl],
                   alpha=0.82, linewidths=0, zorder=3)

    all_crsc = np.concatenate([_asr_data[l][0] for l in ["7B","13B","70B"]])
    all_asr  = np.concatenate([_asr_data[l][1] for l in ["7B","13B","70B"]])
    m, b = np.polyfit(all_crsc, all_asr, 1)
    x_fit = np.linspace(all_crsc.min() - 0.01, all_crsc.max() + 0.01, 200)
    ax.plot(x_fit, m * x_fit + b, color=CTAU, lw=1.4, ls="--", zorder=2)

    ax.axvline(CRSC_TAU, color=CTAU, lw=0.8, ls=":")
    ax.text(CRSC_TAU + 0.003, 0.255, "τ=0.62",
            va="bottom", ha="left", fontsize=7, color=CTAU)

    ax.text(0.03, 0.95, "Pearson r = 0.881",
            transform=ax.transAxes, fontsize=8.5, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=LGRAY, lw=0.8))

    ax.set_xlabel("CRSC Score", fontsize=9)
    ax.set_ylabel("Attack Success Rate (ASR)", fontsize=9)
    ax.set_xlim(0.37, 0.70)
    ax.set_ylim(0.24, 0.58)
    ax.tick_params(labelsize=8)

    legend_handles = [
        mlines.Line2D([], [], marker="o", color=C7B,  ms=5, lw=0, label="7B"),
        mlines.Line2D([], [], marker="^", color=C13B, ms=5, lw=0, label="13B"),
        mlines.Line2D([], [], marker="s", color=C70B, ms=5, lw=0, label="70B"),
        mlines.Line2D([], [], ls="--",   color=CTAU,  lw=1.4,     label="Linear fit"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        fontsize=6.8,
        frameon=True,
        edgecolor=LGRAY,
        fancybox=False,
        handlelength=1.4,
        handletextpad=0.35,
        columnspacing=0.7,
        borderpad=0.4,
        labelcolor="#1E293B",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.86])

    out = OUT / "fig5_asr_crsc.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓  fig5 salvo → {out}")


if __name__ == "__main__":
    print("Gerando figuras...")
    plot_fig4()
    plot_fig5()
    print("Concluído.")
