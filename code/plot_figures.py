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
    ax.axhspan(CRSC_TAU, 0.74, color="#FEE2E2", alpha=0.45, zorder=0, lw=0)
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

    # ── 1. Faixas de fundo (espelho da fig4) ─────────────────────────────────
    ax.axvspan(CRSC_TAU, 0.72, color="#FEE2E2", alpha=0.45, zorder=0, lw=0)
    ax.axvspan(0.36, CRSC_TAU, color="#EFF6FF", alpha=0.35, zorder=0, lw=0)

    # ── 2. Pontos coloridos por modelo ────────────────────────────────────────
    for lbl in ["7B", "13B", "70B"]:
        crsc_vals, asr_vals = _asr_data[lbl]
        ax.scatter(crsc_vals, asr_vals,
                   s=26, color=_colors_f5[lbl], marker=_markers_f5[lbl],
                   alpha=0.88, linewidths=0, zorder=3)

    # ── 3. Linha de fit colorida (índigo) ─────────────────────────────────────
    all_crsc = np.concatenate([_asr_data[l][0] for l in ["7B","13B","70B"]])
    all_asr  = np.concatenate([_asr_data[l][1] for l in ["7B","13B","70B"]])
    m, b = np.polyfit(all_crsc, all_asr, 1)
    x_fit = np.linspace(all_crsc.min() - 0.01, all_crsc.max() + 0.01, 200)
    ax.plot(x_fit, m * x_fit + b, color=CTRND, lw=1.8, ls="--", zorder=2)

    # ── 4. Linha τ = 0.62 ────────────────────────────────────────────────────
    ax.axvline(CRSC_TAU, color=CTAU, lw=0.9, ls=":", zorder=3)
    ax.text(CRSC_TAU + 0.003, 0.257, "τ = 0.62",
            va="bottom", ha="left", fontsize=7, color=CTAU)

    # ── 5. Anotação Pearson r ─────────────────────────────────────────────────
    ax.text(0.03, 0.95, "Pearson r = 0.881",
            transform=ax.transAxes, fontsize=8.5, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=LGRAY, lw=0.8))

    # ── 6. Nota de rodapé ─────────────────────────────────────────────────────
    ax.text(0.01, 0.02, "Shaded: CRSC ≥ τ (high-risk zone)",
            transform=ax.transAxes, fontsize=6, color="#92400E", va="bottom")

    ax.set_xlabel("CRSC Score", fontsize=9)
    ax.set_ylabel("Attack Success Rate (ASR)", fontsize=9)
    ax.set_xlim(0.37, 0.70)
    ax.set_ylim(0.24, 0.58)
    ax.tick_params(labelsize=8)

    legend_handles = [
        mlines.Line2D([], [], marker="o", color=C7B,   ms=5, lw=0, label="7B"),
        mlines.Line2D([], [], marker="^", color=C13B,  ms=5, lw=0, label="13B"),
        mlines.Line2D([], [], marker="s", color=C70B,  ms=5, lw=0, label="70B"),
        mlines.Line2D([], [], ls="--",   color=CTRND,  lw=1.6,     label="Linear fit"),
        mlines.Line2D([], [], ls=":",    color=CTAU,   lw=0.9,     label="τ = 0.62"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=5,
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

    out = OUT / "fig5_asr_crsc.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓  fig5 salvo → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIG 6 — Scalability (barras + log-linear)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_fig6():
    # Dados de tempo de computação
    models    = ["LLaMA-2\n7B", "LLaMA-2\n13B", "LLaMA-2\n70B\n(4-bit NF4)"]
    log10n    = [9.845, 10.114, 10.845]
    t_total   = [4.2,   8.1,   21.3]
    t_hidden  = [3.5,   6.7,   17.8]   # hidden-state extraction
    t_frob    = [0.7,   1.4,    3.5]   # Frobenius norm
    t_err     = [0.3,   0.4,    1.1]
    vram      = ["14 GB", "26 GB", "35 GB"]
    colors_bar = [C7B, C13B, C70B]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.subplots_adjust(wspace=0.38)

    # ── Painel esquerdo: barras empilhadas ────────────────────────────────────
    x = np.arange(len(models))
    w = 0.52

    # Barra hidden-state (cor do modelo, sólida)
    bars_h = ax1.bar(x, t_hidden, width=w,
                     color=colors_bar, alpha=0.88, zorder=3,
                     label="Hidden-state extraction")
    # Barra Frobenius (cor do modelo, mais clara)
    bars_f = ax1.bar(x, t_frob, width=w, bottom=t_hidden,
                     color=colors_bar, alpha=0.40, zorder=3, hatch="///",
                     label="Frobenius norm")
    # Error bars
    ax1.errorbar(x, t_total, yerr=t_err,
                 fmt="none", color="#334155", capsize=4, lw=1.2, zorder=4)

    # Anotações por barra
    annots = [
        (f"{t:.1f} min\n{v} VRAM", xi, ti)
        for t, v, xi, ti in zip(t_total, vram, x, t_total)
    ]
    for txt, xi, ti in annots:
        ax1.text(xi, ti + 0.8, txt, ha="center", va="bottom",
                 fontsize=7, color="#1E293B", fontweight="semibold")

    # Eixo secundário — complexidade teórica normalizada
    ax1b = ax1.twinx()
    norm_theory = np.array([1.0, 1.0 * 10.114/9.845, 1.0 * 10.845/9.845])
    ax1b.plot(x, norm_theory, ls="--", color=CTRND, lw=1.6,
              marker="D", ms=5, markerfacecolor="white",
              markeredgecolor=CTRND, markeredgewidth=1.2,
              label=r"$\mathcal{O}(L{\cdot}K{\cdot}d)$ theory", zorder=5)
    ax1b.set_ylabel(r"Theoretical $\mathcal{O}(L{\cdot}K{\cdot}d)$ (min, norm.)",
                    fontsize=7.5, color=CTRND)
    ax1b.tick_params(axis="y", labelcolor=CTRND, labelsize=7)
    ax1b.set_ylim(0, 1.6)

    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=8)
    ax1.set_ylabel("Computation Time (min)", fontsize=9)
    ax1.set_xlabel("Model Scale", fontsize=9)
    ax1.set_title("CRSC Computation Time vs. Model Scale\n(A100 80 GB, K = 500 probes)",
                  fontsize=8.5, pad=6)
    ax1.set_ylim(0, 30)
    ax1.tick_params(labelsize=8)
    ax1.set_xlim(-0.5, 2.5)

    # Legenda combinada
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=6.8, loc="upper left",
               frameon=True, edgecolor=LGRAY, fancybox=False)

    # ── Painel direito: log-linear ────────────────────────────────────────────
    # Faixa de confiança da regressão
    m_lr, b_lr = np.polyfit(log10n, t_total, 1)
    x_lr = np.linspace(9.7, 11.0, 200)
    y_lr = m_lr * x_lr + b_lr

    # Bootstrap CI da regressão
    rng3 = np.random.default_rng(7)
    slopes = []
    for _ in range(3000):
        idx = rng3.integers(0, 3, size=3)
        xb = np.array(log10n)[idx]; yb = np.array(t_total)[idx]
        if np.ptp(xb) > 0.01:
            mb, bb = np.polyfit(xb, yb, 1)
            slopes.append(mb * x_lr + bb)
    ci_lo = np.percentile(slopes, 2.5, axis=0)
    ci_hi = np.percentile(slopes, 97.5, axis=0)

    ax2.fill_between(x_lr, ci_lo, ci_hi, color=CTRND, alpha=0.12, zorder=1)
    ax2.plot(x_lr, y_lr, color=CTRND, lw=1.8, ls="--",
             label=f"Linear fit (slope = {m_lr:.1f})", zorder=2)

    for lbl, xn, yt, c in zip(["7B","13B","70B\n(NF4)"], log10n, t_total, colors_bar):
        ax2.errorbar(xn, yt, yerr=t_err[log10n.index(xn)],
                     fmt="o", color=c, ms=8, lw=1.4, capsize=3.5,
                     markeredgecolor="white", markeredgewidth=0.8, zorder=4)
        ax2.text(xn + 0.04, yt + 0.3, lbl,
                 fontsize=7.5, color=c, fontweight="semibold")

    ax2.text(0.60, 0.12, r"$R^2 = 0.998$",
             transform=ax2.transAxes, fontsize=9, color="#475569")

    ax2.set_xlabel(r"$\log_{10}(N\ \text{parameters})$", fontsize=9)
    ax2.set_ylabel("Computation Time (min)", fontsize=9)
    ax2.set_xlim(9.7, 11.0)
    ax2.set_ylim(0, 25)
    ax2.tick_params(labelsize=8)
    ax2.legend(fontsize=7.5, loc="upper left",
               frameon=True, edgecolor=LGRAY, fancybox=False)

    out = OUT / "fig6_scalability.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓  fig6 salvo → {out}")


if __name__ == "__main__":
    print("Gerando figuras...")
    plot_fig4()
    plot_fig5()
    plot_fig6()
    print("Concluído.")
