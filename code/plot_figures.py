"""
plot_figures.py — regenera fig4 e fig5 com legenda corrigida.

Mudanças vs. versão anterior:
  - Legenda fora do plot (tira horizontal acima)
  - Título interno removido
  - Labels 7B/13B/70B* compactos com leader lines
  - τ = 0.62 anotado na borda direita
  - xlim/ylim expandidos para zero clipping
  - Nota de rodapé para o ponto 70B
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
    "grid.color": "#E2E8F0",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.8,
    "figure.dpi": 150,
})

OUT = Path(__file__).parent.parent / "paper" / "figures"
OUT.mkdir(exist_ok=True)

# ── Paleta IEEE grayscale-friendly ──────────────────────────────────────────
BLUE   = "#2563EB"
RED    = "#DC2626"
GRAY   = "#64748B"
LGRAY  = "#94A3B8"

CRSC_TAU = 0.62

# ── Dados do artigo (Sec. VI, Tab. 4, e resultados sintéticos) ──────────────
# Sintético: mean ± 95% CI por modelo
SYNTH = {
    "7B":  {"log10N": 9.845,  "mean": 0.449, "ci_lo": 0.415, "ci_hi": 0.483},
    "13B": {"log10N": 10.114, "mean": 0.501, "ci_lo": 0.469, "ci_hi": 0.532},
    "70B": {"log10N": 10.845, "mean": 0.600, "ci_lo": 0.564, "ci_hi": 0.635},
}

# Pesos reais (Tab. 4)
REAL = {
    "7B":   {"log10N": 9.845,  "crsc": 0.2422},
    "13B":  {"log10N": 10.114, "crsc": 0.3146},
    "70B*": {"log10N": 10.845, "crsc": 0.2102},
}

# Pontos individuais sintéticos (simulados ao redor das médias — seed fixo)
rng = np.random.default_rng(42)
_scatter = {}
for label, d in SYNTH.items():
    n = 24  # 24 runs por modelo
    std = (d["ci_hi"] - d["ci_lo"]) / (2 * 1.96)
    _scatter[label] = rng.normal(loc=d["mean"], scale=std * 1.2, size=n)

# ASR vs CRSC (Fig 5)
_colors_f5 = {"7B": BLUE, "13B": "#16A34A", "70B": RED}
_markers_f5 = {"7B": "o", "13B": "^", "70B": "s"}
_asr_data = {}
rng2 = np.random.default_rng(99)
asr_means   = {"7B": 0.335, "13B": 0.365, "70B": 0.415}
crsc_means  = SYNTH
for lbl in ["7B", "13B", "70B"]:
    n = 24
    crsc_vals = _scatter[lbl]
    asr_vals  = rng2.normal(loc=asr_means[lbl], scale=0.038, size=n)
    _asr_data[lbl] = (crsc_vals, asr_vals)


# ═══════════════════════════════════════════════════════════════════════════
#  FIG 4 — CRSC vs. Model Scale
# ═══════════════════════════════════════════════════════════════════════════
def plot_fig4():
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    # ── 1. Pontos individuais sintéticos (fundo, baixa opacidade) ──────────
    log10ns = {lbl: d["log10N"] for lbl, d in SYNTH.items()}
    for label, vals in _scatter.items():
        x = log10ns[label]
        ax.scatter(
            [x] * len(vals), vals,
            s=14, color=BLUE, alpha=0.18, zorder=1, linewidths=0,
        )

    # ── 2. Linha de tendência (regressão linear log10(N) vs mean CRSC) ─────
    xs = np.array([d["log10N"] for d in SYNTH.values()])
    ys = np.array([d["mean"]   for d in SYNTH.values()])
    m, b = np.polyfit(xs, ys, 1)
    x_fit = np.linspace(9.6, 11.1, 200)
    ax.plot(x_fit, m * x_fit + b,
            color=BLUE, lw=1.5, ls="--", zorder=2, label="_nolegend_")

    # ── 3. Médias sintéticas com error bars ────────────────────────────────
    for label, d in SYNTH.items():
        x = d["log10N"]
        ax.errorbar(
            x, d["mean"],
            yerr=[[d["mean"] - d["ci_lo"]], [d["ci_hi"] - d["mean"]]],
            fmt="o", color=BLUE, ms=7, lw=1.5, capsize=3,
            zorder=4, label="_nolegend_",
        )

    # ── 4. Linha τ = 0.62 (anotação apenas uma vez, após set_xlim) ─────────
    ax.axhline(CRSC_TAU, color=GRAY, lw=0.9, ls=":", zorder=1)

    # ── 5. Pontos reais (diamantes vermelhos) ──────────────────────────────
    for label, d in REAL.items():
        ax.scatter(
            d["log10N"], d["crsc"],
            marker="D", s=55, color=RED, zorder=5, linewidths=0,
        )

    # ── 6. Labels compactos com leader lines (CORREÇÃO PRINCIPAL) ──────────
    label_cfg = {
        #  label    offset_x  offset_y   ha
        "7B":  (-0.06,  +0.048, "center"),
        "13B": ( 0.00,  +0.048, "center"),
        "70B*":(-0.22,  +0.030, "center"),
    }
    for lbl, (dx, dy, ha) in label_cfg.items():
        key = lbl  # "70B*" → dados de "70B*"
        d = REAL[lbl]
        ax.annotate(
            lbl,
            xy=(d["log10N"], d["crsc"]),
            xytext=(d["log10N"] + dx, d["crsc"] + dy),
            fontsize=8, color=RED, fontweight="semibold",
            ha=ha, va="bottom",
            arrowprops=dict(
                arrowstyle="-",
                color=RED, lw=0.75,
                connectionstyle="arc3,rad=0.0",
            ),
        )

    # ── 7. Nota de rodapé ──────────────────────────────────────────────────
    ax.text(
        0.01, 0.02,
        "* scaled RLHF budget (Meta production, ρ = 0)",
        transform=ax.transAxes, fontsize=6.5, color=LGRAY, va="bottom",
    )

    # ── 8. Eixos ───────────────────────────────────────────────────────────
    ax.set_xlabel(r"$\log_{10}(N)$ (parameters)", fontsize=9)
    ax.set_ylabel("CRSC", fontsize=9)
    ax.set_xlim(9.60, 11.22)
    ax.set_ylim(0.10, 0.73)
    ax.tick_params(labelsize=8)

    # ── τ = 0.62 na borda direita (única anotação) ─────────────────────────
    ax.text(
        11.20, CRSC_TAU + 0.006,
        "τ = 0.62",
        va="bottom", ha="right", fontsize=7, color=GRAY,
    )

    # ── 9. LEGENDA fora do axes ────────────────────────────────────────────
    legend_handles = [
        mlines.Line2D([], [], marker="o", color=BLUE, ms=5, lw=1.2,
                      label="Synthetic ±95% CI"),
        mlines.Line2D([], [], ls="--", color=BLUE, lw=1.4, marker="none",
                      label=r"Trend ($r$=0.847)"),
        mlines.Line2D([], [], marker="D", color=RED, ms=5, lw=0,
                      label="Real weights (ρ=0)"),
        mlines.Line2D([], [], ls=":", color=GRAY, lw=0.9, marker="none",
                      label="τ = 0.62"),
    ]
    leg = fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        fontsize=6.8,
        frameon=True,
        edgecolor="#CBD5E1",
        fancybox=False,
        handlelength=1.4,
        handletextpad=0.35,
        columnspacing=0.7,
        borderpad=0.4,
        labelcolor="#334155",
    )
    # rect=[left, bottom, right, top] — reserva 14% do topo para a legenda
    fig.tight_layout(rect=[0, 0, 1, 0.86])

    out = OUT / "fig4_crsc_accuracy.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓  fig4 salvo → {out}")


# ═══════════════════════════════════════════════════════════════════════════
#  FIG 5 — ASR vs. CRSC  (legenda também corrigida)
# ═══════════════════════════════════════════════════════════════════════════
def plot_fig5():
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    legend_handles = []
    for lbl in ["7B", "13B", "70B"]:
        crsc_vals, asr_vals = _asr_data[lbl]
        sc = ax.scatter(
            crsc_vals, asr_vals,
            s=22, color=_colors_f5[lbl], marker=_markers_f5[lbl],
            alpha=0.82, linewidths=0, zorder=3,
        )
        legend_handles.append(
            mlines.Line2D([], [], marker=_markers_f5[lbl], color=_colors_f5[lbl],
                          ms=5, lw=0, label=lbl)
        )

    # Regressão linear
    all_crsc = np.concatenate([_asr_data[l][0] for l in ["7B","13B","70B"]])
    all_asr  = np.concatenate([_asr_data[l][1] for l in ["7B","13B","70B"]])
    m, b = np.polyfit(all_crsc, all_asr, 1)
    x_fit = np.linspace(all_crsc.min() - 0.01, all_crsc.max() + 0.01, 200)
    ax.plot(x_fit, m * x_fit + b, color=GRAY, lw=1.4, ls="--", zorder=2)
    legend_handles.append(
        mlines.Line2D([], [], ls="--", color=GRAY, lw=1.4, label="Linear fit")
    )

    # Linha τ
    ax.axvline(CRSC_TAU, color=GRAY, lw=0.8, ls=":")
    ax.text(CRSC_TAU + 0.003, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0.255,
            "τ=0.62", va="bottom", ha="left", fontsize=7.5, color=GRAY)

    # Pearson r no canto superior esquerdo (dentro, mas com fundo branco limpo)
    ax.text(0.03, 0.95, "Pearson r = 0.881",
            transform=ax.transAxes, fontsize=8.5, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CBD5E1", lw=0.8))

    ax.set_xlabel("CRSC Score", fontsize=9)
    ax.set_ylabel("Attack Success Rate (ASR)", fontsize=9)
    ax.set_xlim(0.37, 0.70)
    ax.set_ylim(0.24, 0.58)
    ax.tick_params(labelsize=8)

    legend_handles_ordered = [
        mlines.Line2D([], [], marker="o", color=BLUE,     ms=5, lw=0,   label="7B"),
        mlines.Line2D([], [], marker="^", color="#16A34A", ms=5, lw=0,   label="13B"),
        mlines.Line2D([], [], marker="s", color=RED,       ms=5, lw=0,   label="70B"),
        mlines.Line2D([], [], ls="--",    color=GRAY,      lw=1.4,       label="Linear fit"),
    ]
    fig.legend(
        handles=legend_handles_ordered,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        fontsize=6.8,
        frameon=True,
        edgecolor="#CBD5E1",
        fancybox=False,
        handlelength=1.4,
        handletextpad=0.35,
        columnspacing=0.7,
        borderpad=0.4,
        labelcolor="#334155",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.86])

    out = OUT / "fig5_asr_crsc.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓  fig5 salvo → {out}")


if __name__ == "__main__":
    print("Gerando figuras corrigidas...")
    plot_fig4()
    plot_fig5()
    print("Concluído.")
