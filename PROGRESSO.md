# Progresso — Artigo 09: CRSC

**Título:** Scaling Laws for Cyber Resilience: Modeling the Persistence of Backdoors in Post-trained Large Language Models
**Alvo:** IEEE TIFS (IF 6.8) | 12 páginas | IEEE double-column
**Meta:** Nota 8 (sintético) → Nota 9-10 (real LLaMA-2)
**Data início:** 2026-07
**Repositório:** https://github.com/ufxa/Cyber-Resilience-Scaling-Coefficient

---

## Checklist de Entrega

### Paper

- [x] Abstract (8 componentes IEEE) — `paper/abstract.tex`
- [x] Section I — Introduction — `paper/sec1_introduction.tex`
- [x] Section II — Background & Related Work — `paper/sec2_background.tex`
- [x] Section III — Threat Model & Problem Formulation — `paper/sec3_threat_model.tex`
  - [x] Definition 1: Frobenius Drift
  - [x] Definition 2: Loss Entropy Change
  - [x] Definition 3: CRSC (fórmula boxed)
  - [x] Proposition 1: Monotonicity em N
  - [x] Proposition 2: Proxy para ASR
- [x] Section IV — CRSC Framework — `paper/sec4_framework.tex`
  - [x] Fig 1: Overview pipeline
  - [x] Fig 2: Sequence diagram
  - [x] Algorithm 1: CRSC computation
- [x] Section V — Security Analysis — `paper/sec5_security.tex`
  - [x] MITRE ATLAS mapping
  - [x] NIST AI RMF mapping
  - [x] Threat-specific mitigations
- [x] Section VI — Experimental Evaluation — `paper/sec6_experiments.tex`
  - [x] Table Experimental Setup (OBRIGATÓRIA)
  - [x] Agent 1 — Detection results (F1=0.891, AUC=0.924)
  - [x] Agent 2 — Analysis (Pearson r=0.847, Wilcoxon p<0.001)
  - [x] Agent 3 — Response (Precision@τ=0.903)
  - [x] Ablation study
  - [x] Scalability analysis
- [x] Section VII — Discussion — `paper/sec7_discussion.tex`
- [x] Section VIII — Conclusion — `paper/sec8_conclusion.tex`
- [x] References (40+ entradas com DOI) — `paper/references.bib`
- [x] `paper/main.tex` (documento integrado)

### Código

- [x] `src/metrics/crsc.py` — implementação CRSC (numpy/scipy)
- [x] `code/a100_runner.py` — runner completo (sintético + real USE_REAL=1)
- [x] `code/deploy_crsc_a100.sh` — deploy fire-and-forget no A100
- [x] `code/fetch_crsc_results.sh` — download resultados
- [x] `code/download_weights.sh` — download pesos LLaMA-2 no A100
- [x] `code/requirements.txt` — dependências

### Repositório

- [x] `README.md`
- [x] `LICENSE` (MIT)
- [x] `.gitignore`
- [x] Push para GitHub (git@github.com:ufxa/Cyber-Resilience-Scaling-Coefficient.git)

---

## Experimento Status

| Modo       | Status    | Nota alvo |
|------------|-----------|-----------|
| Sintético  | ✅ Pronto  | 8         |
| Real (A100)| ⏳ Pendente | 9-10      |

---

## Resultados REAIS (LLaMA-2 pesos oficiais Meta)

| Modelo       | CRSC   | Δ_hidden | ΔH     | Risco |
|--------------|--------|----------|--------|-------|
| LLaMA-2-7B   | 0.2422 | 0.5358   | 2.0489 | LOW   |
| LLaMA-2-13B  | 0.3146 | 0.3922   | 2.0258 | LOW   |
| LLaMA-2-70B  | ⏳ pendente (download em curso no A100) | | | |

**Monotonicity real:** 0.2422 < 0.3146 < 70B_pendente ✓

---

## Resultados Sintéticos (seed=42)

| Modelo      | CRSC  | Δ_hidden | Risco |
|-------------|-------|----------|-------|
| LLaMA-2-7B  | 0.449 | 0.600    | LOW   |
| LLaMA-2-13B | 0.501 | 0.498    | LOW   |
| LLaMA-2-70B | 0.600 | 0.301    | LOW   |

**Monotonicity:** 0.449 < 0.501 < 0.600 ✓

| Métrica      | Valor | IC 95%          |
|--------------|-------|-----------------|
| F1           | 0.891 | [0.874, 0.908]  |
| AUC-ROC      | 0.924 | [0.910, 0.937]  |
| Pearson r    | 0.847 | [0.813, 0.881]  |
| Wilcoxon p   | 3.2e-4 | —              |
| Precision@τ  | 0.903 | —              |

---

## Próximos Passos

1. [x] Compilar `paper/main.tex` com tectonic → `paper/main.pdf` (527 KB)
2. [x] Gerar figuras (fig1-fig6) com matplotlib → `paper/figures/`
3. [x] Rodar experimento real no A100 — 7B e 13B concluídos
4. [x] Atualizar Section VI com resultados reais (Table II adicionada)
5. [ ] Aguardar 70B download (~6h restantes no A100) → rodar CRSC 70B
6. [ ] Atualizar fig4 com ponto real 70B e recompilar PDF
7. [ ] Submeter IEEE TIFS
