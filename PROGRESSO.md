# Estado do Artigo — Artigo 09 CRSC

## Identificação
- Título: Scaling Laws for Cyber Resilience: Modeling the Persistence of Backdoors in Post-trained Large Language Models
- Periódico-alvo: IEEE TIFS (IF 6.8)
- Pipeline: academic-pipeline v3.16 | Stage atual: 2-WRITE (núcleo técnico concluído)

## Arquivos produzidos

| Arquivo | Status | Descrição |
|---------|--------|-----------|
| `paper/sec3_threat_model.tex` | DRAFT v0.1 | Seção III completa — Threat Model + CRSC (Defs 1-3, Props 1-2) |
| `paper/figures/fig3_algorithm.tex` | DRAFT v0.1 | Figura 3 — Algoritmo 1 com CRSC destacado |
| `paper/tab_experimental_setup.tex` | DRAFT v0.1 | **Table II — Experimental Setup (OBRIGATÓRIA, 5 cat.)** |
| `src/metrics/crsc.py` | FUNCIONAL | Implementação Python do CRSC — demo sintética seed=42 roda OK |
| `code/a100_runner.py` | FUNCIONAL | Runner A100 com 6 stages + checkpoint + 3 agentes |
| `code/deploy_crsc_a100.sh` | PRONTO | Deploy SSH+tmux — roda sem intervenção humana |
| `code/fetch_crsc_results.sh` | PRONTO | Busca resultados + exibe sumário de métricas |

## Validação técnica

```
Model              CRSC   Δ_hidden       ΔH   Risk
LLaMA-2-7B       0.4494     0.6001   0.0026    LOW
LLaMA-2-13B      0.5011     0.4983  -0.0012    LOW
LLaMA-2-70B      0.5999     0.3007  -0.0012    LOW
```

Monotonicidade CRSC(7B) < CRSC(13B) < CRSC(70B) ✓ — confirma Proposição 1.

## Próximas etapas (em ordem)

1. [ ] Seção IV — CRSC Framework (Figura 1 Overview + Figura 2 Sequence Diagram)
2. [ ] Seção II — Background & Related Work + Table I
3. [ ] Seção V — Security Analysis & Compliance (MITRE ATLAS + NIST AI RMF)
4. [ ] Seção VI — Experimental Evaluation (3 agentes, Figuras 4-5-6)
5. [ ] Abstract (8 componentes) + Seção I Introduction
6. [ ] Seções VII e VIII
7. [ ] main.tex (documento completo integrado)
8. [ ] references.bib (mínimo 40 DOIs verificados)
9. [ ] Stage 2.5 INTEGRITY CHECK
