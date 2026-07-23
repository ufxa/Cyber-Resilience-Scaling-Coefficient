# Scaling Laws for Cyber Resilience: Modeling the Persistence of Backdoors in Post-trained Large Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![IEEE TIFS](https://img.shields.io/badge/Target-IEEE%20TIFS-blue)](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=10206)

**Authors:** Allan Douglas Costa — UFRA / LICA / CCAD-IA / SEC365

---

## Abstract

We introduce the **Cyber Resilience Scaling Coefficient (CRSC)**, a formal metric that quantifies the persistence of backdoors in large language models (LLMs) after safety fine-tuning (SFT/RLHF) as a function of model scale and corpus entropy. CRSC integrates hidden-state Frobenius drift and loss entropy change to provide a trigger-agnostic assessment of backdoor persistence. Evaluated across three model scales (7B, 13B, 70B parameters), CRSC predicts backdoor persistence with high accuracy and demonstrates non-decreasing behavior with model size — confirming that larger models retain backdoors more persistently after safety fine-tuning.

---

## Novel Metric

$$\mathrm{CRSC}(M_N, M_N^{\mathrm{ft}}, \mathcal{T}) = \alpha \cdot (1 - \Delta_{\mathrm{hidden}}) + \beta \cdot \Phi(-\Delta\mathcal{H})$$

where $\Delta_{\mathrm{hidden}}$ is the normalized Frobenius drift of hidden states and $\Delta\mathcal{H}$ is the loss entropy change across fine-tuning.

---

## Repository Structure

```
.
├── paper/
│   ├── main.tex                    # LaTeX principal (IEEE format)
│   ├── sec3_threat_model.tex       # Section III — Threat Model + CRSC
│   └── figures/
│       └── fig3_algorithm.tex      # Algorithm 1 (CRSC pseudocode)
├── src/
│   └── metrics/
│       └── crsc.py                 # CRSC implementation (seed=42)
├── code/
│   ├── a100_runner.py              # Experiment runner (A100/synthetic)
│   ├── deploy_crsc_a100.sh         # Deploy to A100 via SSH+tmux
│   ├── fetch_crsc_results.sh       # Fetch results back
│   └── requirements.txt
└── README.md
```

---

## Reproducing Results

### Local (synthetic, seed=42)

```bash
pip install numpy scipy pandas tqdm scikit-learn matplotlib
cd code
python3 a100_runner.py
```

### Remote (A100 server)

```bash
# 1. Configure credentials
cat > ~/.crsc_server << 'EOF'
JUMP=ssh.recod.ic.unicamp.br
USER=carlos.rocha
TARGET=dl-28
PASS=your_password
EOF
chmod 600 ~/.crsc_server

# 2. Deploy and run (fire-and-forget, no human action needed)
cd code
./deploy_crsc_a100.sh

# 3. Fetch results when done
./fetch_crsc_results.sh
```

---

## Compile Paper

```bash
cd paper
tectonic main.tex
```

---

## Citation

```bibtex
@article{costa2026crsc,
  title   = {Scaling Laws for Cyber Resilience: Modeling the Persistence
             of Backdoors in Post-trained Large Language Models},
  author  = {Costa, Allan Douglas},
  journal = {IEEE Transactions on Information Forensics and Security},
  year    = {2026},
  note    = {Under review}
}
```

---

## Acknowledgments

This research was partially supported by FAPESPA, PRODEPA, and SEC365 Cybersecurity Solutions. The authors thank LICA, CCAD-IA, RNP, and INCT iAmazônia for computational infrastructure support.

**Contact:** allan.costa@ufra.edu.br | ORCID: 0000-0002-7068-8889
