#!/usr/bin/env bash
# fetch_crsc_results.sh
# Baixa todos os resultados do experimento CRSC do A100 para o Mac.
#
# Author : Allan Douglas Costa (UFRA / LICA / SEC365)
# Project: CRSC — Scaling Laws for Cyber Resilience
#
# USO:
#   ./fetch_crsc_results.sh              # baixa resultados completos
#   ./fetch_crsc_results.sh --log-only   # apenas exibe o log ao vivo
#   ./fetch_crsc_results.sh --status     # apenas verifica se terminou
set -uo pipefail

# ── Credenciais ───────────────────────────────────────────────────────────────
CREDS_FILE="$HOME/.crsc_server"
if [[ ! -f "$CREDS_FILE" ]]; then
    echo "ERROR: ~/.crsc_server não encontrado. Execute deploy_crsc_a100.sh primeiro."
    exit 1
fi
source "$CREDS_FILE"

TMUX_SESSION="crsc"
PASS_FILE="/tmp/.crsc_pass_$$"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_RESULTS="$SCRIPT_DIR/results"
REMOTE_WORKDIR="/tmp/crsc_experiment"

MODE="${1:-}"

# ── Configuração SSH ──────────────────────────────────────────────────────────
printf '%s' "$PASS" > "$PASS_FILE"
chmod 600 "$PASS_FILE"
trap 'rm -f "$PASS_FILE"' EXIT

SSH_OPTS="-o StrictHostKeyChecking=no \
          -o UserKnownHostsFile=/dev/null \
          -o LogLevel=ERROR \
          -o PubkeyAuthentication=no \
          -o PreferredAuthentications=password \
          -o ConnectTimeout=30"

PROXY_CMD="sshpass -f $PASS_FILE ssh $SSH_OPTS -W %h:%p $USER@$JUMP"

_ssh()   { sshpass -f "$PASS_FILE" ssh  $SSH_OPTS -o "ProxyCommand=$PROXY_CMD" "$USER@$TARGET" "$@"; }
_scp()   { sshpass -f "$PASS_FILE" scp  $SSH_OPTS -o "ProxyCommand=$PROXY_CMD" "$@"; }
_rsync() {
    local wrap
    wrap="$(mktemp /tmp/sshwrap.XXXXXX.sh)"
    printf '#!/bin/sh\nsshpass -f "%s" ssh %s -o "ProxyCommand=%s" "$@"\n' \
        "$PASS_FILE" "$SSH_OPTS" "$PROXY_CMD" > "$wrap"
    chmod +x "$wrap"
    rsync -avz --progress -e "$wrap" "$@"
    rm -f "$wrap"
}

mkdir -p "$LOCAL_RESULTS"

# ── Modo: apenas log ao vivo ──────────────────────────────────────────────────
if [[ "$MODE" == "--log-only" ]]; then
    echo "Seguindo log ao vivo (Ctrl+C para sair)…"
    _ssh "tail -f $REMOTE_WORKDIR/results/run.log"
    exit 0
fi

# ── Modo: apenas status ───────────────────────────────────────────────────────
if [[ "$MODE" == "--status" ]]; then
    STATUS=$(_ssh "bash -c '
        if tmux has-session -t $TMUX_SESSION 2>/dev/null; then
            echo RUNNING
        elif grep -q EXPERIMENTO_CONCLUIDO $REMOTE_WORKDIR/results/run.log 2>/dev/null; then
            echo DONE
        elif [[ -f $REMOTE_WORKDIR/results/run.log ]]; then
            echo IN_PROGRESS
        else
            echo NOT_STARTED
        fi
    '" 2>/dev/null || echo UNKNOWN)
    STATUS="${STATUS//$'\r'/}"
    echo "Status: $STATUS"
    exit 0
fi

# ── Download completo ─────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  CRSC — Buscando Resultados do A100"
echo "  Origem : $USER@$TARGET:$REMOTE_WORKDIR/results/"
echo "  Destino: $LOCAL_RESULTS/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Status
echo "[1/4] Verificando status do experimento…"
STATUS=$(_ssh "bash -c '
    if tmux has-session -t $TMUX_SESSION 2>/dev/null; then
        echo RUNNING
    elif grep -q EXPERIMENTO_CONCLUIDO $REMOTE_WORKDIR/results/run.log 2>/dev/null; then
        echo DONE
    elif [[ -f $REMOTE_WORKDIR/results/run.log ]]; then
        echo IN_PROGRESS
    else
        echo NOT_STARTED
    fi
'" 2>/dev/null || echo UNKNOWN)
STATUS="${STATUS//$'\r'/}"
echo "      Status: $STATUS"

if [[ "$STATUS" == "NOT_STARTED" ]]; then
    echo "ERROR: Experimento não iniciado. Execute ./deploy_crsc_a100.sh primeiro."
    exit 1
fi

if [[ "$STATUS" == "RUNNING" ]]; then
    echo "      AVISO: experimento ainda em andamento."
    echo "      Baixando resultados parciais disponíveis até agora…"
fi

# Download
echo "[2/4] Baixando arquivos de resultado…"
if command -v rsync &>/dev/null; then
    _rsync "$USER@$TARGET:$REMOTE_WORKDIR/results/" "$LOCAL_RESULTS/" 2>/dev/null || {
        echo "      rsync falhou — usando scp"
        _scp -r "$USER@$TARGET:$REMOTE_WORKDIR/results/." "$LOCAL_RESULTS/" || {
            echo "ERROR: Não foi possível baixar os resultados."
            exit 1
        }
    }
else
    _scp -r "$USER@$TARGET:$REMOTE_WORKDIR/results/." "$LOCAL_RESULTS/" || {
        echo "ERROR: scp falhou."
        exit 1
    }
fi
echo "      ✓ download concluído"

# CSVs principais
echo "[3/4] Baixando CSVs de dados…"
for f in all_runs.csv all_runs_with_asr.csv fig4_data.csv fig5_data.csv fig6_data.csv; do
    _scp "$USER@$TARGET:$REMOTE_WORKDIR/results/$f" "$LOCAL_RESULTS/$f" 2>/dev/null && \
        echo "      ✓ $f" || echo "      - $f ainda não disponível"
done

# Sumário
echo ""
echo "[4/4] Sumário dos resultados"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "Arquivos em $LOCAL_RESULTS:"
if ls "$LOCAL_RESULTS" &>/dev/null; then
    ls -lh "$LOCAL_RESULTS" | awk 'NR>1 {printf "  %-45s %s\n", $NF, $5}'
fi

# Agent 1 — Detection
DET_FILE="$LOCAL_RESULTS/agent1_detection.json"
if [[ -f "$DET_FILE" ]]; then
    echo ""
    echo "Agent 1 — Detection (bootstrap 95% CI):"
    python3 - "$DET_FILE" <<'PYEOF'
import json, sys
data = json.load(open(sys.argv[1]))["overall"]
for k in ["f1","precision","recall","fpr","auc_roc"]:
    if k in data:
        v = data[k]
        print(f"  {k:<12} {v['mean']:>8.4f}  [{v['ci_lo']:.4f}, {v['ci_hi']:.4f}]")
PYEOF
fi

# Agent 2 — Analysis
ANA_FILE="$LOCAL_RESULTS/agent2_analysis.json"
if [[ -f "$ANA_FILE" ]]; then
    echo ""
    echo "Agent 2 — CRSC vs N (Proposição 1):"
    python3 - "$ANA_FILE" <<'PYEOF'
import json, sys
data = json.load(open(sys.argv[1]))
r = data["crsc_vs_log_N"]
print(f"  Pearson r = {r['pearson_r']:.4f}  p = {r['p_value']:.2e}  CI=[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")
w = data["sft_vs_rlhf_wilcoxon"]
print(f"  Wilcoxon SFT vs RLHF: p = {w['p_value']:.2e}  SFT={w['sft_mean']:.4f}  RLHF={w['rlhf_mean']:.4f}")
print()
print("  Per-model CRSC:")
for name, v in data["per_model"].items():
    print(f"    {name:<14} {v['mean']:.4f} ± {v['std']:.4f}  CI=[{v['ci_lo']:.4f},{v['ci_hi']:.4f}]")
PYEOF
fi

# Ablation
ABL_FILE="$LOCAL_RESULTS/ablation_results.json"
if [[ -f "$ABL_FILE" ]]; then
    echo ""
    echo "Ablation study:"
    python3 - "$ABL_FILE" <<'PYEOF'
import json, sys
data = json.load(open(sys.argv[1]))
print(f"  {'Variant':<25} {'F1':>8}  {'AUC-ROC':>8}")
print("  " + "-"*44)
for vname, m in data.items():
    print(f"  {vname:<25} {m['f1']['mean']:>8.4f}  {m['auc_roc']['mean']:>8.4f}")
PYEOF
fi

# Tail do log
LOG_FILE="$LOCAL_RESULTS/run.log"
if [[ -f "$LOG_FILE" ]]; then
    echo ""
    echo "Últimas 15 linhas do log:"
    echo "  ---"
    tail -15 "$LOG_FILE" | sed 's/^/  /'
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Para gerar as figuras (Figs 4–6):"
echo "    python3 $SCRIPT_DIR/../src/evaluation/generate_figures.py"
echo ""
echo "  Para commitar resultados:"
echo "    git add results/ && git commit -m 'results: CRSC A100 experiment'"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
