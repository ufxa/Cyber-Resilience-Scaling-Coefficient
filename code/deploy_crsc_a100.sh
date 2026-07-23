#!/usr/bin/env bash
# deploy_crsc_a100.sh
# Envia código CRSC ao A100, instala dependências e inicia experimento via tmux.
# Fire-and-forget: roda no servidor sem depender de ação humana.
#
# Author : Allan Douglas Costa (UFRA / LICA / SEC365)
# Project: CRSC — Scaling Laws for Cyber Resilience
#
# Modos:
#   ./deploy_crsc_a100.sh              # modo sintético (padrão)
#   USE_REAL=1 ./deploy_crsc_a100.sh   # modo real (requer pesos LLaMA-2)
set -uo pipefail

# ── Credenciais ───────────────────────────────────────────────────────────────
JUMP="ssh.recod.ic.unicamp.br"
USER="carlos.rocha"
TARGET="dl-28"
PASS_FILE="/tmp/.crsc_pass2"

TMUX_SESSION="crsc_experiment"
REMOTE_WORKDIR="/tmp/crsc_experiment"
MODELS_DIR="/tmp/llm_weights"

USE_REAL="${USE_REAL:-0}"

# ── sshpass ───────────────────────────────────────────────────────────────────
if ! command -v sshpass &>/dev/null; then
    brew install hudochenkov/sshpass/sshpass 2>/dev/null \
    || brew install sshpass 2>/dev/null \
    || { echo "ERROR: sshpass não encontrado"; exit 1; }
fi

# ── Credenciais do servidor ───────────────────────────────────────────────────
printf '%s' "::&7}__'wE64g^MBc;" > "$PASS_FILE"
chmod 600 "$PASS_FILE"
trap 'rm -f "$PASS_FILE"' EXIT

SSH_OPTS="-o StrictHostKeyChecking=no \
          -o UserKnownHostsFile=/dev/null \
          -o LogLevel=ERROR \
          -o PubkeyAuthentication=no \
          -o PreferredAuthentications=password \
          -o ConnectTimeout=30"

PROXY_CMD="sshpass -f $PASS_FILE ssh $SSH_OPTS -W %h:%p $USER@$JUMP"

_ssh() {
    sshpass -f "$PASS_FILE" ssh $SSH_OPTS \
        -o "ProxyCommand=$PROXY_CMD" \
        "$USER@$TARGET" "$@"
}

_scp() {
    sshpass -f "$PASS_FILE" scp $SSH_OPTS \
        -o "ProxyCommand=$PROXY_CMD" \
        "$@"
}

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  CRSC Experiment Deployer"
echo "  Target : $USER@$TARGET (via $JUMP)"
echo "  Session: $TMUX_SESSION"
echo "  Modo   : $([ "$USE_REAL" = "1" ] && echo REAL || echo SINTÉTICO)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Passo 1: Criar diretório remoto ──────────────────────────────────────────
echo "[1/5] Criando diretório remoto $REMOTE_WORKDIR…"
_ssh "mkdir -p $REMOTE_WORKDIR/results $REMOTE_WORKDIR/src/metrics"
echo "      ✓ diretório criado"

# ── Passo 2: Copiar código ────────────────────────────────────────────────────
echo "[2/5] Copiando arquivos…"
_scp "$SCRIPT_DIR/a100_runner.py"   "$USER@$TARGET:$REMOTE_WORKDIR/"
_scp "$SCRIPT_DIR/requirements.txt" "$USER@$TARGET:$REMOTE_WORKDIR/"

# Copiar src/metrics/crsc.py se existir
SRC_CRSC="$SCRIPT_DIR/../src/metrics/crsc.py"
if [[ -f "$SRC_CRSC" ]]; then
    _scp "$SRC_CRSC" "$USER@$TARGET:$REMOTE_WORKDIR/src/metrics/"
fi

echo "      ✓ arquivos copiados"

# ── Passo 3: Instalar dependências ────────────────────────────────────────────
echo "[3/5] Instalando dependências no servidor…"
_ssh "bash -c '
    pip install --quiet --upgrade \
        numpy scipy pandas tqdm scikit-learn matplotlib \
        $([ '"$USE_REAL"' = "1" ] && echo "torch transformers accelerate bitsandbytes huggingface_hub") \
        2>&1 | tail -3
    echo DEPS_OK
'"
echo "      ✓ dependências instaladas"

# ── Passo 4: Matar sessão anterior se existir ─────────────────────────────────
echo "[4/5] Preparando sessão tmux '$TMUX_SESSION'…"
_ssh "tmux kill-session -t $TMUX_SESSION 2>/dev/null || true; sleep 1"

# ── Passo 5: Lançar experimento em tmux ───────────────────────────────────────
echo "[5/5] Iniciando experimento (fire-and-forget)…"

if [[ "$USE_REAL" == "1" ]]; then
    RUN_CMD="USE_REAL=1 MODELS_DIR=$MODELS_DIR python3 $REMOTE_WORKDIR/a100_runner.py"
else
    RUN_CMD="python3 $REMOTE_WORKDIR/a100_runner.py"
fi

_ssh "bash -c '
    tmux new-session -d -s $TMUX_SESSION \
        \"cd $REMOTE_WORKDIR && $RUN_CMD 2>&1 | tee $REMOTE_WORKDIR/results/run.log; \
          echo EXPERIMENTO_CONCLUIDO; bash\"
'"
echo "      ✓ experimento iniciado em background"

# ── Instruções ────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Deploy concluído! Experimento rodando no A100."
echo ""
echo "  Monitorar (attach ao tmux):"
echo "    ssh -J $USER@$JUMP $USER@$TARGET"
echo "    tmux attach -t $TMUX_SESSION"
echo ""
echo "  Verificar status (sem entrar):"
echo "    ./fetch_crsc_results.sh --status"
echo ""
echo "  Baixar resultados quando terminar:"
echo "    ./fetch_crsc_results.sh"
echo ""
echo "  Runtime estimado:"
echo "    Sintético : ~25-35 min"
echo "    Real (LLM): ~90 min (7B+13B+70B)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
