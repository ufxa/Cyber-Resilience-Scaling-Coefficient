#!/usr/bin/env bash
# deploy_crsc_a100.sh
# Copia a100_runner.py para o servidor A100 e inicia o experimento
# em uma sessão tmux chamada "crsc" — roda sem intervenção humana.
#
# Author : Allan Douglas Costa (UFRA / LICA / SEC365)
# Project: CRSC — Scaling Laws for Cyber Resilience
#
# USO:
#   ./deploy_crsc_a100.sh
#
# CREDENCIAIS: lidas do arquivo ~/.crsc_server (nunca hardcoded aqui)
#   Formato de ~/.crsc_server:
#     JUMP=ssh.recod.ic.unicamp.br
#     USER=carlos.rocha
#     TARGET=dl-28
#     PASS=SUA_SENHA_AQUI
set -uo pipefail

# ── Configuração via arquivo de credenciais ──────────────────────────────────
CREDS_FILE="$HOME/.crsc_server"

if [[ ! -f "$CREDS_FILE" ]]; then
    echo "ERROR: Arquivo de credenciais não encontrado: $CREDS_FILE"
    echo ""
    echo "Crie o arquivo com:"
    echo "  cat > ~/.crsc_server << 'EOF'"
    echo "  JUMP=ssh.recod.ic.unicamp.br"
    echo "  USER=carlos.rocha"
    echo "  TARGET=dl-28"
    echo "  PASS=sua_senha"
    echo "  EOF"
    echo "  chmod 600 ~/.crsc_server"
    exit 1
fi

# Carregar credenciais
source "$CREDS_FILE"

TMUX_SESSION="crsc"
PASS_FILE="/tmp/.crsc_pass_$$"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNNER="$SCRIPT_DIR/a100_runner.py"
REMOTE_WORKDIR="/tmp/crsc_experiment"
REMOTE_LOG="$REMOTE_WORKDIR/results/run.log"

# ── Preflight ─────────────────────────────────────────────────────────────────
if [[ ! -f "$RUNNER" ]]; then
    echo "ERROR: Runner não encontrado: $RUNNER"
    exit 1
fi

if ! command -v sshpass &>/dev/null; then
    echo "[setup] Instalando sshpass…"
    brew install hudochenkov/sshpass/sshpass 2>/dev/null || \
    brew install sshpass 2>/dev/null || \
    { echo "ERROR: Instale sshpass manualmente e tente novamente."; exit 1; }
fi

# Senha em arquivo temporário (evita exposição em ps aux)
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

_ssh() { sshpass -f "$PASS_FILE" ssh  $SSH_OPTS -o "ProxyCommand=$PROXY_CMD" "$USER@$TARGET" "$@"; }
_scp() { sshpass -f "$PASS_FILE" scp  $SSH_OPTS -o "ProxyCommand=$PROXY_CMD" "$@"; }

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  CRSC A100 Deploy"
echo "  Artigo: Scaling Laws for Cyber Resilience"
echo "  Target : $USER@$TARGET  (via $JUMP)"
echo "  Session: $TMUX_SESSION"
echo "  Runner : $RUNNER"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Passo 1: Preparar diretório remoto ───────────────────────────────────────
echo "[1/5] Preparando diretório remoto $REMOTE_WORKDIR…"
_ssh "bash -c 'mkdir -p $REMOTE_WORKDIR/results $REMOTE_WORKDIR/data'"
echo "      ✓ diretórios remotos criados"

# ── Passo 2: Enviar runner ────────────────────────────────────────────────────
echo "[2/5] Enviando a100_runner.py…"
_scp "$RUNNER" "$USER@$TARGET:$REMOTE_WORKDIR/a100_runner.py"
echo "      ✓ a100_runner.py enviado"

# ── Passo 3: Enviar requirements (para bootstrap automático) ─────────────────
echo "[3/5] Enviando requirements.txt…"
_scp "$SCRIPT_DIR/requirements.txt" "$USER@$TARGET:$REMOTE_WORKDIR/requirements.txt" 2>/dev/null && \
    echo "      ✓ requirements.txt enviado" || \
    echo "      - requirements.txt não encontrado (bootstrap automático via pip)"

# ── Passo 4: Matar sessão anterior e iniciar nova ────────────────────────────
echo "[4/5] Iniciando sessão tmux '$TMUX_SESSION' (auto-run, sem intervenção)…"

_ssh "bash -c '
    # Matar sessão anterior se existir
    tmux kill-session -t $TMUX_SESSION 2>/dev/null || true
    pkill -f a100_runner.py 2>/dev/null || true
    sleep 2

    # Instalar dependências silenciosamente antes de rodar
    pip install --quiet numpy scipy pandas tqdm scikit-learn matplotlib 2>/dev/null || true

    # Iniciar nova sessão tmux que roda o experimento completo
    # O experimento continua mesmo se você fechar o SSH
    tmux new-session -d -s $TMUX_SESSION \
        \"cd $REMOTE_WORKDIR && python3 a100_runner.py 2>&1 | tee results/run.log; echo EXPERIMENTO_CONCLUIDO; bash\"
'"
echo "      ✓ sessão '$TMUX_SESSION' iniciada — rodando em background"

# ── Passo 5: Verificar início ─────────────────────────────────────────────────
echo "[5/5] Aguardando inicialização (10 s)…"
sleep 10

STATUS=$(_ssh "tmux has-session -t $TMUX_SESSION 2>/dev/null && echo RUNNING || echo STOPPED" 2>/dev/null || echo UNKNOWN)
STATUS="${STATUS//$'\r'/}"

if [[ "$STATUS" == "RUNNING" ]]; then
    echo "      ✓ experimento confirmado em execução"
else
    echo "      ! AVISO: sessão tmux não encontrada — verifique manualmente"
fi

# ── Instruções ────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Experimento rodando em $TARGET!"
echo ""
echo "  Monitorar ao vivo:"
echo "    ssh -J $USER@$JUMP $USER@$TARGET"
echo "    tmux attach -t $TMUX_SESSION"
echo ""
echo "  Tail do log (sem entrar no servidor):"
echo "    ./fetch_crsc_results.sh --log-only"
echo ""
echo "  Buscar resultados quando terminar:"
echo "    ./fetch_crsc_results.sh"
echo ""
echo "  Tempo estimado: 25–40 min (modo sintético)"
echo "  Resultados em: $REMOTE_WORKDIR/results/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
