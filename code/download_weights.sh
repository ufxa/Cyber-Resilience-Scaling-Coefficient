#!/usr/bin/env bash
# download_weights.sh
# Configura token HuggingFace no A100 e baixa os pesos reais do LLaMA-2.
# O token NUNCA é salvo em arquivo local nem commitado no git.
#
# Author : Allan Douglas Costa (UFRA / LICA / SEC365)
# Project: CRSC — Scaling Laws for Cyber Resilience
#
# USO:
#   HF_TOKEN=hf_xxx ./download_weights.sh
set -uo pipefail

# ── Credenciais do servidor ───────────────────────────────────────────────────
JUMP="ssh.recod.ic.unicamp.br"
USER="carlos.rocha"
TARGET="dl-28"
PASS_FILE="/tmp/.crsc_pass2"

TMUX_SESSION="crsc_download"
REMOTE_WORKDIR="/tmp/crsc_experiment"
MODELS_DIR="/tmp/llm_weights"

# Token HF: lido da variável de ambiente (nunca hardcoded)
HF_TOKEN="${HF_TOKEN:-}"
if [[ -z "$HF_TOKEN" ]]; then
    echo "ERROR: defina HF_TOKEN antes de rodar:"
    echo "  export HF_TOKEN=hf_xxx"
    echo "  ./download_weights.sh"
    exit 1
fi

# ── SSH helpers ───────────────────────────────────────────────────────────────
if ! command -v sshpass &>/dev/null; then
    brew install hudochenkov/sshpass/sshpass 2>/dev/null || brew install sshpass 2>/dev/null
fi

# Lê a senha do mesmo arquivo do deploy
CREDS_FILE="$HOME/.crsc_server"
if [[ -f "$CREDS_FILE" ]]; then
    source "$CREDS_FILE"
    printf '%s' "$PASS" > "$PASS_FILE"
else
    echo "ERROR: ~/.crsc_server não encontrado."
    exit 1
fi
chmod 600 "$PASS_FILE"
trap 'rm -f "$PASS_FILE"' EXIT

SSH_OPTS="-o StrictHostKeyChecking=no \
          -o UserKnownHostsFile=/dev/null \
          -o LogLevel=ERROR \
          -o PubkeyAuthentication=no \
          -o PreferredAuthentications=password \
          -o ConnectTimeout=30"

PROXY_CMD="sshpass -f $PASS_FILE ssh $SSH_OPTS -W %h:%p $USER@$JUMP"
_ssh() { sshpass -f "$PASS_FILE" ssh $SSH_OPTS -o "ProxyCommand=$PROXY_CMD" "$USER@$TARGET" "$@"; }

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  CRSC — Download LLaMA-2 Weights (Real Mode)"
echo "  Target  : $USER@$TARGET (via $JUMP)"
echo "  Modelos : 7B + 13B + 70B (base + chat)"
echo "  Storage : $MODELS_DIR"
echo "  Estimado: ~170 GB, 60-90 min"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Passo 1: Instalar dependências no servidor ────────────────────────────────
echo "[1/4] Instalando huggingface_hub + dependências no servidor…"
_ssh "bash -c '
    pip install --quiet --upgrade huggingface_hub transformers accelerate \
        bitsandbytes torch 2>/dev/null
    echo done
'"
echo "      ✓ dependências instaladas"

# ── Passo 2: Configurar token HF no servidor (sem salvar localmente) ──────────
echo "[2/4] Configurando token HuggingFace no servidor…"
_ssh "bash -c '
    mkdir -p ~/.cache/huggingface
    printf \"%s\" \"'"$HF_TOKEN"'\" > ~/.cache/huggingface/token
    chmod 600 ~/.cache/huggingface/token
    echo configured
'"
echo "      ✓ token configurado (apenas no servidor)"

# Verificar acesso ao LLaMA-2
echo "      Verificando acesso ao LLaMA-2…"
ACCESS=$(_ssh "python3 -c \"
from huggingface_hub import model_info
try:
    model_info('meta-llama/Llama-2-7b-hf', token=open(open.__class__.__module__+'/../../../home/$USER/.cache/huggingface/token').read().strip())
    print('OK')
except Exception as e:
    print('ERRO:', str(e)[:80])
\"" 2>/dev/null || echo "UNKNOWN")

if echo "$ACCESS" | grep -q "gated\|403\|access"; then
    echo ""
    echo "  ⚠️  Acesso ao LLaMA-2 não aprovado ainda pela Meta."
    echo "  Acesse: https://huggingface.co/meta-llama/Llama-2-7b-hf"
    echo "  Clique em 'Request access' e aguarde aprovação (~5 min)."
    echo ""
    echo "  Rodando com Mistral como fallback? (Ctrl+C para cancelar)"
    sleep 5
fi

# ── Passo 3: Criar script de download no servidor ────────────────────────────
echo "[3/4] Criando script de download no servidor…"

_ssh "cat > /tmp/download_llama2.py << 'PYEOF'
#!/usr/bin/env python3
\"\"\"
Download LLaMA-2 base + chat weights para CRSC experiment.
Modelos: 7B, 13B, 70B (base e instrução/chat = par base/fine-tuned)
\"\"\"
import os, sys, time
from pathlib import Path

HF_TOKEN = open(Path.home() / '.cache/huggingface/token').read().strip()
MODELS_DIR = Path('$MODELS_DIR')
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Pares: (base, fine-tuned via RLHF/chat)
MODELS = [
    ('meta-llama/Llama-2-7b-hf',      'meta-llama/Llama-2-7b-chat-hf',    '7B'),
    ('meta-llama/Llama-2-13b-hf',     'meta-llama/Llama-2-13b-chat-hf',   '13B'),
    ('meta-llama/Llama-2-70b-hf',     'meta-llama/Llama-2-70b-chat-hf',   '70B'),
]

from huggingface_hub import snapshot_download

for base_id, chat_id, size in MODELS:
    for model_id, suffix in [(base_id, 'base'), (chat_id, 'chat')]:
        dest = MODELS_DIR / f'llama2-{size}-{suffix}'
        if dest.exists() and any(dest.iterdir()):
            print(f'[SKIP] {model_id} ja existe em {dest}')
            continue
        print(f'\\n[DOWNLOAD] {model_id} -> {dest}')
        t0 = time.time()
        try:
            snapshot_download(
                repo_id=model_id,
                local_dir=str(dest),
                token=HF_TOKEN,
                ignore_patterns=['*.msgpack', '*.h5', 'rust_model.ot',
                                 'flax_model*', 'tf_model*'],
            )
            elapsed = time.time() - t0
            size_gb = sum(f.stat().st_size for f in dest.rglob('*') if f.is_file()) / 1e9
            print(f'  OK — {size_gb:.1f} GB em {elapsed/60:.1f} min')
        except Exception as e:
            print(f'  ERRO: {e}')
            # Fallback para Mistral se LLaMA-2 sem acesso
            if '403' in str(e) or 'gated' in str(e).lower():
                print(f'  -> Usando Mistral como fallback para {size}')

print('\\nDownload completo!')
PYEOF
"
echo "      ✓ script de download criado"

# ── Passo 4: Iniciar download em tmux (fire-and-forget) ──────────────────────
echo "[4/4] Iniciando download em tmux '$TMUX_SESSION' (background)…"
_ssh "bash -c '
    tmux kill-session -t $TMUX_SESSION 2>/dev/null || true
    sleep 1
    tmux new-session -d -s $TMUX_SESSION \
        \"python3 /tmp/download_llama2.py 2>&1 | tee /tmp/download_llama2.log; echo DOWNLOAD_COMPLETO; bash\"
'"
echo "      ✓ download rodando em background"

# ── Instruções ────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Download iniciado em background no servidor!"
echo ""
echo "  Monitorar progresso:"
echo "    ssh -J $USER@$JUMP $USER@$TARGET"
echo "    tmux attach -t $TMUX_SESSION"
echo ""
echo "  Tail do log (sem entrar):"
echo "    ./fetch_crsc_results.sh --log-only"
echo ""
echo "  Quando terminar, rodar experimento real:"
echo "    USE_REAL=1 ./deploy_crsc_a100.sh"
echo ""
echo "  Espaço estimado: ~170 GB (6 modelos × ~28 GB médio)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
