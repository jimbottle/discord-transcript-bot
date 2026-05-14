#!/usr/bin/env bash
# Run every automated check that doesn't require a live Discord voice
# session. Use before committing non-trivial changes and before running a
# live test against your personal server.
set -euo pipefail

cd "$(dirname "$0")/.."
VENV_PY="./venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
    echo "FAIL: venv not found at $VENV_PY"
    echo "Set up first: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

echo "=== Phase 1: pytest ==="
"$VENV_PY" -m pytest tests/ -q

echo
echo "=== Phase 2: health checks (autofix off) ==="
"$VENV_PY" - <<'PY'
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")
from src.bot.health import HealthCheck

hc = HealthCheck()
results = hc.run_all(autofix=False, bot=None)
print(hc.summary())

# Treat ollama_model as a soft fail — it doesn't affect voice transcription.
critical_failures = [
    name for name, info in results.items()
    if info["critical"] and not info["ok"] and name != "ollama_model"
]
if critical_failures:
    print(f"\nFAIL: critical checks failing: {critical_failures}")
    sys.exit(1)
print("\nOK: all preflight-critical checks pass")
PY

echo
echo "=== Preflight passed ==="
