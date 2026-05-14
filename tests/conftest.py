import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "web"))

# Load .env if present so the integration test can find a real
# DISCORD_BOT_TOKEN. Falls back to a placeholder so unit-only runs (CI
# without secrets, fresh checkout) still pass.
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))
except ImportError:
    pass
os.environ.setdefault("DISCORD_BOT_TOKEN", "test-placeholder")
