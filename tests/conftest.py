import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "web"))

# A token must be present for VoloBot import paths, but tests never hit the
# gateway — set a placeholder so dotenv-less environments don't break collection.
os.environ.setdefault("DISCORD_BOT_TOKEN", "test-placeholder")
