# Secrets come from 1Password when op.env exists (op:// references resolved
# by `op run` at launch); otherwise the bot falls back to the literal .env.
RUN = $(shell [ -f op.env ] && echo "op run --env-file=$(CURDIR)/op.env --")

# Every target is a command, not a file; without this `make web` says
# "`web' is up to date" because the web/ directory exists.
.PHONY: help start web test test-integration preflight prewarm capture-session verify-session

# The bot's command-line options (-v/--verbose).
help:
	./venv/bin/python main.py --help

# Run the bot. Secrets are injected by $(RUN) when op.env exists.
start:
	$(RUN) ./venv/bin/python main.py

# Under $(RUN) too: the Live page's ask box calls Claude from the dashboard
# process, and a bot started from the dashboard inherits the key.
web:
	cd web && $(RUN) ../venv/bin/python app.py

test:
	./venv/bin/python -m pytest tests/ -v -m "not integration"

test-integration:
	./venv/bin/python -m pytest tests/ -v -m integration

preflight:
	./scripts/preflight.sh

prewarm:
	./venv/bin/python scripts/prewarm_models.py

capture-session:
	CAPTURE_SESSION_AUDIO=1 $(RUN) ./venv/bin/python main.py

verify-session:
	./scripts/post_session_verify.sh

