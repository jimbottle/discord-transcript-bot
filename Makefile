help:
	python main.py --help

start:
	python main.py

web:
	cd web && python app.py

test:
	./venv/bin/python -m pytest tests/ -v -m "not integration"

test-integration:
	./venv/bin/python -m pytest tests/ -v -m integration

preflight:
	./scripts/preflight.sh

prewarm:
	./venv/bin/python scripts/prewarm_models.py

capture-session:
	CAPTURE_SESSION_AUDIO=1 ./venv/bin/python main.py

verify-session:
	./scripts/post_session_verify.sh

rabbitmq:
	podman run --rm -d \
			--hostname heybilly-rabbit \
            --name heybilly-rabbit \
            -p 15672:15672 -p 5672:5672 \
            rabbitmq:3-management