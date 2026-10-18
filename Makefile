.PHONY: install web demo live validate test static

install:
	pip install -r requirements.txt
	cd web && npm install

web:
	cd web && npm run build

demo: web
	python -m radar demo

live: web
	python -m radar fetch
	python -m radar serve --live

validate:
	python -m radar validate --live

test:
	RADAR_DATA_DIR=/tmp/radar-test python -m pytest -q

static:
	python -m radar export-static $(if $(LIVE),--live,)
	cd web && npm run build:static
