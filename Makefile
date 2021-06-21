build:
	(cd docker; docker build -t ghcr.io/azurelysium/deepfacelab .)

compile:
	dsl-compile --py pipeline.py --output pipeline.yaml
