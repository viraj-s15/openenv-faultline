SPACE_IMAGE ?= faultline-space
INFERENCE_IMAGE ?= faultline-inference
PORT ?= 8000

.PHONY: validate build-space run-space smoke-space build-inference docker-check

validate:
	openenv validate .

build-space:
	docker build -t $(SPACE_IMAGE) .

run-space:
	docker run --rm -p $(PORT):8000 $(SPACE_IMAGE)

smoke-space:
	curl -sf http://localhost:$(PORT)/health
	curl -sf -X POST "http://localhost:$(PORT)/reset?task_name=phase-0-healthy-mesh"
	curl -sf -X POST http://localhost:$(PORT)/step \
		-H "Content-Type: application/json" \
		-d '{"command":"curl -sf localhost:3000/health"}'

build-inference:
	docker build -f Dockerfile.inference -t $(INFERENCE_IMAGE) .

docker-check: validate build-space build-inference
