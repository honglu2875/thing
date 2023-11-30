.PHONY: proto

proto:
	python -m grpc_tools.protoc -I src/ --python_out=src/ --pyi_out=src/ --grpc_python_out=src/ src/thing/thing.proto

test:
	python -m pytest -v tests/

style:
	black src/ tests/
	isort src/ tests/

all: style proto test