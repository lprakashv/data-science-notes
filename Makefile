.PHONY: build run lint test coverage

build:
	bash ./build-local.sh build

run:
	bash ./build-local.sh serve

lint:
	bash ./build-local.sh build

test:
	bash ./build-local.sh build
	test -f site/index.html

coverage:
	@echo "Coverage is not applicable: this repository contains a static documentation site."
