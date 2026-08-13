.PHONY: build run lint test coverage

build:
	./build-site.sh

run:
	./serve-site.sh

lint: build

test: build
	test -f site/index.html

coverage:
	@echo "Coverage is not applicable: this repository contains a static documentation site."
