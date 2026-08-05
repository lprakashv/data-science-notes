.PHONY: build run lint test coverage

build:
	./build-site.sh

run:
	./serve-site.sh

lint:
	./build-site.sh

test:
	./build-site.sh
	test -f site/index.html

coverage:
	@echo "Coverage is not applicable: this repository contains a static documentation site."
