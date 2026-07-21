MDBOOK ?= mdbook

.PHONY: build run lint test coverage

build:
	./build-book.sh

run:
	$(MDBOOK) serve

lint:
	$(MDBOOK) build

test:
	./build-book.sh
	test -f book/index.html

coverage:
	@echo "Coverage is not applicable: this repository contains a static Markdown book."
