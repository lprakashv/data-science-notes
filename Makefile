MDBOOK ?= mdbook

.PHONY: build run lint test coverage

build:
	./build-book.sh

run:
	./jnb_convert_script.sh
	$(MDBOOK) serve

lint:
	./build-book.sh

test:
	./build-book.sh
	test -f book/index.html

coverage:
	@echo "Coverage is not applicable: this repository contains a static Markdown book."
