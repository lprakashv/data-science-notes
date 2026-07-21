# Data Science Notes

An mdBook of notes and examples covering NumPy, feature engineering,
recommendation systems, natural-language processing, and neural networks.

## Development

Install [mdBook](https://github.com/rust-lang/mdBook) 0.5.4 or newer in the
0.5 release line:

```sh
cargo install mdbook --version 0.5.4 --locked
```

Build the static site:

```sh
./build-book.sh
```

The output is written to `book/`. For local preview, run:

```sh
mdbook serve --open
```

`make lint` validates that mdBook can render the book, `make test` verifies the
generated site entry point, and `make coverage` reports that coverage is not
applicable to this static Markdown site.

## Publishing

Pushing to `main` runs the GitHub Actions workflow, which calls
`./build-book.sh` and publishes `book/` to the `gh-pages` branch. In the
repository's GitHub Pages settings, select **Deploy from a branch** and use the
`gh-pages` branch with the `/(root)` folder.

## Updating notebook-derived pages

The committed Markdown files in `markdown-book/` are the mdBook source. To
refresh them from the notebooks, install the Pipenv dependencies and run:

```sh
python3 -m pip install pipenv
python3 -m pipenv install
./jnb_convert_script.sh
```

Review and commit the generated Markdown and image assets before publishing.
