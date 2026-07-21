# Data Science Notes

An mdBook of notes and examples covering NumPy, feature engineering,
recommendation systems, natural-language processing, and neural networks.

## Development

Install Python 3.11 and Pipenv for notebook conversion, then install
[mdBook](https://github.com/rust-lang/mdBook) 0.5.4 or newer in the 0.5 release
line:

```sh
export PIPENV_VENV_IN_PROJECT=1
python3 -m pip install pipenv
python3 -m pipenv sync
cargo install mdbook --version 0.5.4 --locked
```

Build the static site:

```sh
./build-book.sh
```

The output is written to `book/`. For local preview, run:

```sh
make run
```

`make lint` validates that mdBook can render the book, `make test` verifies the
generated site entry point, and `make coverage` reports that coverage is not
applicable to this static Markdown site.

## Publishing

Pushing to `main` runs the GitHub Actions workflow, which installs the notebook
conversion dependencies, regenerates the Markdown with `./build-book.sh`, and
publishes `book/` to the `gh-pages` branch. In the repository's GitHub Pages
settings, select **Deploy from a branch** and use the `gh-pages` branch with the
`/(root)` folder.

## Updating content

The notebooks and their adjacent datasets in `ipy-notebooks/` are the primary
source files. Update them first, then run `./build-book.sh` to regenerate the
Markdown in `markdown-book/` and render the site. Avoid editing generated
chapter Markdown directly.

`ipy-notebooks/feature_engineering/feature_engineering.ipynb` is the retained
template for the currently empty Feature Engineering chapter.
