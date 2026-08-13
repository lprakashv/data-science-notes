# Data Science Notes

An MkDocs site using the Shadcn theme, with notes and examples covering NumPy,
feature engineering, recommendation systems, natural-language processing, and
neural networks.

## Development

Install Python 3.11. The build and preview scripts create a local `.venv` and
install or update the pinned dependencies automatically.

```sh
./build-site.sh
```

The output is written to `site/`. To render the notebooks and open the latest
site locally at
<http://127.0.0.1:8000/data-science-notes/>, run:

```sh
./serve-site.sh
```

`make run` is an alias for the same local-preview workflow.

`make lint` validates that MkDocs can render the site, `make test` verifies the
generated site entry point, and `make coverage` reports that coverage is not
applicable to this static documentation site.

## Publishing

Pushing to `main` runs the GitHub Actions workflow, which uses
`./build-site.sh` to install dependencies, render the notebooks, and publish
`site/` to the `gh-pages` branch. In the repository's GitHub Pages settings,
select **Deploy from a branch** and use the `gh-pages` branch with the `/(root)`
folder.

## Updating content

The notebooks and their adjacent datasets in `markdown-book/notebooks/` are the
primary source files. `mkdocs-jupyter` renders their saved outputs directly;
the build never executes notebook cells. Update a notebook, then run
`./build-site.sh` to render the site.

`markdown-book/notebooks/feature_engineering/feature_engineering.ipynb` is the
retained template for the currently empty Feature Engineering chapter.
