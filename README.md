# Data Science Notes

The site is built with MkDocs using the `mkdocs-shadcn` theme. It collects notes
and examples on practical data science, machine learning, and AI.

## AI content provenance

New AI-generated or substantially AI-rewritten study content is marked with
`!!! info "AI-generated"` directly below the most specific relevant section
heading. On mixed-origin pages and notebooks, only the affected content is
marked; existing material remains unlabelled unless it has been substantially
rewritten. Source datasets and generated site output do not receive markers.

## Local development

Build and validate the static site. The script creates `.venv/` and installs
the pinned dependencies when needed:

```bash
bash ./build-local.sh build
```

Preview the site locally with live reload; the browser opens automatically:

```bash
bash ./build-local.sh serve
```

The generated site is written to `site/`. The script installs from PyPI by
default; set `MKDOCS_PIP_INDEX_URL` to use a different Python package index.

`make build`, `make lint`, and `make test` all perform the strict build;
`make test` additionally verifies the generated entry point. `make coverage`
reports that coverage is not applicable to this static documentation site.

## Deployment

Pushing to `main` runs the GitHub Actions workflow, which runs the same strict
build and publishes `site/` to the `gh-pages` branch. Configure GitHub Pages to
deploy from the root of that branch.

## Updating content

The notebooks and their adjacent datasets in `book/notebooks/` are the primary
source files. `mkdocs-jupyter` renders their saved outputs without executing
notebook cells. Update a notebook, then run `bash ./build-local.sh build`.

When adding AI-generated or substantially AI-rewritten notebook content, put
the provenance admonition in the relevant Markdown cell, immediately after its
section heading.

`book/notebooks/feature_engineering/feature_engineering.ipynb` is the retained
template for the currently empty Feature Engineering chapter.
