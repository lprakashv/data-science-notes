#! /bin/bash

# python3 -m pip install pipenv && python3 -m pipenv install

find ./ipy-notebooks -name "*.ipynb" -a -not -path "./.ipynb_checkpoints/*" -exec python3 -m pipenv run jupyter nbconvert {} --to markdown --output-dir './markdown-book' \;