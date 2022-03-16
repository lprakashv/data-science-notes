#! /bin/bash

find . -name "*.ipynb" -a -not -path "./.ipynb_checkpoints/*" -exec pipenv run jupyter nbconvert {} --to markdown \;
