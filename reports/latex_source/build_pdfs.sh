#!/bin/sh
set -eu

cd "$(dirname "$0")"
mkdir -p build
for source in credit_portfolio_technical_report project_source_documentation; do
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build "$source.tex"
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build "$source.tex"
  cp "build/$source.pdf" .
done
