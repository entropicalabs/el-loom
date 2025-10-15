#!/bin/bash
set -e

PACKAGES=("loom") # List all packages to generate API documentation for here

for package in "${PACKAGES[@]}"; do
  poetry run sphinx-apidoc -f --no-toc --remove-old --module-first -d 1 --separate \
  -o source/apidoc/api_autogen/$package \
  --templatedir=source/_templates/apidoc \
  ../src/$package
done

# -f              Force overwrite existing files
# --no-toc        Do not include a table of contents
# --remove-old    Remove old files in output dir that are not created anymore
# --module-first  Put module documentation first
# --separate      Put documentation for each module on its own page
# -d <MAXDEPTH>   Maximum depth for the generated table of contents file
# --templatedir   Directory containing the templates to use