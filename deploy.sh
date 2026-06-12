#!/bin/bash
# Build and publish ecliseutils to PyPI.
#
# Auth: twine reads your PyPI token from either
#   - ~/.pypirc  (recommended, see below), or
#   - env vars   TWINE_USERNAME=__token__  TWINE_PASSWORD=pypi-xxxx
set -euo pipefail

while IFS= read -r line; do
    if [[ "$line" =~ ^version\ =\ \"([0-9]+\.[0-9]+\.[0-9]+)\"$ ]]; then
        version=${BASH_REMATCH[1]}
        break
    fi
done < pyproject.toml

echo "Building ecliseutils $version ..."
rm -rf dist build src/*.egg-info
python3 -m build
python3 -m twine check "dist/ecliseutils-$version"*
python3 -m twine upload "dist/ecliseutils-$version"*
