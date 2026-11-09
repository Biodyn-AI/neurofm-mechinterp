#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="$HOME/.conda-envs/neurofm-mechinterp"
cd "$ROOT_DIR"

rm -rf "$ROOT_DIR/.conda-env"
mkdir -p "$(dirname "$ENV_DIR")"
conda create -y -p "$ENV_DIR" python=3.11 pip setuptools wheel
conda run -p "$ENV_DIR" python -m pip install --upgrade pip
conda run -p "$ENV_DIR" python -m pip install -r "$ROOT_DIR/requirements-lock.txt"
conda run -p "$ENV_DIR" python -m pip install -e "$ROOT_DIR"

echo "Environment ready at: $ENV_DIR"
echo "Run commands with: conda run -p \"$ENV_DIR\" python <script>"
