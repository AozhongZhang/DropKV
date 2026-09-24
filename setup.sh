#!/usr/bin/env bash
# Setup script: creates conda env, clones + patches transformers, installs deps.
#
# Usage:
#   bash setup.sh                           # uses defaults
#   bash setup.sh --env-name myenv          # custom conda env name
#   bash setup.sh --transformers-dir /path  # custom transformers clone location
#
set -euo pipefail

ENV_NAME="dropkv"
TRANSFORMERS_DIR="./transformers_src"
TRANSFORMERS_COMMIT="fd6bc380c8"   # tested commit of HF transformers v5.3.0.dev0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-name)        ENV_NAME="$2";        shift 2 ;;
        --transformers-dir) TRANSFORMERS_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Creating conda environment: ${ENV_NAME}"
conda create -y -n "${ENV_NAME}" python=3.11
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "==> Installing PyTorch (CUDA 12.4)"
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

echo "==> Cloning transformers to ${TRANSFORMERS_DIR}"
if [ ! -d "${TRANSFORMERS_DIR}/.git" ]; then
    git clone https://github.com/huggingface/transformers.git "${TRANSFORMERS_DIR}"
fi
cd "${TRANSFORMERS_DIR}"
git checkout "${TRANSFORMERS_COMMIT}"

echo "==> Applying DropKV patches"
if git diff --quiet; then
    git apply "${SCRIPT_DIR}/patches/dropkv_transformers.patch"
else
    echo "    (patch already applied or working tree dirty, skipping git apply)"
fi

echo "==> Installing patched transformers (editable)"
pip install -e .
cd "${SCRIPT_DIR}"

echo "==> Installing DropKV and evaluation dependencies"
pip install -e "${SCRIPT_DIR}[triton,test]"
pip install lm-eval accelerate packaging wonderwords nltk

echo ""
echo "Done! Activate with:  conda activate ${ENV_NAME}"
echo "Run evaluation with:  bash run_eval.sh --model <model_path>"
echo "Run Triton tests with: python -m pytest tests/test_dropkv_triton.py -v"
