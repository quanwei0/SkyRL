#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-retriever}"

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda create -n "${ENV_NAME}" python=3.10 -y
fi
conda activate "${ENV_NAME}"

conda install -y numpy==1.26.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

pip install transformers datasets pyserini huggingface_hub
conda install -y faiss-gpu==1.8.0 -c pytorch -c nvidia
pip install uvicorn fastapi

echo "Retriever env '${ENV_NAME}' is ready."
conda deactivate
