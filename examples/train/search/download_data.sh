#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="${LOCAL_DIR:-${HOME}/data/searchR1}"
ENV_NAME="${ENV_NAME:-retriever}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKYRL_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

mkdir -p "${LOCAL_DIR}"
cd "${SKYRL_DIR}"

# 1) Prepare dataset (uses project uv env, isolated).
uv run --isolated examples/train/search/searchr1_dataset.py --local_dir "${LOCAL_DIR}"

# 2) Download index with the retriever conda env.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

python examples/train/search/searchr1_download.py --local_dir "${LOCAL_DIR}"
cat "${LOCAL_DIR}"/part_* > "${LOCAL_DIR}/e5_Flat.index"
gzip -df "${LOCAL_DIR}/wiki-18.jsonl.gz"

echo "Data ready in ${LOCAL_DIR}"
conda deactivate
