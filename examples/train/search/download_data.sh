#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="${LOCAL_DIR:-${HOME}/data/searchR1}"
ENV_NAME="${ENV_NAME:-retriever}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKYRL_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

mkdir -p "${LOCAL_DIR}"
cd "${SKYRL_DIR}"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    source "${HOME}/.local/bin/env" 2>/dev/null || export PATH="${HOME}/.local/bin:${PATH}"
fi

uv run --isolated examples/train/search/searchr1_dataset.py --local_dir "${LOCAL_DIR}"

# 2) Download index with the retriever conda env.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

python examples/train/search/searchr1_download.py --local_dir "${LOCAL_DIR}"
cat "${LOCAL_DIR}"/part_* > "${LOCAL_DIR}/e5_Flat.index"
gzip -df "${LOCAL_DIR}/wiki-18.jsonl.gz"

echo "Data ready in ${LOCAL_DIR}"
conda deactivate
