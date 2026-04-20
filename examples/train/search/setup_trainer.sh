#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKYRL_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    source "${HOME}/.local/bin/env" 2>/dev/null || export PATH="${HOME}/.local/bin:${PATH}"
fi

cd "${SKYRL_DIR}"
uv venv --python 3.12
# shellcheck disable=SC1091
source .venv/bin/activate
uv sync --active --extra fsdp

echo "Trainer env ready at ${SKYRL_DIR}/.venv"
deactivate
