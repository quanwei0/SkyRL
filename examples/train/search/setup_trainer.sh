#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKYRL_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${SKYRL_DIR}"
uv venv --python 3.12
# shellcheck disable=SC1091
source .venv/bin/activate
uv sync --active --extra fsdp

echo "Trainer env ready at ${SKYRL_DIR}/.venv"
deactivate
