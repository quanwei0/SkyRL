curl -LsSf https://astral.sh/uv/install.sh | sh

# creates a venv at .venv/
uv venv --python 3.12
source .venv/bin/activate
uv sync --active --extra fsdp

deactivate