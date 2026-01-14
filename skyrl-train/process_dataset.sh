source .venv/bin/activate

local_dir=~/data/searchR1
uv run --isolated examples/search/searchr1_dataset.py --local_dir $local_dir
