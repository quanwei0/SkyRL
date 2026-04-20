## Setup Retriever

### Prepare Datasets 
```bash
local_dir=~/data/searchR1
uv run --isolated examples/train/search/searchr1_dataset.py --local_dir $local_dir
```
### Retriever Environments 
```bash
# Create and activate the retriever environment with Python 3.10
conda create -n retriever python=3.10 -y
conda activate retriever

# Install PyTorch (with GPU support) and related libraries
conda install numpy==1.26.4 # needed to stop incompatible version of numpy from being installed via pip
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install other Python packages
pip install transformers datasets pyserini huggingface_hub

# Install the GPU version of faiss
conda install faiss-gpu==1.8.0 -c pytorch -c nvidia -y

# Install the API service framework
pip install uvicorn fastapi
```

### Download Index
```bash
conda activate retriever

local_dir=~/data/searchR1
python examples/train/search/searchr1_download.py --local_dir $local_dir
cat $local_dir/part_* > $local_dir/e5_Flat.index
gzip -d $local_dir/wiki-18.jsonl.gz
```

## Setup Trainer

```bash
cd SkyRL
uv venv --python 3.12
source .venv/bin/activate
uv sync --active --extra fsdp
```


## Run Trainer

```bash
bash examples/train/search/run_search_multi_reward_rwppo_7b.sh  2>&1 | tee experiment_search.log
```