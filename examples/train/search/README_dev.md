## Setup Retriever

```bash
bash examples/train/search/setup_retriever.sh
```

## Download Data & Index

```bash
bash examples/train/search/download_data.sh
```

## Setup Trainer

```bash
bash examples/train/search/setup_trainer.sh
```

## Run Trainer

```bash
bash examples/train/search/run_search_multi_reward_rwppo_7b.sh 2>&1 | tee experiment_search.log
```
