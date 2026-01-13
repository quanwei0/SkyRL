# export CUDA_VISIBLE_DEVICES=0,1,2,3

# source /mnt/home/siliang/miniconda3/bin/activate

# conda init
conda activate retriever

# redirect the output to a file to avoid cluttering the terminal
# we have observed outputting to the terminal causing spikes in server response times
bash examples/search/retriever/retrieval_launch.sh > retrieval_server.log 2>&1 &