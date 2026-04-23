set -x

# Colocated RWPPO training+generation with multi-reward (answer + format + retrieval)
# for Qwen2.5-7B on SearchR1 data.
# Follow the instructions in docs/content/docs/recipes/searchr1.mdx for setup.
#
# Usage:
#   export WANDB_API_KEY=<your_key_here>
#   bash examples/train/search/run_search_multi_reward_rwppo_7b.sh
#
# Configurable knobs (override via env vars or command-line args):
#   USE_CONVERSATION_MULTI_TURN - set to "true" to use conversation multi-turn format (default: false)
#   STEP_WISE - set to "true" to enable step-wise training (default: false)
#     Requires USE_CONVERSATION_MULTI_TURN=true.
#   SAVE - set to "false" to disable checkpoint saving (default: true)

# export CUDA_VISIBLE_DEVICES=0,1,2,3
conda run -n retriever bash examples/train/search/retriever/retrieval_launch.sh > retrieval_server.log 2>&1 &

export WANDB_API_KEY="810f91e58aa0fd1d03b11c60b0d1cffbb1d941f4"
export WANDB_ENTITY="rl_agent"

DATA_DIR="$HOME/data/searchR1"

PROJECT_NAME="skyrl-search-rwpo"
RUN_NAME="skyrl-search-7b-multi-reward-rwppo"
BASE_DIR=$HOME/experiments/$PROJECT_NAME/$RUN_NAME

TIS_TYPE=token
TIS_IMP_RATIO_CAP=2.0

: "${USE_CONVERSATION_MULTI_TURN:=false}"
: "${STEP_WISE:=false}"
: "${SAVE:=true}"

MULTI_TURN_ARGS=""
if [ "$USE_CONVERSATION_MULTI_TURN" = "true" ]; then
  MULTI_TURN_ARGS="generator.use_conversation_multi_turn=true generator.append_eos_token_after_stop_str_in_multi_turn=true"
else
  MULTI_TURN_ARGS="generator.use_conversation_multi_turn=false"
fi

STEP_WISE_ARGS=""
if [ "$STEP_WISE" = "true" ]; then
  STEP_WISE_ARGS="generator.step_wise_trajectories=true"
  if [ "$USE_CONVERSATION_MULTI_TURN" != "true" ]; then
    echo "WARNING: STEP_WISE=true requires USE_CONVERSATION_MULTI_TURN=true. Enabling it automatically."
    MULTI_TURN_ARGS="generator.use_conversation_multi_turn=true generator.append_eos_token_after_stop_str_in_multi_turn=true"
  fi
fi

SAVE_ARGS=""
if [ "$SAVE" = "true" ]; then
  SAVE_ARGS="trainer.ckpt_interval=20 trainer.hf_save_interval=100 trainer.max_ckpts_to_keep=-1 trainer.resume_mode=latest trainer.ckpt_path=$BASE_DIR"
else
  SAVE_ARGS="trainer.ckpt_interval=-1 trainer.hf_save_interval=-1 trainer.resume_mode=null"
fi

NUM_GPUS=8
MODEL_NAME="Qwen/Qwen2.5-7B"

uv run --isolated --frozen --extra fsdp -m skyrl.train.entrypoints.main_base \
  data.train_data="['${DATA_DIR}/train.parquet']" \
  data.val_data="['${DATA_DIR}/validation.parquet']" \
  trainer.algorithm.advantage_estimator="rwppo" \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.off_policy_correction.tis_ratio_type=$TIS_TYPE \
  trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high=$TIS_IMP_RATIO_CAP \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.policy.optimizer_config.num_warmup_steps=94 \
  trainer.policy.model.path=${MODEL_NAME} \
  trainer.critic.model.path=${MODEL_NAME} \
  trainer.critic.optimizer_config.lr=1.0e-5 \
  trainer.critic.model_config_kwargs.n_value_heads=3 \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.policy.fsdp_config.cpu_offload=true \
  trainer.ref.fsdp_config.cpu_offload=true \
  trainer.critic.fsdp_config.cpu_offload=true \
  trainer.placement.policy_num_gpus_per_node=${NUM_GPUS} \
  trainer.placement.ref_num_gpus_per_node=${NUM_GPUS} \
  trainer.placement.critic_num_gpus_per_node=${NUM_GPUS} \
  generator.inference_engine.num_engines=${NUM_GPUS} \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.5 \
  trainer.epochs=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=512 \
  trainer.policy_mini_batch_size=256 \
  trainer.critic_mini_batch_size=256 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.max_prompt_length=2048 \
  generator.max_input_length=4096 \
  generator.sampling_params.max_generate_length=500 \
  generator.inference_engine.async_engine=true \
  generator.batched=false \
  $MULTI_TURN_ARGS \
  $STEP_WISE_ARGS \
  generator.n_samples_per_prompt=1 \
  generator.max_turns=4 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.stop='["</search>", "</answer>"]' \
  environment.env_class="search" \
  environment.skyrl_gym.max_env_workers=16 \
  environment.skyrl_gym.search.multi_reward=true \
  environment.skyrl_gym.search.log_requests=false \
  environment.skyrl_gym.search.search_url="http://127.0.0.1:8000/retrieve" \
  environment.skyrl_gym.search.topk=3 \
  trainer.logger="wandb" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.run_name="${RUN_NAME}" \
  $SAVE_ARGS \
  trainer.eval_batch_size=256 \
  trainer.eval_before_train=false \
  generator.eval_sampling_params.temperature=0 \
  generator.eval_sampling_params.stop='["</search>", "</answer>"]' \
  generator.eval_sampling_params.max_generate_length=500 \
  trainer.export_path="$BASE_DIR/exports" \
  trainer.eval_interval=50 \
  $@
