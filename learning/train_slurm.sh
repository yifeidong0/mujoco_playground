#!/usr/bin/env bash
#SBATCH -A NAISS2025-5-43
#SBATCH -p alvis
#SBATCH --gpus-per-node=A100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=168:00:00
#SBATCH --job-name=leapcube-jax-ppo
#SBATCH --output=/mimer/NOBACKUP/groups/softenable-design/yifeidong/mujoco_playground/logs/%A_slurm.out
#SBATCH --error=/mimer/NOBACKUP/groups/softenable-design/yifeidong/mujoco_playground/logs/%A_slurm.err

echo "=== SLURM job started on $(hostname) at $(date) ==="
nvidia-smi

# ------------------------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------------------------
module purge
source /mimer/NOBACKUP/groups/softenable-design/yifeidong/mujoco_playground/.venv/bin/activate

# allow pip inside conda
# unset PIP_REQUIRE_VIRTUALENV
# export PIP_REQUIRE_VIRTUALENV=0

# ensure wandb works
export WANDB_MODE=online
export WANDB_ENTITY=yifeidong
export WANDB_PROJECT=leapcube_jax_ppo

# ------------------------------------------------------------------------------
# Move to repo
# ------------------------------------------------------------------------------
REPO_DIR="/mimer/NOBACKUP/groups/softenable-design/yifeidong/mujoco_playground"
cd "${REPO_DIR}"

PYTHON_BIN="$(which python)"
echo "Using python: ${PYTHON_BIN}"

# ------------------------------------------------------------------------------
# Run training
# ------------------------------------------------------------------------------

${PYTHON_BIN} learning/train_jax_ppo.py \
  --env_name=LeapCubeReorient \
  --impl=jax \
  --num_timesteps=100000000 \
  --num_evals=20 \
  --num_minibatches=32 \
  --unroll_length=40 \
  --num_updates_per_batch=4 \
  --discounting=0.99 \
  --learning_rate=3e-4 \
  --entropy_cost=1e-2 \
  --num_envs=8192 \
  --num_eval_envs=128 \
  --batch_size=256 \
  --policy_hidden_layer_sizes=512,256,128 \
  --value_hidden_layer_sizes=512,256,128 \
  --policy_obs_key=state \
  --value_obs_key=privileged_state \
  --episode_length=1000 \
  --action_repeat=1 \
  --reward_scaling=0.1 \
  --normalize_observations=True \
  --use_wandb=True \
  --rscope_envs 16 \
  --deterministic_rscope=True

STATUS=$?
echo "=== Job ${SLURM_JOB_ID} finished at $(date) with status ${STATUS} ==="
exit ${STATUS}
