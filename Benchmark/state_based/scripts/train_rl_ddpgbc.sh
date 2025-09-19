#!/usr/bin/env bash
set -e

cd Benchmark/state_based/rl

export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1

# Replace these with the actual .npz paths printed by 02_generate_demos.sh
NEEDLEPICK_DEMO="/home/ubuntu/ysh/Code/SurRoL/Benchmark/state_based/surrol/data/demo/data_NeedlePickRL-v0_random_200.npz"
PEGTRANSFER_DEMO="/home/ubuntu/ysh/Code/SurRoL/Benchmark/state_based/surrol/data/demo/data_PegTransferRL-v0_random_200.npz"

workers=8

# Train RL from scratch (DDPG is a stable starting point)
mpirun -np ${workers} python train_rl.py task=NeedlePickRL-v0 agent=ddpg

# Train RL with demonstrations (DDPGBC is a stable starting point)
# mpirun -np ${workers} python train_rl.py task=NeedlePickRL-v0 agent=ddpgbc demo_path="${NEEDLEPICK_DEMO}"

# for resuming training from the latest checkpoint, use:
# mpirun -np ${workers} python train_rl.py task=NeedlePickRL-v0 agent=ddpgbc demo_path="${NEEDLEPICK_DEMO}" n_train_steps=$((50*2000)) n_eval=$((8*workers)) n_save=$((4*workers)) n_log=$((20*workers)) clean_model_dir=False ckpt_episode=latest

# python evaluate_rl.py task=NeedlePickRL-v0 agent=ddpgbc num_demo=200 ckpt_episode=latest n_eval_episodes=100

# Then try PegTransfer
# python train_rl.py task=PegTransferRL-v0 agent=ddpgbc demo_path="${PEGTRANSFER_DEMO}"

# echo ""
# echo "[TIP] If learning is unstable, try agent=awac:"
# echo "python train_rl.py task=PegTransferRL-v0 agent=awac demo_path=\"${PEGTRANSFER_DEMO}\""