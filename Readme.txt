./main.py --debug
tmux kill-session -t
tmux attach-session -t 
tensorboard --logdir=~/pytorch-ddpg/ray_results --port 6006
tensorboard --logdir=log/Seaquest-ram-v4/dqn/best_reward/ --port 6006
tensorboard --logdir=log/Seaquest-ram-v4/dqn/baseline/ --port 6006
tensorboard --logdir=log/Seaquest-ram-v4/dqn/ --port 6006
echo $RAY_RESULTS_DIR
export RAY_RESULTS_DIR="~/pytorch-ddpg/ray_results"
conda env create -f tianshou_env.yaml
conda activate tianshou_env