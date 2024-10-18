./main.py --debug
echo $RAY_RESULTS_DIR
export RAY_RESULTS_DIR="~/pytorch-ddpg/ray_results"
tmux kill-session -t
tmux attach-session -t 
tensorboard --logdir=log/Seaquest-ram-v4/dqn_icm/ --port 6006
tensorboard --logdir=log/Seaquest-ram-v4/dqn/best_reward/ --port 6006
tensorboard --logdir=log/Seaquest-ram-v4/dqn/baseline/ --port 6006
tensorboard --logdir=log/Seaquest-ram-v4/dqn/ --port 6006
tensorboard --logdir='log/Seaquest-ram-v4/dqn/sslrl/smooth L2 False241016-135521/' --port 6006
conda env create -f tianshou_env.yaml
conda activate tianshou_env