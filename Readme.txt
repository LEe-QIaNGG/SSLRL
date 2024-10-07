./main.py --debug
tmux kill-session -t
tensorboard --logdir=~/pytorch-ddpg/ray_results --port 6006
tensorboard --logdir=~/pytorch-ddpg/log/MontezumaRevenge-ram-v4/dqn/0 --port 6006
tensorboard --logdir=~/pytorch-ddpg/log/ --port 6006
echo $RAY_RESULTS_DIR
export RAY_RESULTS_DIR="~/pytorch-ddpg/ray_results"
conda env create -f tianshou_env.yaml