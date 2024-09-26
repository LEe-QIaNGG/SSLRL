./main.py --debug
tmux kill-session -t
tensorboard --logdir=~/pytorch-ddpg/ray_results --port 6006
echo $RAY_RESULTS_DIR
export RAY_RESULTS_DIR="~/pytorch-ddpg/ray_results"