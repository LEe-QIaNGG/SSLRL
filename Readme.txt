./main.py --debug
echo $RAY_RESULTS_DIR
export RAY_RESULTS_DIR="~/pytorch-ddpg/ray_results"
conda env create -f tianshou_env.yaml
tmux kill-session -t
tmux attach-session -t 
tensorboard --logdir=log/MontezumaRevenge-ram-v4/framework_test/ --port 6006
tensorboard --logdir=log/Seaquest-ram-v4/DA_test/ --port 6006
tensorboard --logdir=log/Venture-ram-v4/DA_test --port 6006
tensorboard --logdir=log/MontezumaRevenge-ram-v4/DA_test/ --port 6006
tensorboard --logdir='log/Venture-ram-v4/framework_test' --port 6006
tensorboard --logdir=log/Hero-ram-v4/framework_test/ --port 6006
tensorboard --logdir="log/Hero-ram-v4/DA_test/" --port 6006
tensorboard --logdir="log/Seaquest-ram-v4/framework_test/" --port 6006
conda activate tianshou_env
gpu-interactive
python main.py --task Seaquest-ram-v4 --is_L2 1 --data_augmentation cutout --is_store 1 --test_type framework_test --logdir log
python baseline.py --task Venture-ram-v4 --logdir log


是不是epsilon有问题
Venture应该降低更新频率/venture shannon降低了效果更差

score curve更新
framework  Hero MontezumaRevenge
DA   MontezumaRevenge