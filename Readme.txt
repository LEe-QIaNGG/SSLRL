./main.py --debug
echo $RAY_RESULTS_DIR
export RAY_RESULTS_DIR="~/pytorch-ddpg/ray_results"
conda env create -f tianshou_env.yaml
tmux kill-session -t
tmux attach-session -t 
tensorboard --logdir=log/MontezumaRevenge-ram-v4/framework_test/ --port 6006
tensorboard --logdir=log/Pitfall-ram-v4/framework_test/ --port 6006
tensorboard --logdir=log/Venture-ram-v4/DA_test --port 6006
tensorboard --logdir=log/MontezumaRevenge-ram-v4/DA_test/ --port 6006
tensorboard --logdir='log/Seaquest-ram-v4/L2_test/' --port 6006
tensorboard --logdir=log/Hero-ram-v4/framework_test/ --port 6006
tensorboard --logdir="log/Hero-ram-v4/framework_test/cutout L2 True241101-140105/" --port 6006
tensorboard --logdir="log/Hero-ram-v4/DA_test/scale L2 False241108-142400" --port 6006
tensorboard --logdir="log/Hero-ram-v4/framework_test/" --port 6006
conda activate tianshou_env
gpu-interactive
python main.py --task MontezumaRevenge-ram-v4 --is_L2 0 --data_augmentation scale --is_store 0 --test_type DA_test --logdir log
python baseline.py --task Venture-ram-v4 