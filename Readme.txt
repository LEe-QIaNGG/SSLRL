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
python main.py --task Hero-ram-v4 --is_L2 1 --data_augmentation shannon --is_store 0 --test_type framework_test --logdir log
python ICM.py --task Venture-ram-v4 

Venture应该降低更新频率
score curve更新
framework  Hero MontezumaRevenge
DA   MontezumaRevenge