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
tensorboard --logdir='logbeta0.7' --port 6006
tensorboard --logdir=log/Hero-ram-v4/framework_test/ --port 6006
tensorboard --logdir="log/Hero-ram-v4/DA_test/" --port 6006
tensorboard --logdir="log/Seaquest-ram-v4/framework_test/" --port 6006
conda activate tianshou_env
gpu-interactive
python main.py --task FetchReach-v3 --is_L2 1 --data_augmentation cutout --is_store False --test_type framework_test --logdir log_test
python robotics_test.py --task FetchReach-v3 --replay-buffer normal --buffer-size 50000 --epoch 2000 --step-per-epoch 800 --logdir log/beta0.7
python baseline.py --task FetchReach-v3 --buffer-type normal  --logdir log

pip install pytorch-lightning -i https://pypi.tuna.tsinghua.edu.cn/simple

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/yangyangjun/.mujoco/mujoco210/bin

python ./RCP/trainer.py --implementation RCP --gamename MontezumaRevenge-ram-v4 --exp_name debug --num_workers 1 --seed 25


!