./main.py --debug
echo $RAY_RESULTS_DIR
export RAY_RESULTS_DIR="~/pytorch-ddpg/ray_results"
conda env create -f tianshou_env.yaml

tensorboard --logdir=log/MontezumaRevenge-ram-v4/framework_test/ --port 6006
tensorboard --logdir=log/Seaquest-ram-v4/DA_test/ --port 6006
tensorboard --logdir=log/Venture-ram-v4/DA_test --port 6006
tensorboard --logdir=log/MontezumaRevenge-ram-v4/DA_test/ --port 6006
tensorboard --logdir=log/Hero-ram-v4/framework_test/ --port 6006
tensorboard --logdir="log/Hero-ram-v4/DA_test/" --port 6006
tensorboard --logdir="log/Seaquest-ram-v4/framework_test/" --port 6006

tensorboard --logdir='log/FetchReach-v3/normal250101-230352' --port 6006

conda activate tianshou_env
tmux kill-session -t
tmux attach-session -t 

gpu-interactive
python main.py --task FetchReach-v3 --is_L2 1 --data_augmentation cutout --is_store False --test_type framework_test --logdir log_test
python robotics_test.py --task FetchReach-v3 --replay-buffer normal --buffer-size 100000 --epoch 500 --step-per-epoch 5000 --icm-lr-scale 0.2 --logdir log
python baseline.py --task FetchReach-v3 --buffer-type normal  --logdir log
python robotics_test.py --task FetchReach-v3 --watch --resume-path log/FetchReach-v3/normal250101-230352/policy.pth --render 0.02 --icm-lr-scale 0 --replay-buffer normal

pip install pytorch-lightning -i https://pypi.tuna.tsinghua.edu.cn/simple

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/yangyangjun/.mujoco/mujoco210/bin

python ./RCP/trainer.py --implementation RCP --gamename MontezumaRevenge-ram-v4 --exp_name debug --num_workers 1 --seed 25


