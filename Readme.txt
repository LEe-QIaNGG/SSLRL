tensorboard --logdir="log/Hero-ram-v4/DA_test/" --port 6006
tensorboard --logdir="log/Seaquest-ram-v4/framework_test/" --port 6006

tensorboard --logdir='log/Seaquest-ram-v4/framework_test/42' --port 6006

conda activate tianshou_env
tmux kill-session -t
tmux attach-session -t 
pip install gymnasium[atari]

gpu-interactive
python main.py --task Seaquest-ram-v4 --seed 0 --is_L2 1 --data_augmentation cutout --is_store False  --logdir log;python main.py --task Seaquest-ram-v4 --seed 1 --is_L2 1 --data_augmentation cutout --is_store False  --logdir log;
python robotics_test.py --task FetchReach-v3 --replay-buffer normal --buffer-size 100000 --epoch 500 --step-per-epoch 5000 --icm-lr-scale 0 --logdir log
FetchPickAndPlace-v3   FetchReach-v3
python baseline.py --task Seaquest-ram-v4 --seed 0 --buffer-type normal  --logdir log

python robotics_test.py --task FetchReach-v3 --watch --resume-path log/FetchReach-v3/normal250101-230352/policy.pth --render 0.02 --icm-lr-scale 0 --replay-buffer normal

pip install pytorch-lightning -i https://pypi.tuna.tsinghua.edu.cn/simple

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/yangyangjun/.mujoco/mujoco210/bin

python ./RCP/trainer.py --implementation RCP --gamename MontezumaRevenge-ram-v4 --exp_name debug --num_workers 1 --seed 25
./main.py --debug
echo $RAY_RESULTS_DIR
export RAY_RESULTS_DIR="~/pytorch-ddpg/ray_results"
conda env create -f tianshou_env.yaml


