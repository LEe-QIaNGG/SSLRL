# -*- coding: utf-8 -*-

import ray
from ray import air, tune
from ray.rllib.algorithms.dqn import DQNConfig

# ��ʼ�� Ray
ray.init()

# DQN �㷨����
config = (
    DQNConfig()
    .environment("CartPole-v1")  # ���û���
    .framework("torch")          # ʹ�� PyTorch ��ܣ�����ʹ�� "tf" ��Ӧ TensorFlow
    .env_runners(num_env_runners=1,
                 exploration_config={         
            "epsilon_timesteps": 10000,  # epsilon �� 1.0 ˥���� 0.1 �Ĳ���
            "final_epsilon": 0.1,        # epsilon ����ֵ
        })  # ���� worker ������
    .training(
        gamma=0.99,                  # �ۿ�����
        lr=1e-3,                     # ѧϰ��
        train_batch_size=32,          # ������С
        replay_buffer_config={
            "capacity": 50000         # �طŻ�������С
        },
        target_network_update_freq=500,  # Ŀ������ĸ���Ƶ��
    )
    .resources(num_gpus=0)            # ʹ�� GPU ��������0 ��ʾֻʹ�� CPU
)

# ����ѵ��
tuner = tune.Tuner(
    "DQN",
    param_space=config.to_dict(),      # ������תΪ�ֵ���ʽ����
    run_config=air.RunConfig(stop={"episode_reward_mean": 200}, 
                             checkpoint_config=air.CheckpointConfig(
            checkpoint_at_end=True      # ʹ���µ� CheckpointConfig �������
        ))
)

results = tuner.fit()

# ��ȡ����ӡ��������
best_result = results.get_best_result()
print("Best config: ", best_result.config)

# �ر� Ray
ray.shutdown()
