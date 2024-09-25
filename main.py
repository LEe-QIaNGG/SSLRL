import ray
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
import gym

# ����Ray
ray.init()

# ע�ỷ��
def atari_env_creator(env_config):
    env = gym.make('PongNoFrameskip-v4')  # ѡ������Ҫ��Atari��Ϸ
    env = wrap_deepmind(env, frame_stack=True)  # ʹ��DeepMind��װ������Ԥ����
    return env

tune.register_env("atari_env", atari_env_creator)

# ����DQN�㷨
config = {
    "env": "atari_env",
    "framework": "torch",  # ���� "torch" ��ʹ�� PyTorch
    "num_gpus": 0,  # �����GPU��������Ϊ1
    "num_workers": 3,  # ���л��������ߵ�������Խ��Խ�ܼ���ѵ��
    "train_batch_size": 32,
    "exploration_config": {
        "epsilon_timesteps": 1000000,  # ̽����ʱ�䲽��
        "final_epsilon": 0.01,  # ���յ�epsilonֵ
    },
    "dueling": True,  # ʹ��Dueling DQN
    "double_q": True,  # ʹ��Double Q-Learning
    "prioritized_replay": True,  # ʹ�����Ⱦ���ط�
    "lr": 1e-4,  # ѧϰ��
    "gamma": 0.99,  # �ۿ�����
    "buffer_size": 50000,  # ����طŻ�������С
    "learning_starts": 10000,  # ���ٲ�֮��ʼѧϰ
    "target_network_update_freq": 500,  # Ŀ������ĸ���Ƶ��
}

# ��ʼѵ��
tune.run(DQNTrainer, config=config)
