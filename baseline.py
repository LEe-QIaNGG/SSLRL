# -*- coding: utf-8 -*-
import argparse
import datetime
import os
import pprint
import sys
from ale_py import ALEInterface
ale = ALEInterface()
import gymnasium as gym 
# import gymnasium_robotics
from gymnasium import spaces

import numpy as np

import torch
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.policy import DiscreteSACPolicy
from atari_network import DQN
from gymnasium.spaces import Box, Discrete, MultiBinary, MultiDiscrete

import tianshou as ts
from tianshou.env import ShmemVectorEnv, TruncatedAsTerminated
from tianshou.data import Collector, CollectStats, VectorReplayBuffer,PrioritizedReplayBuffer,HERReplayBuffer
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.env import SubprocVectorEnv
from tianshou.utils.logger.tensorboard import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter
from tianshou.policy import DQNPolicy
from tianshou.policy.base import BasePolicy
from tianshou.policy.modelbased.icm import ICMPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.discrete import IntrinsicCuriosityModule
from tianshou.env.venvs import BaseVectorEnv
from tianshou.utils.space_info import ActionSpaceInfo
from tianshou.utils.space_info import SpaceInfo


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="FetchReach-v3")
    parser.add_argument("--buffer-type",type=str,default='per')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale-obs", type=int, default=0)
    parser.add_argument("--eps-test", type=float, default=0.005)
    parser.add_argument("--eps-train", type=float, default=1.0)
    parser.add_argument("--eps-train-final", type=float, default=0.05)
    parser.add_argument("--buffer-size", type=int, default=30000)  
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=400)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=2000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)  
    parser.add_argument("--training-num", type=int, default=1)  
    parser.add_argument("--test-num", type=int, default=1)  
    parser.add_argument("--logdir", type=str, default='log_test')
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--frames-stack", type=int, default=4)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="atari.benchmark")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    parser.add_argument("--save-buffer-name", type=str, default=None)
    parser.add_argument(
        "--icm-lr-scale",
        type=float,
        default=0.0,
        help="use intrinsic curiosity module with this lr scale",
    )
    parser.add_argument(
        "--icm-reward-scale",
        type=float,
        default=0.01,
        help="scaling factor for intrinsic curiosity reward",
    )
    parser.add_argument(
        "--icm-forward-loss-weight",
        type=float,
        default=0.2,
        help="weight for the forward model loss in ICM",
    )
    parser.add_argument("--reward-distribution", type=bool, default=False)
    parser.add_argument("--actor-lr", type=float, default=1e-5)
    parser.add_argument("--critic-lr", type=float, default=1e-5)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--auto-alpha", action="store_true", default=True)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=512)
    return parser.parse_args()


# 自定义 Wrapper，用于展平 observation
class FlattenObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_space = env.observation_space
        # 提取原始 observation_space 的低维和高维信息
        flat_dim = np.prod(obs_space["observation"].shape) + \
                   np.prod(obs_space["achieved_goal"].shape) + \
                   np.prod(obs_space["desired_goal"].shape)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32
        )

    def observation(self, observation):
        # 将字典形式的 observation 展平为数组
        return np.concatenate([
            observation["observation"],
            observation["achieved_goal"],
            observation["desired_goal"]
        ])

# 创建离散化后的 FetchReach 环境，并展平 observation
class DiscretizedFetchReachEnv(gym.Wrapper):
    def __init__(self, env, n=3):
        super().__init__(FlattenObservation(env))
        self.n = n
        self.action_space = gym.spaces.Discrete(n**3)
        self._discrete_actions = self._create_discrete_actions(n)

    def _create_discrete_actions(self, n):
        actions = []
        for dx in np.linspace(-1, 1, n):
            for dy in np.linspace(-1, 1, n):
                for dz in np.linspace(-1, 1, n):
                    # 添加抓取动作（grip）为0
                    actions.append([dx, dy, dz, 0.0])  # 添加第四个维度
        return np.array(actions)

    def step(self, action):
        continuous_action = self._discrete_actions[action]
        obs, reward, done, truncated, info = self.env.step(continuous_action)
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

# 创建离散化后的 FetchReach 环境
def make_fetchreach_env(n_envs=1, n_discretize=3):
    def _make_env():
        return DiscretizedFetchReachEnv(gym.make("FetchReach-v3"), n=n_discretize)
    return SubprocVectorEnv([_make_env for _ in range(n_envs)])

def make_fetch_env(
    task: str,
    training_num: int,
    test_num: int,
) -> tuple[gym.Env, BaseVectorEnv, BaseVectorEnv]:
    env = TruncatedAsTerminated(gym.make(task))
    train_envs = ShmemVectorEnv(
        [lambda: TruncatedAsTerminated(gym.make(task)) for _ in range(training_num)],
    )
    test_envs = ShmemVectorEnv(
        [lambda: TruncatedAsTerminated(gym.make(task)) for _ in range(test_num)],
    )
    return env, train_envs, test_envs

def main(args: argparse.Namespace = get_args()) -> None:
    env = gym.make(args.task)
    def compute_reward_fn(ag: np.ndarray, g: np.ndarray) -> np.ndarray:
        return env.compute_reward(ag, g, {})
    if args.task=="FetchReach-v3":
        train_envs = DiscretizedFetchReachEnv(gym.make("FetchReach-v3"), n=3)
        test_envs = DiscretizedFetchReachEnv(gym.make("FetchReach-v3"), n=3)
        state_shape = train_envs.observation_space.shape[0]  # 使用包装后的环境的观察空间维度
        action_shape = train_envs.action_space.n
    else:
        train_envs = ts.env.DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.training_num)])
        test_envs = ts.env.DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.test_num)])
        args.state_shape = env.observation_space.shape #or env.observation_space.n
        args.action_shape = env.action_space.shape or env.action_space.n
        space_info = SpaceInfo.from_env(env)
        state_shape = space_info.observation_info.obs_shape
        action_shape = space_info.action_info.action_shape
        print("Observations shape:", args.state_shape)
        print("Actions shape:", args.action_shape)
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
        # 离散动作空间
    net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[128, 128, 128], device=args.device)


    # define policy
    net = DQN(
        *args.state_shape,
        args.action_shape,
        device=args.device,
        features_only=True,
        output_dim_added_layer=args.hidden_size,
    )
    actor = Actor(net, args.action_shape, device=args.device, softmax_output=False)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1 = Critic(net, last_size=args.action_shape, device=args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = Critic(net, last_size=args.action_shape, device=args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    # define policy
    if args.auto_alpha:
        target_entropy = 0.98 * np.log(np.prod(args.action_shape))
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    policy = DiscreteSACPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic=critic1,
        critic_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        action_space=env.action_space,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        estimation_step=args.n_step,
    ).to(args.device)

    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    if args.buffer_type == "normal":
        buffer = VectorReplayBuffer(
            args.buffer_size,
            buffer_num=args.training_num,
            ignore_obs_next=True,
        )
    elif args.buffer_type == "per":
        buffer = PrioritizedReplayBuffer(
            size=args.buffer_size,
            alpha=0.6, beta=0.4,
            buffer_num=args.training_num,
            ignore_obs_next=True,
        )
    elif args.buffer_type == "her":
        buffer = HERReplayBuffer(
            size=args.buffer_size,
            buffer_num=args.training_num,
            compute_reward_fn=compute_reward_fn,
            horizon=50,
            future_k=8,
        )



    train_collector = Collector[CollectStats](policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector[CollectStats](policy, test_envs, exploration_noise=True)
    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "dqn_icm" if args.icm_lr_scale > 0 else "dqn"
    log_name = os.path.join(args.task, 'framework_test',str(args.seed), 'baseline_'+args.buffer_type+'_'+ now)
    log_path = os.path.join(args.logdir, log_name)

    # logger
    logger = TensorboardLogger(SummaryWriter(log_path),train_interval=10000,test_interval=10000,update_interval=10000,save_interval=10000)

    def save_best_fn(policy: BasePolicy) -> None:
        pass

    def stop_fn(mean_rewards: float) -> bool:
        if env.spec.reward_threshold:
            return mean_rewards >= env.spec.reward_threshold
        if "Pong" in args.task:
            return mean_rewards >= 20
        return False

    # def train_fn(epoch: int, env_step: int) -> None:
        # nature DQN setting, linear decay in the first 1M steps
        # env_step就是每个epoch的step,这里一个epoch执行200次，
        # num_steps=args.step_per_epoch*args.epoch
        # if env_step <= num_steps:
        #     eps = args.eps_train - env_step / num_steps * (args.eps_train - args.eps_train_final)
        # else:
        #     eps = args.eps_train_final
        # policy.set_eps(eps)
        # if args.reward_distribution and epoch%200==0:
        # # 保存buffer中的reward值
        #     reward_distribution_path = os.path.join("log", "reward_distribution", args.task, "baseline_per_")
        #     os.makedirs(reward_distribution_path, exist_ok=True)
            
        #     # 获取buffer中的所有reward
        #     rewards = buffer.rew
            
        #     # 将reward保存到文件中
        #     np.save(os.path.join(reward_distribution_path, f"rewards_epoch_{epoch}.npy"), rewards)


    # def test_fn(epoch: int, env_step: int | None) -> None:
    #     policy.set_eps(args.eps_test)

    def save_checkpoint_fn(epoch: int, env_step: int, gradient_step: int) -> str:
        pass


    # trainer
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        # train_fn=train_fn,
        # test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        verbose=False,
        update_per_step=args.update_per_step,
        test_in_train=False,
        resume_from_log=args.resume_id is not None,
        save_checkpoint_fn=save_checkpoint_fn,
    ).run()

    pprint.pprint(result)


if __name__ == "__main__":
    main(get_args())