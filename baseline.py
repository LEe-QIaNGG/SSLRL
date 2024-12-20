# -*- coding: utf-8 -*-
import argparse
import datetime
import os
import pprint
import sys
import gymnasium as gym 

import numpy as np

import torch
from tianshou.utils.net.common import Net
from gymnasium.spaces import Box, Discrete, MultiBinary, MultiDiscrete

import tianshou as ts
from tianshou.data import Collector, CollectStats, VectorReplayBuffer,PrioritizedReplayBuffer,HERReplayBuffer
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.utils.logger.tensorboard import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter
from tianshou.policy import DQNPolicy
from tianshou.policy.base import BasePolicy
from tianshou.policy.modelbased.icm import ICMPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.discrete import IntrinsicCuriosityModule
from tianshou.utils.space_info import SpaceInfo


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Pitfall-ram-v4")
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
    return parser.parse_args()


def main(args: argparse.Namespace = get_args()) -> None:

    env = gym.make(args.task)
    train_envs = ts.env.DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.training_num)])
    test_envs = ts.env.DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.test_num)])

    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # should be N_FRAMES x H x W
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    assert isinstance(env.action_space, Discrete)
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # define model
    # net = DQN(*args.state_shape, action_shape=args.action_shape, device=args.device).to(args.device)
    space_info = SpaceInfo.from_env(env)
    state_shape = space_info.observation_info.obs_shape
    action_shape = space_info.action_info.action_shape
    net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[128, 128, 128],device=args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    # define policy
    policy= DQNPolicy(
        model=net,
        optim=optim,
        action_space=env.action_space,
        discount_factor=args.gamma,
        estimation_step=args.n_step,
        target_update_freq=args.target_update_freq,
    ).to(args.device)

    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    if args.buffer_type == "vector":
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


    # collector
    train_collector = Collector[CollectStats](policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector[CollectStats](policy, test_envs, exploration_noise=True)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "dqn_icm" if args.icm_lr_scale > 0 else "dqn"
    log_name = os.path.join(args.task, 'framework_test', 'baseline_per_'+ now)
    log_path = os.path.join(args.logdir, log_name)

    # logger
    logger = TensorboardLogger(SummaryWriter(log_path),train_interval=200000,test_interval=200000,update_interval=200000,save_interval=200000)

    def save_best_fn(policy: BasePolicy) -> None:
        pass

    def stop_fn(mean_rewards: float) -> bool:
        if env.spec.reward_threshold:
            return mean_rewards >= env.spec.reward_threshold
        if "Pong" in args.task:
            return mean_rewards >= 20
        return False

    def train_fn(epoch: int, env_step: int) -> None:
        # nature DQN setting, linear decay in the first 1M steps
        # env_step就是每个epoch的step,这里一个epoch执行200次，
        num_steps=args.step_per_epoch*args.epoch
        if env_step <= 1e6:
            eps = args.eps_train - env_step / 1e6 * (args.eps_train - args.eps_train_final)
        else:
            eps = args.eps_train_final
        policy.set_eps(eps)
        if args.reward_distribution and epoch%200==0:
        # 保存buffer中的reward值
            reward_distribution_path = os.path.join("log", "reward_distribution", args.task, "baseline_per_")
            os.makedirs(reward_distribution_path, exist_ok=True)
            
            # 获取buffer中的所有reward
            rewards = buffer.rew
            
            # 将reward保存到文件中
            np.save(os.path.join(reward_distribution_path, f"rewards_epoch_{epoch}.npy"), rewards)

    def compute_reward_fn(ag: np.ndarray, g: np.ndarray) -> np.ndarray:
        return env.compute_reward(ag, g, {})    

    def test_fn(epoch: int, env_step: int | None) -> None:
        policy.set_eps(args.eps_test)

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
        train_fn=train_fn,
        test_fn=test_fn,
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