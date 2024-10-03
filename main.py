import argparse
import datetime
import os
import pprint
import sys
import gymnasium as gym 

import numpy as np
import envpool
import torch
from atari_network import DQN
from atari_wrapper import make_atari_env
from tianshou.utils.net.common import Net
from gymnasium.spaces import Box, Discrete, MultiBinary, MultiDiscrete

import tianshou as ts
from tianshou.data import Collector, CollectStats, VectorReplayBuffer
from tianshou.highlevel.logger import LoggerFactoryDefault
from Policy import DQNPolicy
from tianshou.policy.base import BasePolicy
from tianshou.policy.modelbased.icm import ICMPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.discrete import IntrinsicCuriosityModule
from tianshou.utils.space_info import SpaceInfo
from training_functions import Reward_Estimator


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="MontezumaRevenge-ram-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale-obs", type=int, default=0)
    parser.add_argument("--eps-test", type=float, default=0.005)
    parser.add_argument("--eps-train", type=float, default=1.0)
    parser.add_argument("--eps-train-final", type=float, default=0.05)
    parser.add_argument("--buffer-size", type=int, default=50000)  
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=500)
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--step-per-epoch", type=int, default=5000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)  
    parser.add_argument("--training-num", type=int, default=2)  
    parser.add_argument("--test-num", type=int, default=2) 
    parser.add_argument("--logdir", type=str, default="log")
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
    return parser.parse_args()


def main(args: argparse.Namespace = get_args()) -> None:
    # print(envpool.list_all_envs())
    # env, train_envs, test_envs = make_atari_env(
    #     args.task,
    #     args.seed,
    #     args.training_num,
    #     args.test_num,
    #     scale=args.scale_obs,
    #     frame_stack=args.frames_stack,
    # )

    env = gym.make('MontezumaRevenge-ram-v4')
    train_envs = ts.env.DummyVectorEnv([lambda: gym.make('MontezumaRevenge-ram-v4') for _ in range(args.training_num)])
    test_envs = ts.env.DummyVectorEnv([lambda: gym.make('MontezumaRevenge-ram-v4') for _ in range(args.test_num)])
    reward_estimator=Reward_Estimator()

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
    net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[128, 128, 128]).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    # define policy
    policy= DQNPolicy(
        model=net,
        optim=optim,
        action_space=env.action_space,
        discount_factor=args.gamma,
        estimation_step=args.n_step,#  n step  DQN
        target_update_freq=args.target_update_freq,
        reward_estimator=reward_estimator,
    )

    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)
    # replay buffer: `save_last_obs` and `stack_num` can be removed together
    # when you have enough RAM
    buffer = VectorReplayBuffer(
        args.buffer_size,
        buffer_num=args.training_num,
        #如下参数导致采样的记录不是obs shape形状�??
        ignore_obs_next=True,
        # save_only_last_obs=True,
        # stack_num=args.frames_stack,
    )
    # collector
    train_collector = Collector[CollectStats](policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector[CollectStats](policy, test_envs, exploration_noise=True)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "dqn_icm" if args.icm_lr_scale > 0 else "dqn"
    log_name = os.path.join(args.task, args.algo_name, str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)

    # logger
    logger_factory = LoggerFactoryDefault()
    if args.logger == "wandb":
        logger_factory.logger_type = "wandb"
        logger_factory.wandb_project = args.wandb_project
    else:
        logger_factory.logger_type = "tensorboard"

    logger = logger_factory.create_logger(
        log_dir=log_path,
        experiment_name=log_name,
        run_id=args.resume_id,
        config_dict=vars(args),
    )

    def save_best_fn(policy: BasePolicy) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards: float) -> bool:
        if env.spec.reward_threshold:
            return mean_rewards >= env.spec.reward_threshold
        if "Pong" in args.task:
            return mean_rewards >= 20
        return False

    def train_fn(epoch: int, env_step: int) -> None:
        # nature DQN setting, linear decay in the first 1M steps
        if env_step <= 1e6:
            eps = args.eps_train - env_step / 1e6 * (args.eps_train - args.eps_train_final)
        else:
            eps = args.eps_train_final
        policy.set_eps(eps)
        if env_step % 1000 == 0:
            logger.write("train/env_step", env_step, {"train/eps": eps})
        
    def test_fn(epoch: int, env_step: int | None) -> None:
        policy.set_eps(args.eps_test)

    def save_checkpoint_fn(epoch: int, env_step: int, gradient_step: int) -> str:
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
        torch.save({"model": policy.state_dict()}, ckpt_path)
        return ckpt_path

    # watch agent's performance
    def watch() -> None:
        print("Setup test envs ...")
        policy.set_eps(args.eps_test)
        test_envs.seed(args.seed)
        if args.save_buffer_name:
            print(f"Generate buffer with size {args.buffer_size}")
            buffer = VectorReplayBuffer(
                args.buffer_size,
                buffer_num=len(test_envs),
                ignore_obs_next=True,
                save_only_last_obs=True,
                stack_num=args.frames_stack,
            )
            collector = Collector[CollectStats](policy, test_envs, buffer, exploration_noise=True)
            result = collector.collect(n_step=args.buffer_size)
            print(f"Save buffer into {args.save_buffer_name}")
            # Unfortunately, pickle will cause oom with 1M buffer size
            buffer.save_hdf5(args.save_buffer_name)
        else:
            print("Testing agent ...")
            test_collector.reset()
            result = test_collector.collect(n_episode=args.test_num, render=args.render)
        result.pprint_asdict()

    if args.watch:
        watch()
        sys.exit(0)


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
        update_per_step=args.update_per_step,
        test_in_train=False,
        resume_from_log=args.resume_id is not None,
        save_checkpoint_fn=save_checkpoint_fn,
    ).run()

    pprint.pprint(result)
    watch()


if __name__ == "__main__":
    main(get_args())