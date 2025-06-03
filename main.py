import argparse
import datetime
import os
import pprint
import sys
from ale_py import ALEInterface
ale = ALEInterface()
import gymnasium as gym 
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.exploration import GaussianNoise
# import gymnasium_robotics

import numpy as np
import torch
from tianshou.utils.net.common import Net
from atari_network import DQN
from gymnasium.spaces import Box, Discrete, MultiBinary, MultiDiscrete
from robotics_test import make_fetch_env

import tianshou as ts
from tianshou.data import Collector, CollectStats    , VectorReplayBuffer
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.utils.logger.tensorboard import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter
from Policy import DQNPolicy,CusSACPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.space_info import SpaceInfo
from training_functions import Reward_Estimator
from tianshou.utils.space_info import ActionSpaceInfo


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Pong-ram-v4")
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
    parser.add_argument("--training-num", type=int, default=10)  
    parser.add_argument("--test-num", type=int, default=2) 
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
    parser.add_argument(
        "--is_L2",
        type=bool,
        default=False,
        help="weight for the forward model loss in ICM",
    )
    parser.add_argument(
        "--data_augmentation",
        type=str,
        default="cutout",
        help="cutout,shannon,smooth,scale,translate,flip",
    )
    parser.add_argument(
        "--is_store",
        type=bool,
        default=False,
        help="buffer,reward distribution",
    )
    parser.add_argument(
        "--test_type",
        type=str,
        default="framework_test",
        help="DA_test,framework_test",
    )
    parser.add_argument("--actor-lr", type=float, default=1e-5)
    parser.add_argument("--critic-lr", type=float, default=1e-5)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--auto-alpha", action="store_true", default=True)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=512)
    return parser.parse_args()


def main(args: argparse.Namespace = get_args()) -> None:
    if args.task=="FetchReach-v3":
        env, train_envs, test_envs = make_fetch_env(args.task, args.training_num, args.test_num)
        args.state_shape = {
        "observation": env.observation_space["observation"].shape,
        "achieved_goal": env.observation_space["achieved_goal"].shape,
        "desired_goal": env.observation_space["desired_goal"].shape,
    }
        action_info = ActionSpaceInfo.from_space(env.action_space)
        args.action_shape = action_info.action_shape
        args.max_action = action_info.max_action       
    else:
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
    space_info = SpaceInfo.from_env(env)
    state_shape = space_info.observation_info.obs_shape
    action_shape = space_info.action_info.action_shape
    net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[128, 128, 128],device=args.device)
    reward_estimator=Reward_Estimator(args)
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

    policy = CusSACPolicy(
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
        reward_estimator=reward_estimator,
        args=args,
    ).to(args.device)


    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    buffer = VectorReplayBuffer(
        args.buffer_size,
        buffer_num=args.training_num,
    )
    # collector
    train_collector = Collector[CollectStats](policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector[CollectStats](policy, test_envs, exploration_noise=True)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "dqn_icm" if args.icm_lr_scale > 0 else "dqn"
    log_name = os.path.join(args.task, args.test_type,str(args.seed),args.data_augmentation+' L2 '+str(args.is_L2)+now)
    log_path = os.path.join(args.logdir, log_name)


    logger = TensorboardLogger(SummaryWriter(log_path),train_interval=10000,test_interval=10000,update_interval=10000,save_interval=10000)

    def save_best_fn(policy: BasePolicy) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    # def stop_fn(mean_rewards: float) -> bool:
    #     pass

    # def train_fn(epoch: int, env_step: int) -> None:
    #     # nature DQN setting, linear decay in the first 1M steps
    #     num_steps = args.step_per_epoch*args.epoch
    #     if env_step <= num_steps:
    #         eps = args.eps_train - env_step / num_steps * (args.eps_train - args.eps_train_final)
    #     else:
    #         eps = args.eps_train_final
    #     policy.set_eps(eps)

        
    # def test_fn(epoch: int, env_step: int | None) -> None:
    #     policy.set_eps(args.eps_test)


    # watch agent's performance
    def watch() -> None:
        print("Setup test envs ...")
        # policy.set_eps(args.eps_test)
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
        # train_fn=train_fn,
        # test_fn=test_fn,
        # stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        verbose=False,
        update_per_step=args.update_per_step,
        test_in_train=False,
        resume_from_log=args.resume_id is not None,
        # save_checkpoint_fn=save_checkpoint_fn,
    ).run()

    pprint.pprint(result)
    watch()


if __name__ == "__main__":
    main(get_args())