from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Generic, Literal, Self, TypeVar, cast
from collections.abc import Sequence
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tianshou.data import Batch, to_torch
from tianshou.utils.net.common import MLP, BaseActor, Net, TActionShape, get_output_dim
from tianshou.utils.net.continuous import Actor, Critic
from tianshou.exploration import BaseNoise, GaussianNoise
import warnings


from tianshou.data import Batch, ReplayBuffer
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import (
    ActBatchProtocol,
    ActStateBatchProtocol,
    BatchWithReturnsProtocol,
    ModelOutputBatchProtocol,
    ObsBatchProtocol,
    RolloutBatchProtocol,
)
from tianshou.policy import BasePolicy,DQNPolicy,DDPGPolicy
from tianshou.policy.base import TLearningRateScheduler, TrainingStats

from training_functions import Reward_Estimator
from tianshou.policy.modelbased.icm import ICMPolicy

class IntrinsicCuriosityModule(nn.Module):
    """Implementation of Intrinsic Curiosity Module. arXiv:1705.05363.

    Args:
        feature_net (nn.Module): a feature encoder network.
        feature_dim (int): input dimension of forward and inverse networks.
        action_dim (int): action space dimension.
        hidden_sizes (list): hidden sizes for forward and inverse networks.
        device (str): device for tensor allocation.
        discrete_action (bool): whether the action space is discrete.
    """

    def __init__(
        self,
        feature_net: nn.Module,
        feature_dim: int,
        action_dim: int,
        hidden_sizes: list[int],
        device: str | int | torch.device = "cpu",
        discrete_action: bool = True,
    ) -> None:
        super().__init__()
        self.feature_net = feature_net
        self.action_dim = action_dim
        self.discrete_action = discrete_action
        self.device = device
        
        # 前向模型
        forward_input_dim = feature_dim + (action_dim if not discrete_action else action_dim)
        self.forward_model = MLP(
            forward_input_dim,
            output_dim=feature_dim,
            hidden_sizes=hidden_sizes,
            device=device
        )
        
        # 逆向模型
        self.inverse_model = MLP(
            feature_dim * 2,
            output_dim=action_dim,
            hidden_sizes=hidden_sizes,
            device=device
        )

    def forward(
        self,
        s1: torch.Tensor,
        act: torch.Tensor,
        s2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute forward loss and inverse loss."""
        # 获取状态特征
        phi1 = self.feature_net(s1)
        phi2 = self.feature_net(s2)
        
        # 处理动作输入
        if self.discrete_action:
            act_one_hot = F.one_hot(act.long(), num_classes=self.action_dim).to(phi1.device)
            forward_input = torch.cat([phi1, act_one_hot], dim=1)
        else:
            # 对于连续动作，确保将 act 转换为 Tensor，并移动到相同设备
            act_tensor = torch.tensor(act, dtype=torch.float32, device=phi1.device)
            forward_input = torch.cat([phi1, act_tensor], dim=1)

        # 前向预测
        phi2_hat = self.forward_model(forward_input)
        mse_loss = 0.5 * F.mse_loss(phi2_hat, phi2, reduction="none").sum(1)
        act_hat = self.inverse_model(torch.cat([phi1, phi2], dim=1))   

        # forward_loss = 0.5 * torch.mean(torch.sum((phi2_hat - phi2.detach()) ** 2, dim=1))
        
        # # 逆向预测
        # inverse_input = torch.cat([phi1, phi2], dim=1)
        # act_hat = self.inverse_model(inverse_input)
        
        # if self.discrete_action:
        #     inverse_loss = F.cross_entropy(act_hat, act.long())
        # else:
        #     if act.ndim() == 1:  # 如果 act 是 1D 向量
        #         act = act.unsqueeze(-1)  # 添加一个维度
        #     inverse_loss = F.mse_loss(act_hat, act)  # 确保形状一致
            
        return mse_loss, act_hat 

class CusDQNPolicy(DQNPolicy):

    def __init__(self, reward_estimator: Reward_Estimator, args,  **kwargs: Any) -> None:
        super().__init__( **kwargs)  # 调用父类的初始化方法
        self.reward_estimator = reward_estimator
        self.args = args

    def post_process_fn(
        self,
        batch: BatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> None:
        """Post-process the data from the provided replay buffer.

        Typical usage is to update the sampling weight in prioritized
        experience replay. Used in :meth:`update`.
        """
        # 一共 epoch*step_per_epoch*update_per_step 次 iter, 200000
        # alpha 从 0 到 0.5
        num_iter = self.args.epoch * self.args.step_per_epoch * self.args.update_per_step
        if self._iter < num_iter * 0.6:
            alpha = 1 - np.exp(-self._iter / num_iter)  
            self.reward_estimator.update(batch, buffer, alpha, self._iter)
        else:
            alpha = 0.5
            self.reward_estimator.update(batch, buffer, alpha, self._iter)
    

# 添加新的 CustomICMPolicy 类
class CustomICMPolicy(ICMPolicy):
    def process_fn(self, batch, buffer, indices):
            # 处理字典观察空间
        if not isinstance(batch.obs, np.ndarray):
                # batch.obs = batch.obs.observation
                batch.obs=np.concatenate((batch.obs.observation,batch.obs.achieved_goal,batch.obs.desired_goal),axis=1)
        else:
            print('ndarray')
        if not isinstance(batch.obs_next, np.ndarray):
                batch.obs_next=np.concatenate((batch.obs_next.observation,batch.obs_next.achieved_goal,batch.obs_next.desired_goal),axis=1)
        # batch.obs=batch.obs.flatten()
        # batch.obs_next=batch.obs_next.flatten()
        # 调用父类的 process_fn
        return super().process_fn(batch, buffer, indices)   

@dataclass(kw_only=True)
class DDPGTrainingStats(TrainingStats):
    actor_loss: float
    critic_loss: float


TDDPGTrainingStats = TypeVar("TDDPGTrainingStats", bound=DDPGTrainingStats)


# class CusDDPGPolicy(DDPGPolicy):
#     def __init__(
#         self,
#         reward_estimator: Reward_Estimator,
#         args,
#         **kwargs: Any,
#     ) -> None:
#         super().__init__(**kwargs)  # 调用父类的初始化方法
#         self.iter = 0
#         self.reward_estimator = reward_estimator
#         self.args = args

#     def post_process_fn(
#         self,
#         batch: BatchProtocol,
#         buffer: ReplayBuffer,
#         indices: np.ndarray,
#     ) -> None:
#         """Post-process the data from the provided replay buffer.

#         Typical usage is to update the sampling weight in prioritized
#         experience replay. Used in :meth:`update`.
#         """
#         # 一共 epoch*step_per_epoch*update_per_step 次 iter
#         num_iter = self.args.epoch * self.args.step_per_epoch * self.args.update_per_step
#         if self.iter % 10 == 0:
#             if self.iter < num_iter * 0.6:
#                 alpha = 1 - np.exp(-self.iter / num_iter)
#                 self.reward_estimator.update(batch, buffer, alpha, self.iter)
#             else:
#                 alpha = 0.5
#                 self.reward_estimator.update(batch, buffer, alpha, self.iter)
#         self.iter += 1

@dataclass(kw_only=True)
class DDPGTrainingStats(TrainingStats):
    actor_loss: float
    critic_loss: float
TDDPGTrainingStats = TypeVar("TDDPGTrainingStats", bound=DDPGTrainingStats)
class CusDDPGPolicy(BasePolicy[TDDPGTrainingStats], Generic[TDDPGTrainingStats]):
    """Implementation of Deep Deterministic Policy Gradient. arXiv:1509.02971.
    :param actor: The actor network following the rules (s -> actions)
    :param actor_optim: The optimizer for actor network.
    :param critic: The critic network. (s, a -> Q(s, a))
    :param critic_optim: The optimizer for critic network.
    :param action_space: Env's action space.
    :param tau: Param for soft update of the target network.
    :param gamma: Discount factor, in [0, 1].
    :param exploration_noise: The exploration noise, added to the action. Defaults
        to ``GaussianNoise(sigma=0.1)``.
    :param estimation_step: The number of steps to look ahead.
    :param observation_space: Env's observation space.
    :param action_scaling: if True, scale the action from [-1, 1] to the range
        of action_space. Only used if the action_space is continuous.
    :param action_bound_method: method to bound action to range [-1, 1].
        Only used if the action_space is continuous.
    :param lr_scheduler: if not None, will be called in `policy.update()`.
    .. seealso::
        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """
    def __init__(
        self,
        *,
        actor: torch.nn.Module | Actor,
        actor_optim: torch.optim.Optimizer,
        critic: torch.nn.Module | Critic,
        critic_optim: torch.optim.Optimizer,
        action_space: gym.Space,
        tau: float = 0.005,
        gamma: float = 0.99,
        exploration_noise: BaseNoise | Literal["default"] | None = "default",
        estimation_step: int = 1,
        observation_space: gym.Space | None = None,
        action_scaling: bool = True,
        # tanh not supported, see assert below
        action_bound_method: Literal["clip"] | None = "clip",
        lr_scheduler: TLearningRateScheduler | None = None,
        reward_estimator: Reward_Estimator,
        args,
    ) -> None:
        assert 0.0 <= tau <= 1.0, f"tau should be in [0, 1] but got: {tau}"
        assert 0.0 <= gamma <= 1.0, f"gamma should be in [0, 1] but got: {gamma}"
        assert action_bound_method != "tanh", (  # type: ignore[comparison-overlap]
            "tanh mapping is not supported"
            "in policies where action is used as input of critic , because"
            "raw action in range (-inf, inf) will cause instability in training"
        )
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            lr_scheduler=lr_scheduler,
        )
        if action_scaling and not np.isclose(actor.max_action, 1.0):
            warnings.warn(
                "action_scaling and action_bound_method are only intended to deal"
                "with unbounded model action space, but find actor model bound"
                f"action space with max_action={actor.max_action}."
                "Consider using unbounded=True option of the actor model,"
                "or set action_scaling to False and action_bound_method to None.",
            )
        self.actor = actor
        self.actor_old = deepcopy(actor)
        self.actor_old.eval()
        self.actor_optim = actor_optim
        self.critic = critic
        self.critic_old = deepcopy(critic)
        self.critic_old.eval()
        self.critic_optim = critic_optim
        self.tau = tau
        self.gamma = gamma
        self.iter=0
        self.reward_estimator=reward_estimator
        self.args=args
        if exploration_noise == "default":
            exploration_noise = GaussianNoise(sigma=0.1)
        # TODO: IMPORTANT - can't call this "exploration_noise" because confusingly,
        #  there is already a method called exploration_noise() in the base class
        #  Now this method doesn't apply any noise and is also not overridden. See TODO there
        self._exploration_noise = exploration_noise
        # it is only a little difference to use GaussianNoise
        # self.noise = OUNoise()
        self.estimation_step = estimation_step
    def set_exp_noise(self, noise: BaseNoise | None) -> None:
        """Set the exploration noise."""
        self._exploration_noise = noise
    def train(self, mode: bool = True) -> Self:
        """Set the module in training mode, except for the target network."""
        self.training = mode
        self.actor.train(mode)
        self.critic.train(mode)
        return self
    def sync_weight(self) -> None:
        """Soft-update the weight for the target network."""
        self.soft_update(self.actor_old, self.actor, self.tau)
        self.soft_update(self.critic_old, self.critic, self.tau)
    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        obs_next_batch = Batch(
            obs=buffer[indices].obs_next,
            info=[None] * len(indices),
        )  # obs_next: s_{t+n}
        return self.critic_old(obs_next_batch.obs, self(obs_next_batch, model="actor_old").act)
    def process_fn(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> RolloutBatchProtocol | BatchWithReturnsProtocol:
        return self.compute_nstep_return(
            batch=batch,
            buffer=buffer,
            indices=indices,
            target_q_fn=self._target_q,
            gamma=self.gamma,
            n_step=self.estimation_step,
        )
    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        model: Literal["actor", "actor_old"] = "actor",
        **kwargs: Any,
    ) -> ActStateBatchProtocol:
        """Compute action over the given batch data.
        :return: A :class:`~tianshou.data.Batch` which has 2 keys:
            * ``act`` the action.
            * ``state`` the hidden state.
        .. seealso::
            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        model = getattr(self, model)
        actions, hidden = model(batch.obs, state=state, info=batch.info)
        return cast(ActStateBatchProtocol, Batch(act=actions, state=hidden))
    @staticmethod
    def _mse_optimizer(
        batch: RolloutBatchProtocol,
        critic: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """A simple wrapper script for updating critic network."""
        weight = getattr(batch, "weight", 1.0)
        current_q = critic(batch.obs, batch.act).flatten()
        target_q = batch.returns.flatten()
        td = current_q - target_q
        # critic_loss = F.mse_loss(current_q1, target_q)
        critic_loss = (td.pow(2) * weight).mean()
        optimizer.zero_grad()
        critic_loss.backward()
        optimizer.step()
        return td, critic_loss
    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> TDDPGTrainingStats:  # type: ignore
        # critic
        td, critic_loss = self._mse_optimizer(batch, self.critic, self.critic_optim)
        batch.weight = td  # prio-buffer
        # actor
        actor_loss = -self.critic(batch.obs, self(batch).act).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.sync_weight()
        return DDPGTrainingStats(actor_loss=actor_loss.item(), critic_loss=critic_loss.item())  # type: ignore[return-value]
    _TArrOrActBatch = TypeVar("_TArrOrActBatch", bound="np.ndarray | ActBatchProtocol")
    def exploration_noise(
        self,
        act: _TArrOrActBatch,
        batch: ObsBatchProtocol,
    ) -> _TArrOrActBatch:
        if self._exploration_noise is None:
            return act
        if isinstance(act, np.ndarray):
            return act + self._exploration_noise(act.shape)
        warnings.warn("Cannot add exploration noise to non-numpy_array action.")
        return act
    
    def post_process_fn(
        self,
        batch: BatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> None:
        """Post-process the data from the provided replay buffer.
        Typical usage is to update the sampling weight in prioritized
        experience replay. Used in :meth:`update`.
        """
        #一共epoch*step_per_epoch*update_per_step次iter,200000,每个epoch step都调用一次
        #alpha从0到0.5
        num_iter=self.args.epoch*self.args.step_per_epoch*self.args.update_per_step
        if self.iter%10==0:
            if self.iter < num_iter*0.8:
                alpha = 0.2 + (0.7 - 0.2) * (self.iter / (num_iter * 0.8))  # alpha 从 0.2 逐步增加到 0.7
                self.reward_estimator.update(batch,buffer, alpha,self.iter)
            else:
                alpha = 0.7  # 保持在 0.7
                self.reward_estimator.update(batch,buffer, alpha,self.iter)
        self.iter+=1