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


class CusDDPGPolicy(DDPGPolicy):
    def __init__(
        self,
        reward_estimator: Reward_Estimator,
        args,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)  # 调用父类的初始化方法
        self.iter = 0
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
        # 一共 epoch*step_per_epoch*update_per_step 次 iter
        num_iter = self.args.epoch * self.args.step_per_epoch * self.args.update_per_step
        if self.iter % 10 == 0:
            if self.iter < num_iter * 0.6:
                alpha = 1 - np.exp(-self.iter / num_iter)
                self.reward_estimator.update(batch, buffer, alpha, self.iter)
            else:
                alpha = 0.5
                self.reward_estimator.update(batch, buffer, alpha, self.iter)
        self.iter += 1