import torch
import torch.nn as nn


class FlowMatchingModel(nn.Module):

    def __init__(
        self,
        action_horizon=4,
        action_dim=3,
        hidden_dim=64
    ):
        super().__init__()

        self.action_horizon = action_horizon
        self.action_dim = action_dim

        self.action_size = action_horizon * action_dim

        self.net = nn.Sequential(
            nn.Linear(self.action_size + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,self.action_size)

        )

    def forward(self, action, tau):

        # TODO 1: 取得 batch size
        B = action.size(0)

        # TODO 2: 把动作展开成 (B, action_size)
        action_flat =action.reshape(B,-1)

        # TODO 3: 把时间参数展开成 (B, 1)

        tau_flat=tau.reshape(B,1)
        # TODO 4: 拼接动作和时间参数
        x =torch.cat([action_flat,tau_flat],dim=1)

        # TODO 5: 通过神经网络预测向量场
        v_pred = self.net(x)

        # TODO 6: 恢复成原来的动作形状
        v_pred = v_pred.reshape(B,self.action_horizon,self.action_dim)

        return v_pred