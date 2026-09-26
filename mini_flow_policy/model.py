import torch
import torch.nn as nn


class FlowPolicyModel(nn.Module):

    def __init__(
        self,
        obs_dim=2,
        action_horizon=4,
        action_dim=2,
        hidden_dim=64
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_horizon = action_horizon

        self.action_dim = action_dim        # TODO 1
        # 计算 action_size
        self.action_size=self.action_dim*self.action_horizon

        # TODO 2
        # 计算总输入维度
        #
        # action_flat
        # + obs
        # + tau
        self.input_size=self.action_size+self.obs_dim+1


        # TODO 3
        # 建立 MLP
        #
        # 输入维度：上面的总输入维度
        # 输出维度：action_size
        self.net=nn.Sequential(
            nn.Linear(self.input_size,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2,self.action_size)
        )


    def forward(self, action, obs, tau):

        # TODO 4
        B=action.size(0)
        H=action.size(1)
        D=action.size(2)

        # TODO 5
        # action 展平
        action_flat=action.reshape(B,-1)

        # TODO 6
        # tau 整理成 (B, 1)
        tau=tau.reshape(B,1)


        # TODO 7
        # 拼接 action_flat、obs、tau
        x=torch.cat([action_flat,obs,tau],dim=1) 

        # TODO 8
        # MLP
        v_pred=self.net(x)

        # TODO 9
        # reshape 回 (B, H, D)
        v_pred=v_pred.reshape(B,H,D)


        return v_pred
    