import torch
from torch.utils.data import Dataset


class FlowPolicyDataset(Dataset):

    def __init__(
        self,
        num_samples=4096,
        obs_dim=2,
        action_horizon=4,
        action_dim=2
    ):
        super().__init__()

        self.num_samples = num_samples
        self.obs_dim = obs_dim
        self.action_horizon = action_horizon
        self.action_dim = action_dim

        # TODO 1
        # 随机生成机器人当前位置 observation
        #
        # 希望：
        # observations.shape == (num_samples, obs_dim)
        #
        # 范围可以设为 [-1, 1]
        self.obs=torch.rand(num_samples,obs_dim)
        self.obs=self.obs*2-1


        # TODO 2
        # 根据 observation 生成对应的 action chunk
        #
        # actions.shape:
        # (num_samples, action_horizon, action_dim)
        #
        # 暂时设计为：
        #
        # action ≈ - observation
        #
        # 意思：
        # 如果机器人在右边，就往左移动
        # 如果机器人在上面，就往下移动
        #
        # 但我们希望 4 个 action 稍微有变化，
        # 不要完全一模一样。
        direction=-self.obs
        direction=direction.unsqueeze(1)
        scales=torch.tensor([
            0.4,
            0.3,
            0.2,
            0.1
        ])
        scales=scales.reshape(1,4,1)
        self.action=direction*scales
        


    def __len__(self):

        # TODO
        return self.num_samples


    def __getitem__(self, idx):

        # TODO
        #
        # 返回：
        #
        # observation, action
        #
        # 注意不是以前单独返回 action

        return self.obs[idx],self.action[idx]

if __name__ == "__main__":

    dataset = FlowPolicyDataset()

    obs, action = dataset[0]

    print("obs =", obs)
    print("action =", action)

    print("obs shape =", obs.shape)
    print("action shape =", action.shape)