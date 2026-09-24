import torch
from torch.utils.data import Dataset


class ActionDataset(Dataset):

    def __init__(
        self,
        num_samples=4096,
        action_horizon=4,
        action_dim=3
    ):
        super().__init__()

        self.num_samples = num_samples
        self.action_horizon = action_horizon
        self.action_dim = action_dim

        # 创建目标动作
        self.mu = torch.zeros(
            action_horizon,
            action_dim
        )

        # TODO 1:
        # 将 self.mu 每一行的第 0 个元素设置为 1
        self.mu[:,0]=1


    def __len__(self):

        # TODO 2:
        # 返回数据集大小

        return self.num_samples


    def __getitem__(self, idx):

        # TODO 3:
        # 生成与 self.mu 形状相同的高斯扰动
        eta =torch.randn_like(self.mu)

        # TODO 4:
        # 根据 A = mu + 0.1 * eta 生成动作
        action = self.mu+0.1*eta

        return action

if __name__ == "__main__":

    dataset = ActionDataset()

    print("Dataset size:", len(dataset))

    action = dataset[0]

    print("Action shape:", action.shape)
    print("Action:", action)