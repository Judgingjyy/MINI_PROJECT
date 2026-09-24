import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from model import FlowMatchingModel
from dataset import ActionDataset


def train():

    # ========== 1. 超参数 ==========

    batch_size = 64
    epochs = 10
    lr = 0.001

    action_horizon = 4
    action_dim = 3

    min_loss=2
    # ========== 2. 数据集 ==========

    dataset = ActionDataset(
        num_samples=4096,
        action_horizon=action_horizon,
        action_dim=action_dim
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # ========== 3. 创建模型 ==========

    model = FlowMatchingModel(
        action_horizon=action_horizon,
        action_dim=action_dim
    )

    # ========== 4. 损失函数与优化器 ==========

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    # ========== 5. 训练循环 ==========

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0

        for A in loader:

            # TODO 1: 获取当前 batch size
            B=A.size(0)


            # TODO 2: 生成与 A 形状相同的高斯噪声
            noise =torch.randn_like(A)

            # TODO 3: 为每条样本生成一个随机时间参数
            # 目标形状：(B, 1, 1)
            tau = torch.rand(B)
            tau=tau.unsqueeze(1)
            tau=tau.unsqueeze(1)

            # TODO 4: 构造插值动作
            Ar = tau*A+(1-tau)*noise

            # TODO 5: 计算目标向量场
            v_target = A-noise

            # TODO 6: 前向传播
            v_pred = model(Ar,tau)

            # TODO 7: 计算 Flow Matching 损失
            loss =loss_fn(v_target,v_pred)

            # TODO 8: 梯度清零、反向传播、参数更新

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            # 记录本次损失
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if avg_loss<min_loss:
            torch.save(
                model.state_dict(),
                "checkpoints/flow_matching.pth"
            )
            print(f"{epoch} model is saved")
        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {avg_loss:.6f}"
        )


if __name__ == "__main__":
    train()