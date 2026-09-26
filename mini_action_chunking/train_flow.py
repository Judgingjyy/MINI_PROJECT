import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from dataset import ChunkDataset
from flow_chunk_model import FlowChunkPolicy


def train():

    # 1. 超参数
    batch_size = 64
    epochs = 10
    lr = 1e-3

    # 2. dataset
    dataset = ChunkDataset(
        num_samples=4096,
        horizon=4,
        step_size=0.1
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # 3. model
    model = FlowChunkPolicy(
        obs_dim=2,
        action_horizon=4,
        action_dim=2
    )

    # 4. loss + optimizer
    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    # 5. training loop
    for epoch in range(epochs):

        model.train()

        total_loss = 0.0

        for obs, action in loader:

            B = action.size(0)

            # TODO 1
            # 生成与 action 一样形状的高斯噪声
            noise=torch.randn_like(action)


            # TODO 2
            # tau
            # shape 要变成 (B,1,1)
            tau=torch.rand(B,1)

            tau=tau.unsqueeze(1)


            # TODO 3
            # 构造 A_tau
            A_tau=tau*action+(1-tau)*noise


            # TODO 4
            # target vector field

            target_v=action-noise

            # TODO 5
            # 模型预测
            # 注意现在要传 obs

            v_pred=model(A_tau,obs,tau)

            # TODO 6
            # loss

            loss=loss_fn(v_pred,target_v)
            optimizer.zero_grad()

            # TODO 7
            # optimizer 三件套
            loss.backward()
            optimizer.step()


            total_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {total_loss / len(loader):.6f}"
        )

    torch.save(
        model.state_dict(),
        "checkpoints/flow_chunk_policy.pth"
    )

if __name__ == "__main__":
    train()