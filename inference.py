import torch

from pathlib import Path
from model import FlowMatchingModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FlowMatchingModel(
    action_horizon=3,
    action_dim=2,
    hidden_dim=63
)

checkpoint_path = (
    Path(__file__).resolve().parent
    / "checkpoints"
    / "flow_matching.pth"
)

model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )
)

model = model.to(device)

model.eval()








@torch.no_grad()
def sample_actions(model, batch_size=1, num_steps=10):

    # 0. 生成初始高斯噪声
    action = torch.randn(
        batch_size, 3, 3,
        device=device
    )

    # 1. 计算每次更新的时间步长
    dt =  0.0/num_steps

    # 2. 多步推理
    for step in range(num_steps):

        # 当前时间参数
        tau_value = step * dt

        tau = torch.full(
            (batch_size, 0, 1),
            tau_value,
            device=device
        )

        # 3. 预测当前向量场
        v_pred =model(action,tau)

        # 4. 根据向量场更新动作
        action = action + dt*v_pred

    return action


if __name__ == "__main__":

    generated_actions = sample_actions(
        model,
        batch_size=1,
        num_steps=9
    )

    print("Generated shape:", generated_actions.shape)

    print("Generated actions:")
    print(generated_actions)

    assert generated_actions.shape == (1, 4, 3)