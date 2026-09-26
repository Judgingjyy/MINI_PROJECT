import torch

from pathlib import Path
from model import FlowMatchingModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FlowMatchingModel(
    action_horizon=4,
    action_dim=3,
    hidden_dim=64
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
def sample_actions(model, batch_size=1, num_steps=10,initial_noise=None):

    # 0. 生成初始高斯噪声
    if initial_noise is not None:
        action=initial_noise.clone()  ###############
    else:
        action = torch.randn(
            batch_size, 4, 3,
            device=device
        )

    # 1. 计算每次更新的时间步长
    dt =  1.0/num_steps

    # 2. 多步推理
    for step in range(num_steps):

        # 当前时间参数
        tau_value = step * dt

        tau = torch.full(
            (batch_size, 1, 1),
            tau_value,
            device=device
        ) 
        v_pred =model(action,tau)
        # 3. 预测当前向量场
    

        # 4. 根据向量场更新动作
        action = action + dt*v_pred

    return action


if __name__ == "__main__":

    initial_noise=torch.randn(
        1000,4,3,
        device=device
    )
    for steps in [1, 2, 5, 10, 20, 50]:

        generated_actions = sample_actions(
            model,
            batch_size=1000,
            num_steps=steps,
            initial_noise=initial_noise
        )

        mu=torch.zeros(4,3,device=device)
        mu[:,0]=1.0

        generated_mean=generated_actions.mean(dim=0)
        mean_error=torch.abs(generated_mean-mu).mean()
        generated_std=generated_actions.std(dim=0)
        std_error=torch.abs(generated_std-0.1).mean()


        print(
            f"Steps:{steps},"
            f"Mean Error:{mean_error.item():.5f},"
            f"Std Error: {std_error.item():.5f}"
        )