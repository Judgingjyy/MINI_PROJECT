import torch

from model import FlowPolicyModel
from inference import sample_action

obs_dim=2
action_horizon=4
action_dim=2

model = FlowPolicyModel(
    obs_dim=obs_dim,
    action_horizon=action_horizon,
    action_dim=action_dim
)

model.load_state_dict(
    torch.load(
        "flow_policy.pth",
        weights_only=True
    )
)

model.eval()

num_samples=1000
obs=torch.rand(num_samples,obs_dim)
obs=obs*2-1
direction=-obs
scales=torch.tensor([
    0.4,0.3,0.2,0.1
])
scales=scales.reshape(1,4,1)
direction=direction.unsqueeze(1)
target_action=direction*scales
initial_noise=torch.randn(
    num_samples,
    action_horizon,
    action_dim
)

steps_list=[1,5,10,20,50]

for steps in steps_list:

    generated_action=sample_action(
        model,
        obs,
        num_steps=steps,
        initial_noise=initial_noise
    )
    mae=torch.abs(generated_action-target_action).mean()
    print(
        f"Steps:{steps:2d},MAE:{mae.item():.6f}"
    )