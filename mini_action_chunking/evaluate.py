import torch

from model import ChunkPolicy
from expert import expert_action_chunk

model = ChunkPolicy(
    obs_dim=2,
    action_horizon=4,
    action_dim=2,
    hidden_dim=128
)

model.load_state_dict(
    torch.load(
        "chunk_policy.pth",
        map_location="cpu",
        weights_only=True
    )
)

model.eval()

num_tests = 1000

states = torch.empty(
    num_tests,
    2
).uniform_(-1.0, 1.0)
goal=torch.ones(2)

horizon_errors=torch.zeros(4)

for state in states:

    expert_chunk = expert_action_chunk(
        state=state,
        goal=goal,
        horizon=4,
        step_size=0.1
    )

    with torch.no_grad():

        pred_chunk = model(
            state.unsqueeze(0)
        ).squeeze(0)

    abs_error=torch.abs(
        pred_chunk-expert_chunk
    )
    per_action_mae=abs_error.mean(dim=1)
    horizon_errors += per_action_mae
horizon_errors /= num_tests
for i, error in enumerate(horizon_errors):

    print(
        f"Action {i} MAE: "
        f"{error.item():.6f}"
    )
    