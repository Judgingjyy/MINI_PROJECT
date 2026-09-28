import torch

from model import ChunkPolicy
from expert import expert_action_chunk
from dataset import ChunkDataset

torch.manual_seed(31)
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

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = ChunkPolicy(
    obs_dim=2,
    action_horizon=4,
    action_dim=2,
    hidden_dim=64,
).to(device)

model.load_state_dict(
    torch.load(
        "flow_policy.pth",
        map_location=device
    )
)

model.eval()


dataset = ChunkDataset()

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    shuffle=False
)

obs, target_action = next(iter(loader))
obs = obs.to(device)
target_action = target_action.to(device)


