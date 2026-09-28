import torch
import time
from flow_chunk_model import FlowChunkPolicy
from dataset import ChunkDataset
from flow_inference2com import sample_action_chunk
device=torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
model = FlowChunkPolicy(
    obs_dim=2,
    action_horizon=4,
    action_dim=2,
    hidden_dim=128
).to(device)

model.load_state_dict(
    torch.load(
        "checkpoints/flow_chunk_policy.pth",
        map_location="cpu",
        weights_only=True
    )
)

model.eval()
torch.manual_seed(31)




dataset = ChunkDataset()

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    shuffle=False
)
total_abs_error=0
total_num=0
if device.type=="cuda":
    torch.cuda.synchronize()
start_time=time.perf_counter()
for obs, target_action in loader:
    obs = obs.to(device)
    target_action=target_action.to(device)
    pred_action = sample_action_chunk(
        model=model,
        obs=obs,
        action_horizon=4,
        action_dim=2,
        num_steps=10
    )   
    abs_error=torch.abs(pred_action - target_action)
    total_abs_error+=abs_error.sum().item()
    
    total_num+=abs_error.numel()

if device.type=="cuda":
    torch.cuda.synchronize()
end_time=time.perf_counter()

print("TIME: ",end_time-start_time)
print("MAE: ",total_abs_error/total_num)

