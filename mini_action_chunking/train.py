import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ChunkDataset
from model import ChunkPolicy


num_epochs=50



dataset=ChunkDataset(
    num_samples=4096,
    horizon=4,
    step_size=0.1
)

dataloader=DataLoader(
    dataset,
    batch_size=64,
    shuffle=True
)

model=ChunkPolicy(
    obs_dim=2,
    action_horizon=4,
    action_dim=2,
    hidden_dim=128
)

criterion=nn.MSELoss()
optimizer=torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

for epoch in range(num_epochs):
    model.train()
    total_loss=0.0

    for obs,target_chunk in dataloader:
        pred_chunk=model(obs)
        loss=criterion(
            pred_chunk,
            target_chunk
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    avg_loss=total_loss/len(dataloader)
    print(
        f"Epoch:{epoch+1},Loss:{avg_loss:.6f}"
    )
    torch.save(
        model.state_dict(),
        "checkpoints/chunk_policy.pth"
    )
    print("Model saved.")