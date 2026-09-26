import torch
import torch.nn as nn

class ChunkPolicy(nn.Module):
    def __init__(
        self,
        obs_dim=2,
        action_horizon=4,
        action_dim=4,
        hidden_dim=128
    ):
        super().__init__()
        self.action_horizon=action_horizon
        self.action_dim=action_dim

        self.net=nn.Sequential(
            nn.Linear(
                obs_dim,
                hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                action_horizon*action_dim
            )
        )
    def forward(self,obs):
        x=self.net(obs)
        action_chunk=x.view(
            -1,
            self.action_horizon,
            self.action_dim
        )
        return action_chunk

if __name__ == "__main__":

    model = ChunkPolicy(
        obs_dim=2,
        action_horizon=4,
        action_dim=2
    )

    obs = torch.randn(32, 2)

    action_chunk = model(obs)

    print("Obs shape:")
    print(obs.shape)

    print("Action chunk shape:")
    print(action_chunk.shape)