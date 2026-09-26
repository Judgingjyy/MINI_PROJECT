import torch
from torch.utils.data import Dataset

from expert import expert_action_chunk


class ChunkDataset(Dataset):

    def __init__(
        self,
        num_samples=4096,
        horizon=4,
        step_size=0.1
    ):
        self.num_samples=num_samples
        self.horizon=horizon
        self.step_size=step_size
        self.goal=torch.tensor(
            [1.0,1.0],
            dtype=torch.float32
        )
        num_global_obs=(int)(num_samples*0.7)
        num_near_goal=num_samples-num_global_obs
        global_obs=torch.empty(
            num_global_obs,
            2
        ).uniform_(-1.0,1.2)

        near_goal_obs=self.goal+(torch.empty(num_near_goal,2)).uniform_(-0.2,0.2)
        self.obs=torch.cat([global_obs,near_goal_obs],dim=0)
        chunks=[]
        for obs in self.obs:
            chunk=expert_action_chunk(
                state=obs,
                goal=self.goal,
                horizon=self.horizon,
                step_size=self.step_size
            )
            chunks.append(chunk)

        self.chunks=torch.stack(chunks)

    def __len__(self):
        return self.num_samples

    def __getitem__(self,idx):
        obs=self.obs[idx]
        chunk=self.chunks[idx]
        return obs,chunk

if __name__ == "__main__":

    dataset = ChunkDataset(
        num_samples=10,
        horizon=4,
        step_size=0.1
    )

    obs, chunk = dataset[0]

    print("Dataset size:", len(dataset))

    print("Obs:")
    print(obs)
    print("Obs shape:", obs.shape)

    print("Chunk:")
    print(chunk)
    print("Chunk shape:", chunk.shape)

    print("All obs shape:", dataset.obs.shape)
    print("All chunks shape:", dataset.chunks.shape)