import torch

class PointEnv:
    def __init__(
        self,
        start,
        goal,
        noise_std=0.0
    ):
        self.state=torch.tensor(
            start,
            dtype=torch.float32
        )
        self.goal=torch.tensor(
            goal,
            dtype=torch.float32
        )
        self.noise_std=noise_std
    def get_obs(self):
        return self.state.clone()
    def step(self,action):
        noise=torch.randn_like(action)*self.noise_std
        actual_action=action+noise
        self.state=self.state+actual_action
        return self.get_obs()
    def distance_to_goal(self):
        distance=torch.norm(
            self.goal-self.state
        )
        return distance.item()

if __name__ == "__main__":

    env = PointEnv(
        start=[0.0, 0.0],
        goal=[1.0, 1.0]
    )

    print("Initial state:", env.get_obs())

    action = torch.tensor([0.1, 0.2])

    new_state = env.step(action)

    print("New state:", new_state)
    print("Distance:", env.distance_to_goal())