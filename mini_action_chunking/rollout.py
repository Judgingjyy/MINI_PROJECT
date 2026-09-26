import torch
from env import PointEnv
from expert import expert_action_chunk

def rollout(
        env,
        horizon=4,
        execute_horizon=1,
        step_size=0.1,
        max_steps=100
):
    over=0
    env_step=0
    for step in range(max_steps):
        obs=env.get_obs()
        action_chunk=expert_action_chunk(
            state=obs,
            goal=env.goal,
            horizon=horizon,
            step_size=step_size
        )
        actions_to_execute=action_chunk[:execute_horizon]
        for action in actions_to_execute:
            env.step(action)

            env_step+=1
            distance = env.distance_to_goal()

            print(
                f"Step: {step},Env_step: {env_step}  "
                f"State: {env.get_obs()}, "
                f"Distance: {distance:.4f}"
            )
            if distance<1e-3:
                over=1
                print("You have reached")
                break
        if over:
            break

if __name__=="__main__":
    torch.manual_seed(42)
    env=PointEnv(
        start=[0.0,0.0],
        goal=[1.0,1.0],
        noise_std=0.001
    )
    rollout(
        env=env,
        horizon=4,
        execute_horizon=4,
        step_size=0.1,
        max_steps=100
    )