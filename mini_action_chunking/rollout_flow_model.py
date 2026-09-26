import torch
from flow_inference import sample_action_chunk
from env import PointEnv
from flow_chunk_model import FlowChunkPolicy
model = FlowChunkPolicy(
    obs_dim=2,
    action_horizon=4,
    action_dim=2,
    hidden_dim=64
)

model.load_state_dict(
    torch.load(
        "checkpoints/flow_chunk_policy.pth",
        map_location="cpu"
    )
)

model.eval()

def rollout(
    env,
    model,
    execute_horizon=1,
    max_steps=100
):

    env_step = 0
    total_path_length=0.0
    for plan_step in range(max_steps):

        obs = env.get_obs()


        with torch.no_grad():
            action_chunk = sample_action_chunk(
                model=model,
                obs=obs,
                action_horizon=4,
                action_dim=2,
                num_steps=10
            )

        actions_to_execute = action_chunk[:execute_horizon]

        for action in actions_to_execute:
            old_state=env.get_obs()
            env.step(action)
            new_state=env.get_obs()
            step_distance=torch.norm(
                new_state-old_state
            ).item()
            total_path_length+=step_distance
            env_step += 1
            
            # if env_step == 8:
            #     env.state += torch.tensor([0.0, 0.5])

            #     print("===== DISTURBANCE! =====")
            #     print("New state:", env.get_obs())
            
            distance = env.distance_to_goal()

            print(
                f"Plan Step: {plan_step}, "
                f"Env Step: {env_step}, "
                f"State: {env.get_obs()}, "
                f"Distance: {distance:.4f}"
            )

            if distance < 0.02:
                print("Goal reached!")
                return env_step,distance,total_path_length
    return env_step,distance,total_path_length

if __name__ == "__main__":

    for execute_horizon in [1,2,4]:
        torch.manual_seed(42)

        env = PointEnv(
            start=[-0.5, -0.5],
            goal=[1.0, 1.0],
            noise_std=0.0
        )

        env_steps,final_distance,path_length=rollout(
            env=env,
            model=model,
            execute_horizon=execute_horizon,
            max_steps=100
        )
        print(
            f"Execute Horizon: {execute_horizon}, "
            f"Env Steps: {env_steps}, "
            f"Final Distance: {final_distance:.4f}, "
            f"Path Length: {path_length:.4f}"
        )
