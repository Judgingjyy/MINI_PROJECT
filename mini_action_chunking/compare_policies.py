import time
import torch

from env import PointEnv

from model import ChunkPolicy
from flow_chunk_model import FlowChunkPolicy
from flow_inference import sample_action_chunk


# ============================================================
# 1. Load Direct Chunk Policy
# ============================================================

direct_model = ChunkPolicy(
    obs_dim=2,
    action_horizon=4,
    action_dim=2,
    hidden_dim=128
)

direct_model.load_state_dict(
    torch.load(
        "checkpoints/chunk_policy.pth",
        map_location="cpu",
        weights_only=True
    )
)

direct_model.eval()


# ============================================================
# 2. Load Flow Chunk Policy
# ============================================================

flow_model = FlowChunkPolicy(
    obs_dim=2,
    action_horizon=4,
    action_dim=2,
    hidden_dim=64
)

flow_model.load_state_dict(
    torch.load(
        "checkpoints/flow_chunk_policy.pth",
        map_location="cpu",
        weights_only=True
    )
)

flow_model.eval()


# ============================================================
# 3. Direct Policy Action Generator
# ============================================================

def sample_direct_chunk(
    model,
    obs
):
    """
    obs:
        shape = [2]

    return:
        action_chunk
        shape = [4, 2]
    """

    obs_batch = obs.unsqueeze(0)

    with torch.no_grad():

        action_chunk = model(
            obs_batch
        )

    action_chunk = action_chunk.squeeze(0)

    return action_chunk


# ============================================================
# 4. Flow Policy Action Generator
# ============================================================

def sample_flow_chunk(
    model,
    obs,
    num_steps=10
):
    """
    obs:
        shape = [2]

    return:
        action_chunk
        shape = [4, 2]
    """

    action_chunk = sample_action_chunk(
        model=model,
        obs=obs,
        action_horizon=4,
        action_dim=2,
        num_steps=num_steps
    )

    return action_chunk


# ============================================================
# 5. Unified Rollout
# ============================================================

def rollout_policy(
    env,
    chunk_fn,
    execute_horizon=1,
    max_plan_steps=100,
    success_threshold=0.02
):

    env_step = 0
    plan_count = 0

    total_path_length = 0.0
    total_inference_time = 0.0

    success = False

    for plan_step in range(max_plan_steps):

        # --------------------------------
        # Get current observation
        # --------------------------------

        obs = env.get_obs()

        # --------------------------------
        # Generate Action Chunk
        # --------------------------------

        start_time = time.perf_counter()

        action_chunk = chunk_fn(obs)

        end_time = time.perf_counter()

        total_inference_time += (
            end_time - start_time
        )

        plan_count += 1

        # --------------------------------
        # Receding Horizon
        # --------------------------------

        actions_to_execute = (
            action_chunk[:execute_horizon]
        )

        for action in actions_to_execute:

            old_state = env.get_obs()

            env.step(action)

            new_state = env.get_obs()

            env_step += 1

            # --------------------------------
            # Path Length
            # --------------------------------

            step_length = torch.norm(
                new_state - old_state
            ).item()

            total_path_length += step_length

            # --------------------------------
            # Distance to Goal
            # --------------------------------

            distance = env.distance_to_goal()

            if distance < success_threshold:

                success = True

                return {
                    "success": success,
                    "env_steps": env_step,
                    "plan_count": plan_count,
                    "final_distance": distance,
                    "path_length": total_path_length,
                    "inference_time": total_inference_time
                }

    # 如果一直没有成功

    distance = env.distance_to_goal()

    return {
        "success": success,
        "env_steps": env_step,
        "plan_count": plan_count,
        "final_distance": distance,
        "path_length": total_path_length,
        "inference_time": total_inference_time
    }


# ============================================================
# 6. Single Experiment
# ============================================================

def run_single_experiment(
    policy_name,
    execute_horizon=1,
    flow_steps=10,
    seed=42
):

    torch.manual_seed(seed)

    # --------------------------------
    # Environment
    # --------------------------------

    env = PointEnv(
        start=[-0.5, -0.5],
        goal=[1.0, 1.0],
        noise_std=0.0
    )

    # --------------------------------
    # Choose Action Generator
    # --------------------------------

    if policy_name == "direct":

        def chunk_fn(obs):

            return sample_direct_chunk(
                model=direct_model,
                obs=obs
            )

        forwards_per_plan = 1

    elif policy_name == "flow":

        def chunk_fn(obs):

            return sample_flow_chunk(
                model=flow_model,
                obs=obs,
                num_steps=flow_steps
            )

        forwards_per_plan = flow_steps

    else:

        raise ValueError(
            f"Unknown policy name: {policy_name}"
        )

    # --------------------------------
    # Rollout
    # --------------------------------

    result = rollout_policy(
        env=env,
        chunk_fn=chunk_fn,
        execute_horizon=execute_horizon,
        max_plan_steps=100,
        success_threshold=0.02
    )

    # --------------------------------
    # Extra Metrics
    # --------------------------------

    result["policy"] = policy_name

    result["execute_horizon"] = (
        execute_horizon
    )

    result["model_forwards"] = (
        result["plan_count"]
        * forwards_per_plan
    )

    if policy_name == "flow":

        result["flow_steps"] = (
            flow_steps
        )

    else:

        result["flow_steps"] = 1

    return result


# ============================================================
# 7. Main Comparison
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 100)
    print("Direct Chunk Policy vs Flow Chunk Policy")
    print("=" * 100)

    for policy_name in [
        "direct",
        "flow"
    ]:

        for execute_horizon in [
            1,
            2,
            4
        ]:

            result = run_single_experiment(
                policy_name=policy_name,
                execute_horizon=execute_horizon,
                flow_steps=10,
                seed=42
            )

            print(
                f"{result['policy']:8s} | "
                f"execute={result['execute_horizon']} | "
                f"success={str(result['success']):5s} | "
                f"env_steps={result['env_steps']:3d} | "
                f"plans={result['plan_count']:3d} | "
                f"forwards={result['model_forwards']:4d} | "
                f"final_dist={result['final_distance']:.4f} | "
                f"path={result['path_length']:.4f} | "
                f"time={result['inference_time']:.6f}s"
            )