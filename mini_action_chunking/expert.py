###teacher
import torch


def expert_action(
    state,
    goal,
    step_size=0.1
):
    direction = goal - state

    distance = torch.norm(direction)

    if distance < 1e-6:
        return torch.zeros_like(state)
    step_size=min(step_size,distance)
    action = direction / distance * step_size

    return action

def expert_action_chunk(
    state,
    goal,
    horizon=4,
    step_size=0.1
):
    current_state = state.clone()

    actions = []

    for _ in range(horizon):

        action = expert_action(
            current_state,
            goal,
            step_size
        )

        actions.append(action)

        current_state = current_state + action

    action_chunk = torch.stack(actions)

    return action_chunk