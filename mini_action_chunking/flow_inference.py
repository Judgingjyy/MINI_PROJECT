import torch


def sample_action_chunk(
    model,
    obs,
    action_horizon=4,
    action_dim=2,
    num_steps=10
):
    obs_batch = obs.unsqueeze(0)
    action = torch.randn(
        1,
        action_horizon,
        action_dim
    )
    dt=1.0/num_steps
    with torch.no_grad():

        for step in range(num_steps):

            # 1. 当前 tau
            tau = torch.full(
                (1, 1, 1),
                dt * step,
                dtype=action.dtype,
                device=action.device
            )
            # 2. model 预测 velocity
            v_pred=model(action,
                         obs_batch,
                         tau)

            # 3. Euler 更新 action
            action+=dt*v_pred
    action=action.squeeze(0)
    return action 