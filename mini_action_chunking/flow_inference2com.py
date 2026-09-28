import torch


def sample_action_chunk(
    model,
    obs,
    action_horizon=4,
    action_dim=2,
    num_steps=10
):
    device=torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model=model.to(device)
    obs=obs.to(device)
    B=obs.shape[0]
    action = torch.randn(
        B, 
        action_horizon,
        action_dim,
        device=device
    )
    dt=1.0/num_steps
    with torch.no_grad():

        for step in range(num_steps):


            # 1. 当前 tau
            tau = torch.full(
                (B, 1, 1),
                dt * step,
                dtype=action.dtype,
                device=action.device
            )
            tau=tau.to(device)
            # 2. model 预测 velocity
            v_pred=model(action,
                         obs,
                         tau)

            # 3. Euler 更新 action
            action+=dt*v_pred
    return action 