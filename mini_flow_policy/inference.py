import torch

from model import FlowPolicyModel


model = FlowPolicyModel(
    obs_dim=2,
    action_horizon=4,
    action_dim=2
)

model.load_state_dict(
    torch.load(
        "flow_policy.pth",
        weights_only=True
    )
)

model.eval()


@torch.no_grad()
def sample_action(
    model,
    obs,
    num_steps=10,
    initial_noise=None
):

    # TODO 1
    # 得到 batch size B
    B=obs.size(0)

    # TODO 2
    # 生成初始 action noise
    #
    # shape:
    # (B, 4, 2)
    if initial_noise is None:
        noise=torch.randn(B,4,2)
    else:
        noise=initial_noise.clone()

    # TODO 3
    # dt
    dt=1.0/num_steps


    # TODO 4
    # 循环 num_steps 次
    for i in range(num_steps):
        # 当前 tau
        tau_value=dt*i

        ##
        tau=torch.full(
            (B,1,1),
            tau_value
        )
        # TODO 5
        # 预测 vector field
        #
        # 注意：
        # model(action, obs, tau)
        v_pred=model(noise,obs,tau)

        # TODO 6
        # Euler 更新 action
        noise=noise+dt*v_pred
    action=noise

    return action

if __name__ == "__main__":

    obs = torch.tensor([
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0]
    ])

    actions = sample_action(
        model,
        obs,
        num_steps=10
    )

    print("obs:")
    print(obs)

    print("generated actions:")
    print(actions)