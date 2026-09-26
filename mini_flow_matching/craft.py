import torch 
import torch.nn as nn

A=torch.randn(2,4,3)
noise=torch.randn_like(A)
r=torch.rand(2)
r=r.unsqueeze(1)
r=r.unsqueeze(1)
Ar=r*A+(1-r)*noise
print("A shape",Ar.shape)
print(r,'',r.shape)
v_target=A-noise
Ar_flat=Ar.reshape(2,12)
r_flat=r.reshape(2,1)
X=torch.cat([Ar_flat,r_flat],dim=1)
model=nn.Sequential(
    nn.Linear(13,16),
    nn.ReLU(),
    nn.Linear(16,12)
) 

v_pred=model(X)
loss_fn=nn.MSELoss()
loss=loss_fn(v_pred,v_target.reshape(2,12))
print("loss: ",loss.item())
optimizer=torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
optimizer.zero_grad()
loss.backward()
optimizer.step()