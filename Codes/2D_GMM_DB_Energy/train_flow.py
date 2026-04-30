import torch
from utilities import *
from parameters import *

torch.manual_seed(0)

optimizer = torch.optim.AdamW(flow.parameters(), lr=LR)

for epoch in range(EPOCH):
    # x ~ mu_0 (prior space); R_F(x) = U(F(x)) - U_0(x) - log|det J_F|
    x_batch = gmm.samples(N=BATCH).requires_grad_(True)
    grv     = gradient_ratio_v(source=target, target=gmm, x=x_batch, flow_t=flow.t())
    loss    = grv.abs().mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"  [DB-E]  Epoch {epoch+1}/{EPOCH}, Loss: {loss.item():.4e}")


torch.save(flow.state_dict(), f"flow_{name}.pt")
