import torch
from utilities import *
from parameters import *

torch.manual_seed(0)

optimizer = torch.optim.AdamW(flow.parameters(), lr=LR)

for epoch in range(EPOCH):
    # reverse-KL energy loss: y ~ ν, minimize V(F(y)) - log|det J_F(y)|
    y_batch = gmm_frozen.samples(N=BATCH)
    x, ladj = flow.t().call_and_ladj(y_batch)
    loss    = (target(x) - ladj).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"  [KL]  Epoch {epoch+1}/{EPOCH}, Loss: {loss.item():.4e}")


torch.save(flow.state_dict(), f"flow_{name}.pt")
