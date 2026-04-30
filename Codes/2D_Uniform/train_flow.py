import torch
from utilities import *
from parameters import *

torch.manual_seed(0)

# initialize training
samples = uniform_frozen.samples(N=N_TRAIN)
optimizer = torch.optim.AdamW(flow.parameters(), lr=LR)

for epoch in range(EPOCH):
    # select random batch
    idx     = torch.randperm(N_TRAIN)[:BATCH]
    x_batch = samples[idx].clone().requires_grad_(True)

    # compute detailed balance loss
    loss    = detailed_balance_loss(source=uniform, source_frozen=uniform_frozen, target=target, x=x_batch, flow_t=flow.t())

    # gradient descent step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"  [TV]  Epoch {epoch+1}/{EPOCH}, Loss: {loss.item():.4e}")


torch.save(flow.state_dict(), f"flow_{name}.pt")
