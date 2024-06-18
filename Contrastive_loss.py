import torch
torch.set_printoptions(precision=2)
identity_vec = torch.rand([10, 9], requires_grad=True)
gt_identity  = torch.randint(0, 4, [10])
sorted_index, indices = torch.sort(gt_identity)
num_masks = torch.bincount(sorted_index)


cumsum_index = torch.cumsum(num_masks, dim=0)


def compute_loss(x):
    start = 0
    sim_loss = 0.
    dis_loss = 0.
    for end in cumsum_index:
        if start == end:
            continue
        sim_loss += torch.mean(1 - x[start:end, start:end])
        if end == x.shape[1]:
            pass
        else:
            dis = torch.nn.ReLU()(x[start:end, end:]- 0.5)
            dis_loss += torch.mean(torch.nn.ReLU()(x[start:end, end:]- 0.5)) 
        start = end
    return sim_loss+dis_loss
optimizer = torch.optim.Adam([identity_vec])
for epoch in range(10000):
    identity_vec_oredered = torch.gather(identity_vec, dim=0, index=indices.view(-1, 1).expand(-1, 9))
    identity_vec_oredered = torch.nn.functional.normalize(identity_vec_oredered, dim=-1)
    similarity = torch.mm(identity_vec_oredered, identity_vec_oredered.T)
    loss = compute_loss(similarity)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (epoch % 1000 == 0):
        print(loss.item())


print(identity_vec)
print(gt_identity)
print(indices)
print(num_masks)
print(identity_vec_oredered)
print(similarity)
