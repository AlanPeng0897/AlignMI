import torch
import torch.nn.functional as F

def max_margin_loss(out, iden):
    real = out.gather(1, iden.unsqueeze(1)).squeeze(1)
    tmp1 = torch.argsort(out, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == iden, tmp1[:, -2], tmp1[:, -1])
    margin = out.gather(1, new_y.unsqueeze(1)).squeeze(1)

    return (-1 * real).mean() + margin.mean()