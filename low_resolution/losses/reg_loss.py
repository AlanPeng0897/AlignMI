import torch

def reg_loss(featureT, fea_mean, fea_logvar):
    fea_reg = reparameterize(fea_mean, fea_logvar)
    fea_reg = fea_mean.repeat(featureT.shape[0], 1)
    loss_reg = torch.mean((featureT - fea_reg).pow(2))
    # print('loss_reg',loss_reg)
    return loss_reg

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)

    return eps * std + mu