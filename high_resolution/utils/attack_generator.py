import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


def TRADES_loss(adv_logits, natural_logits, target, beta):
    adv_logits = torch.clamp(adv_logits, min=-100, max=100)
    natural_logits = torch.clamp(natural_logits, min=-100, max=100)
    natural_logits = natural_logits - natural_logits.max(dim=1, keepdim=True)[0]

    batch_size = len(target)
    criterion_kl = nn.KLDivLoss(size_average=False).cuda()
    loss_natural = nn.CrossEntropyLoss(reduction='mean')(natural_logits, target)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(adv_logits, dim=1),
                                                    F.softmax(natural_logits, dim=1))
        
    loss = loss_natural + beta * loss_robust
    return loss


def MART_loss(adv_logits, natural_logits, target, beta):
    adv_logits = torch.clamp(adv_logits, min=-100, max=100)
    natural_logits = torch.clamp(natural_logits, min=-100, max=100)
    kl = nn.KLDivLoss(reduction='none')
    batch_size = len(target)
    adv_probs = F.softmax(adv_logits, dim=1)
    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == target, tmp1[:, -2], tmp1[:, -1])
    loss_adv = F.cross_entropy(adv_logits, target) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
    nat_probs = F.softmax(natural_logits, dim=1)
    true_probs = torch.gather(nat_probs, 1, (target.unsqueeze(1)).long()).squeeze()
    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
    loss = loss_adv + float(beta) * loss_robust
    return loss


def cwloss(output, target, confidence=50, num_classes=10):
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
    loss = torch.sum(loss)
    return loss


def PGD(model, data, target, epsilon, step_size, num_steps, loss_fn, category, rand_init):
    model.eval()
    Kappa = torch.ones(len(data))

    # If data is in the range [-1, 1]
    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()
        nat_output = model(data)

    if category == "Madry":
        # Adjust the initialization of adversarial examples to stay in [-1, 1]
        x_adv = data.detach() + torch.from_numpy(
            np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        # Ensure x_adv is clamped within [-1, 1] instead of [0, 1]
        x_adv = torch.clamp(x_adv, -1.0, 1.0)

    for k in range(num_steps):
        x_adv.requires_grad_()
        output = model(x_adv)
        predict = output.max(1, keepdim=True)[1]
        # Update Kappa
        for p in range(len(x_adv)):
            if predict[p] == target[p]:
                Kappa[p] += 1

        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)
            elif loss_fn == "cw":
                loss_adv = cwloss(output, target)
            elif loss_fn == "kl":
                criterion_kl = nn.KLDivLoss(size_average=False).cuda()
                loss_adv = criterion_kl(F.log_softmax(output, dim=1), F.softmax(nat_output, dim=1))

        loss_adv.backward(retain_graph=True)
        
        eta = step_size * x_adv.grad.sign()

        # Update adversarial data
        x_adv = x_adv.detach() + eta
        # Ensure x_adv stays within [-1, 1] after update
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, -1.0, 1.0)

    x_adv = Variable(x_adv, requires_grad=False)

    return x_adv, Kappa


'''
def PGD(model, data, target, epsilon, step_size, num_steps,loss_fn,category,rand_init):
    model.eval()
    Kappa = torch.ones(len(data))
    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()
        nat_output = model(data)
    
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    for k in range(num_steps):
        x_adv.requires_grad_()
        output = model(x_adv)
        predict = output.max(1, keepdim=True)[1]
        # Update Kappa
        for p in range(len(x_adv)):
            if predict[p] == target[p]:
                Kappa[p] += 1
        
        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)
            if loss_fn == "cw":
                loss_adv = cwloss(output,target)
            if loss_fn == "kl":
                criterion_kl = nn.KLDivLoss(size_average=False).cuda()
                loss_adv = criterion_kl(F.log_softmax(output, dim=1),F.softmax(nat_output, dim=1))
        
        loss_adv.backward(retain_graph=True)
        eta = step_size * x_adv.grad.sign()
        # Update adversarial data
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    x_adv = Variable(x_adv, requires_grad=False)
    
    return x_adv, Kappa

'''


def eval_clean(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def eval_robust(model, test_loader, perturb_steps, epsilon, step_size, loss_fn, category, random):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.enable_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            x_adv, _ = PGD(model, data, target, epsilon, step_size, perturb_steps, loss_fn, category, rand_init=random)
            output = model(x_adv)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy
