"""
import numpy as np
import torch
from torch import optim
import torch.nn.functional as nnF
from nnunet.IMA.RobustDNN_PGD import clip_norm_, normalize_grad_
# https://arxiv.org/pdf/1901.08573.pdf
#https://github.com/yaodongyu/TRADES/blob/master/trades.py
#%%
def run_model(model, X):
    Z=model(X)
    if len(Z.size()) <= 1:
        Yp = (Z.data>0).to(torch.int64)
    else:
        Yp = Z.data.max(dim=1)[1]
    return Z, Yp
#%%
def kl_div(Zn, Z, reduction='batchmean'):
    if len(Z.size()) <= 1:
        Zn = torch.cat([-Zn.view(-1,1), Zn.view(-1,1)], dim=1)
        Z = torch.cat([-Z.view(-1,1), Z.view(-1,1)], dim=1)
    #------------------------
    return nnF.kl_div(nnF.log_softmax(Zn, dim=1), nnF.softmax(Z, dim=1), reduction=reduction)
#%%
def find_Xn(model, X, Y, max_norm, norm_type, max_iter, step, sigma=0.001,
            clip_X_min=0, clip_X_max=1, use_optimizer=False):
    X=X.detach() # not need to clone
    noise_init = sigma*torch.randn_like(X)
    clip_norm_(noise_init, norm_type, max_norm)
    Xn = X + noise_init
    Z, Yp = run_model(model, X)
    Z=Z.detach()
    #-----------------
    noise_new=(Xn-X).detach()
    if use_optimizer == True:
        optimizer = optim.Adamax([noise_new], lr=step)
    #-----------------
    for iter in range(0, max_iter):
        Xn=Xn.detach()
        Xn.requires_grad=True
        Zn, Ypn=run_model(model, Xn)
        dist=kl_div(Zn, Z, reduction='sum')
        #dist.backward() will update W.grad
        grad_n=torch.autograd.grad(dist, Xn)[0]
        normalize_grad_(grad_n, norm_type)
        if use_optimizer == False:
            Xnew = Xn.detach() + step*grad_n.detach()
            noise_new = Xnew-X
        else:
            noise_new.grad=-grad_n.detach() #grad ascend
            optimizer.step()
        clip_norm_(noise_new, norm_type, max_norm)
        Xn = torch.clamp(X+noise_new, clip_X_min, clip_X_max)
        noise_new.data -= noise_new.data-(Xn-X).data
        Xn=Xn.detach()
        #---------------------------
    return Xn
#%%
def TRADES_loss_old(model, loss_fn, X, Y, max_norm, norm_type, max_iter, step,
                sigma=0.001, beta=0.5, clip_X_min=0, clip_X_max=1, use_optimizer=False,
                model_eval_attack=False, model_eval_Xn=False):
    Z, Yp = run_model(model, X)
    #-------------------------------------------
    train_mode=model.training# record the mode
    if model_eval_attack == True and train_mode == True:
        model.eval()#no BN, no dropout, etc
    #-----------------
    Xn=find_Xn(model, X, Y, max_norm, norm_type, max_iter, step, sigma, clip_X_min, clip_X_max, use_optimizer)
    #-----------------
    if model_eval_Xn == True:
        if model.training == True:
            model.eval()
    else:
        if train_mode == True and model.training == False:
            model.train()
    #-----------------
    Zn, Ypn=run_model(model, Xn)

    loss1=loss_fn(Z, Y).sum()  
    loss2=kl_div(Zn, Z)
    loss=loss1+ beta*loss2
    #--------------------------------------------
    if train_mode == True and model.training == False:
        model.train()
    #--------------------------------------------
    return loss, loss1, loss2
"""


import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def TRADES_loss(model,
                loss_fn,
                criterion_kl,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    # define KL-loss
    #criterion_kl = nn.KLDivLoss(size_average=False)
    
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(model(x_adv),model(x_natural))
                
                
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(model(adv),model(x_natural))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = loss_fn(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(model(x_adv),model(x_natural))
    loss = loss_natural + beta * loss_robust
    return loss, loss_natural, loss_robust, 