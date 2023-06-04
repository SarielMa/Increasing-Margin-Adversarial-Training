import numpy as np
import torch
from torch import optim
import torch.nn.functional as nnF
from RobustDNN_PGD import clip_norm_, normalize_grad_
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
def TRADES_loss(model, X, Y, max_norm, norm_type, max_iter, step,
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
    if len(Z.size()) <= 1:
        loss1=nnF.binary_cross_entropy_with_logits(Z, Y.to(X.dtype))
    else:
        loss1=nnF.cross_entropy(Z, Y)
    loss2=kl_div(Zn, Z)
    loss=loss1+ beta*loss2
    #--------------------------------------------
    if train_mode == True and model.training == False:
        model.train()
    #--------------------------------------------
    return loss, loss1, loss2