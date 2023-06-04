# -*- coding: utf-8 -*-
"""
Created on Sun May 19 03:31:58 2019

@author: liang
"""
#%%
import numpy as np
import torch
from torch import optim
import torch.nn.functional as nnF
#%%
def loss_reduction(loss, reduction):
    #print(loss.shape)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'sum_squared':
        return torch.sum(loss**2)
    elif reduction == 'sum_abs':
        return torch.sum(loss.abs())
    elif reduction == 'mean_squared':
        return torch.mean(loss**2)
    elif reduction == None or reduction == 'none':
        return loss
    else:
        raise NotImplementedError("unkown reduction "+reduction)
#%%
def binary_cross_entropy_with_logits(Z, Y, reduction='sum'):
    Y=Y.to(Z.dtype)
    return nnF.binary_cross_entropy_with_logits(Z, Y, reduction=reduction)
#%%
def logit_margin_loss_binary(Z, Y, reduction='sum'):
    Y=Y.to(Z.dtype)
    t = Y*2-1
    loss =- Z*t
    return loss_reduction(loss, reduction)
#%%
def logit_margin_loss_old(Z, Y, reduction='sum'):
    num_classes=Z.size(1)
    Zy=torch.gather(Z, 1, Y[:,None])
    Zy=Zy.view(-1)
    idxTable=torch.zeros((Z.size(0), num_classes-1), dtype=torch.int64)
    idxlist = torch.arange(0, num_classes, dtype=torch.int64)
    Ycpu=Y.cpu()
    for n in range(Z.size(0)):
        idxTable[n]=idxlist[idxlist!=Ycpu[n]]
    Zother=torch.gather(Z, 1, idxTable.to(Z.device))
    Zother_max = Zother.max(dim=1)[0]
    loss=Zother_max-Zy
    return loss_reduction(loss, reduction)
#%%
def logit_margin_loss(Z, Y, reduction='sum'):
    # Z.shape is (N,C) or (N,C,d1,d2,...,dK)
    N=Z.shape[0]
    C=Z.shape[1]
    #-------------------------------
    # Z.shape is (N,C), Y.shape is (N,)
    if len(Z.shape)==2:
        val, idx=Z.topk(2, dim=1)
        Zy=Z[torch.arange(N), Y]
        Zother_max=((Y != idx[:, 0]).to(Z.dtype) * val[:, 0]
                   +(Y == idx[:, 0]).to(Z.dtype) * val[:, 1])
        loss=Zother_max-Zy
        return loss_reduction(loss, reduction)
    #------------------------------
    # Z.shape is (N,C,d1,d2,...,dK)
    # Y.shape is (N,1,d1,d2,...,dK)
    ZZ=Z.unsqueeze(len(Z.shape))
    ZZ=ZZ.transpose(1, len(ZZ.shape)-1).reshape(-1, C)
    YY=Y.view(-1)
    val, idx=ZZ.topk(2, dim=1)
    ZZy=ZZ[torch.arange(ZZ.shape[0]), YY]
    ZZother_max=((YY != idx[:, 0]).to(ZZ.dtype) * val[:, 0]
                +(YY == idx[:, 0]).to(ZZ.dtype) * val[:, 1])
    loss=ZZother_max-ZZy
    loss=loss.view(Y.shape)
    #print(loss.shape)
    return loss_reduction(loss, reduction)
#%%
def soft_logit_margin_loss_old(Z, Y, reduction='sum'):
    num_classes=Z.size(1)
    if len(Y.shape)==1:
        Y=Y.view(-1,1)
    Zy=torch.gather(Z, 1, Y)
    Zy=Zy.view(-1)
    idxTable=torch.zeros((Z.size(0), num_classes-1), dtype=torch.int64)
    idxlist = torch.arange(0, num_classes, dtype=torch.int64)
    Ycpu=Y.cpu()
    for n in range(Z.size(0)):
        idxTable[n]=idxlist[idxlist!=Ycpu[n]]
    Zother=torch.gather(Z, 1, idxTable.to(Z.device))
    loss=torch.logsumexp(Zother, dim=1)-Zy
    return loss_reduction(loss, reduction)
#%%
def soft_logit_margin_loss(Z, Y, reduction='sum'):
    # Z.shape is (N,C) or (N,C,d1,d2,...,dK)
    N=Z.shape[0]
    C=Z.shape[1]
    #-------------------------------
    # Z.shape is (N,C), Y.shape is (N,)
    if len(Z.shape)==2:
        Zy=Z[torch.arange(N), Y]
        mask=torch.ones_like(Z).bool()
        mask[torch.arange(N), Y]=False
        Zother=Z[mask].view(N, C-1)
        loss=torch.logsumexp(Zother, dim=1)-Zy
        return loss_reduction(loss, reduction)
    #------------------------------
    # Z.shape is (N,C,d1,d2,...,dK)
    # Y.shape is (N,1,d1,d2,...,dK)
    ZZ=Z.unsqueeze(len(Z.shape))
    ZZ=ZZ.transpose(1, len(ZZ.shape)-1).reshape(-1, C)
    YY=Y.view(-1)
    ZZy=ZZ[torch.arange(ZZ.shape[0]), YY]
    mask=torch.ones_like(ZZ).bool()
    mask[torch.arange(ZZ.shape[0]), YY]=False
    ZZother=ZZ[mask].view(ZZ.shape[0], C-1)
    loss=torch.logsumexp(ZZother, dim=1)-ZZy
    loss=loss.view(Y.shape)
    return loss_reduction(loss, reduction)    
#%%
def get_pgd_loss_fn_by_name(loss_fn):
    if loss_fn is None:
        loss_fn=torch.nn.CrossEntropyLoss(reduction="sum")
    elif isinstance(loss_fn, str):
        if loss_fn == 'none' or loss_fn == 'ce':
            loss_fn=torch.nn.CrossEntropyLoss(reduction="sum")
        elif loss_fn == 'bce':
            #loss_fn=torch.nn.BCEWithLogitsLoss(reduction="sum")
            loss_fn=binary_cross_entropy_with_logits
        elif loss_fn =='logit_margin_loss_binary' or loss_fn == 'lmb':
            loss_fn=logit_margin_loss_binary
        elif loss_fn == 'logit_margin_loss' or loss_fn == 'lm':
            loss_fn=logit_margin_loss
        elif loss_fn == 'soft_logit_margin_loss' or loss_fn == 'slm':
            loss_fn=soft_logit_margin_loss
        else:
            raise NotImplementedError("not implemented.")
    return loss_fn
#%%
def clip_norm_(noise, norm_type, norm_max):
    if not isinstance(norm_max, torch.Tensor):
        clip_normA_(noise, norm_type, norm_max)
    else:
        clip_normB_(noise, norm_type, norm_max)
#%%
def clip_normA_(noise, norm_type, norm_max):
    # noise is a tensor modified in place, noise.size(0) is batch_size
    # norm_type can be np.inf, 1 or 2, or p
    # norm_max is a scalar noise level
    if noise.size(0) == 0:
        return noise
    with torch.no_grad():
        if norm_type == np.inf or norm_type == 'Linf':
            noise.clamp_(-norm_max, norm_max)
        elif norm_type == 2 or norm_type == 'L2':
            N=noise.view(noise.size(0), -1)
            l2_norm= torch.sqrt(torch.sum(N**2, dim=1, keepdim=True))
            temp = (l2_norm > norm_max).squeeze()
            if temp.sum() > 0:
                N[temp]*=norm_max/l2_norm[temp]
        else:
            raise NotImplementedError("other norm clip is not implemented.")
    #-----------
    return noise
#%%
def clip_normB_(noise, norm_type, norm_max):
    # noise is a tensor modified in place, noise.size(0) is batch_size
    # norm_type can be np.inf, 1 or 2, or p
    # norm_max is 1D tensor, norm_max[k] is the maximum noise level for noise[k]
    if noise.size(0) == 0:
        return noise
    with torch.no_grad():
        if norm_type == np.inf or norm_type == 'Linf':
            #for k in range(noise.size(0)):
            #    noise[k].clamp_(-norm_max[k], norm_max[k])
            N=noise.view(noise.size(0), -1)
            norm_max=norm_max.view(norm_max.size(0), -1)
            N=torch.max(torch.min(N, norm_max), -norm_max)
            N=N.view(noise.size())
            noise-=noise-N
        elif norm_type == 2 or norm_type == 'L2':
            N=noise.view(noise.size(0), -1)
            l2_norm= torch.sqrt(torch.sum(N**2, dim=1, keepdim=True))
            norm_max=norm_max.view(norm_max.size(0), 1)
            #print(l2_norm.shape, norm_max.shape)
            temp = (l2_norm > norm_max).squeeze()
            if temp.sum() > 0:
                norm_max=norm_max[temp]
                norm_max=norm_max.view(norm_max.size(0), -1)
                N[temp]*=norm_max/l2_norm[temp]
        else:
            raise NotImplementedError("not implemented.")
        #-----------
    return noise
#%%
def add_noise_to_X_and_clip_norm_(noise, norm_type, norm_max, X, clip_X_min=0, clip_X_max=1):
    #noise and X are modified in place
    if X.size(0) == 0:
        return noise, X
    with torch.no_grad():
        clip_norm_(noise, norm_type, norm_max)
        Xnew = torch.clamp(X+noise, clip_X_min, clip_X_max)
        noise -= noise-(Xnew-X)
        X -= X-Xnew
    return noise, X
#%%
def normalize_grad_(x_grad, norm_type, eps=1e-12):
    #x_grad is modified in place
    #x_grad.size(0) is batch_size
    #eps=1e-12 is from cleverhans/attacks/fast_gradient_method.py
    with torch.no_grad():
        if norm_type == np.inf or norm_type == 'Linf':
            x_grad-=x_grad-x_grad.sign()
        elif norm_type == 2 or norm_type == 'L2':
            g=x_grad.view(x_grad.size(0), -1)
            l2_norm=torch.sqrt(torch.sum(g**2, dim=1, keepdim=True))
            l2_norm = torch.max(l2_norm, torch.tensor(eps, dtype=l2_norm.dtype, device=l2_norm.device))
            g *= 1/l2_norm
        else:
            raise NotImplementedError("not implemented.")
    return x_grad
#%%
def normalize_noise_(noise, norm_type, eps=1e-12):
    if noise.size(0) == 0:
        return noise
    with torch.no_grad():
        N=noise.view(noise.size(0), -1)
        if norm_type == np.inf or norm_type == 'Linf':
            linf_norm=N.abs().max(dim=1, keepdim=True)[0]
            N *= 1/(linf_norm+eps)
        elif norm_type == 2 or norm_type == 'L2':
            l2_norm=torch.sqrt(torch.sum(N**2, dim=1, keepdim=True))
            l2_norm = torch.max(l2_norm, torch.tensor(eps, dtype=l2_norm.dtype, device=l2_norm.device))
            N *= 1/l2_norm
        else:
            raise NotImplementedError("not implemented.")
    return noise
#%%
def get_noise_init(norm_type, noise_norm, init_norm, X):
    #noise_norm is a saclar or a 1D tensor (noise_norm.shape[0] is X.shape[0])
    #init_norm is a saclar or a 1D tensor  (init_norm.shape[0] is X.shape[0])
    if isinstance(noise_norm, torch.Tensor):
        if isinstance(init_norm, torch.Tensor):
            norm_max=torch.max(noise_norm, init_norm)
        else:
            norm_max=torch.clamp(noise_norm, max=init_norm)
    else:
        if isinstance(init_norm, torch.Tensor):
            norm_max=torch.clamp(init_norm, max=noise_norm)
        else:
            norm_max=max(noise_norm, init_norm)
    noise_init=2*torch.rand_like(X)-1
    if isinstance(norm_max, torch.Tensor):
        noise_init=norm_max.view(X.shape[0],-1)*noise_init.view(X.shape[0],-1)
    else:
        noise_init=norm_max*noise_init.view(X.shape[0],-1)
    noise_init=noise_init.view(X.size())
    clip_norm_(noise_init, norm_type, norm_max)
    return noise_init
#%%
def ifgsm_attack(model, X, Y, noise, max_iter=None, step=None,
                 clip_X_min=0, clip_X_max=1,
                 model_eval_attack=False, return_Xn_grad=False):
    if max_iter is None and step is None and clip_X_min==0 and clip_X_max==1:
        max_iter=int(min(255*noise+4, 1.25*255*noise))
        step=1/255
    #-----------------------------------------
    train_mode=model.training# record the mode
    if model_eval_attack == True:
        model.eval()#set model to evaluation mode
    Xn = X.clone().detach()
    for n in range(0, max_iter):
        Xn.requires_grad = True
        #if Xn.grad is not None:
        #    print(Xn.grad.abs().sum().item())
        Zn = model(Xn)
        if len(Zn.size()) <= 1:
            loss = nnF.binary_cross_entropy_with_logits(Zn, Y.to(X.dtype))
        else:
            loss = nnF.cross_entropy(Zn, Y)
        #loss.backward() will update dLdW
        Xn_grad=torch.autograd.grad(loss, Xn)[0]
        Xn = Xn.detach() + step*Xn_grad.sign().detach()
        Xn = torch.clamp(Xn, clip_X_min, clip_X_max)
        Xn = Xn.detach()
    if model_eval_attack == True and train_mode == True:
        model.train()
    if return_Xn_grad == False:
        return Xn
    else:
        return Xn, Xn_grad
#%%
def pgd_attack(model, X, Y, noise_norm, norm_type, max_iter, step,
               rand_init=True, rand_init_norm=None, targeted=False,
               clip_X_min=0, clip_X_max=1, use_optimizer=False,
               loss_fn=None, model_eval_attack=False):
    #-----------------------------------------------------
    loss_fn=get_pgd_loss_fn_by_name(loss_fn)
    #-----------------------------------------------------
    train_mode=model.training# record the mode
    if model_eval_attack == True and train_mode == True:
        model.eval()#set model to evaluation mode
    #-----------------
    X = X.detach()
    #-----------------
    if rand_init == True:
        init_norm=rand_init_norm
        if rand_init_norm is None:
            init_norm=noise_norm
        noise_init=get_noise_init(norm_type, noise_norm, init_norm, X)
        Xn = torch.clamp(X+noise_init, clip_X_min, clip_X_max)
    else:
        Xn = X.clone().detach() # must clone
    #-----------------
    noise_new=(Xn-X).detach()
    if use_optimizer == True:
        optimizer = optim.Adamax([noise_new], lr=step)
    #-----------------
    for n in range(0, max_iter):
        Xn = Xn.detach()
        Xn.requires_grad = True
        Zn = model(Xn)
        loss = loss_fn(Zn, Y)
        #---------------------------
        if targeted == True:
            loss=-loss
        #---------------------------
        #loss.backward() will update W.grad
        grad_n=torch.autograd.grad(loss, Xn)[0]
        grad_n=normalize_grad_(grad_n, norm_type)
        if use_optimizer == True:
            noise_new.grad=-grad_n.detach() #grad ascent to maximize loss
            optimizer.step()
        else:
            Xnew = Xn.detach() + step*grad_n.detach()
            noise_new = Xnew-X
        #---------------------
        clip_norm_(noise_new, norm_type, noise_norm)
        Xn = torch.clamp(X+noise_new, clip_X_min, clip_X_max)
        noise_new.data -= noise_new.data-(Xn-X).data
        Xn=Xn.detach()
    #---------------------------
    if train_mode == True and model.training == False:
        model.train()
    #---------------------------
    return Xn
#%%
def repeated_pgd_attack(model, X, Y, noise_norm, norm_type, max_iter, step,
                        rand_init=True, rand_init_norm=None, targeted=False,
                        clip_X_min=0, clip_X_max=1, use_optimizer=False,
                        loss_fn=None, model_eval_attack=False,
                        num_repeats=1):
    for m in range(0, num_repeats):
        Xm = pgd_attack(model, X, Y, noise_norm, norm_type, max_iter, step,
                        rand_init, rand_init_norm, targeted,
                        clip_X_min, clip_X_max, use_optimizer,
                        loss_fn, model_eval_attack)
        if m == 0:
            Xn=Xm
        else:
            Zm=model(Xm)
            if len(Zm.size()) <= 1:
                Ypm = (Zm.data>0).to(torch.int64)
            else:
                Ypm = Zm.data.max(dim=1)[1]
            if targeted==False:
                adv=Ypm!=Y
            else:
                adv=Ypm==Y
            Xn[adv]=Xm[adv].data
    #--------
    return Xn
#%%