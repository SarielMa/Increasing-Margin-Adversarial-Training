# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 20:15:47 2019

@author: liang
"""
import numpy as np
import torch
from torch import optim
import torch.nn.functional as nnF
from RobustDNN_PGD import pgd_attack as pgd_attack
#%%
def run_model(model, X):
    Z=model(X)
    if len(Z.size()) <= 1:
        Yp = (Z.data>0).to(torch.int64)
    else:
        Yp = Z.data.max(dim=1)[1]
    return Z, Yp
#%%
def IAAT_loss(model, X, Y, margin, norm_type, max_iter, step, gamma, beta,
              clip_margin_min , clip_margin_max, clip_X_min=0, clip_X_max=1,
              use_optimizer=False, pgd_loss_fn=None,
              model_eval_attack=False, model_eval_Xn=False):
    if isinstance(step, torch.Tensor):
        temp=tuple([1]*len(X[0].size()))
        step=step.view(-1, *temp)
    #----------------------------
    Z, Yp=run_model(model, X)
    if len(Z.size()) <= 1:
        loss_fn=torch.nn.BCEWithLogitsLoss(reduction="sum")
    else:
        loss_fn=torch.nn.CrossEntropyLoss(reduction="sum")
    #----------------------------
    Yp_e_Y=Yp==Y
    Yp_ne_Y=Yp!=Y
    #----------------------------
    train_mode=model.training# record the mode
    if model_eval_attack == True and train_mode == True:
        model.eval()#freeze BN, etc
    #----------------------------
    Xid=torch.arange(0, X.size(0), device=X.device)
    margin_new=margin.clone().detach()
    #----------------------------
    Xn1 = pgd_attack(model, X, Y,
                     norm_type=norm_type, noise_norm=margin+gamma,
                     max_iter=max_iter, step=step,
                     clip_X_min=clip_X_min, clip_X_max=clip_X_max,
                     use_optimizer=use_optimizer, loss_fn=pgd_loss_fn)
    Zn1, Ypn1=run_model(model, Xn1)
    margin_new[Ypn1==Y]+=gamma
    margin_new.clamp_(max=clip_margin_max)
    #--------------------------------------------
    Ypn1_ne_Y = Ypn1!=Y
    if Ypn1_ne_Y.sum().item()>0 and beta>0:
        step_=step
        if isinstance(step, torch.Tensor):
            step_=step[Ypn1_ne_Y]
        Xn2 = pgd_attack(model, X[Ypn1_ne_Y], Y[Ypn1_ne_Y],
                         norm_type=norm_type, noise_norm=margin[Ypn1_ne_Y],
                         max_iter=max_iter, step=step_,
                         clip_X_min=clip_X_min, clip_X_max=clip_X_max,
                         use_optimizer=use_optimizer, loss_fn=pgd_loss_fn)
        Zn2, Ypn2=run_model(model, Xn2)
        Ypn2_ne_Y = Ypn2!=Y[Ypn1_ne_Y]
        if Ypn2_ne_Y.sum().item()>0:
            temp=Xid[Ypn1_ne_Y][Ypn2_ne_Y]
            margin_new[temp]-=gamma
            margin_new.clamp_(min=clip_margin_min)
    #--------------------------------------------
    margin_new=(1-beta)*margin+beta*margin_new
    Xn = pgd_attack(model, X, Y,
                    norm_type=norm_type, noise_norm=margin_new,
                    max_iter=max_iter, step=step,
                    clip_X_min=clip_X_min, clip_X_max=clip_X_max,
                    use_optimizer=use_optimizer, loss_fn=pgd_loss_fn)
    #--------------------------------------------
    if model_eval_Xn == True:
        if model.training == True:
            model.eval()
    else:
        if train_mode == True and model.training == False:
            model.train()
    #--------------------------------------------
    Zn, Ypn=run_model(model, Xn)
    loss1=0
    if Yp_e_Y.sum().item()>0:
        loss1=loss_fn(Zn[Yp_e_Y], Y[Yp_e_Y])
    loss2=0
    if Yp_ne_Y.sum().item()>0:
        loss2=loss_fn(Z[Yp_ne_Y], Y[Yp_ne_Y])
    loss=(loss1+loss2)/X.size(0)
    #--------------------------------------------
    if train_mode == True and model.training == False:
        model.train()
    #--------------------------------------------
    return loss, margin_new, Yp, Ypn