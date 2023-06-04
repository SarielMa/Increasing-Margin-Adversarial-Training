# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 20:15:47 2019

@author: liang
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import optim
from RobustDNN_PGD import get_pgd_loss_fn_by_name, get_noise_init, normalize_grad_, clip_norm_
#%%
def run_model(model, X):
    Z=model(X)
    if len(Z.size()) <= 1:
        Yp = (Z.data>0).to(torch.int64)
    else:
        Yp = Z.data.max(dim=1)[1]
    return Z, Yp
#%%
def pgd_attack(model, X, Y, noise_norm, norm_type, max_iter, step,
               rand_init_norm=None, rand_init_Xn=None,
               targeted=False, clip_X_min=0, clip_X_max=1,
               refine_Xn_max_iter=10,
               Xn1_equal_X=False,
               Xn2_equal_Xn=False,
               stop_near_boundary=False,
               stop_if_label_change=False,
               stop_if_label_change_next_step=False,
               use_optimizer=False,
               loss_fn=None,
               model_eval_attack=False):
    #-------------------------------------------
    loss_fn=get_pgd_loss_fn_by_name(loss_fn)
    #-------------------------------------------
    train_mode=model.training# record the mode
    if model_eval_attack == True and train_mode == True:
        model.eval()#set model to evaluation mode
    #-----------------
    X = X.detach()
    #-----------------
    advc=torch.zeros(X.size(0), dtype=torch.int64, device=X.device)
    #-----------------
    if rand_init_norm is not None:
        noise_init=get_noise_init(norm_type, noise_norm, rand_init_norm, X)
        Xn = torch.clamp(X+noise_init, clip_X_min, clip_X_max)
    elif rand_init_Xn is not None:
        Xn = torch.clamp(rand_init_Xn.detach(), clip_X_min, clip_X_max)
    else:
        raise ValueError('invalid input')
    #-----------------
    Xn1=X.detach().clone() # about to across decision boundary
    Xn2=X.detach().clone() # just across decision boundary
    Ypn_old=Y # X is correctly classified
    #-----------------
    noise=(Xn-X).detach()
    if use_optimizer == True:
        optimizer = optim.Adamax([noise], lr=step)
    #-----------------
    for n in range(0, max_iter+1):
        Xn = Xn.detach()
        Xn.requires_grad = True
        Zn, Ypn=run_model(model, Xn)
        loss = loss_fn(Zn, Y)
        Ypn_e_Y=(Ypn==Y)
        Ypn_ne_Y=(Ypn!=Y)
        Ypn_old_e_Y=(Ypn_old==Y)
        Ypn_old_ne_Y=(Ypn_old!=Y)
        #---------------------------
        #targeted attack, Y should be filled with targeted class label
        if targeted == False:
            A=Ypn_e_Y
            A_old=Ypn_old_e_Y
            B=Ypn_ne_Y
        else:
            A=Ypn_ne_Y
            A_old=Ypn_old_ne_Y
            B=Ypn_e_Y
            loss=-loss
        #---------------------------
        temp1=(A&A_old)&(advc<1)
        Xn1[temp1]=Xn[temp1].data
        temp2=(B&A_old)&(advc<1)
        Xn2[temp1]=Xn[temp1].data
        Xn2[temp2]=Xn[temp2].data
        advc[B]+=1
        #---------------------------
        if n < max_iter:
            #loss.backward() will update W.grad
            grad_n=torch.autograd.grad(loss, Xn)[0]
            grad_n=normalize_grad_(grad_n, norm_type)
            if use_optimizer == True:
                noise.grad=-grad_n #grad ascent to maximize loss
                optimizer.step()
            else:
                Xnew = Xn + step*grad_n
                noise = Xnew-X
            #---------------------
            clip_norm_(noise, norm_type, noise_norm)
            Xn = torch.clamp(X+noise, clip_X_min, clip_X_max)
            noise.data -= noise.data-(Xn-X).data
            Ypn_old=Ypn
    #---------------------------
    Xn_out = Xn.detach()
    if Xn1_equal_X:
        Xn1=X.detach().clone()
    if Xn2_equal_Xn:
        Xn2=Xn
    if stop_near_boundary == True:
        temp=advc>0
        if temp.sum()>0:
            Xn_out=refine_Xn_onto_boundary(model, Xn1, Xn2, Y, refine_Xn_max_iter)
    elif stop_if_label_change == True:
        temp=advc>0
        if temp.sum()>0:
            Xn_out=refine_Xn2_onto_boundary(model, Xn1, Xn2, Y, refine_Xn_max_iter)
    elif stop_if_label_change_next_step == True:
        temp=advc>0
        if temp.sum()>0:
            Xn_out=refine_Xn1_onto_boundary(model, Xn1, Xn2, Y, refine_Xn_max_iter)
    #---------------------------
    if train_mode == True and model.training == False:
        model.train()
    #---------------------------
    return Xn_out, advc

#%%
def TRADES_attack(model, X, Y, noise_norm, norm_type, max_iter, step,
               rand_init_norm=None, rand_init_Xn=None,
               targeted=False, clip_X_min=0, clip_X_max=1,
               refine_Xn_max_iter=10,
               Xn1_equal_X=False,
               Xn2_equal_Xn=False,
               stop_near_boundary=False,
               stop_if_label_change=False,
               stop_if_label_change_next_step=False,
               use_optimizer=False,
               loss_fn=None,
               model_eval_attack=False,
               loss_opt='ce_z'):
    #-------------------------------------------
    loss_fn=get_pgd_loss_fn_by_name(loss_fn)
    #-------------------------------------------
    train_mode=model.training# record the mode
    if model_eval_attack == True and train_mode == True:
        model.eval()#set model to evaluation mode
    #-----------------
    X = X.detach()
    #-----------------
    advc=torch.zeros(X.size(0), dtype=torch.int64, device=X.device)
    #-----------------
    if rand_init_norm is not None:
        noise_init=get_noise_init(norm_type, noise_norm, rand_init_norm, X)
        Xn = torch.clamp(X+noise_init, clip_X_min, clip_X_max)
    elif rand_init_Xn is not None:
        Xn = torch.clamp(rand_init_Xn.detach(), clip_X_min, clip_X_max)
    else:
        raise ValueError('invalid input')
    #-----------------
    Xn1=X.detach().clone() # about to across decision boundary
    Xn2=X.detach().clone() # just across decision boundary
    Ypn_old=Y # X is correctly classified
    #-----------------
    noise=(Xn-X).detach()
    if use_optimizer == True:
        optimizer = optim.Adamax([noise], lr=step)
    #-----------------
    for n in range(0, max_iter+1):
        Xn = Xn.detach()
        Xn.requires_grad = True
        Zn, Ypn=run_model(model, Xn)
        #loss = loss_fn(Zn, Y)
        if loss_opt == 'ce_y':
            loss = loss_fn(Zn, Y)
        elif loss_opt == 'ce_z':
            loss = cross_entropy_z(Zn, Z, reduction='mean')
        elif loss_opt == 'mse_z':
            z_size=Z.size()
            if len(z_size) <=1:
                loss=((Zn-Z)**2).mean()
            else:
                loss=((Zn-Z)**2).mean()
        else:
            raise NotImplementedError        
           
        
        Ypn_e_Y=(Ypn==Y)
        Ypn_ne_Y=(Ypn!=Y)
        Ypn_old_e_Y=(Ypn_old==Y)
        Ypn_old_ne_Y=(Ypn_old!=Y)
        #---------------------------
        #targeted attack, Y should be filled with targeted class label
        if targeted == False:
            A=Ypn_e_Y
            A_old=Ypn_old_e_Y
            B=Ypn_ne_Y
        else:
            A=Ypn_ne_Y
            A_old=Ypn_old_ne_Y
            B=Ypn_e_Y
            loss=-loss
        #---------------------------
        temp1=(A&A_old)&(advc<1)
        Xn1[temp1]=Xn[temp1].data
        temp2=(B&A_old)&(advc<1)
        Xn2[temp1]=Xn[temp1].data
        Xn2[temp2]=Xn[temp2].data
        advc[B]+=1
        #---------------------------
        if n < max_iter:
            #loss.backward() will update W.grad
            grad_n=torch.autograd.grad(loss, Xn)[0]
            grad_n=normalize_grad_(grad_n, norm_type)
            if use_optimizer == True:
                noise.grad=-grad_n #grad ascent to maximize loss
                optimizer.step()
            else:
                Xnew = Xn + step*grad_n
                noise = Xnew-X
            #---------------------
            clip_norm_(noise, norm_type, noise_norm)
            Xn = torch.clamp(X+noise, clip_X_min, clip_X_max)
            noise.data -= noise.data-(Xn-X).data
            Ypn_old=Ypn
    #---------------------------
    Xn_out = Xn.detach()
    if Xn1_equal_X:
        Xn1=X.detach().clone()
    if Xn2_equal_Xn:
        Xn2=Xn
    if stop_near_boundary == True:
        temp=advc>0
        if temp.sum()>0:
            Xn_out=refine_Xn_onto_boundary(model, Xn1, Xn2, Y, refine_Xn_max_iter)
    elif stop_if_label_change == True:
        temp=advc>0
        if temp.sum()>0:
            Xn_out=refine_Xn2_onto_boundary(model, Xn1, Xn2, Y, refine_Xn_max_iter)
    elif stop_if_label_change_next_step == True:
        temp=advc>0
        if temp.sum()>0:
            Xn_out=refine_Xn1_onto_boundary(model, Xn1, Xn2, Y, refine_Xn_max_iter)
    #---------------------------
    if train_mode == True and model.training == False:
        model.train()
    #---------------------------
    return Xn_out, advc
#%%
def refine_onto_boundary(model, Xn1, Xn2, Y, max_iter):
#note: Xn1 and Xn2 will be modified
    with torch.no_grad():
        Xn=(Xn1+Xn2)/2
        for k in range(0, max_iter):
            Zn, Ypn=run_model(model, Xn)
            Ypn_e_Y=Ypn==Y
            Ypn_ne_Y=Ypn!=Y
            Xn1[Ypn_e_Y]=Xn[Ypn_e_Y]
            Xn2[Ypn_ne_Y]=Xn[Ypn_ne_Y]
            Xn=(Xn1+Xn2)/2
    return Xn, Xn1, Xn2
#%%
def refine_Xn_onto_boundary(model, Xn1, Xn2, Y, max_iter):
#note: Xn1 and Xn2 will be modified
    Xn, Xn1, Xn2=refine_onto_boundary(model, Xn1, Xn2, Y, max_iter)
    return Xn
#%%
def refine_Xn1_onto_boundary(model, Xn1, Xn2, Y, max_iter):
#note: Xn1 and Xn2 will be modified
    Xn, Xn1, Xn2=refine_onto_boundary(model, Xn1, Xn2, Y, max_iter)
    return Xn1
#%%
def refine_Xn2_onto_boundary(model, Xn1, Xn2, Y, max_iter):
#note: Xn1 and Xn2 will be modified
    Xn, Xn1, Xn2=refine_onto_boundary(model, Xn1, Xn2, Y, max_iter)
    return Xn2
#%%
def repeated_pgd_attack(model, X, Y, noise_norm, norm_type, max_iter, step,
                        rand_init_norm=None, rand_init_Xn=None,
                        targeted=False, clip_X_min=0, clip_X_max=1,
                        refine_Xn_max_iter=10,
                        Xn1_equal_X=False,
                        Xn2_equal_Xn=False,
                        stop_near_boundary=False,
                        stop_if_label_change=False,
                        stop_if_label_change_next_step=False,
                        use_optimizer=False,
                        loss_fn=None,
                        model_eval_attack=False,
                        num_repeats=1):
    for m in range(0, num_repeats):
        Xm, advcm = pgd_attack(model, X, Y, noise_norm, norm_type, max_iter, step,
                               rand_init_norm, rand_init_Xn,
                               targeted, clip_X_min, clip_X_max,
                               refine_Xn_max_iter,
                               Xn1_equal_X,
                               Xn2_equal_Xn,
                               stop_near_boundary,
                               stop_if_label_change,
                               stop_if_label_change_next_step,
                               use_optimizer,
                               loss_fn,
                               model_eval_attack)
        if m == 0:
            Xn=Xm
            advc=advcm
        else:
            temp=advcm>0
            advc[temp]=advcm[temp]
            Xn[temp]=Xm[temp]
    #--------
    return Xn, advc
#%%
def get_loss_function(z_size, reduction='sum'):
    if len(z_size) <=1:
        return torch.nn.BCEWithLogitsLoss(reduction=reduction)
    else:
        return torch.nn.CrossEntropyLoss(reduction=reduction)
    
def get_loss_function2(z_size, reduction='sum'):
    if len(z_size) <=1:
        return torch.nn.BCEWithLogitsLoss(reduction=reduction)
    else:
        return torch.nn.CrossEntropyLoss(reduction=reduction)
#%%  
def IMA_exploration(model, X, Y, margin, norm_type, max_iter, step,
             rand_init_norm=None, rand_init_Xn=None,
             clip_X_min=0, clip_X_max=1,
             refine_Xn_max_iter=10,
             Xn1_equal_X=False,
             Xn2_equal_Xn=False,
             stop_near_boundary=False,
             stop_if_label_change=False,
             stop_if_label_change_next_step=False,
             use_optimizer=False,
             beta=0.5, beta_position=1,
             pgd_loss_fn=None,
             pgd_num_repeats=1,
             model_eval_attack=False,
             model_eval_Xn=False,
             model_Xn_advc_p=False):
    
    #----------------------------------
    if isinstance(step, torch.Tensor):
        if use_optimizer:
            raise ValueError('incompatible')
        else:
            step=step.view(-1, *tuple([1]*len(X[0].size())))
    Xn, _ = repeated_pgd_attack(model, X, Y,
                            noise_norm=margin, norm_type=norm_type,
                            max_iter=max_iter, step=step,
                            rand_init_norm=rand_init_norm, rand_init_Xn=rand_init_Xn,
                            clip_X_min=clip_X_min, clip_X_max=clip_X_max,
                            refine_Xn_max_iter=refine_Xn_max_iter,
                            Xn1_equal_X=Xn1_equal_X,
                            Xn2_equal_Xn=Xn2_equal_Xn,
                            stop_near_boundary=stop_near_boundary,
                            stop_if_label_change=stop_if_label_change,
                            stop_if_label_change_next_step=stop_if_label_change_next_step,
                            use_optimizer=use_optimizer,
                            loss_fn=pgd_loss_fn,
                            num_repeats=pgd_num_repeats)

    return Xn

#%%
def IMA_loss(model, X, Y, margin, norm_type, max_iter, step,
             rand_init_norm=None, rand_init_Xn=None,
             clip_X_min=0, clip_X_max=1,
             refine_Xn_max_iter=10,
             Xn1_equal_X=False,
             Xn2_equal_Xn=False,
             stop_near_boundary=False,
             stop_if_label_change=False,
             stop_if_label_change_next_step=False,
             use_optimizer=False,
             beta=0.5, beta_position=1,
             pgd_loss_fn=None,
             pgd_num_repeats=1,
             model_eval_attack=False,
             model_eval_Xn=False,
             model_Xn_advc_p=False,
             loss3_opt='ce_y'):
    #----------------------------------
    if isinstance(step, torch.Tensor):
        if use_optimizer:
            raise ValueError('incompatible')
        else:
            step=step.view(-1, *tuple([1]*len(X[0].size())))
    #-----------------------------------
    Z, Yp=run_model(model, X)
    Yp_e_Y=Yp==Y.to(torch.int64)
    Yp_ne_Y=Yp!=Y.to(torch.int64)
    #-----------------------------------
    loss_fn=get_loss_function(Z.size())
    loss1=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    loss2=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    loss3=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    Xn=torch.tensor([], dtype=X.dtype, device=X.device)
    Ypn=torch.tensor([], dtype=Y.dtype, device=X.device)
    advc=torch.zeros(X.size(0), dtype=torch.int64, device=X.device)
    idx_n=torch.tensor([], dtype=torch.int64, device=X.device)
    #----------------------------------
    if Yp_ne_Y.sum().item()>0:
        loss1 = loss_fn(Z[Yp_ne_Y], Y[Yp_ne_Y])/X.size(0)
    if Yp_e_Y.sum().item()>0:
        loss2 = loss_fn(Z[Yp_e_Y], Y[Yp_e_Y])/X.size(0)
    #---------------------------------
    train_mode=model.training# record the mode
    if model_eval_attack == True and train_mode == True:
        model.eval()#BN, dropout, etc
        #re-run the model in eval mode
        _Z_, _Yp_=run_model(model, X)
        Yp_e_Y=_Yp_==Y.to(torch.int64)
        Yp_ne_Y=_Yp_!=Y.to(torch.int64)
    #---------------------------------
    enable_loss3=False
    if Yp_e_Y.sum().item()>0 and beta>0:
         enable_loss3=True
    #----------------------------------
    if enable_loss3 == True:
        Xn, advc = repeated_pgd_attack(model, X, Y,
                                       noise_norm=margin, norm_type=norm_type,
                                       max_iter=max_iter, step=step,
                                       rand_init_norm=rand_init_norm, rand_init_Xn=rand_init_Xn,
                                       clip_X_min=clip_X_min, clip_X_max=clip_X_max,
                                       refine_Xn_max_iter=refine_Xn_max_iter,
                                       Xn1_equal_X=Xn1_equal_X,
                                       Xn2_equal_Xn=Xn2_equal_Xn,
                                       stop_near_boundary=stop_near_boundary,
                                       stop_if_label_change=stop_if_label_change,
                                       stop_if_label_change_next_step=stop_if_label_change_next_step,
                                       use_optimizer=use_optimizer,
                                       loss_fn=pgd_loss_fn,
                                       num_repeats=pgd_num_repeats)
        #--------------------------------------------
        if model_eval_Xn == True:
            if model.training == True:
                model.eval()
        else:
            if train_mode == True and model.training == False:
                model.train()
        #--------------------------------------------
        
        Zn, Ypn=run_model(model, Xn)
        if model_Xn_advc_p == True:
            idx_n=torch.arange(0,X.size(0))[(advc>0)&(Yp_e_Y)]
        else:
            idx_n=torch.arange(0,X.size(0))[Yp_e_Y]
        # to be consistant with the output of IMA_loss in RobustDNN_IMA
        Xn=Xn[idx_n]; Zn=Zn[idx_n]; Ypn=Ypn[idx_n]
        if idx_n.size(0)>0:
            loss3 = loss_fn(Zn, Y[idx_n])/X.size(0)
    #--------------------------------------------
    if beta_position == 0:
        loss=(1-beta)*loss1+(beta*0.5)*(loss2+loss3)
    elif beta_position == 1:
        loss=(1-beta)*(loss1+loss2)+beta*loss3
    elif beta_position == 2:
        loss=loss1+(1-beta)*loss2+beta*loss3
    elif beta_position == 3:
        loss=(1-beta)*loss1+beta*loss3
    else:
        raise ValueError('unknown beta_position')
    #--------------------------------------------
    if train_mode == True and model.training == False:
        model.train()
    #--------------------------------------------
    return loss, loss1, loss2, loss3, Yp, advc, Xn, Ypn, idx_n

#%%
def IMA_loss_grad_weighted(model, X, Y, margin, norm_type, max_iter, step,
             rand_init_norm=None, rand_init_Xn=None,
             clip_X_min=0, clip_X_max=1,
             refine_Xn_max_iter=10,
             Xn1_equal_X=False,
             Xn2_equal_Xn=False,
             stop_near_boundary=False,
             stop_if_label_change=False,
             stop_if_label_change_next_step=False,
             use_optimizer=False,
             beta=0.5, beta_position=1,
             pgd_loss_fn=None,
             pgd_num_repeats=1,
             model_eval_attack=False,
             model_eval_Xn=False,
             model_Xn_advc_p=False):
    #----------------------------------
    if isinstance(step, torch.Tensor):
        if use_optimizer:
            raise ValueError('incompatible')
        else:
            step=step.view(-1, *tuple([1]*len(X[0].size())))
    #-----------------------------------
    Z, Yp=run_model(model, X)
    Yp_e_Y=Yp==Y.to(torch.int64)
    Yp_ne_Y=Yp!=Y.to(torch.int64)
    #-----------------------------------
    loss_fn=get_loss_function(Z.size())
    loss1=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    loss2=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    loss3=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    Xn=torch.tensor([], dtype=X.dtype, device=X.device)
    Ypn=torch.tensor([], dtype=Y.dtype, device=X.device)
    advc=torch.zeros(X.size(0), dtype=torch.int64, device=X.device)
    idx_n=torch.tensor([], dtype=torch.int64, device=X.device)
    #----------------------------------
    if Yp_ne_Y.sum().item()>0:
        loss1 = loss_fn(Z[Yp_ne_Y], Y[Yp_ne_Y])/Yp_ne_Y.sum().item()
    if Yp_e_Y.sum().item()>0:
        loss2 = loss_fn(Z[Yp_e_Y], Y[Yp_e_Y])/Yp_e_Y.sum().item()
    #---------------------------------
    train_mode=model.training# record the mode
    if model_eval_attack == True and train_mode == True:
        model.eval()#BN, dropout, etc
        #re-run the model in eval mode
        _Z_, _Yp_=run_model(model, X)
        Yp_e_Y=_Yp_==Y.to(torch.int64)
        Yp_ne_Y=_Yp_!=Y.to(torch.int64)
    #---------------------------------
    enable_loss3=False
    if Yp_e_Y.sum().item()>0 and beta>0:
         enable_loss3=True
    #----------------------------------
    if enable_loss3 == True:
        Xn, advc = repeated_pgd_attack(model, X, Y,
                                       noise_norm=margin, norm_type=norm_type,
                                       max_iter=max_iter, step=step,
                                       rand_init_norm=rand_init_norm, rand_init_Xn=rand_init_Xn,
                                       clip_X_min=clip_X_min, clip_X_max=clip_X_max,
                                       refine_Xn_max_iter=refine_Xn_max_iter,
                                       Xn1_equal_X=Xn1_equal_X,
                                       Xn2_equal_Xn=Xn2_equal_Xn,
                                       stop_near_boundary=stop_near_boundary,
                                       stop_if_label_change=stop_if_label_change,
                                       stop_if_label_change_next_step=stop_if_label_change_next_step,
                                       use_optimizer=use_optimizer,
                                       loss_fn=pgd_loss_fn,
                                       num_repeats=pgd_num_repeats)
        #--------------------------------------------
        if model_eval_Xn == True:
            if model.training == True:
                model.eval()
        else:
            if train_mode == True and model.training == False:
                model.train()
        #--------------------------------------------
        temp = Xn.detach().clone().requires_grad_(True)
        Zn, Ypn=run_model(model, temp)
        idx_n=torch.arange(0,X.size(0))[(advc>0)&(Yp_e_Y)]
        # to be consistant with the output of IMA_loss in RobustDNN_IMA
        Xn=Xn[idx_n]; Zn=Zn[idx_n]; Ypn=Ypn[idx_n]
        weight = advc[idx_n]/advc[idx_n].sum() # sample-wise weights
        loss_fn3 = None
        if len(Z.size()) <=1:
            loss_fn3 = torch.nn.BCEWithLogitsLoss(reduction= "none")
        else:
            loss_fn3 = torch.nn.CrossEntropyLoss(reduction= "none")
        
        if idx_n.size(0)>0:
            loss3 = (loss_fn3(Zn, Y[idx_n])*weight).sum()
            
    #--------------------------------------------
    if beta_position == 0:
        loss=(1-beta)*loss1+(beta*0.5)*(loss2+loss3)
    elif beta_position == 1:
        loss=(1-beta)*(loss1+loss2)+beta*loss3
    elif beta_position == 2:
        loss=loss1+(1-beta)*loss2+beta*loss3
    elif beta_position == 3:
        loss=(1-beta)*loss1+beta*loss3
    else:
        raise ValueError('unknown beta_position')
    #--------------------------------------------
    if train_mode == True and model.training == False:
        model.train()
    #--------------------------------------------
    if enable_loss3 and idx_n.size(0)>0:
        grads_n = torch.autograd.grad(loss, temp, retain_graph=True)[0]
        grads_n = torch.max(abs(grads_n).reshape(grads_n.shape[0], -1), dim = 1)[0] # thegradient can be negative, but magnitude is large
    else:
        grads_n = torch.zeros(X.size(0), dtype=torch.int64, device=X.device)
    return loss, loss1, loss2, loss3, Yp, advc, Xn, Ypn, idx_n, grads_n

#%%
def IMA_loss_grad(model, X, Y, margin, norm_type, max_iter, step,
             rand_init_norm=None, rand_init_Xn=None,
             clip_X_min=0, clip_X_max=1,
             refine_Xn_max_iter=10,
             Xn1_equal_X=False,
             Xn2_equal_Xn=False,
             stop_near_boundary=False,
             stop_if_label_change=False,
             stop_if_label_change_next_step=False,
             use_optimizer=False,
             beta=0.5, beta_position=1,
             pgd_loss_fn=None,
             pgd_num_repeats=1,
             model_eval_attack=False,
             model_eval_Xn=False,
             model_Xn_advc_p=False):
    #----------------------------------
    if isinstance(step, torch.Tensor):
        if use_optimizer:
            raise ValueError('incompatible')
        else:
            step=step.view(-1, *tuple([1]*len(X[0].size())))
    #-----------------------------------
    Z, Yp=run_model(model, X)
    Yp_e_Y=Yp==Y.to(torch.int64)
    Yp_ne_Y=Yp!=Y.to(torch.int64)
    #-----------------------------------
    loss_fn=get_loss_function(Z.size())
    loss1=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    loss2=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    loss3=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    Xn=torch.tensor([], dtype=X.dtype, device=X.device)
    Ypn=torch.tensor([], dtype=Y.dtype, device=X.device)
    advc=torch.zeros(X.size(0), dtype=torch.int64, device=X.device)
    idx_n=torch.tensor([], dtype=torch.int64, device=X.device)
    #----------------------------------
    if Yp_ne_Y.sum().item()>0:
        loss1 = loss_fn(Z[Yp_ne_Y], Y[Yp_ne_Y])/X.size(0)
    if Yp_e_Y.sum().item()>0:
        loss2 = loss_fn(Z[Yp_e_Y], Y[Yp_e_Y])/X.size(0)
    #---------------------------------
    train_mode=model.training# record the mode
    if model_eval_attack == True and train_mode == True:
        model.eval()#BN, dropout, etc
        #re-run the model in eval mode
        _Z_, _Yp_=run_model(model, X)
        Yp_e_Y=_Yp_==Y.to(torch.int64)
        Yp_ne_Y=_Yp_!=Y.to(torch.int64)
    #---------------------------------
    enable_loss3=False
    if Yp_e_Y.sum().item()>0 and beta>0:
         enable_loss3=True
    #----------------------------------
    if enable_loss3 == True:
        Xn, advc = repeated_pgd_attack(model, X, Y,
                                       noise_norm=margin, norm_type=norm_type,
                                       max_iter=max_iter, step=step,
                                       rand_init_norm=rand_init_norm, rand_init_Xn=rand_init_Xn,
                                       clip_X_min=clip_X_min, clip_X_max=clip_X_max,
                                       refine_Xn_max_iter=refine_Xn_max_iter,
                                       Xn1_equal_X=Xn1_equal_X,
                                       Xn2_equal_Xn=Xn2_equal_Xn,
                                       stop_near_boundary=stop_near_boundary,
                                       stop_if_label_change=stop_if_label_change,
                                       stop_if_label_change_next_step=stop_if_label_change_next_step,
                                       use_optimizer=use_optimizer,
                                       loss_fn=pgd_loss_fn,
                                       num_repeats=pgd_num_repeats)
        #--------------------------------------------
        if model_eval_Xn == True:
            if model.training == True:
                model.eval()
        else:
            if train_mode == True and model.training == False:
                model.train()
        #--------------------------------------------
        temp = Xn.detach().clone().requires_grad_(True)
        Zn, Ypn=run_model(model, temp)
        if model_Xn_advc_p == True:
            idx_n=torch.arange(0,X.size(0))[(advc>0)&(Yp_e_Y)]
        else:
            idx_n=torch.arange(0,X.size(0))[Yp_e_Y]
        # to be consistant with the output of IMA_loss in RobustDNN_IMA
        Xn=Xn[idx_n]; Zn=Zn[idx_n]; Ypn=Ypn[idx_n]
        if idx_n.size(0)>0:
            loss3 = loss_fn(Zn, Y[idx_n])/X.size(0)
    #--------------------------------------------
    if beta_position == 0:
        loss=(1-beta)*loss1+(beta*0.5)*(loss2+loss3)
    elif beta_position == 1:
        loss=(1-beta)*(loss1+loss2)+beta*loss3
    elif beta_position == 2:
        loss=loss1+(1-beta)*loss2+beta*loss3
    elif beta_position == 3:
        loss=(1-beta)*loss1+beta*loss3
    else:
        raise ValueError('unknown beta_position')
    #--------------------------------------------
    if train_mode == True and model.training == False:
        model.train()
    #--------------------------------------------
    if enable_loss3 and idx_n.size(0)>0:
        grads_n = torch.autograd.grad(loss, temp, retain_graph=True)[0]
        grads_n = torch.max(abs(grads_n).reshape(grads_n.shape[0], -1), dim = 1)[0] # thegradient can be negative, but magnitude is large
    else:
        grads_n = torch.zeros(X.size(0), dtype=torch.int64, device=X.device)
    return loss, loss1, loss2, loss3, Yp, advc, Xn, Ypn, idx_n, grads_n

#%%
def IMA_loss_grad_faster(model, X, Y, margin, norm_type, max_iter, step,
             rand_init_norm=None, rand_init_Xn=None,
             clip_X_min=0, clip_X_max=1,
             refine_Xn_max_iter=10,
             Xn1_equal_X=False,
             Xn2_equal_Xn=False,
             stop_near_boundary=False,
             stop_if_label_change=False,
             stop_if_label_change_next_step=False,
             use_optimizer=False,
             beta=0.5, beta_position=1,
             pgd_loss_fn=None,
             pgd_num_repeats=1,
             model_eval_attack=False,
             model_eval_Xn=False,
             model_Xn_advc_p=False):
    #----------------------------------
    if isinstance(step, torch.Tensor):
        if use_optimizer:
            raise ValueError('incompatible')
        else:
            step=step.view(-1, *tuple([1]*len(X[0].size())))
    #-----------------------------------
    Z, Yp=run_model(model, X)
    Yp_e_Y=Yp==Y.to(torch.int64)
    Yp_ne_Y=Yp!=Y.to(torch.int64)
    #-----------------------------------
    loss_fn=get_loss_function(Z.size())
    loss1=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    loss2=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    loss3=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    #Xn=torch.tensor([], dtype=X.dtype, device=X.device)
    Xn = X.detach().clone()
    Ypn=torch.tensor([], dtype=Y.dtype, device=X.device)
    advc=torch.zeros(X.size(0), dtype=torch.int64, device=X.device)
    idx_n=torch.tensor([], dtype=torch.int64, device=X.device)
    #----------------------------------
    if Yp_ne_Y.sum().item()>0:
        loss1 = loss_fn(Z[Yp_ne_Y], Y[Yp_ne_Y])/X.size(0)
    if Yp_e_Y.sum().item()>0:
        loss2 = loss_fn(Z[Yp_e_Y], Y[Yp_e_Y])/X.size(0)
    #---------------------------------
    train_mode=model.training# record the mode
    if model_eval_attack == True and train_mode == True:
        model.eval()#BN, dropout, etc
        #re-run the model in eval mode
        _Z_, _Yp_=run_model(model, X)
        Yp_e_Y=_Yp_==Y.to(torch.int64)
        Yp_ne_Y=_Yp_!=Y.to(torch.int64)
    #---------------------------------
    enable_loss3=False
    if Yp_e_Y.sum().item()>0 and beta>0:
         enable_loss3=True
    #----------------------------------
    if enable_loss3 == True:
        Xn[Yp_e_Y], advc[Yp_e_Y] = repeated_pgd_attack(model, X[Yp_e_Y], Y[Yp_e_Y],
                                       noise_norm=margin[Yp_e_Y], norm_type=norm_type,
                                       max_iter=max_iter, step=step[Yp_e_Y],
                                       rand_init_norm=rand_init_norm[Yp_e_Y], rand_init_Xn=rand_init_Xn,
                                       clip_X_min=clip_X_min, clip_X_max=clip_X_max,
                                       refine_Xn_max_iter=refine_Xn_max_iter,
                                       Xn1_equal_X=Xn1_equal_X,
                                       Xn2_equal_Xn=Xn2_equal_Xn,
                                       stop_near_boundary=stop_near_boundary,
                                       stop_if_label_change=stop_if_label_change,
                                       stop_if_label_change_next_step=stop_if_label_change_next_step,
                                       use_optimizer=use_optimizer,
                                       loss_fn=pgd_loss_fn,
                                       num_repeats=pgd_num_repeats)
        #--------------------------------------------
        if model_eval_Xn == True:
            if model.training == True:
                model.eval()
        else:
            if train_mode == True and model.training == False:
                model.train()
        #--------------------------------------------
        temp = Xn.detach().clone().requires_grad_(True)
        Zn, Ypn=run_model(model, temp)
        if model_Xn_advc_p == True:
            idx_n=torch.arange(0,X.size(0))[(advc>0)&(Yp_e_Y)]
        else:
            idx_n=torch.arange(0,X.size(0))[Yp_e_Y]
        # to be consistant with the output of IMA_loss in RobustDNN_IMA
        Xn=Xn[idx_n]; Zn=Zn[idx_n]; Ypn=Ypn[idx_n]
        if idx_n.size(0)>0:
            loss3 = loss_fn(Zn, Y[idx_n])/X.size(0)
    #--------------------------------------------
    if beta_position == 0:
        loss=(1-beta)*loss1+(beta*0.5)*(loss2+loss3)
    elif beta_position == 1:
        loss=(1-beta)*(loss1+loss2)+beta*loss3
    elif beta_position == 2:
        loss=loss1+(1-beta)*loss2+beta*loss3
    elif beta_position == 3:
        loss=(1-beta)*loss1+beta*loss3
    else:
        raise ValueError('unknown beta_position')
    #--------------------------------------------
    if train_mode == True and model.training == False:
        model.train()
    #--------------------------------------------
    if enable_loss3 and idx_n.size(0)>0:
        grads_n = torch.autograd.grad(loss, temp, retain_graph=True)[0]
        grads_n = torch.max(abs(grads_n).reshape(grads_n.shape[0], -1), dim = 1)[0] # thegradient can be negative, but magnitude is large
    else:
        grads_n = torch.zeros(X.size(0), dtype=torch.int64, device=X.device)
    return loss, loss1, loss2, loss3, Yp, advc, Xn, Ypn, idx_n, grads_n
#%%
def IMA_loss_noBinary(model, X, Y, margin, norm_type, max_iter, step,
             rand_init_norm=None, rand_init_Xn=None,
             clip_X_min=0, clip_X_max=1,
             refine_Xn_max_iter=10,
             Xn1_equal_X=False,
             Xn2_equal_Xn=False,
             stop_near_boundary=False,
             stop_if_label_change=False,
             stop_if_label_change_next_step=False,
             use_optimizer=False,
             beta=0.5, beta_position=1,
             pgd_loss_fn=None,
             pgd_num_repeats=1,
             model_eval_attack=False,
             model_eval_Xn=False,
             model_Xn_advc_p=False):
    #----------------------------------
    if isinstance(step, torch.Tensor):
        if use_optimizer:
            raise ValueError('incompatible')
        else:
            step=step.view(-1, *tuple([1]*len(X[0].size())))
    #-----------------------------------
    Z, Yp=run_model(model, X)
    Yp_e_Y=Yp==Y.to(torch.int64)
    Yp_ne_Y=Yp!=Y.to(torch.int64)
    #-----------------------------------
    loss_fn=get_loss_function(Z.size())
    loss1=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    loss2=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    loss3=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    Xn=torch.tensor([], dtype=X.dtype, device=X.device)
    Ypn=torch.tensor([], dtype=Y.dtype, device=X.device)
    advc=torch.zeros(X.size(0), dtype=torch.int64, device=X.device)
    idx_n=torch.tensor([], dtype=torch.int64, device=X.device)
    #----------------------------------
    if Yp_ne_Y.sum().item()>0:
        loss1 = loss_fn(Z[Yp_ne_Y], Y[Yp_ne_Y])/X.size(0)
    if Yp_e_Y.sum().item()>0:
        loss2 = loss_fn(Z[Yp_e_Y], Y[Yp_e_Y])/X.size(0)
    #---------------------------------
    train_mode=model.training# record the mode
    if model_eval_attack == True and train_mode == True:
        model.eval()#BN, dropout, etc
        #re-run the model in eval mode
        _Z_, _Yp_=run_model(model, X)
        Yp_e_Y=_Yp_==Y.to(torch.int64)
        Yp_ne_Y=_Yp_!=Y.to(torch.int64)
    #---------------------------------
    enable_loss3=False
    if Yp_e_Y.sum().item()>0 and beta>0:
         enable_loss3=True
    #----------------------------------
    if enable_loss3 == True:
        Xn, advc = repeated_pgd_attack(model, X, Y,
                                       noise_norm=margin, norm_type=norm_type,
                                       max_iter=max_iter, step=step,
                                       rand_init_norm=rand_init_norm, rand_init_Xn=rand_init_Xn,
                                       clip_X_min=clip_X_min, clip_X_max=clip_X_max,
                                       refine_Xn_max_iter=refine_Xn_max_iter,
                                       Xn1_equal_X=Xn1_equal_X,
                                       Xn2_equal_Xn=Xn2_equal_Xn,
                                       stop_near_boundary=stop_near_boundary,
                                       stop_if_label_change=stop_if_label_change,
                                       stop_if_label_change_next_step=stop_if_label_change_next_step,
                                       use_optimizer=use_optimizer,
                                       loss_fn=pgd_loss_fn,
                                       num_repeats=pgd_num_repeats)
        #--------------------------------------------
        if model_eval_Xn == True:
            if model.training == True:
                model.eval()
        else:
            if train_mode == True and model.training == False:
                model.train()
        #--------------------------------------------

        Zn, Ypn=run_model(model, Xn)
        if model_Xn_advc_p == True:
            idx_n=torch.arange(0,X.size(0))[(advc>0)&(Yp_e_Y)]
        else:
            idx_n=torch.arange(0,X.size(0))[Yp_e_Y]
        # to be consistant with the output of IMA_loss in RobustDNN_IMA
        Xn=Xn[idx_n]; Zn=Zn[idx_n]; Ypn=Ypn[idx_n]
        if idx_n.size(0)>0:
            loss3 = loss_fn(Zn, Y[idx_n])/X.size(0)
    #--------------------------------------------
    if beta_position == 0:
        loss=(1-beta)*loss1+(beta*0.5)*(loss2+loss3)
    elif beta_position == 1:
        loss=(1-beta)*(loss1+loss2)+beta*loss3
    elif beta_position == 2:
        loss=loss1+(1-beta)*loss2+beta*loss3
    elif beta_position == 3:
        loss=(1-beta)*loss1+beta*loss3
    else:
        raise ValueError('unknown beta_position')
    #--------------------------------------------
    if train_mode == True and model.training == False:
        model.train()
    #--------------------------------------------
    return loss, loss1, loss2, loss3, Yp, advc, Xn, Ypn, idx_n

#%%
def IMA_check_margin(model, device, dataloader,
                     margin, norm_type, max_iter, step, rand_init_norm=None,
                     clip_X_min=0, clip_X_max=1,
                     refine_Xn_max_iter=10,
                     Xn1_equal_X=False,
                     Xn2_equal_Xn=False,
                     stop_near_boundary=True,
                     use_optimizer=False,
                     pgd_loss_fn=None, pgd_num_repeats=1,
                     model_eval_check=False):
    #----------------------------------------------
    if isinstance(step, torch.Tensor):
        if use_optimizer:
            raise ValueError('incompatible')
    #----------------------------------------------
    train_mode=model.training# record the mode
    if model_eval_check == True and train_mode == True:
        model.eval()#no BN, no dropout, etc
    #----------------------------------------------
    try:
        sample_count=len(dataloader.dataset)# no duplication if dataloader is using sampler
    except:
        sample_count=dataloader.get_n_samples()
    margin_new=margin.detach().clone()
    flag1=torch.zeros(sample_count, dtype=torch.float32)
    #flag1[k] is 1: no adv is found for sample k
    #flag1[k] is 0: adv is found for sample k
    flag2=torch.zeros(sample_count, dtype=torch.float32)
    #flag2[k] is 1: correctly classified sample k
    #flag2[k] is 0: wrongly classified sample k
    #--------------------------------
    if rand_init_norm is None:
        rand_init_norm=margin.detach().clone()
    #--------------------------------
    for batch_idx, (X, Y, Idx) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)
        #--------------------
        Z, Yp=run_model(model, X)
        Yp_e_Y=Yp==Y.to(torch.int64)
        #--------------------
        advc=torch.zeros(X.shape[0], dtype=torch.int64, device=X.device)
        if Yp_e_Y.sum().item()>0:
            noise_norm=margin[Idx].to(device)
            rand_init_norm_=rand_init_norm
            if isinstance(rand_init_norm, torch.Tensor):
                rand_init_norm_=rand_init_norm[Idx].to(X.device)
            step_=step
            if isinstance(step, torch.Tensor):
                step_=step[Idx].view(-1, *tuple([1]*len(X[0].size()))).to(device)
            Xn, advc = repeated_pgd_attack(model, X, Y,
                                           noise_norm=noise_norm, norm_type=norm_type,
                                           max_iter=max_iter, step=step_,
                                           rand_init_norm=rand_init_norm_,
                                           clip_X_min=clip_X_min, clip_X_max=clip_X_max,
                                           refine_Xn_max_iter=refine_Xn_max_iter,
                                           Xn1_equal_X=Xn1_equal_X,
                                           Xn2_equal_Xn=Xn2_equal_Xn,
                                           stop_near_boundary=stop_near_boundary,
                                           use_optimizer=use_optimizer,
                                           loss_fn=pgd_loss_fn,
                                           num_repeats=pgd_num_repeats)
            if stop_near_boundary == True:
                temp=Xn[advc>0]-X[advc>0]
                if temp.shape[0]>0:
                    temp=torch.norm(temp.view(temp.shape[0], -1), p=norm_type, dim=1).cpu()
                    idx_adv=Idx[advc>0]
                    #margin_new[idx_adv]=torch.min(margin_new[idx_adv], temp)
                    #better if data aug is used
                    margin_new[idx_adv]=(margin_new[idx_adv]+temp)/2
        #----------------------
        flag1[Idx[advc==0]]=1
        flag2[Idx[Yp_e_Y]]=1
    #--------------------------------------------
    if train_mode == True and model.training == False:
        #print('restore train mode')
        model.train()
    #-------------------------------
    return flag1, flag2, margin_new

#%%
def IMA_check_margin_f(model, device, dataloader,
                     margin, norm_type, max_iter, step, rand_init_norm=None,
                     clip_X_min=0, clip_X_max=1,
                     refine_Xn_max_iter=10,
                     Xn1_equal_X=False,
                     Xn2_equal_Xn=False,
                     stop_near_boundary=True,
                     use_optimizer=False,
                     pgd_loss_fn=None, pgd_num_repeats=1,
                     model_eval_check=False):
    #----------------------------------------------
    if isinstance(step, torch.Tensor):
        if use_optimizer:
            raise ValueError('incompatible')
    #----------------------------------------------
    train_mode=model.training# record the mode
    if model_eval_check == True and train_mode == True:
        model.eval()#no BN, no dropout, etc
    #----------------------------------------------
    try:
        sample_count=len(dataloader.dataset)# no duplication if dataloader is using sampler
    except:
        sample_count=dataloader.get_n_samples()
    margin_new=margin.detach().clone()
    flag1=torch.zeros(sample_count, dtype=torch.float32)
    #flag1[k] is 1: no adv is found for sample k
    #flag1[k] is 0: adv is found for sample k
    flag2=torch.zeros(sample_count, dtype=torch.float32)
    #flag2[k] is 1: correctly classified sample k
    #flag2[k] is 0: wrongly classified sample k
    #--------------------------------
    if rand_init_norm is None:
        rand_init_norm=margin.detach().clone()
    #--------------------------------
    for batch_idx, (X, Y, Idx) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)
        #--------------------
        Z, Yp=run_model(model, X)
        Yp_e_Y=Yp==Y.to(torch.int64)
        #--------------------
        advc=torch.zeros(X.shape[0], dtype=torch.int64, device=X.device)
        if Yp_e_Y.sum().item()>0:
            noise_norm=margin[Idx].to(device)
            rand_init_norm_=rand_init_norm
            if isinstance(rand_init_norm, torch.Tensor):
                rand_init_norm_=rand_init_norm[Idx].to(X.device)
            step_=step
            if isinstance(step, torch.Tensor):
                step_=step[Idx].view(-1, *tuple([1]*len(X[0].size()))).to(device)
            Xn, advc = repeated_pgd_attack(model, X, Y,
                                           noise_norm=noise_norm, norm_type=norm_type,
                                           max_iter=max_iter, step=step_,
                                           rand_init_norm=rand_init_norm_,
                                           clip_X_min=clip_X_min, clip_X_max=clip_X_max,
                                           refine_Xn_max_iter=refine_Xn_max_iter,
                                           Xn1_equal_X=Xn1_equal_X,
                                           Xn2_equal_Xn=Xn2_equal_Xn,
                                           stop_near_boundary=stop_near_boundary,
                                           use_optimizer=use_optimizer,
                                           loss_fn=pgd_loss_fn,
                                           num_repeats=pgd_num_repeats)
            if stop_near_boundary == True:
                temp=Xn[advc>0]-X[advc>0]
                if temp.shape[0]>0:
                    temp=torch.norm(temp.view(temp.shape[0], -1), p=norm_type, dim=1).cpu()
                    idx_adv=Idx[advc>0]
                    #margin_new[idx_adv]=torch.min(margin_new[idx_adv], temp)
                    #better if data aug is used
                    margin_new[idx_adv]=(margin_new[idx_adv]+temp)/2
        #----------------------
    #--------------------------------------------
    if train_mode == True and model.training == False:
        model.train()
    #-------------------------------
    return margin_new
#%%
def IMA_update_margin(margin, delta, max_margin, flag1, flag2, margin_new):
    # margin: to be updated
    # delta: margin expansion step size
    # max_margin: maximum margin
    # flag1, flag2, margin_new: from IMA_check_margin
    expand=(flag1==1)&(flag2==1)
    no_expand=(flag1==0)&(flag2==1)
    margin[expand]+=delta
    margin[no_expand]=margin_new[no_expand]
    #disable it if data-aug is used because flag2 may vary for different x_aug of the same x
    #margin[flag2==0]=delta
    #margin.clamp_(min=delta, max=max_margin)
    margin.clamp_(min=0, max=max_margin)
#%%
def IMA_update_margin2(margin, delta, max_margin, flag1, flag2, margin_new, delta_decay = None):
    # margin: to be updated
    # delta: margin expansion step size
    # max_margin: maximum margin
    # flag1, flag2, margin_new: from IMA_check_margin
    expand=(flag1==1)&(flag2==1)
    no_expand=(flag1==0)&(flag2==1)
    if delta_decay is not None:
        margin[expand] += delta*delta_decay[expand]
    else:
        margin[expand]+=delta
    margin[no_expand]=margin_new[no_expand]
    margin.clamp_(min=0, max=max_margin)
#%%
def IMA_update_margin_no_upperbound(margin, delta, flag1, flag2, margin_new, delta_decay = None):
    # margin: to be updated
    # delta: margin expansion step size
    # max_margin: maximum margin
    # flag1, flag2, margin_new: from IMA_check_margin
    expand=(flag1==1)&(flag2==1)
    no_expand=(flag1==0)&(flag2==1)
    if delta_decay is not None:
        margin[expand] += delta*delta_decay[expand]
    else:
        margin[expand]+=delta
    margin[no_expand]=margin_new[no_expand]
    assert margin.min() >=0, "margin contains negetives"
    #margin.clamp_(min=0, max=max_margin)
    
#%%
def IMA_update_margin_no_upperbound_r(margin, delta, flag1, flag2, margin_new, delta_decay = None):
    # margin: to be updated
    # delta: margin expansion step size
    # max_margin: maximum margin
    # flag1, flag2, margin_new: from IMA_check_margin
    expand=(flag1==1)&(flag2==1)&(margin<margin.mean())
    no_expand=(flag1==0)&(flag2==1)
    if delta_decay is not None:
        margin[expand] += delta*delta_decay[expand]
    else:
        margin[expand]+=delta
    margin[no_expand]=margin_new[no_expand]
    assert margin.min() >=0, "margin contains negetives"
    return expand.sum().item()
    #margin.clamp_(min=0, max=max_margin)
#%%
def IMA_update_margin_no_upperbound2(margin, delta, flag1, flag2, margin_new, delta_decay = None):
    # margin: to be updated
    # delta: margin expansion step size
    # max_margin: maximum margin
    # flag1, flag2, margin_new: from IMA_check_margin
    expand=(flag1==1)&(flag2==1)
    no_expand=(flag1==0)&(flag2==1)
    if delta_decay is not None:
        margin[expand] += delta*delta_decay[expand]
    else:
        margin[expand]+=delta
    #margin[no_expand]=margin_new[no_expand]
    assert margin.min() >=0, "margin contains negetives"
    #margin.clamp_(min=0, max=max_margin)
#%%
def IMA_estimate_margin(model, device, dataloader,
                        margin_level, norm_type, max_iter, step,
                        clip_X_min=0, clip_X_max=1,
                        refine_Xn_max_iter=10,
                        use_optimizer=False,
                        pgd_loss_fn=None, pgd_num_repeats=1,
                        model_eval_check=False):
    try:
        sample_count=len(dataloader.dataset)# no duplication if dataloader is using sampler
    except:
        sample_count=dataloader.get_n_samples()
    sample_margin=torch.zeros(sample_count, dtype=torch.float32)
    flag_found=torch.zeros(sample_count, dtype=torch.float32)
    #margin_level[0] is 0 for wrongly classified samples
    for n in range(1, len(margin_level)):
        E=margin_level[n]*torch.ones(sample_count, dtype=torch.float32, device=device)
        flag1, flag2, Enew =IMA_check_margin(model, device, dataloader,
                                             margin=E, norm_type=norm_type,
                                             max_iter=max_iter, step=step, rand_init_norm=None,
                                             clip_X_min=clip_X_min, clip_X_max=clip_X_max,
                                             refine_Xn_max_iter=refine_Xn_max_iter,
                                             Xn1_equal_X=False,
                                             Xn2_equal_Xn=False,
                                             stop_near_boundary=True,
                                             use_optimizer=use_optimizer,
                                             pgd_loss_fn=pgd_loss_fn,
                                             pgd_num_repeats=pgd_num_repeats,
                                             model_eval_check=model_eval_check)
        temp=(flag_found==0)&(flag1==0)&(flag2==1)
        sample_margin[temp]=Enew[temp]
        flag_found[temp]=1
        #-------------
        if n == len(margin_level)-1:
            temp=(flag_found==0)&(flag2==1)
            sample_margin[temp]=Enew[temp]
        #-------------
        wrong=flag2==0
        sample_margin[wrong]=margin_level[0]
        flag_found[wrong]=1
    #-------------------------------
    return sample_margin
#%%
def plot_margin_hist(margin, noise_norm_list, normalize=False, ax=None):
    #noise_norm_list[0] is 0
    if isinstance(margin, torch.Tensor):
        margin=margin.detach().cpu().numpy()
    histc=np.zeros(len(noise_norm_list), dtype=np.float32)
    width=max(noise_norm_list)
    for k in range(0, len(noise_norm_list)):
        m1=noise_norm_list[k]
        if k < len(noise_norm_list)-1:
            m2=noise_norm_list[k+1]
            histc[k]=((margin>=m1)&(margin<m2)).sum()
            width = min(m2-m1, width)
        else:
            histc[k]=(margin>=m1).sum()
    width/=4
    if normalize == True:
        histc/=histc.sum()
    if ax is None:
        fig, ax = plt.subplots()
    ax.bar(noise_norm_list, histc, width=width)
    return ax.figure, ax
#%% it should be used on training set
def estimate_accuracy_from_margin(margin, noise_norm_list, flag):
    #noise_norm_list[0] is 0
    #flag[n]: 1(True) if correctly classified, 0(False) if wrongly classified
    #margin[n]: margin of sample n
    if isinstance(margin, torch.Tensor):
        margin=margin.detach().cpu().numpy()
    margin=margin.copy()
    margin[flag<1]=-1
    accuracy=np.zeros(len(noise_norm_list), dtype=np.float32)
    for n in range(len(noise_norm_list)):
       accuracy[n]=(margin>=noise_norm_list[n]).sum()
    accuracy/=len(margin)
    return accuracy
#%%
def estimate_accuracy_by_margin(margin, noise_norm_list, model, device, loader):
    if isinstance(margin, torch.Tensor):
        margin=margin.detach().cpu().numpy()
    margin=margin.copy()
    with torch.no_grad():
        for batch_idx, (X, Y, Idx) in enumerate(loader):
            X, Y = X.to(device), Y.to(device)
            Z = model(X)
            if len(Z.size()) <= 1:
                Yp = (Z.data>0).to(torch.int64) #binary/sigmoid
            else:
                Yp = Z.data.max(dim=1)[1] #multiclass/softmax
                temp=Idx[Yp!=Y].numpy()
            margin[temp]=-1
    accuracy=np.zeros(len(noise_norm_list), dtype=np.float32)
    for n in range(len(noise_norm_list)):
       accuracy[n]=(margin>=noise_norm_list[n]).sum()
    accuracy/=len(margin)
    return accuracy
