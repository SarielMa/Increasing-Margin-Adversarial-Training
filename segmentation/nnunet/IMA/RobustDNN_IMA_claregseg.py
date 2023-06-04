
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as nnF
from torch import optim
from nnunet.IMA.RobustDNN_PGD import get_noise_init, normalize_grad_, clip_norm_
#%% classification, see more advanced loss functions in RobustDNN_IMA
def one_hot_output(net_output, gt):
    shp_x = net_output.shape
    shp_y = gt.shape
    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)
    return y_onehot# this is a transform for gt

def loss_bce_cla(Z, Y, reduction):
    Y=Y.to(Z.dtype)
    loss=nnF.binary_cross_entropy_with_logits(Z, Y, reduction='none')
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'none':
        pass
    else:
        raise ValueError('error')
    return loss
#
def loss_ce_cla(Z, Y, reduction):
    Y=(Y>0).to(torch.int64)
    loss=nnF.cross_entropy(Z, Y, reduction='none')
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'none':
        pass
    else:
        raise ValueError('error')
    return loss

def run_model_std_cla(model, X, Y=None, return_loss=False, reduction='none'):
    Z=model(X)
    if type(Z)==tuple:
        Z = Z[0]
                       
    if return_loss == True:
        Y = one_hot_output(Z, Y)
        if len(Z.shape) <= 1:
            loss_ce=loss_bce_cla(Z, Y, reduction)
            Yp = (Z.data>0).to(torch.int64)
        else:
            loss_ce=loss_ce_cla(Z, Y, reduction)
            Yp = Z.data.max(dim=1)[1]
        return Yp, loss_ce
    else:
        if len(Z.shape) <= 1:
            Yp = (Z.data>0).to(torch.int64)
        else:
            Yp = Z.data.max(dim=1)[1]
        return Yp
#
def run_model_adv_cla(model, X, Y=None, return_loss=False, reduction='sum'):
    return run_model_std_cla(model, X, Y, return_loss, reduction)
#
def classify_model_std_output_cla(Yp, Y):    
    Yp_e_Y=Yp==Y
    return Yp_e_Y
#
def classify_model_adv_output_cla(Ypn, Y):
    Ypn_e_Y=Ypn==Y
    return Ypn_e_Y
#%% regression
def loss_mae_reg(Yp, Y, reduction):
    Yp=Yp.view(Yp.shape[0], -1)
    Y=Y.view(Y.shape[0], -1)
    loss_mae=(Yp-Y).abs().mean(dim=1)
    if reduction == 'mean':
        loss_mae=loss_mae.mean()
    elif reduction == 'sum':
        loss_mae=loss_mae.sum()
    elif reduction == 'none':
        pass
    else:
        raise ValueError('error')
    return loss_mae
#
def loss_mse_reg(Yp, Y, reduction):
    Yp=Yp.view(Yp.shape[0], -1)
    Y=Y.view(Y.shape[0], -1)
    loss_mse=((Yp-Y)**2).mean(dim=1)
    if reduction == 'mean':
        loss_mse=loss_mse.mean()
    elif reduction == 'sum':
        loss_mse=loss_mse.sum()
    elif reduction == 'none':
        pass
    else:
        raise ValueError('error')
    return loss_mse

def run_model_std_reg(model, X, Y=None, return_loss=False, reduction='sum'):
    Yp=model(X)
    
    if type(Yp)==tuple:
        Yp = Yp[0]
             
    if return_loss == True:
        Y = one_hot_output(Yp, Y)
        loss=loss_mae_reg(Yp, Y, reduction)        
        return Yp, loss
    else:
        return Yp
#
def run_model_adv_reg(model, X, Y=None, return_loss=False, reduction='sum'):
    Yp=model(X)
    if type(Yp)==tuple:
        Yp = Yp[0]
      
    if return_loss == True:
        Y = one_hot_output(Yp, Y)
        loss=loss_mae_reg(Yp, Y, reduction)        
        return Yp, loss
    else:
        return Yp
#
def classify_model_std_output_reg(Yp, Y):
    threshold=0.1
    loss=loss_mae_reg(Yp, Y, reduction='none')
    Yp_e_Y=(loss<=threshold)
    return Yp_e_Y
#
def classify_model_adv_output_reg(Ypn, Y):
    #Y could be Ytrue or Ypred
    threshold=0.1
    loss=loss_mae_reg(Ypn, Y, reduction='none')
    Ypn_e_Y=(loss<=threshold)
    return Ypn_e_Y
#%% segmentation
def dice_seg(Yp, Y, reduction):
    #-----
    if type(Yp)==tuple:
        Yp = Yp[0]    
    Y = one_hot_output(Yp, Y)      
    #-----
    if len(Yp.shape)==4:
        #2D segmentation, shape: NxKxHxW
        temp=(2,3)
    elif len(Yp.shape)==5:
        #3D segmentation, shape: NxKxHxWxD
        temp=(2,3,4)
    
    intersection = (Yp*Y).sum(dim=temp)
    smooth=1
    dice = (2*intersection+smooth) / (Yp.sum(dim=temp) + Y.sum(dim=temp)+smooth)
    dice=dice.mean(dim=1)
    if reduction == 'mean':
        dice = dice.mean()
    elif reduction == 'sum':
        dice = dice.sum()
    elif reduction == 'none':
        pass
    else:
        raise ValueError('error')
    return dice
#
def loss_dice_seg(Yp, Y, reduction):
    return 1-dice_seg(Yp, Y, reduction)
#
def loss_bce_seg(Z, Y, reduction):
    N=Z.shape[0]
    Y=Y.to(Z.dtype)
    loss=nnF.binary_cross_entropy_with_logits(Z.view(N, -1), Y.view(N,-1), reduction='none')
    loss=loss.mean(dim=1)
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'none':
        pass
    else:
        raise ValueError('error')
    return loss
#
def loss_ce_seg(Z, Y, reduction):
    N=Z.shape[0]
    C=Z.shape[1]
    Y=(Y>0).to(torch.int64)
    loss=nnF.cross_entropy(Z.view(N, C, -1), Y.view(N,-1), reduction='none')
    loss=loss.mean(dim=1)
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'none':
        pass
    else:
        raise ValueError('error')
    return loss
    

#%%
# this is in fact binary PGD attack
def pgd_attack(task, model, X, Y, noise_norm, norm_type, max_iter, step,
               rand_init_norm=None, rand_init_Xn=None,
               targeted=False, clip_X_min=0, clip_X_max=1,
               refine_Xn_max_iter=10,
               Xn1_equal_X=False, Xn2_equal_Xn=False,
               stop_near_boundary=False,
               stop_if_label_change=False,
               stop_if_label_change_next_step=False,
               use_optimizer=False,
               run_model=None, classify_model_output=None,               
               model_eval_attack=False):
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
        Xn = X + noise_init
    elif rand_init_Xn is not None:
        Xn = rand_init_Xn.clone().detach()
    else:
        raise ValueError('invalid input')
    #-----------------
    Xn1=X.detach().clone()
    Xn2=X.detach().clone()
    Ypn_old_e_Y=torch.ones(Y[0].shape[0], dtype=torch.bool, device=Y[0].device)
    Ypn_old_ne_Y=~Ypn_old_e_Y
    #-----------------
    noise=(Xn-X).detach()
    if use_optimizer == True:
        optimizer = optim.Adamax([noise], lr=step)
    #-----------------
    for n in range(0, max_iter+1):
        Xn = Xn.detach()
        Xn.requires_grad = True        
        Ypn, loss=run_model(model, Xn, Y, return_loss=True, reduction='sum')
        Ypn_e_Y=classify_model_output(Ypn, Y, task)
        Ypn_ne_Y=~Ypn_e_Y
        #---------------------------
        #targeted attack, Y should be filled with targeted output
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
        Xn1[temp1]=Xn[temp1].data# last right and this right
        temp2=(B&A_old)&(advc<1)
        Xn2[temp1]=Xn[temp1].data# last right and this right
        Xn2[temp2]=Xn[temp2].data# last right and this wrong
        advc[B]+=1#
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
            #Xn = torch.clamp(X+noise, clip_X_min, clip_X_max)
            Xn = X+noise
            noise.data -= noise.data-(Xn-X).data
            #---------------------
            Ypn_old_e_Y=Ypn_e_Y
            Ypn_old_ne_Y=Ypn_ne_Y
    #---------------------------
    Xn_out = Xn.detach()
    if Xn1_equal_X:
        Xn1=X.detach().clone()
    if Xn2_equal_Xn:
        Xn2=Xn
    if stop_near_boundary == True:
        temp=advc>0
        if temp.sum()>0:
            Xn_out=refine_Xn_onto_boundary(task, model, Xn1, Xn2, Y, refine_Xn_max_iter, run_model, classify_model_output)
    elif stop_if_label_change == True:
        temp=advc>0
        if temp.sum()>0:
            Xn_out=refine_Xn2_onto_boundary(task, model, Xn1, Xn2, Y, refine_Xn_max_iter, run_model, classify_model_output)
    elif stop_if_label_change_next_step == True:
        temp=advc>0
        if temp.sum()>0:
            Xn_out=refine_Xn1_onto_boundary(task, model, Xn1, Xn2, Y, refine_Xn_max_iter, run_model, classify_model_output)
    #---------------------------
    if train_mode == True and model.training == False:
        model.train()
    #---------------------------
    return Xn_out, advc
#%%
def refine_onto_boundary(task, model, Xn1, Xn2, Y, max_iter, run_model, classify_model_output):
#note: Xn1 and Xn2 will be modified
    with torch.no_grad():
        Xn=(Xn1+Xn2)/2
        for k in range(0, max_iter):
            Ypn=run_model(model, Xn, return_loss=False)            
            Ypn_e_Y=classify_model_output(Ypn, Y, task)
            Ypn_ne_Y=~Ypn_e_Y
            Xn1[Ypn_e_Y]=Xn[Ypn_e_Y]
            Xn2[Ypn_ne_Y]=Xn[Ypn_ne_Y]
            Xn=(Xn1+Xn2)/2
    return Xn, Xn1, Xn2
#%%
def refine_Xn_onto_boundary(task, model, Xn1, Xn2, Y, max_iter, run_model, classify_model_output):
#note: Xn1 and Xn2 will be modified
    Xn, Xn1, Xn2=refine_onto_boundary(task,model, Xn1, Xn2, Y, max_iter, run_model, classify_model_output)
    return Xn
#%%
def refine_Xn1_onto_boundary(task, model, Xn1, Xn2, Y, max_iter, run_model, classify_model_output):
#note: Xn1 and Xn2 will be modified
    Xn, Xn1, Xn2=refine_onto_boundary(task,model, Xn1, Xn2, Y, max_iter, run_model, classify_model_output)
    return Xn1
#%%
def refine_Xn2_onto_boundary(task, model, Xn1, Xn2, Y, max_iter, run_model, classify_model_output):
#note: Xn1 and Xn2 will be modified
    Xn, Xn1, Xn2=refine_onto_boundary(task, model, Xn1, Xn2, Y, max_iter, run_model, classify_model_output)
    return Xn2
#%%
def repeated_pgd_attack(task, model, X, Y, noise_norm, norm_type, max_iter, step,
                        rand_init_norm=None, rand_init_Xn=None,
                        targeted=False, clip_X_min=0, clip_X_max=1,
                        refine_Xn_max_iter=10,
                        Xn1_equal_X=False,
                        Xn2_equal_Xn=False,
                        stop_near_boundary=False,
                        stop_if_label_change=False,
                        stop_if_label_change_next_step=False,
                        use_optimizer=False,
                        run_model=None, classify_model_output=None,
                        model_eval_attack=False,
                        num_repeats=1):
    for m in range(0, num_repeats):
        Xm, advcm = pgd_attack(task, model, X, Y, noise_norm, norm_type, max_iter, step,
                               rand_init_norm, rand_init_Xn,
                               targeted, clip_X_min, clip_X_max,
                               refine_Xn_max_iter,
                               Xn1_equal_X,
                               Xn2_equal_Xn,
                               stop_near_boundary,
                               stop_if_label_change,
                               stop_if_label_change_next_step,
                               use_optimizer,
                               run_model, classify_model_output,
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
def IMA_loss(task, model, X, Y, margin, norm_type, max_iter, step,
             rand_init_norm=None, rand_init_Xn=None,
             clip_X_min=0, clip_X_max=1,
             refine_Xn_max_iter=10,
             Xn1_equal_X=False,
             Xn2_equal_Xn=False,
             stop_near_boundary=True,
             stop_if_label_change=False,
             stop_if_label_change_next_step=False,
             use_optimizer=False,
             beta=0.5, beta_position=1,
             run_model_std=None, classify_model_std_output=None,
             run_model_adv=None, classify_model_adv_output=None,
             pgd_num_repeats=1,
             pgd_replace_Y_with_Yp=False,
             model_eval_attack=False,
             model_eval_Xn=False,
             model_Xn_advc_p=False):
    #----------------------------------
    if isinstance(step, torch.Tensor):
        if use_optimizer:
            raise ValueError('incompatible')
        else:
            temp=tuple([1]*len(X[0].size()))
            step=step.view(-1, *temp)
    #-----------------------------------
    Yp, loss_X=run_model_std(model, X, Y, return_loss=True, reduction='none')

    Yp_e_Y=classify_model_std_output(Yp, Y)
    Yp_ne_Y=~Yp_e_Y    
    #-----------------------------------
    loss1=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    loss2=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    loss3=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    Xn=torch.tensor([], dtype=X.dtype, device=X.device)
    Ypn=torch.tensor([], dtype=Y[0].dtype, device=X.device)
    advc=torch.zeros(X.size(0), dtype=torch.int64, device=X.device)
    idx_n=torch.tensor([], dtype=torch.int64, device=X.device)

    #----------------------------------
    if Yp_ne_Y.sum().item()>0:
        loss1 = loss_X[Yp_ne_Y].sum()/X.size(0)
    if Yp_e_Y.sum().item()>0:
        loss2 = loss_X[Yp_e_Y].sum()/X.size(0)
    #---------------------------------
    train_mode=model.training# record the mode
    if model_eval_attack == True and train_mode == True:
        model.eval()#BN, dropout, etc
        #re-run the model in eval mode
        _Yp=run_model_std(model, X, return_loss=False)
        Yp_e_Y=classify_model_std_output(_Yp, Y)
        Yp_ne_Y=~Yp_e_Y
    #---------------------------------
    enable_loss3=False
    if Yp_e_Y.sum().item()>0 and beta>0:
         enable_loss3=True
    #----------------------------------
    if enable_loss3 == True:
        Ypgd=Y
        if pgd_replace_Y_with_Yp == True:
            Ypgd=Yp.detach()
        Xn, advc = repeated_pgd_attack(task,model, X, Y=Ypgd,
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
                                       run_model=run_model_adv, classify_model_output=classify_model_adv_output,
                                       num_repeats=pgd_num_repeats)
        #--------------------------------------------
        if model_eval_Xn == True:
            if model.training == True:
                model.eval()
        else:
            if train_mode == True and model.training == False:
                model.train()
        #--------------------------------------------
        Ypn, loss_Xn=run_model_std(model, Xn, Y, return_loss=True, reduction='none')#use std not adv
        if model_Xn_advc_p == True:
            idx_n=torch.arange(0,X.size(0))[(advc>0)&(Yp_e_Y)]                
        else:
            idx_n=torch.arange(0,X.size(0))[Yp_e_Y]# for those margin=0, they are not adversarial samples, should not be here
        # to be consistant with the output of IMA_loss in RobustDNN_IMA
        Xn=Xn[idx_n] 
        #Ypn=Ypn[idx_n]
        if idx_n.size(0)>0:   
            loss3 = loss_Xn[idx_n].sum()/Xn.size(0)
    #--------------------------------------------
    #tuning the beta based on the proportion of the adversarial samples
    """
    nAdvSamples = ((Yp_e_Y)&(margin>args.delta)).sum().item()
    nCleanSamples = X.size(0)
    beta = beta/((nAdvSamples+1e8)/(nCleanSamples+1e8))#expand the beta as required
    beta = beta/(0.5+beta)
    """
    #-----------------------------------------
    if beta_position == 0:
        loss=(1-beta)*loss1+(beta*0.5)*(loss2+loss3)
    elif beta_position == 1:
        loss=(1-beta)*(loss1+loss2)+beta*loss3# almost the same as the vanilla PGD adv, diff: only add adv to part of the data
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
def IMA_check_margin_one_batch(model, X, Y, margin, norm_type, max_iter, step,
                               rand_init_norm=None,
                               clip_X_min=0, clip_X_max=1,
                               refine_Xn_max_iter=10,
                               Xn1_equal_X=False,
                               Xn2_equal_Xn=False,
                               use_optimizer=False,
                               run_model_std=None, classify_model_std_output=None,
                               run_model_adv=None, classify_model_adv_output=None,
                               pgd_num_repeats=1,
                               pgd_replace_Y_with_Yp=False,
                               model_eval_check=False):
    #----------------------------------------------
    #note: model_eval_check is True by default
    #----------------------------------------------
    if isinstance(step, torch.Tensor):
        if use_optimizer:
            raise ValueError('incompatible')
    #----------------------------------------------
    train_mode=model.training# record the mode
    if model_eval_check == True and train_mode == True:
        model.eval()#no BN, no dropout, etc
    #----------------------------------------------
    batch_size=X.shape[0]
    margin_new=margin.detach().clone()
    flag1=torch.zeros(batch_size, dtype=torch.float32)
    #flag1[k] is 1: no adv is found for sample k
    #flag1[k] is 0: adv is found for sample k
    flag2=torch.zeros(batch_size, dtype=torch.float32)
    #flag2[k] is 1: correctly classified sample k
    #flag2[k] is 0: wrongly classified sample k
    #--------------------------------
    if rand_init_norm is None:
        rand_init_norm=margin.detach().clone()
    #--------------------
    Yp=run_model_std(model, X, return_loss=False)
    Yp_e_Y=classify_model_std_output(Yp, Y)
    #--------------------
    advc=torch.zeros(X.shape[0], dtype=torch.int64, device=X.device)
    step_=step
    if isinstance(step, torch.Tensor):
        step_=step.view(-1, tuple([1]*len(X[0].size()))).to(X.device)
    Ypgd=Y
    if pgd_replace_Y_with_Yp == True:
        Ypgd=Yp.detach()
    Xn, advc = repeated_pgd_attack(model, X, Ypgd,
                                   noise_norm=margin,
                                   norm_type=norm_type,
                                   max_iter=max_iter,
                                   step=step_,
                                   rand_init_norm=rand_init_norm,
                                   clip_X_min=clip_X_min,
                                   clip_X_max=clip_X_max,
                                   refine_Xn_max_iter=refine_Xn_max_iter,
                                   Xn1_equal_X=Xn1_equal_X,
                                   Xn2_equal_Xn=Xn2_equal_Xn,
                                   stop_near_boundary=True,
                                   use_optimizer=use_optimizer,
                                   run_model=run_model_adv,
                                   classify_model_output=classify_model_adv_output,
                                   num_repeats=pgd_num_repeats)
    temp=Xn[advc>0]-X[advc>0]
    if temp.shape[0]>0:
        temp=torch.norm(temp.view(temp.shape[0], -1), p=norm_type, dim=1).cpu()
        margin_new[advc>0]=torch.min(margin_new[advc>0], temp)
    #----------------------
    flag1[advc==0]=1
    flag2[Yp_e_Y]=1
    Yp_ne_Y=~Yp_e_Y
    margin_new[Yp_ne_Y]=margin[Yp_ne_Y]
    #--------------------------------------------
    if train_mode == True and model.training == False:
        #print('restore train mode')
        model.train()
    #-------------------------------
    return flag1, flag2, margin_new
#%% you may need to re-write this function
def IMA_check_margin(model, device, dataloader,
                     margin, norm_type, max_iter, step, rand_init_norm=None,
                     clip_X_min=0, clip_X_max=1,
                     refine_Xn_max_iter=10,
                     Xn1_equal_X=False,
                     Xn2_equal_Xn=False,
                     use_optimizer=False,
                     run_model_std=None, classify_model_std_output=None,
                     run_model_adv=None, classify_model_adv_output=None,
                     pgd_num_repeats=1,
                     pgd_replace_Y_with_Yp=False,
                     model_eval_check=False):
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
        margin_=margin[Idx].to(device)
        rand_init_norm_=rand_init_norm
        if isinstance(rand_init_norm, torch.Tensor):
            rand_init_norm_=rand_init_norm[Idx].to(device)
        step_=step
        if isinstance(step, torch.Tensor):
            step_=step[Idx].to(device)
        flag1_, flag2_, margin_ = IMA_check_margin_one_batch(model, X, Y, margin_, norm_type, max_iter, step_,
                                                             rand_init_norm=rand_init_norm_,
                                                             clip_X_min=clip_X_min, clip_X_max=clip_X_max,
                                                             refine_Xn_max_iter=refine_Xn_max_iter,
                                                             Xn1_equal_X=Xn1_equal_X,
                                                             Xn2_equal_Xn=Xn2_equal_Xn,
                                                             use_optimizer=use_optimizer,
                                                             run_model_std=run_model_std,
                                                             classify_model_std_output=classify_model_std_output,
                                                             run_model_adv=run_model_adv,
                                                             classify_model_adv_output=classify_model_adv_output,
                                                             pgd_num_repeats=pgd_num_repeats,
                                                             pgd_replace_Y_with_Yp=pgd_replace_Y_with_Yp,
                                                             model_eval_check=model_eval_check)
        flag1[Idx]=flag1_
        flag2[Idx]=flag2_
        margin_new[Idx]=margin_
    #-------------------------------
    return flag1, flag2, margin_new
#%%
def IMA_update_margin_OLD(margin, delta, max_margin, flag1, flag2, margin_new):
    # margin: to be updated
    # delta: margin expansion step size
    # max_margin: maximum margin
    # flag1, flag2, margin_new: from IMA_check_margin
    expand=(flag1==1)&(flag2==1)
    no_expand=(flag1==0)&(flag2==1)
    margin[expand]+=delta
    margin[no_expand]=margin_new[no_expand]
    #
    margin[flag2==0]=delta
    margin.clamp_(min=0, max=max_margin)
    
def IMA_update_margin(args, delta, max_margin, flag1, flag2, margin_new):
    # margin: to be updated
    # delta: margin expansion step size
    # max_margin: maximum margin
    # flag1, flag2, margin_new: from IMA_check_margin
    expand=(flag1==1)&(flag2==1)
    no_expand=(flag1==0)&(flag2==1)
    args.E[expand]+=delta
    args.E[no_expand]=margin_new[no_expand]
    #when wrongly classified, do not re-initialize
    #args.E[flag2==0]=delta
    args.E.clamp_(min=0, max=max_margin)
#%%
def IMA_estimate_margin(model, device, dataloader,
                        margin_level, norm_type, max_iter, step,
                        clip_X_min=0, clip_X_max=1,
                        refine_Xn_max_iter=10,
                        use_optimizer=False,
                        run_model_std=None, classify_model_std_output=None,
                        run_model_adv=None, classify_model_adv_output=None,
                        pgd_num_repeats=1,
                        pgd_replace_Y_with_Yp=False,
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
                                             use_optimizer=use_optimizer,
                                             run_model_std=run_model_std, 
                                             classify_model_std_output=classify_model_std_output,
                                             run_model_adv=run_model_adv, 
                                             classify_model_adv_output=classify_model_adv_output,
                                             pgd_num_repeats=pgd_num_repeats,
                                             pgd_replace_Y_with_Yp=pgd_replace_Y_with_Yp,
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

