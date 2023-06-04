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
from RobustDNN_PGD import ifgsm_attack, repeated_pgd_attack, get_pgd_loss_fn_by_name
#%%
def cal_AUC_robustness(acc_list, noise_level_list):
    #noise_level_list[0] is 0
    #acc_list[0] is acc on clean data
    auc=0
    for n in range(1, len(acc_list)):
        auc+= (acc_list[n]+acc_list[n-1])*(noise_level_list[n]-noise_level_list[n-1])*0.5
    auc/=noise_level_list[n]
    return auc
#%%
def cal_performance(confusion, class_balanced_acc=False):
    num_classes=confusion.shape[0]
    if class_balanced_acc == True:
        confusion=confusion.copy()
        for m in range(0, num_classes):
            confusion[m]/=confusion[m].sum()+1e-8
    acc = confusion.diagonal().sum()/confusion.sum()
    sens=np.zeros(num_classes)
    prec=np.zeros(num_classes)
    for m in range(0, num_classes):
        sens[m]=confusion[m,m]/(np.sum(confusion[m,:])+1e-8)
        prec[m]=confusion[m,m]/(np.sum(confusion[:,m])+1e-8)
    return acc, sens, prec
#%%
def update_confusion(confusion, Y, Yp):
    Y=Y.detach().cpu().numpy()
    Yp=Yp.detach().cpu().numpy()
    num_classes=confusion.shape[0]
    if num_classes <= Y.shape[0]:
        for i in range(0, num_classes):
            for j in range(0, num_classes):
                confusion[i,j]+=np.sum((Y==i)&(Yp==j))
    else:
        for n in range(0, Y.shape[0]):
            confusion[Y[n],Yp[n]]+=1
#%%
def test(model, device, dataloader, num_classes, class_balanced_acc=False):
    model.eval()#set model to evaluation mode
    sample_count=0
    sample_idx_wrong=[]
    confusion=np.zeros((num_classes,num_classes))
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            X, Y = batch_data[0].to(device), batch_data[1].to(device)
            Z = model(X)#forward pass
            if len(Z.size()) <= 1:
                Yp = (Z.data>0).to(torch.int64) #binary/sigmoid
            else:
                Yp = Z.data.max(dim=1)[1] #multiclass/softmax
            update_confusion(confusion, Y, Yp)
            #------------------
            for n in range(0,X.size(0)):
                if Y[n] != Yp[n]:
                    sample_idx_wrong.append(sample_count+n)
            sample_count+=X.size(0)
    #------------------
    acc, sens, prec = cal_performance(confusion, class_balanced_acc)
    result={}
    result['confusion']=confusion
    result['acc']=acc
    result['sens']=sens
    result['prec']=prec
    result['sample_idx_wrong']=sample_idx_wrong
    print('testing')
    print('acc', result['acc'])
    print('sens', result['sens'])
    print('prec', result['prec'])
    return result
#%%
def rand_uniform_attack_old(model, X, Y, noise_norm, max_iter, clip_X_min=0, clip_X_max=1):
    with torch.no_grad():
        Xout=X.detach().clone()
        for n in range(0, max_iter):
            Xn = X + noise_norm*(2*torch.rand_like(X)-1)
            Xn.clamp_(clip_X_min, clip_X_max)
            Zn = model(Xn)
            if len(Zn.size()) <= 1:
                Ypn = (Zn.data>0).to(torch.int64) #binary/sigmoid
            else:
                Ypn = Zn.data.max(dim=1)[1] #multiclass/softmax
            Ypn_ne_Y=Ypn!=Y
            Xout[Ypn_ne_Y]=Xn[Ypn_ne_Y]
    return Xout
#%%
def clip_norm_(noise, norm_type, norm_max):
    if not isinstance(norm_max, torch.Tensor):
        return clip_normA_(noise, norm_type, norm_max)
    else:
        return clip_normB_(noise, norm_type, norm_max)
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
def rand_uniform_attack(model, X, Y, noise_norm, max_iter, clip_X_min=0, clip_X_max=1, norm_type = 2):
    with torch.no_grad():
        Xout=X.detach().clone()
        for n in range(0, max_iter):
            if norm_type == 2:
                deltaX = torch.rand_like(X)
                deltaX = clip_norm_(deltaX, norm_type, noise_norm)
                Xn = X + deltaX
            else:
                Xn = X + noise_norm*(torch.rand_like(X)) # rand_like returns uniform noise in [0,1]
                
            Xn.clamp_(clip_X_min, clip_X_max)
            Zn = model(Xn)
            if len(Zn.size()) <= 1:
                Ypn = (Zn.data>0).to(torch.int64) #binary/sigmoid
            else:
                Ypn = Zn.data.max(dim=1)[1] #multiclass/softmax
            Ypn_ne_Y=Ypn!=Y
            Xout[Ypn_ne_Y]=Xn[Ypn_ne_Y]
            if Ypn_ne_Y.sum().item() == Xn.shape[0]:
                break
    return Xout
#%%
def rand_attack_Gaussian(model, X, Y, noise_norm, max_iter, clip_X_min=0, clip_X_max=1):
    with torch.no_grad():
        Xout=X.detach().clone()
        for n in range(0, max_iter):
            Xn = X + noise_norm*(2*torch.randn_like(X)-1)
            Xn.clamp_(clip_X_min, clip_X_max)
            Zn = model(Xn)
            if len(Zn.size()) <= 1:
                Ypn = (Zn.data>0).to(torch.int64) #binary/sigmoid
            else:
                Ypn = Zn.data.max(dim=1)[1] #multiclass/softmax
            Ypn_ne_Y=Ypn!=Y
            Xout[Ypn_ne_Y]=Xn[Ypn_ne_Y]
    return Xout
#%%
def test_rand(model, device, dataloader, num_classes, noise_norm, norm_type, max_iter,
              clip_X_min=0, clip_X_max=1, class_balanced_acc=False):
    model.eval()#set model to evaluation mode
    sample_count=0
    adv_sample_count=0
    sample_idx_wrong=[]
    sample_idx_attack=[]
    confusion_clean=np.zeros((num_classes,num_classes))
    confusion_noisy=np.zeros((num_classes,num_classes))
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            X, Y = batch_data[0].to(device), batch_data[1].to(device)
            Xn=rand_uniform_attack(model, X, Y, noise_norm, max_iter, clip_X_min, clip_X_max, norm_type)
            Z = model(X)
            Zn = model(Xn)
            if len(Z.size()) <= 1: #binary/sigmoid
                Yp = (Z.data>0).to(torch.int64)
                Ypn = (Zn.data>0).to(torch.int64)
            else:  #multiclass/softmax
                Yp = Z.data.max(dim=1)[1]  
                Ypn = Zn.data.max(dim=1)[1]        
            #------------------
            #do not attack x that is missclassified
            Ypn_ = Ypn.clone().detach()
            Zn_=Zn.clone().detach()
            temp=(Yp!=Y)
            Ypn_[temp]=Yp[temp]
            Zn_[temp]=Z[temp]
            update_confusion(confusion_noisy, Y, Ypn_)
            update_confusion(confusion_clean, Y, Yp)
            #------------------
            for m in range(0,X.size(0)):
                idx=sample_count+m
                if Y[m] != Yp[m]:
                    sample_idx_wrong.append(idx)
                elif Ypn[m] != Yp[m]:
                    sample_idx_attack.append(idx)
            #------------------
            sample_count+=X.size(0)
            adv_sample_count+=torch.sum((Yp==Y)&(Ypn!=Y)).item()
    #------------------
    acc_clean, sens_clean, prec_clean = cal_performance(confusion_clean, class_balanced_acc)
    acc_noisy, sens_noisy, prec_noisy = cal_performance(confusion_noisy, class_balanced_acc)
    #------------------
    result={}
    result['method']='rand'
    result['noise_norm']=noise_norm
    result['sample_count']=sample_count
    result['adv_sample_count']=adv_sample_count
    result['sample_idx_wrong']=sample_idx_wrong
    result['sample_idx_attack']=sample_idx_attack
    result['confusion_clean']=confusion_clean
    result['acc_clean']=acc_clean
    result['sens_clean']=sens_clean
    result['prec_clean']=prec_clean
    result['confusion_noisy']=confusion_noisy
    result['acc_noisy']=acc_noisy
    result['sens_noisy']=sens_noisy
    result['prec_noisy']=prec_noisy
    #------------------
    print('testing robustness rand, adv%=', adv_sample_count/sample_count, sep='')
    print('noise_norm:', noise_norm)
    print('acc_clean', result['acc_clean'], ', acc_noisy', result['acc_noisy'])
    print('sens_clean', result['sens_clean'])
    print('sens_noisy', result['sens_noisy'])
    print('prec_clean', result['prec_clean'])
    print('prec_noisy', result['prec_noisy'])
    return result
#%%
def cal_mean_L2norm_X(dataloader):
    mean_norm=0
    sample_count=0
    for batch_idx, (X, Y) in enumerate(dataloader):
        X=X.view(X.size(0), -1)
        mean_norm+=torch.sum(torch.sqrt(torch.sum(X**2, dim=1))).item()
        sample_count+=X.size(0)
    mean_norm/=sample_count
    return mean_norm
#%%
def cal_mean_L1norm_X(dataloader):
    mean_norm=0
    sample_count=0
    for batch_idx, (X, Y) in enumerate(dataloader):
        X=X.view(X.size(0), -1)
        mean_norm+=torch.sum(torch.sum(X.abs(), dim=1)).item()
        sample_count+=X.size(0)
    mean_norm/=sample_count
    return mean_norm
#%%
def test_adv(model, device, dataloader, num_classes, noise_norm, norm_type, max_iter, step, method,
             targeted=False, clip_X_min=0, clip_X_max=1,
             use_optimizer=False, adv_loss_fn=None, num_repeats=1,
             save_model_output=False, class_balanced_acc=False):
    model.eval()#set model to evaluation mode
    confusion_clean=np.zeros((num_classes,num_classes))
    confusion_noisy=np.zeros((num_classes,num_classes))
    sample_count=0
    adv_sample_count=0
    sample_idx_wrong=[]
    sample_idx_attack=[]
    if save_model_output == True:
        y_list=[]
        z_list=[]
        yp_list=[]
        adv_z_list=[]
        adv_yp_list=[]
    #---------------------
    print('testing robustness wba ', method, '(', num_repeats, ')', sep='')
    print('norm_type:', norm_type, ', noise_norm:', noise_norm, ', max_iter:', max_iter, ', step:', step, sep='')
    adv_loss_fn=get_pgd_loss_fn_by_name(adv_loss_fn)
    print('adv_loss_fn', adv_loss_fn)
    #---------------------
    for batch_idx, batch_data in enumerate(dataloader):
        X, Y = batch_data[0].to(device), batch_data[1].to(device)
        #------------------
        #classify the clean sample X
        Z = model(X)
        if len(Z.size()) <= 1:
            Yp = (Z.data>0).to(torch.int64) #binary/sigmoid
        else:
            Yp = Z.data.max(dim=1)[1] #multiclass/softmax
        #------------------
        if method == 'ifgsm':
            Xn = ifgsm_attack(model, X, Y, noise_norm=noise_norm, norm_type=norm_type,
                              max_iter=max_iter, step=step, targeted=targeted,
                              clip_X_min=clip_X_min, clip_X_max=clip_X_max,
                              use_optimizer=use_optimizer, loss_fn=adv_loss_fn)
        elif method == 'pgd':
            Xn = repeated_pgd_attack(model, X, Y, noise_norm=noise_norm, norm_type=norm_type,
                                     max_iter=max_iter, step=step, targeted=targeted,
                                     clip_X_min=clip_X_min, clip_X_max=clip_X_max,
                                     use_optimizer=use_optimizer, loss_fn=adv_loss_fn,
                                     num_repeats=num_repeats)
        else:
            raise NotImplementedError("other method is not implemented.")
        #------------------
        #classify the noisy sample Xn
        Zn = model(Xn)
        if len(Zn.size()) <= 1:
            Ypn = (Zn.data>0).to(torch.int64) #binary/sigmoid
        else:
            Ypn = Zn.data.max(dim=1)[1] #multiclass/softmax
        #------------------
        #do not attack x that is missclassified
        Ypn_ = Ypn.clone().detach()
        Zn_=Zn.clone().detach()
        if targeted == False:
            temp=(Yp!=Y)
            Ypn_[temp]=Yp[temp]
            Zn_[temp]=Z[temp]
        update_confusion(confusion_noisy, Y, Ypn_)
        update_confusion(confusion_clean, Y, Yp)
        #------------------
        for m in range(0,X.size(0)):
            idx=sample_count+m
            if Y[m] != Yp[m]:
                sample_idx_wrong.append(idx)
            elif Ypn[m] != Yp[m]:
                sample_idx_attack.append(idx)
        sample_count+=X.size(0)
        adv_sample_count+=torch.sum((Yp==Y)&(Ypn!=Y)).item()
        #------------------
        if save_model_output == True:
            y_list.append(Y.detach().to('cpu').numpy())
            z_list.append(Z.detach().to('cpu').numpy())
            yp_list.append(Yp.detach().to('cpu').numpy())
            adv_z_list.append(Zn_.detach().to('cpu').numpy())
            adv_yp_list.append(Ypn_.detach().to('cpu').numpy())
        #------------------
    #------------------
    acc_clean, sens_clean, prec_clean = cal_performance(confusion_clean, class_balanced_acc)
    acc_noisy, sens_noisy, prec_noisy = cal_performance(confusion_noisy, class_balanced_acc)
    #------------------
    result={}
    result['method']=method
    result['noise_norm']=noise_norm
    result['norm_type']=norm_type
    result['max_iter']=max_iter
    result['step']=step
    result['sample_count']=sample_count
    result['adv_sample_count']=adv_sample_count
    result['sample_idx_wrong']=sample_idx_wrong
    result['sample_idx_attack']=sample_idx_attack
    result['confusion_clean']=confusion_clean
    result['acc_clean']=acc_clean
    result['sens_clean']=sens_clean
    result['prec_clean']=prec_clean
    result['confusion_noisy']=confusion_noisy
    result['acc_noisy']=acc_noisy
    result['sens_noisy']=sens_noisy
    result['prec_noisy']=prec_noisy
    #------------------
    if save_model_output == True:
        y_list = np.concatenate(y_list, axis=0).squeeze().astype('int64')
        z_list=np.concatenate(z_list, axis=0).squeeze()
        yp_list = np.concatenate(yp_list, axis=0).squeeze().astype('int64')
        adv_z_list=np.concatenate(adv_z_list, axis=0).squeeze()
        adv_yp_list = np.concatenate(adv_yp_list, axis=0).squeeze().astype('int64')
        result['y']=y_list
        result['z']=z_list
        result['yp']=yp_list
        result['adv_z']=adv_z_list
        result['adv_yp']=adv_yp_list
    #------------------
    print('testing robustness wba ', method, '(', num_repeats, '), adv%=', adv_sample_count/sample_count, sep='')
    print('norm_type:', norm_type, ', noise_norm:', noise_norm, ', max_iter:', max_iter, ', step:', step, sep='')
    print('acc_clean', result['acc_clean'], ', acc_noisy', result['acc_noisy'])
    print('sens_clean', result['sens_clean'])
    print('sens_noisy', result['sens_noisy'])
    print('prec_clean', result['prec_clean'])
    print('prec_noisy', result['prec_noisy'])
    return result
#%%