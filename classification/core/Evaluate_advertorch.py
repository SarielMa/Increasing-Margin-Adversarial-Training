# -*- coding: utf-8 -*-
"""
Created on Sun May 19 03:31:58 2019

@author: liang
"""
#%%
import numpy as np
import torch
from advertorch.attacks import MomentumIterativeAttack, PGDAttack, DDNL2Attack, LinfSPSAAttack, L2BasicIterativeAttack, LinfBasicIterativeAttack 
from Evaluate import cal_performance, update_confusion
from RobustDNN_PGD import get_pgd_loss_fn_by_name
from advertorch.attacks.utils import MarginalLoss
from advertorch.loss import CWLoss
from myautoattack import AutoAttack
#https://github.com/fra31/auto-attack
#%%
def pgd_attack(model, X, Y, noise_norm, norm_type, max_iter, step,
               rand_init=True, targeted=False, clip_X_min=0, clip_X_max=1, loss_fn=None, model_eval_attack=False):
    train_mode=model.training# record the mode
    if model_eval_attack == True and train_mode == True:
        model.eval()
    #--------------------------------------------
    loss_fn=get_pgd_loss_fn_by_name(loss_fn)
    #--------------------------------------------
    adversary = PGDAttack(model, loss_fn=loss_fn, eps=noise_norm,
                          nb_iter=max_iter, eps_iter=step, rand_init=rand_init,
                          clip_min=clip_X_min, clip_max=clip_X_max,
                          targeted=targeted, ord=norm_type)
    #--------------------------------------------
    Xn = adversary.perturb(X, Y) # inside: loss.backward()
    Xn = Xn.detach()
    model.zero_grad()
    #--------------------------------------------
    if train_mode == True and model.training == False:
        model.train()
    #--------------------------------------------
    return Xn
#%%
def repeated_attack(model, adversary, X, Y, num_repeats, targeted, return_output=True):
    Xn = X
    Zn = Y
    Ypn = Y
    for m in range(0, num_repeats):
        Xm = adversary.perturb(X, Y) # inside: loss.backward()
        Xm = Xm.detach()
        model.zero_grad()
        Zm = model(Xm)
        if len(Zm.size()) <= 1:
            Ypm = (Zm>0).to(torch.int64) #binary/sigmoid
        else:
            Ypm = Zm.data.max(dim=1)[1] #multiclass/softmax
        if m ==0:
            Xn=Xm.detach()
            Zn=Zm.detach()
            Ypn=Ypm.detach()
        else:
            if targeted == False:
                adv=Ypm!=Y
            else:
                adv=Ypm==Y
            Xn[adv]=Xm[adv].data
            Zn[adv]=Zm[adv].data
            Ypn[adv]=Ypm[adv].data
    if return_output == True:
        return Xn, Zn, Ypn
    else:
        return Xn
#%%
def repeated_attackN(model, adversary_list, X, Y, num_repeats, targeted, return_output):
    N=len(adversary_list)
    adversary0=adversary_list[0]
    Xn, Zn, Ypn = repeated_attack(model, adversary0, X, Y, int(num_repeats/N), targeted, True)
    for adversary in adversary_list[1:]:
        Xn1, Zn1, Ypn1 = repeated_attack(model, adversary, X, Y, int(num_repeats/N), targeted, True)
        if targeted == False:
            adv=Ypn1!=Y
        else:
            adv=Ypn1==Y
        #-----------------------------------
        Xn[adv]=Xn1[adv]
        Zn[adv]=Zn1[adv]
        Ypn[adv]=Ypn1[adv]
        #-----------------------------------        
    if return_output == True:
        return Xn, Zn, Ypn
    else:
        return Xn
#%%
def auto_attack(model, adversary_list, X, Y, targeted, return_output, num_classes):
    assert targeted == False
    adversary0=adversary_list[0]
    if num_classes <3:# fab-t and apgd-dlr can't be used when number of classes < 4 
        adversary0.attacks_to_run = ['apgd-ce','square']
    if X.shape[0] < 1024:
        Xn, Ypn = adversary0.run_standard_evaluation(X, Y, bs = X.shape[0], return_labels = True)
    else:
        Xn, Ypn = adversary0.run_standard_evaluation(X, Y, bs = 256, return_labels = True)
    
    if return_output == True:
        #Zn = adversary0.get_logits(Xn)
        return Xn, Ypn
    else:
        return Xn
#%%
def test_adv_auto(model, device, dataloader, num_classes, noise_norm, norm_type, max_iter, step, method,
             spsa_delta=0.01, spsa_samples=2048, spsa_iters=1, spsa_max_batch_size=128,
             targeted=False, clip_X_min=0, clip_X_max=1, adv_loss_fn=None, num_repeats=1,
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
    print('Auto Attack')
    print('testing robustness wba ', method,  sep='')
    print('norm_type:', norm_type, ', noise_norm:', noise_norm)
    #---------------------

    if method == 'auto_attack' or method == 'auto':
        loss_fn=get_pgd_loss_fn_by_name(adv_loss_fn)
        if norm_type == 2:
            auto_norm = "L2"
        elif norm_type == np.inf:
            auto_norm = "Linf"
        else:
            raise Exception("undefined norm type")
        adversary = AutoAttack(model, norm = auto_norm, eps = noise_norm, version='standard')
        adversary_list=[adversary]
    else:
        raise NotImplementedError("other method is not implemented.")
    #---------------------
    X = None
    Y = None
    Yp = None
    Z = None
    for batch_idx, batch_data in enumerate(dataloader):
        #if batch_idx >=3:
        #    break
        X_b, Y_b = batch_data[0].to(device), batch_data[1].to(device)
        #------------------
        Z_b = model(X_b)#classify the 'clean' signal X
        if len(Z_b.size()) <= 1:
            Yp_b = (Z_b.data>0).to(torch.int64) #binary/sigmoid
        else:
            Yp_b = Z_b.data.max(dim=1)[1] #multiclass/softmax
            
        if X == None:
            X = X_b.detach().cpu()
            Y = Y_b.detach().cpu()
            Yp = Yp_b.detach().cpu()
            Z = Z_b.detach().cpu()
        else:
            X = torch.cat((X, X_b.detach().cpu()), dim = 0)
            Y = torch.cat((Y, Y_b.detach().cpu()), dim = 0)
            Z = torch.cat((Z, Z_b.detach().cpu()), dim = 0)
            Yp = torch.cat((Yp, Yp_b.detach().cpu()), dim = 0)
    #------------------
    X = X.to(device)
    Y = Y.to(device)
    Z = Z.to(device)
    Yp = Yp.to(device)
    if method == 'auto_attack' or method == 'auto':
        Xn, Ypn = auto_attack(model, adversary_list, X, Y, targeted = targeted, return_output = True,num_classes = num_classes)
    else:
        raise NotImplementedError("other method is not implemented.")      
    #------------------
    #do not attack x that is missclassified
    Ypn_ = Ypn.clone().detach()
    #Zn_=Zn.clone().detach()
    if targeted == False:
        temp=(Yp!=Y)
        Ypn_[temp]=Yp[temp]
        #Zn_[temp]=Z[temp]
    #------------------
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
        #adv_z_list.append(Zn_.detach().to('cpu').numpy())
        adv_yp_list.append(Ypn_.detach().to('cpu').numpy())
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
        #adv_z_list=np.concatenate(adv_z_list, axis=0).squeeze()
        adv_yp_list = np.concatenate(adv_yp_list, axis=0).squeeze().astype('int64')
        result['y']=y_list
        result['z']=z_list
        result['yp']=yp_list
        #result['adv_z']=adv_z_list
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
def test_adv(model, device, dataloader, num_classes, noise_norm, norm_type, max_iter, step, method,
             spsa_delta=0.01, spsa_samples=2048, spsa_iters=1, spsa_max_batch_size=128,
             targeted=False, clip_X_min=0, clip_X_max=1, adv_loss_fn=None, num_repeats=1,
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
    print('Evaluate_advertorch')
    print('testing robustness wba ', method, '(', num_repeats, ')', sep='')
    print('norm_type:', norm_type, ', noise_norm:', noise_norm, ', max_iter:', max_iter, ', step:', step, sep='')
    #---------------------
    if method == 'MomentumIterativeAttack':
        loss_fn=get_pgd_loss_fn_by_name(adv_loss_fn)
        adversary = MomentumIterativeAttack(model, loss_fn=loss_fn, eps=noise_norm,
                                            nb_iter=max_iter, eps_iter=step,
                                            clip_min=clip_X_min, clip_max=clip_X_max,
                                            targeted=targeted, ord=norm_type)
        adversary_list=[adversary]
    elif method == 'ifgsm':
        # the default number of iteration is 10
        if norm_type == 2:
            loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
            adversary = L2BasicIterativeAttack(model, loss_fn = loss_fn, eps = noise_norm, 
                                               nb_iter=max_iter, eps_iter=step,
                                               clip_min=clip_X_min, clip_max=clip_X_max,
                                               targeted=targeted)
        else: 
            loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
            adversary = LinfBasicIterativeAttack(model, loss_fn = loss_fn, eps = noise_norm, 
                                               nb_iter=max_iter, eps_iter=step,
                                               clip_min=clip_X_min, clip_max=clip_X_max,
                                               targeted=targeted)
            
        adversary_list=[adversary]
    elif method == 'ifgsmb':
        # the default number of iteration is 10
        loss_fn=get_pgd_loss_fn_by_name('bce')
        adversary = L2BasicIterativeAttack(model, loss_fn = loss_fn, eps = noise_norm, 
                                           nb_iter=max_iter, eps_iter=step,
                                           clip_min=clip_X_min, clip_max=clip_X_max,
                                           targeted=targeted)
        adversary_list=[adversary]
    #elif method == 'CarliniWagnerL2Attack' or method == 'cwl2':
    #    loss_fn = loss_fn1=torch.nn.CrossEntropyLoss(reduction="sum")
    #    adversary = CarliniWagnerL2Attack(model, num_classes, clip_min=clip_X_min, clip_max=clip_X_max, )
    elif method == 'PGDAttack' or method == 'pgd':
        loss_fn=get_pgd_loss_fn_by_name(adv_loss_fn)
        adversary = PGDAttack(model, loss_fn=loss_fn, eps=noise_norm,
                              nb_iter=max_iter, eps_iter=step,
                              clip_min=clip_X_min, clip_max=clip_X_max,
                              targeted=targeted, ord=norm_type)
        adversary_list=[adversary]
    elif method == 'auto_attack' or method == 'auto':
        loss_fn=get_pgd_loss_fn_by_name(adv_loss_fn)
        if norm_type == 2:
            auto_norm = "L2"
        elif norm_type == np.inf:
            auto_norm = "Linf"
        else:
            raise Exception("undefined norm type")
        adversary = AutoAttack(model, norm = auto_norm, eps = noise_norm, version='standard')
        adversary_list=[adversary]
        
    elif method == 'PGDAttack_CE_M' or method == 'pgd_ce_m':
        loss_fn1=torch.nn.CrossEntropyLoss(reduction="sum")
        loss_fn2=MarginalLoss(reduction='sum')
        adversary1 = PGDAttack(model, loss_fn=loss_fn1, eps=noise_norm,
                               nb_iter=max_iter, eps_iter=step,
                               clip_min=clip_X_min, clip_max=clip_X_max,
                               targeted=targeted, ord=norm_type)
        adversary2 = PGDAttack(model, loss_fn=loss_fn2, eps=noise_norm,
                               nb_iter=max_iter, eps_iter=step,
                               clip_min=clip_X_min, clip_max=clip_X_max,
                               targeted=targeted, ord=norm_type)
        adversary_list=[adversary1, adversary2]
    elif method == 'PGDAttack_BCE_M' or method == 'pgd_bce_m':
        loss_fn1=get_pgd_loss_fn_by_name('bce')
        loss_fn2=get_pgd_loss_fn_by_name('lmb')
        adversary1 = PGDAttack(model, loss_fn=loss_fn1, eps=noise_norm,
                               nb_iter=max_iter, eps_iter=step,
                               clip_min=clip_X_min, clip_max=clip_X_max,
                               targeted=targeted, ord=norm_type)
        adversary2 = PGDAttack(model, loss_fn=loss_fn2, eps=noise_norm,
                               nb_iter=max_iter, eps_iter=step,
                               clip_min=clip_X_min, clip_max=clip_X_max,
                               targeted=targeted, ord=norm_type)
        adversary_list=[adversary1, adversary2]
    elif method == 'PGDAttack_CE_CW' or method == 'pgd_ce_cw':
        loss_fn1=torch.nn.CrossEntropyLoss(reduction="sum")
        loss_fn2=CWLoss(reduction='sum')
        adversary1 = PGDAttack(model, loss_fn=loss_fn1, eps=noise_norm,
                               nb_iter=max_iter, eps_iter=step,
                               clip_min=clip_X_min, clip_max=clip_X_max,
                               targeted=targeted, ord=norm_type)
        adversary2 = PGDAttack(model, loss_fn=loss_fn2, eps=noise_norm,
                               nb_iter=max_iter, eps_iter=step,
                               clip_min=clip_X_min, clip_max=clip_X_max,
                               targeted=targeted, ord=norm_type)
        adversary_list=[adversary1, adversary2]
    elif method == 'LinfSPSAAttack' or method == 'spsa':
        if norm_type != np.inf:
            raise NotImplementedError("only Linf is supported for spsa")
        print('It is suggested that you set step to 0.01 and set max_iter to 100')
        loss_fn=get_pgd_loss_fn_by_name(adv_loss_fn)
        adversary=LinfSPSAAttack(model, eps=noise_norm, nb_sample=spsa_samples, nb_iter=max_iter,
                                 delta=spsa_delta, lr=step, max_batch_size=spsa_max_batch_size,
                                 clip_min=clip_X_min, clip_max=clip_X_max, targeted=targeted, loss_fn=loss_fn)
        adversary_list=[adversary]
    elif method == 'DDNL2Attack' or method == 'ddn':
        gamma=1-(1.5/255)**(1/max_iter)
        adversary = DDNL2Attack(model, nb_iter=max_iter, gamma=gamma, init_norm=1,
                                quantize=True, levels=256,
                                clip_min=clip_X_min, clip_max=clip_X_max, targeted=targeted)
        adversary_list=[adversary]
    else:
        raise NotImplementedError("other method is not implemented.")
    #---------------------
    for batch_idx, batch_data in enumerate(dataloader):
        X, Y = batch_data[0].to(device), batch_data[1].to(device)
        #------------------
        Z = model(X)#classify the 'clean' signal X
        if len(Z.size()) <= 1:
            Yp = (Z.data>0).to(torch.int64) #binary/sigmoid
        else:
            Yp = Z.data.max(dim=1)[1] #multiclass/softmax
        #------------------
        if method == 'auto_attack' or method == 'auto':
            assert method!= "auto_attack", "please use the test_adv_auto() instead"
        else:
            Xn, Zn, Ypn = repeated_attackN(model, adversary_list, X, Y, num_repeats, targeted, True)      
        #------------------
        if method == 'DDNL2Attack' or method == 'ddn':
            N=(Xn-X).detach()
            N=N.view(N.size(0), -1)
            L2normN=torch.norm(N, p=2, dim=1)
            temp=L2normN>noise_norm #only attack X inside noise-ball
            Xn=X[temp]
            Zn=Z[temp]
            Ypn=Yp[temp]
        #------------------
        #do not attack x that is missclassified
        Ypn_ = Ypn.clone().detach()
        Zn_=Zn.clone().detach()
        if targeted == False:
            temp=(Yp!=Y)
            Ypn_[temp]=Yp[temp]
            Zn_[temp]=Z[temp]
        #------------------
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
def Estimate_L2Margin_by_DDN(model, device, dataloader, num_classes,
                             nb_iter=100, gamma=0.05, init_norm=1., quantize=True, levels=256,
                             clip_X_min=0, clip_X_max=1, class_balanced_acc=False):
    model.eval()
    confusion_clean=np.zeros((num_classes,num_classes))
    confusion_noisy=np.zeros((num_classes,num_classes))
    sample_count=0
    adv_sample_count=0
    sample_idx_wrong=[]
    sample_idx_attack=[]
    #------------------------------------------------------------------------
    if gamma is None:
        gamma=1-(1.5/255)**(1/nb_iter)
    adversary = DDNL2Attack(model, nb_iter=nb_iter, gamma=gamma, init_norm=init_norm, quantize=quantize, levels=levels,
                            clip_min=clip_X_min, clip_max=clip_X_max)
    margin=torch.zeros(len(dataloader.dataset), dtype=torch.float32)
    for batch_idx, batch_data in enumerate(dataloader):
        X, Y, Idx = batch_data[0].to(device), batch_data[1].to(device), batch_data[2].to(device)
        #------------------
        Z = model(X)#classify the 'clean' signal X
        Yp = Z.data.max(dim=1)[1] #multiclass/softmax
        #------------------
        Xn = adversary.perturb(X, Y)
        Zn = model(Xn)
        Ypn = Zn.data.max(dim=1)[1] #multiclass/softmax
        #------------------
        margin[Idx]=torch.norm((Xn-X).view(X.size(0),-1), p=2, dim=1).cpu()
        margin[Idx[Yp!=Y]]=-1
        #------------------
        #do not attack x that is missclassified
        Ypn_ = Ypn.clone().detach()
        Zn_=Zn.clone().detach()
        temp=(Yp!=Y)
        Ypn_[temp]=Yp[temp]
        Zn_[temp]=Z[temp]
        #------------------        
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
    #------------------
    acc_clean, sens_clean, prec_clean = cal_performance(confusion_clean, class_balanced_acc)
    acc_noisy, sens_noisy, prec_noisy = cal_performance(confusion_noisy, class_balanced_acc)
    #------------------
    result={}
    result['method']='ddn'
    result['margin']=margin
    result['norm_type']=2
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
    print('Estimate_L2Margin_by_DDN, adv%=', adv_sample_count/sample_count)
    print('acc_clean', result['acc_clean'], ', acc_noisy', result['acc_noisy'])
    print('sens_clean', result['sens_clean'])
    print('sens_noisy', result['sens_noisy'])
    print('prec_clean', result['prec_clean'])
    print('prec_noisy', result['prec_noisy'])

    return result
#%%
def Estimate_Accuracy_from_L2Margin(margin, noise_list):
    margin=margin.numpy()
    margin=margin[margin>=0]
    acc_list=np.zeros(len(noise_list), dtype=np.float32)
    for n in range(len(noise_list)):
       acc_list[n]=float((margin>=noise_list[n]).sum())/len(margin)
    return acc_list
