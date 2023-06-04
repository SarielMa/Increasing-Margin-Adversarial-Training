
import sys
from COVID19a_Dataset import get_dataloader
from COVID19a_CNN import main, get_filename
sys.path.append('../../core')
#%%
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import torch
from RobustDNN_IMA_bn import IMA_loss, IMA_update_margin, estimate_accuracy_from_margin, plot_margin_hist
from Evaluate import cal_AUC_robustness
#%%
random_seed=1
#%%
#https://pytorch.org/docs/stable/notes/randomness.html
#https://pytorch.org/docs/stable/cuda.html
import random
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(random_seed)
#%%
loader_check=get_dataloader(batch_size=1024, return_idx=(True, False, False))
loader_check=loader_check[0]#training set
#%%
def train_one_epoch(model, device, optimizer, dataloader, epoch, arg):
    norm_type=arg['norm_type']
    noise=arg['noise']
    beta=arg['beta']
    beta_position=arg['beta_position']
    E=arg['E']
    delta=arg['delta']
    max_iter=arg['max_iter']
    alpha=arg['alpha']
    pgd_loss_fn=arg['pgd_loss_fn']
    model_eval_attack=arg['model_eval_attack']
    model_eval_Xn=arg['model_eval_Xn']
    model_Xn_advc_p=arg['model_Xn_advc_p']
    Xn1_equal_X=arg['Xn1_equal_X']
    Xn2_equal_Xn=arg['Xn2_equal_Xn']
    #--------------
    refine_Xn_max_iter=arg['refine_Xn_max_iter']
    stop=arg['stop']
    stop_near_boundary=False
    stop_if_label_change=False
    stop_if_label_change_next_step=False
    if stop==1:
        stop_near_boundary=True
    elif stop==2:
        stop_if_label_change=True
    elif stop==3:
        stop_if_label_change_next_step=True
    #-------------
    print('noise', noise, 'epoch', epoch, 'delta', delta, 'stop', stop,
          'max_iter', max_iter, 'alpha', alpha)
    #--------------
    model.train()
    loss_train=0
    loss1_train=0
    loss2_train=0
    loss3_train=0
    acc1_train =0
    acc2_train =0
    sample1_count=0
    sample2_count=0
    #--------------
    sample_count=len(dataloader.dataset)# no duplication if dataloader is using sampler
    E_new=E.detach().clone()
    flag1=torch.zeros(sample_count, dtype=torch.float32)
    #flag1[k] is 1: no adv is found for sample k
    #flag1[k] is 0: adv is found for sample k
    flag2=torch.zeros(sample_count, dtype=torch.float32)
    #flag2[k] is 1: correctly classified sample k
    #--------------------------
    import time
    t1 = time.time()
    for batch_idx, (X, Y, Idx) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)
        #----------------------------
        model.zero_grad()
        #----------------------------
        rand_init_norm=torch.clamp(E[Idx]-delta, min=1e-3).to(device)
        margin=E[Idx].to(device)
        step=alpha*margin/max_iter
        loss, loss1, loss2, loss3, Yp, advc, Xn, Ypn, idx_n = IMA_loss(model, X, Y,
                                                                       norm_type=norm_type,
                                                                       rand_init_norm=rand_init_norm,
                                                                       margin=margin,
                                                                       max_iter=max_iter,
                                                                       step=step,
                                                                       refine_Xn_max_iter=refine_Xn_max_iter,
                                                                       Xn1_equal_X=Xn1_equal_X,
                                                                       Xn2_equal_Xn=Xn2_equal_Xn,
                                                                       stop_near_boundary=stop_near_boundary,
                                                                       stop_if_label_change=stop_if_label_change,
                                                                       stop_if_label_change_next_step=stop_if_label_change_next_step,
                                                                       beta=beta, beta_position=beta_position,
                                                                       use_optimizer=False,
                                                                       pgd_loss_fn=pgd_loss_fn,
                                                                       model_eval_attack=model_eval_attack,
                                                                       model_eval_Xn=model_eval_Xn,
                                                                       model_Xn_advc_p=model_Xn_advc_p)
        loss.backward()
        optimizer.step()
        #---------------------------
        Yp_e_Y=Yp==Y
        flag1[Idx[advc==0]]=1
        flag2[Idx[Yp_e_Y]]=1
        if idx_n.shape[0]>0:
            temp=torch.norm((Xn-X[idx_n]).view(Xn.shape[0], -1), p=norm_type, dim=1).cpu()
            #E_new[Idx[idx_n]]=torch.min(E_new[Idx[idx_n]], temp)
            E_new[Idx[idx_n]]=(E_new[Idx[idx_n]]+temp)/2
        #---------------------------
        loss_train+=loss.item()
        loss1_train+=loss1.item()
        loss2_train+=loss2.item()
        loss3_train+=loss3.item()
        acc1_train+= torch.sum(Yp==Y).item()
        acc2_train+= torch.sum(Ypn==Y[idx_n]).item()
        sample1_count+=X.size(0)
        sample2_count+=Xn.size(0)
        if batch_idx % 50 == 0:
            print('''Train Epoch: {} [{:.0f}%]\tLoss: {:.6f}\tLoss1: {:.6f}\tLoss2: {:.6f}\tLoss3: {:.6f}'''.format(
                   epoch, 100. * batch_idx / len(dataloader), loss.item(), loss1.item(), loss2.item(), loss3.item()))
    #---------------------------
    loss_train/=len(dataloader)
    loss1_train/=len(dataloader)
    loss2_train/=len(dataloader)
    loss3_train/=len(dataloader)
    acc1_train/=sample1_count
    if sample2_count > 0:
        acc2_train/=sample2_count
    loss_train=(loss_train, loss1_train, loss2_train, loss3_train)
    acc_train=(acc1_train, acc2_train)
    #---------------------------
    t2 = time.time()
    print ("time cost one epoch is ",t2-t1)
    return loss_train, acc_train, flag1, flag2, E_new

    
#%%

def update_lr(optimizer, new_lr):
    for g in optimizer.param_groups:
        g['lr']=new_lr
        print('new lr=', g['lr'])
def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    if epoch+1 == 40:
        update_lr(optimizer, 0.03)
    if epoch+1 == 60:
        update_lr(optimizer, 0.01)
    if epoch+1 == 80:
        update_lr(optimizer, 0.003)
#%%
def train(model, device, optimizer, dataloader, epoch, arg):
    norm_type=arg['norm_type']
    noise=arg['noise']
    epoch_freeze=arg['epoch_freeze']
    E=arg['E']
    delta=arg['delta']
    #--------------
    if arg['optimizer'] == 'SGD':
        adjust_learning_rate(optimizer, epoch)
    
    #--------------
    loss_train, acc_train, flag1, flag2, E_new = train_one_epoch(model, device, optimizer, dataloader, epoch, arg)
    #---------------------------
    if epoch < epoch_freeze: # and acc_train[0]>0.8 and acc_train[1]>0.5:
        IMA_update_margin(E, delta, noise, flag1, flag2, E_new)
        print('IMA_update_margin: done, margin updated')
    #---------------------------
    return loss_train, acc_train
#%% ------ use this line, and then this file can be used as a Python module --------------------
if __name__ == '__main__':
#%%
    parser = argparse.ArgumentParser(description='Input Parameters:')
    parser.add_argument('--noise', default=12.0, type=float)  # 3.0L2
    parser.add_argument('--norm_type', default=2, type=float)
    parser.add_argument('--epoch_start', default=0, type=int)
    parser.add_argument('--epoch_refine', default=100, type=int)
    parser.add_argument('--epoch_freeze', default=100, type=int)
    parser.add_argument('--epoch_end', default=100, type=int)
    parser.add_argument('--max_iter', default=20, type=int)
    parser.add_argument('--alpha', default=4, type=float)
    parser.add_argument('--stop', default=3, type=int)
    parser.add_argument('--refine_Xn_max_iter', default=10, type=int)
    parser.add_argument('--beta', default=0.5, type=float)
    parser.add_argument('--beta_position', default=1, type=int)
    parser.add_argument('--pgd_loss_fn', default='ce', type=str)
    parser.add_argument('--model_eval_attack', default=0, type=int)
    parser.add_argument('--model_eval_Xn', default=0, type=int)
    parser.add_argument('--model_Xn_advc_p', default=1, type=int)
    parser.add_argument('--Xn1_equal_X', default=1, type=int)
    parser.add_argument('--Xn2_equal_Xn', default=1, type=int)
    parser.add_argument('--optimizer', default='SGD', type=str)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--cuda_id', default=1, type=int)
    parser.add_argument('--data_aug', default=True, type=bool)
    parser.add_argument('--net_name', default='resnet18a2', type=str)
    parser.add_argument('--pretrained_model', default="none",type = str)
    arg = parser.parse_args()
    print(arg)

    #%%
    #-------------------------------------------
    sample_count_train=len(loader_check.dataset)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.cuda_id)
    device=torch.device('cuda'if torch.cuda.is_available() else "cpu")
    arg.delta=2.0
    loss_name=(str(arg.beta)+'IMAc_noPre_'+str(arg.noise)+'L'+str(arg.norm_type)
               +'_'+str(arg.max_iter)+'a'+str(arg.alpha)
               +'_s'+str(arg.stop)
               +'b'+str(arg.beta_position)
               +'e'+str(arg.epoch_refine)
               +'e'+str(arg.epoch_freeze)
               +'m'+str(arg.refine_Xn_max_iter)
               +str(bool(arg.model_eval_attack))[0]
               +str(bool(arg.model_eval_Xn))[0]
               +str(bool(arg.model_Xn_advc_p))[0]
               +str(bool(arg.Xn1_equal_X))[0]
               +str(bool(arg.Xn2_equal_Xn))[0]
               +str(bool(arg.data_aug))[0]
               +'_'+arg.pgd_loss_fn
               +'_'+arg.optimizer)
    if arg.pretrained_model != 'none':
        loss_name+='_ptm'
    if random_seed >0:
        loss_name+='_rs'+str(random_seed)
    #-------------------------------------------
    #stop=0 if every is False
    #stop=1 if stop_near_boundary=True
    #stop=2 if stop_if_label_change=True
    #stop=3 if stop_if_label_change_next_step=True
    #-------------------------------------------
    arg=vars(arg)
    arg['E']=None
    if arg['epoch_start'] == 0:
        arg['E']=arg['delta']*torch.ones(sample_count_train, dtype=torch.float32)
    arg['return_idx']=(True, False, False)
    arg['loss_name']=loss_name
    arg['device'] = device
    main(epoch_start=arg['epoch_start'], epoch_end=arg['epoch_end'], train=train, arg=arg, 
         evaluate_model=False)
#%%
