
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim
from CIFAR10_CNNM import main, update_lr
#%%
#https://pytorch.org/docs/stable/notes/randomness.html
#https://pytorch.org/docs/stable/cuda.html
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(0)
#%%

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    if epoch+1 == 60:
        update_lr(optimizer, 0.09)
    if epoch+1 == 90:
        update_lr(optimizer, 0.03)
    if epoch+1 == 120:
        update_lr(optimizer, 0.009)
        
def train(model, device, optimizer, dataloader, epoch, arg):
    #--------------
    adjust_learning_rate(optimizer, epoch)
    #update_lr(optimizer, arg['lr']/2**(epoch//25))    
    #----------------------------------
    model.train()
    loss_train=0
    acc_train =0
    sample_count=0
    import time
    t1 = time.time()
    for batch_idx, (X, Y) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)
        #----------------------------
        model.zero_grad()
        Z=model(X)
        loss=nnF.cross_entropy(Z, Y)        
        loss.backward()
        optimizer.step()
        #---------------------------
        Yp=Z.data.max(dim=1)[1]
        loss_train+=loss.item()
        acc_train+= torch.sum(Yp==Y).item()
        sample_count+=X.size(0)
        if batch_idx % 100 == 0:
            print('''Train Epoch: {} [{:.0f}%]\tLoss: {:.6f}'''.format(
                   epoch, 100. * batch_idx / len(dataloader), loss.item()))
    #---------------------------
    loss_train/=len(dataloader)
    acc_train/=sample_count
    #---------------------------
    t2 = time.time()
    print ("time cost one epoch is ",t2-t1)
    return loss_train, acc_train
#%% ------ use this line, and then this file can be used as a Python module --------------------
if __name__ == '__main__':
#%%
    parser = argparse.ArgumentParser(description='Input Parameters:')
    parser.add_argument('--epoch_start', default=0, type=int)
    parser.add_argument('--epoch_end', default=150, type=int)
    parser.add_argument('--optimizer', default='SGD', type=str)
    parser.add_argument('--lr', default=0.3, type=float)
    parser.add_argument('--cuda_id', default=1, type=int)
    parser.add_argument('--norm_type', default=np.inf, type=float)
    parser.add_argument('--net_name', default='mmacifar10', type=str)
    arg = parser.parse_args()
    epoch_start=arg.epoch_start
    epoch_end=arg.epoch_end
    optimizer=arg.optimizer
    lr=arg.lr
    cuda_id=arg.cuda_id
    print(arg)
    #-------------------------------------------
    device=torch.device('cuda:'+str(cuda_id) if torch.cuda.is_available() else "cpu")
    #-------------------------------------------
    arg=vars(arg)
    arg['device'] = device
    arg['optimizer']=optimizer
    arg['lr']=lr
    arg['loss_name']='ce_'+optimizer
    main(epoch_start=arg['epoch_start'], epoch_end=arg['epoch_end'], 
         train=train, arg=arg, evaluate_model=True)
#%%