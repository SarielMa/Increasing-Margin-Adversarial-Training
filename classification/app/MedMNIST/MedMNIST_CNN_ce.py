from MedMNIST_Dataset import get_dataloader
from MedMNIST_CNN import main
import sys
sys.path.append('../../core')
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(0)
#%%
def train(model, device, optimizer, dataloader, epoch, train_arg):
    model.train()
    #--------------
    loss_train=0
    acc_train =0
    sample_count=0
    for batch_idx, (X, Y, idx) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)
        #----------------------------
        model.zero_grad()
        #----------------------------
        Z=model(X)
        Yp=Z.data.max(dim=1)[1]
        loss=nnF.cross_entropy(Z, Y)
        loss.backward()
        #---------------------------
        optimizer.step()
        #---------------------------
        loss_train+=loss.item()
        acc_train+= torch.sum(Yp==Y).item()
        sample_count+=X.size(0)
        if batch_idx % 50 == 0:
            print('''Train Epoch: {} [{:.0f}%]\tLoss: {:.6f}'''.format(
                   epoch, 100. * batch_idx / len(dataloader), loss.item()))
    #---------------------------
    loss_train/=len(dataloader)
    acc_train/=sample_count
    #---------------------------
    return loss_train, acc_train
#%% ------ use this line, and then this file can be used as a Python module --------------------
if __name__ == '__main__':
    #%%
    parser = argparse.ArgumentParser(description='Input Parameters:')
    parser.add_argument('--cuda_id', default=1, type=int)
    parser.add_argument('--epoch_start', default=0, type=int)
    parser.add_argument('--epoch_end', default=100, type=int)
    parser.add_argument('--optimizer', default='Adam', type=str)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--net_name', default='resnet18', type=str)
    arg = parser.parse_args() 
    print(arg)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=str(arg.cuda_id)
    #-------------------------------------------
    device=torch.device('cuda' if torch.cuda.is_available() else "cpu")
    #-------------------------------------------
    arg=vars(arg)
    arg['norm_type']=2
    arg['device'] = device
    arg['return_idx']=(True, False, False)
    arg['loss_name']='ce_'+str(arg['optimizer'])
    main(epoch_start=arg['epoch_start'], epoch_end=arg['epoch_end'], train=train, arg=arg, evaluate_model=True)