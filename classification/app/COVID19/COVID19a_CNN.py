#%%
import sys
sys.path.append('../../core')
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim
import torchvision.models as tv_models
import time
from sklearn.metrics import roc_curve
from Evaluate import test, test_rand, cal_AUC_robustness
from Evaluate_advertorch import test_adv
from COVID19a_Dataset import get_dataloader
from tqdm import tqdm
#%%
class Net(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name=name
        if name == 'resnet18' or name == 'resnet18a1' or name == 'resnet18a2':
            self.model = tv_models.resnet18(pretrained=False)
            self.model.conv1=nn.Conv2d(1,64,7,2,3, bias=False)
            if name == 'resnet18' or name == 'resnet18a1':
                self.out=1
                self.model.fc=torch.nn.Linear(512, 1)
            else:
                self.out=2
                self.model.fc=torch.nn.Linear(512, 2)
            #
            self.model.bn1 = nn.GroupNorm(64, 64)
            self.model.layer1[0].bn1 = nn.GroupNorm(64, 64)
            self.model.layer1[0].bn2 = nn.GroupNorm(64, 64)
            self.model.layer1[1].bn1 = nn.GroupNorm(64, 64)
            self.model.layer1[1].bn2 = nn.GroupNorm(64, 64)
            #
            self.model.layer2[0].bn1 = nn.GroupNorm(128, 128)
            self.model.layer2[0].bn2 = nn.GroupNorm(128, 128)
            self.model.layer2[0].downsample[1]=nn.GroupNorm(128, 128)
            self.model.layer2[1].bn1 = nn.GroupNorm(128, 128)
            self.model.layer2[1].bn2 = nn.GroupNorm(128, 128)
            #
            self.model.layer3[0].bn1 = nn.GroupNorm(256, 256)
            self.model.layer3[0].bn2 = nn.GroupNorm(256, 256)
            self.model.layer3[0].downsample[1]=nn.GroupNorm(256, 256)
            self.model.layer3[1].bn1 = nn.GroupNorm(256, 256)
            self.model.layer3[1].bn2 = nn.GroupNorm(256, 256)
            #
            self.model.layer4[0].bn1 = nn.GroupNorm(512, 512)
            self.model.layer4[0].bn2 = nn.GroupNorm(512, 512)
            self.model.layer4[0].downsample[1]=nn.GroupNorm(512, 512)
            self.model.layer4[1].bn1 = nn.GroupNorm(512, 512)
            self.model.layer4[1].bn2 = nn.GroupNorm(512, 512)
        self.return_feature=False
        
    def forward(self,x):
        if self.return_feature == True:
            return self.forward_(x)
        #---------------------------
        x=(x-0.5)/0.5
        x=self.model(x)
        if self.out == 1:
            x=x.view(-1)
        return x
    
    def forward_(self, x):
        
        x=(x-0.5)/0.5
        
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        
        if self.return_feature == False:
            x = self.model.fc(x)
        return x
#%%
def save_checkpoint(filename, model, optimizer, result, epoch):
    #-------------------------------------------
    #https://github.com/pytorch/pytorch/issues/9176
    try:
        state_dict = model.module.state_dict()
    except AttributeError:
        state_dict = model.state_dict()
    #-------------------------------------------
    torch.save({'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'result':result},
               filename)
    print('saved:', filename)
#%%
def plot_result(loss_train_list, acc_train_list,
                acc_val_list, adv_acc_val_list, acc_test_list):
    fig, ax = plt.subplots(1, 3, figsize=(9,3))
    ax[0].set_title('loss v.s. epoch')
    ax[0].plot(loss_train_list, '-b', label='train loss')
    ax[0].set_xlabel('epoch')
    #ax[0].legend()
    ax[0].grid(True)
    ax[1].set_title('accuracy v.s. epoch')
    ax[1].plot(acc_train_list, '-b', label='train acc')
    ax[1].plot(acc_val_list, '-r', label='val acc')
    ax[1].plot(acc_test_list, '-c', label='test acc')
    ax[1].set_xlabel('epoch')
    #ax[1].legend()
    ax[1].grid(True)
    ax[2].set_title('accuracy v.s. epoch')
    ax[2].plot(adv_acc_val_list, '-m', label='adv val acc')
    ax[2].set_xlabel('epoch')
    #ax[2].legend()
    ax[2].grid(True)
    return fig, ax
#%%
def get_filename(net_name, loss_name, epoch=None, pre_fix='result/COVID19a_CNN_'):
    if epoch is None:
        filename=pre_fix+net_name+'_'+loss_name
    else:
        filename=pre_fix+net_name+'_'+loss_name+'_epoch'+str(epoch)
    return filename
#%%
def main(epoch_start, epoch_end, train, arg, evaluate_model):
    main_train(epoch_start, epoch_end, train, arg)
    if evaluate_model == True:
        main_evaluate(epoch_end-1, arg)
#%%
def main_evaluate(epoch, arg):
    device=arg['device']
    norm_type=arg['norm_type']
    net_name=arg['net_name']
    loss_name=arg['loss_name']
    loader_train, loader_val, loader_test = get_dataloader()
    #del loader_train, loader_val
    #main_evaluate_rand(net_name, loss_name, epoch, device, loader_test, (0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.3))
    if norm_type == np.inf:
        noise_norm_list=(0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.3)
        print('Linf norm noise_norm_list', noise_norm_list)
    else:
        #noise_norm_list=(1.0, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)
        #noise_norm_list=(0.5,1.0,1.5,2.0,2.5,3.0)
        noise_norm_list=(1.0,2.0,3.0)
        print('L2 norm noise_norm_list', noise_norm_list)
    #main_evaluate_bba_spsa(net_name, loss_name, epoch, device, loader_bba, norm_type, noise_norm_list)
    #main_evaluate_wba(net_name, loss_name, epoch, device, 'bba', loader_bba, norm_type, noise_norm_list)
    main_evaluate_wba(net_name, loss_name, epoch, device, 'test', loader_test, norm_type, noise_norm_list)
#%%
def main_train(epoch_start, epoch_end, train, arg):
#%%
    net_name=arg['net_name']
    loss_name=arg['loss_name']
    filename=get_filename(net_name, loss_name)
    print('train model: '+filename)
    if epoch_start == epoch_end:
        print('epoch_end is epoch_start, exist main_train')
        return
    #---------------------------------------
    device=arg['device']
    lr=arg['lr']
    if 'reset_optimizer' not in arg.keys():
        arg['reset_optimizer']=False
    reset_optimizer=arg['reset_optimizer']
    if 'batch_size' not in arg.keys():
        arg['batch_size']=32
    batch_size=arg['batch_size']
    if 'return_idx' not in arg.keys():
        arg['return_idx']=(False, False, False)
    return_idx=arg['return_idx']
    norm_type=arg['norm_type']
#%%
    num_classes=2
    if norm_type == np.inf:
        noise_norm=0.05
        max_iter=1
        step=1.0
    elif norm_type == 2:
        noise_norm=10.0
        max_iter=1
        step=1.0
#%%
    loader_train, loader_val, loader_test = get_dataloader(batch_size=batch_size, return_idx=return_idx)
#%%
    loss_train_list=[]
    acc_train_list=[]
    acc_val_list=[]
    adv_acc_val_list=[]
    acc_test_list=[]
    epoch_save=epoch_start-1
#%%
    model=Net(net_name)
    if epoch_start > 0:
        print('load', filename+'_epoch'+str(epoch_save)+'.pt')
        checkpoint=torch.load(filename+'_epoch'+str(epoch_save)+'.pt', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        #------------------------
        loss_train_list=checkpoint['result']['loss_train_list']
        acc_train_list=checkpoint['result']['acc_train_list']
        acc_val_list=checkpoint['result']['acc_val_list']
        adv_acc_val_list=checkpoint['result']['adv_acc_val_list']
        if 'E' in arg.keys():
            arg['E']=checkpoint['result']['arg']['E']
            print('load E')
    #------------------------
    model.to(device)
    #------------------------
    if arg['optimizer']=='Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif arg['optimizer']=='AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    elif arg['optimizer']=='Adamax':
        optimizer = optim.Adamax(model.parameters(), lr=lr)
    elif arg['optimizer']=='SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.001, nesterov=True)
    else:
        raise NotImplementedError('unknown optimizer')
    if epoch_start > 0 and reset_optimizer == False:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('load optimizer')
#%%
    for epoch in range(epoch_save+1, epoch_end):
    #for epoch in tqdm(range(epoch_save+1, epoch_end), initial=epoch_save+1, total=epoch_end):
        #-------- training --------------------------------
        start = time.time()
        loss_train, acc_train =train(model, device, optimizer, loader_train, epoch, arg)
        loss_train_list.append(loss_train)
        acc_train_list.append(acc_train)
        print('epoch', epoch, 'training loss:', loss_train, 'acc:', acc_train)
        end = time.time()
        print('time cost:', end - start)
        #-------- validation --------------------------------
        """
        result_val = test_adv(model, device, loader_val, num_classes=num_classes,
                              noise_norm=noise_norm, norm_type=norm_type,
                              max_iter=max_iter, step=step, method='pgd', adv_loss_fn='lm')
        acc_val_list.append(result_val['acc_clean'])
        adv_acc_val_list.append(result_val['acc_noisy'])
        """
        #--------test-------------------------
        """
        result_test = test(model, device, loader_test, num_classes=num_classes)
        acc_test_list.append(result_test['acc'])
        """
        #--------save model-------------------------
        result={}
        result['arg']=arg
        result['loss_train_list'] =loss_train_list
        result['acc_train_list'] =acc_train_list
        result['acc_val_list'] =acc_val_list
        result['adv_acc_val_list'] =adv_acc_val_list
        result['acc_test_list']=acc_test_list
        if (epoch+1)%10 == 0:
            save_checkpoint(filename+'_epoch'+str(epoch)+'.pt', model, optimizer, result, epoch)
        epoch_save=epoch
        #------- show result ----------------------
        #plt.close('all')
        display.clear_output(wait=False)
        fig, ax = plot_result(loss_train_list, acc_train_list,
                              acc_val_list, adv_acc_val_list, acc_test_list)
        display.display(fig)
        fig.savefig(filename+'_epoch'+str(epoch)+'.png')
        plt.close(fig)
   
def main_evaluate_wba(net_name, loss_name, epoch, device, data_name, loader, norm_type, noise_norm_list):
#%%
    filename=get_filename(net_name, loss_name, epoch)
    checkpoint=torch.load(filename+'.pt', map_location=torch.device('cpu'))
    x=loader.dataset[0][0]
    model=Net(net_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(x.dtype).to(device)
    model.eval()
    print('evaluate_wba model in '+filename+'.pt')
    print(noise_norm_list)
    #%%IFGSM
    
    print ("IFGSM is testing...")
    num_repeats=1
    result_ifgsm=[]  
    for noise_norm in noise_norm_list:
        start = time.time()
        result_ifgsm.append(test_adv(model, device, loader, 2, noise_norm=noise_norm, norm_type=norm_type,
                                       max_iter=10, step=noise_norm/4, method='ifgsm', num_repeats=num_repeats))
        end = time.time()
        print('time cost:', end - start)    
    
    #%% 100pgd
    print ("pgd100 is testing...")
    num_repeats=2
    result_100pgd=[]
    
    for noise_norm in noise_norm_list:
        start = time.time()
        result_100pgd.append(test_adv(model, device, loader, num_classes=2,
                                      noise_norm=noise_norm, norm_type=norm_type,
                                      max_iter=100, step=noise_norm/10, method='pgd_ce_m',
                                      adv_loss_fn='lm',
                                      num_repeats=num_repeats,
                                      save_model_output=True))
        end = time.time()
        print('time cost:', end - start)

    #%%
 

    filename=filename+'_result_wba_L'+str(norm_type)+'_'+data_name+'.pt'
    torch.save({'result_100pgd':result_100pgd, 
                'result_ifgsm': result_ifgsm}, filename)
    print('saved:', filename)

def main_evaluate_rand(net_name, loss_name, epoch, device, loader, noise_norm_list):
#%%
    filename=get_filename(net_name, loss_name, epoch)
    checkpoint=torch.load(filename+'.pt', map_location=torch.device('cpu'))
    x=loader.dataset[0][0]
    model=Net(net_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(x.dtype).to(device)
    model.eval()
    print('evaluate_rand model in '+filename+'.pt')
    print(noise_norm_list)
    result_rand=[]
    for noise_norm in noise_norm_list:
        result_rand.append(test_rand(model, device, loader, num_classes=2, noise_norm=noise_norm, max_iter=100))
    noise=[0]
    acc=[result_rand[0]['acc_clean']]
    adv=[0]
    for k in range(0, len(result_rand)):
        noise.append(result_rand[k]['noise_norm'])
        acc.append(result_rand[k]['acc_noisy'])
        adv.append(result_rand[k]['adv_sample_count']/result_rand[k]['sample_count'])
    auc=cal_AUC_robustness(acc, noise)
    result_rand[0]['auc']=auc
    fig, ax = plt.subplots(1,2)
    ax[0].plot(noise, acc, '.-b')
    ax[0].set_ylim(0, 1)
    ax[0].set_yticks(np.arange(0, 1.05, step=0.05))
    ax[0].grid(True)
    ax[0].set_title('rand')
    ax[0].set_xlabel(filename)
    ax[1].plot(noise, adv, '.-b')
    ax[1].grid(True)
    ax[1].set_title('rand adv%'+' auc='+str(auc))
    display.display(fig)
    fig.savefig(filename+'_rand.png')
    plt.close(fig)
    #------------------------------------
    filename=filename+'_result_rand.pt'
    torch.save({'result_rand':result_rand}, filename)
    print('saved:', filename)
