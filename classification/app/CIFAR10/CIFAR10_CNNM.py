import sys
sys.path.append('../../core')
#%%
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim
import time
from CIFAR10_Dataset_new import get_dataloader
from Evaluate import test, test_rand, cal_AUC_robustness
from Evaluate_advertorch import test_adv, test_adv_auto

#%%
'''
#%% https://github.com/meliketoy/wide-resnet.pytorch
class CIFAR(nn.Module):
    def __init__(self, features, n_channel, num_classes):
        super(CIFAR, self).__init__()
        assert isinstance(features, nn.Sequential), type(features)
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(n_channel, num_classes)
        )
        print(self.features)
        print(self.classifier)

    def forward(self, x):
        x=(x-0.5)/0.5
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            padding = v[1] if isinstance(v, tuple) else 1
            out_channels = v[0] if isinstance(v, tuple) else v
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(out_channels, affine=False), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = out_channels
    return nn.Sequential(*layers)

def Net(n_channel=128, pretrained=False):
    cfg = [n_channel, n_channel, 'M', 2*n_channel, 2*n_channel, 'M', 4*n_channel, 4*n_channel, 'M', (8*n_channel, 0), 'M']
    layers = make_layers(cfg, batch_norm=True)
    model = CIFAR(layers, n_channel=8*n_channel, num_classes=10)
    if pretrained:
        print('load pretrained CIFAR10')
        data=torch.load('../pytorch-playground-master/cifar10-d875770b.pth')
        model.load_state_dict(data)
    return model
'''
#%%
def Net(net_name):
    if net_name == 'mmacifar10':
        from advertorch_examples.models import get_cifar10_wrn28_widen_factor
        model = get_cifar10_wrn28_widen_factor(4)
    elif net_name == 'mmacifar10g':
        from models import get_cifar10_wrn28_widen_factor
        model = get_cifar10_wrn28_widen_factor(4)
    elif net_name == 'wrn28_10':
        from advertorch_examples.models import get_cifar10_wrn28_widen_factor
        model = get_cifar10_wrn28_widen_factor(10)
    else:
        raise ValueError('invalid net_name ', net_name)
    return model
#%%
def update_lr(optimizer, new_lr):
    for g in optimizer.param_groups:
        g['lr']=new_lr
        print('new lr=', g['lr'])
#%%
def save_checkpoint(filename, model, optimizer, result, epoch):
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'result':result},
               filename)
    print('saved:', filename)
#%%
def plot_result(loss_train_list, acc_train_list,
                acc_val_list, adv_acc_val_list, acc_test_list):
    fig, ax = plt.subplots(1, 3, figsize=(9,3))
    ax[0].set_title('loss v.s. epoch')
    ax[0].plot(loss_train_list, '-b', label='train')
    ax[0].set_xlabel('epoch')
    #ax[0].legend()
    ax[0].grid(True)
    ax[1].set_title('accuracy v.s. epoch')
    ax[1].plot(acc_train_list, '-b', label='train')
    ax[1].plot(acc_val_list, '-m', label='val')
    ax[1].plot(acc_test_list, '-r', label='test')
    ax[1].set_xlabel('epoch')
    ax[1].set_ylim(0.5, 1)
    #ax[1].legend()
    ax[1].grid(True)
    ax[2].set_title('accuracy v.s. epoch')
    ax[2].plot(adv_acc_val_list, '-m', label='adv val')
    ax[2].set_xlabel('epoch')
    ax[2].set_ylim(0, 0.8)
    #ax[2].legend()
    ax[2].grid(True)
    return fig, ax
#%%
def get_filename(net_name, loss_name, epoch=None, pre_fix='result/CIFAR10_'):
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
def get_noise_norm_list(norm_type):
    if norm_type == np.inf:
        noise_norm_list=[2/255, 4/255, 8/255]
    else:
        #noise_norm_list=[0.5, 1.0, 1.5, 2.0, 2.5]
        noise_norm_list = [0.1,0.3,0.5]
    return noise_norm_list
#%%
def main_evaluate(epoch, arg):
    net_name=arg['net_name']
    loss_name=arg['loss_name']
    device=arg['device']
    norm_type=arg['norm_type']
    noise_norm_list=get_noise_norm_list(norm_type)
    loader_train, loader_val, loader_test = get_dataloader()
    del loader_train, loader_val
    main_evaluate_wba(net_name, loss_name, epoch, device, 'test', loader_test, norm_type, noise_norm_list)
#%%
def main_train(epoch_start, epoch_end, train, arg):
#%%
    net_name=arg['net_name']
    loss_name=arg['loss_name']
    filename=get_filename(net_name, loss_name)
    print('train model: '+filename)
    device=arg['device']
    if 'pretrained_model' not in arg.keys():
        arg['pretrained_model']='none'
    pretrained_model=arg['pretrained_model']
    if 'reset_optimizer' not in arg.keys():
        arg['reset_optimizer']=False
    reset_optimizer=arg['reset_optimizer']
    if 'batch_size' not in arg.keys():
        arg['batch_size']=128
    batch_size=arg['batch_size']
    if 'return_idx' not in arg.keys():
        arg['return_idx']=(False, False, False)
    return_idx=arg['return_idx']
    norm_type=arg['norm_type']
    #---------------------------------------
    if 'DataParallel' not in arg.keys():
        arg['DataParallel']=False
    DataParallel=arg['DataParallel']
    #-----
    if 'data_aug' not in arg.keys():
        arg['data_aug']=True
    data_aug=arg['data_aug']
#%%
    if norm_type == np.inf:
        noise_norm=8/255
    elif norm_type == 2:
        noise_norm=1.0
#%%
    loader_train, loader_val, loader_test = get_dataloader(batch_size=batch_size, return_idx=return_idx, data_aug=data_aug)
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
            if arg['E'] is None:
                arg['E']=checkpoint['result']['arg']['E']
                print('load E')
    elif pretrained_model != 'none':
        print('load pretrained_model', pretrained_model)
        checkpoint=torch.load(pretrained_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
    #------------------------
    if DataParallel == True:
        print('DataParallel')
        torch.cuda.set_device(arg['device_ids'][0])
        model=nn.DataParallel(model, device_ids=arg['device_ids'])
        model.to(torch.device('cuda'))
    else:
        model.to(device)
    #------------------------
    if arg['optimizer']=='Adam':
        optimizer = optim.Adam(model.parameters(), lr=arg['lr'])
    elif arg['optimizer']=='AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=arg['lr'])
    elif arg['optimizer']=='Adamax':
        optimizer = optim.Adamax(model.parameters(), lr=arg['lr'])
    elif arg['optimizer']=='SGD':
        optimizer = optim.SGD(model.parameters(), lr=arg['lr'], momentum=0.9, weight_decay=0.0001, nesterov=False)
    else:
        raise NotImplementedError('unknown optimizer')
    if epoch_start > 0 and reset_optimizer == False:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('load optimizer state')
        update_lr(optimizer, arg['lr'])
#%%
    train_time = 0
    for epoch in range(epoch_save+1, epoch_end):
        start = time.time()
        #-------- training --------------------------------
        loss_train, acc_train =train(model, device, optimizer, loader_train, epoch, arg)
        train_time += (time.time() - start)
        loss_train_list.append(loss_train)
        acc_train_list.append(acc_train)
        print('epoch', epoch, 'training loss:', loss_train, 'acc:', acc_train)
        #-------- validation --------------------------------
        #result_val=test(model, device, loader_val, num_classes=10)
        #acc_val_list.append(result_val['acc'])
        if (epoch+1)%50 == 0:
            result_val = test_adv(model, device, loader_val, num_classes=10,
                                  noise_norm=noise_norm, norm_type=norm_type,
                                  max_iter=100, step=noise_norm, method='pgd')
            acc_val_list.append(result_val['acc_clean'])
            adv_acc_val_list.append(result_val['acc_noisy'])
        
        result_test=test(model, device, loader_test, num_classes=10)
        acc_test_list.append(result_test['acc'])
        #--------save model-------------------------
        result={}
        result['arg']=arg
        result['loss_train_list'] =loss_train_list
        result['acc_train_list'] =acc_train_list
        result['acc_val_list'] =acc_val_list
        result['adv_acc_val_list'] =adv_acc_val_list
        if 'E' in arg.keys():
            result['E']=arg['E']
        if (epoch+1)%10 == 0 or (epoch+1) == 98 :
            save_checkpoint(filename+'_epoch'+str(epoch)+'.pt', model, optimizer, result, epoch)
        epoch_save=epoch
        #------- show result ----------------------
        fig, ax = plot_result(loss_train_list, acc_train_list,
                              acc_val_list, adv_acc_val_list, acc_test_list)
        display.display(fig)
        fig.savefig(filename+'_epoch'+str(epoch)+'.png')
        plt.close(fig)
        end = time.time()
        print('time cost:', end - start)
        #-------check if termination is needed----------------------------------------------
        if 'termination_condition' in arg:
            if arg['termination_condition'] == arg['no_expand_times']:
                print ("termination condition is met, terminate training")
                save_checkpoint(filename+'_epoch'+str(epoch)+'.pt', model, optimizer, result, epoch)
                break
    print ("================== train time: ",train_time,"=============================================")
    
#%%
def main_evaluate_wba(net_name, loss_name, epoch, device, data_name, loader, norm_type, noise_norm_list):
    #%%
    filename=get_filename(net_name, loss_name, epoch)
    checkpoint=torch.load(filename+'.pt', map_location=torch.device('cpu'))
    model=Net(net_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print('evaluate_wba model in '+filename+'.pt')
    print(noise_norm_list)
    result_100pgd=[]
    result_auto=[]
    result_ifgsm=[]
    #%% auto attack
    num_repeats=1
    
    for noise_norm in noise_norm_list:
        start = time.time()
        result_auto.append(test_adv_auto(model, device, loader, 10, noise_norm=noise_norm, norm_type=norm_type,
                                      max_iter=100, step=noise_norm/4, method='auto', num_repeats=num_repeats))
        end = time.time()
        print('time cost:', end - start)
    noise=[0]
    acc=[result_auto[0]['acc_clean']]
    for k in range(0, len(result_auto)):
        noise.append(result_auto[k]['noise_norm'])
        acc.append(result_auto[k]['acc_noisy'])
    auc=cal_AUC_robustness(acc, noise)
    print('auto auc is ', auc) 
    
    # can add other attacks here
#%%
    
    fig, ax = plt.subplots()
    ax.plot(noise, acc, '.-b')
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.05, step=0.05))
    ax.grid(True)
    title='wba_100pgd_norm_type_'+str(norm_type)+'_r'+str(num_repeats)+' auc='+str(auc)+' '+data_name
    ax.set_title(title)
    ax.set_xlabel(filename)
    display.display(fig)
    
    fig.savefig(filename+'_'+title+'.png')
    plt.close(fig)
    
    #%%
    filename=filename+'_result_wba_L'+str(norm_type)+'_r'+str(num_repeats)+'_'+data_name+'.pt'
    torch.save({'result_auto':result_auto, 
                'result_100pgd_ce_cw':result_100pgd,
                'result_100pgd': result_ifgsm}, filename)
    print('saved:', filename)
#%%

#%% add rand noise to image
def main_evaluate_rand(net_name, loss_name, epoch, device, loader, noise_norm_list):
#%%
    filename=get_filename(net_name, loss_name, epoch)
    checkpoint=torch.load(filename+'.pt', map_location=torch.device('cpu'))
    model=Net(net_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print('evaluate_rand model in '+filename+'.pt')
    result_rand=[]
    for noise_norm in noise_norm_list:
        result_rand.append(test_rand(model, device, loader, 10, noise_norm=noise_norm))
    noise=[0]
    acc=[result_rand[0]['acc_clean']]
    adv=[0]
    for k in range(0, len(result_rand)):
        noise.append(result_rand[k]['noise_norm'])
        acc.append(result_rand[k]['acc_noisy'])
        adv.append(result_rand[k]['adv_sample_count']/result_rand[k]['sample_count'])
    fig, ax = plt.subplots(1,2)
    ax[0].plot(noise, acc, '.-b')
    ax[0].set_ylim(0, 1)
    ax[0].set_yticks(np.arange(0, 1.05, step=0.05))
    ax[0].grid(True)
    ax[0].set_title('rand')
    ax[0].set_xlabel(filename)
    ax[1].plot(noise, adv, '.-b')
    ax[1].grid(True)
    ax[1].set_title('rand adv%')
    display.display(fig)
    fig.savefig(filename+'_rand.png')
    plt.close(fig)
    #------------------------------------
    filename=filename+'_result_rand.pt'
    torch.save({'result_rand':result_rand}, filename)
    print('saved:', filename)
#%%
