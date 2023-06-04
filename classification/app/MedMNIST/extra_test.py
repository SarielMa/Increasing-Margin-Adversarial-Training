#%%
import os
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
from MedMNIST_Dataset import get_dataloader
from Evaluate import test, test_rand, cal_AUC_robustness
from Evaluate_advertorch import test_adv, test_adv_auto
from advertorch_examples.models import get_cifar10_wrn28_widen_factor
import csv
from MedMNIST_CNN import Net

#%%


#%%
def main(norm_type, noise_norm_list):
    net_name = "resnet18"
    device= device=torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model=Net(net_name)
    loader_train, loader_val, loader_test = get_dataloader()
    del loader_train, loader_val
    base = "results_CMPB_revision/"    
    tasks = ["STD", "IMA","GAI","FAT","LBGAT","TRADES","SAT","TE","IAAT"]
    #tasks = ["MMA"]
    modelToTests = []
    for task in tasks:
        modelToTests.extend([name.split(".pt")[0] for name in os.listdir(base) if task in name and "test" not in name])
    assert len(modelToTests) == len(tasks), "model not found"
    print ("all models found!")
    savedcsv = base+"L"+str(norm_type)+"non_aa_result.csv"
    head = ["methods","fgsm", "ifgsm", "pgd", "pgdcw"]
    with open(savedcsv, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(head)   
    for i, model_name in enumerate(modelToTests):
        main_evaluate(base+model_name, tasks[i], model, device, 'test', loader_test, norm_type, noise_norm_list, savedcsv)

#%%

def main_evaluate(filename, task, model, device, data_name, loader, norm_type, noise_norm_list, savedcsv):
    checkpoint=torch.load(filename+'.pt', map_location=torch.device('cpu'))    
    if "MMA" in task:
        model.load_state_dict(checkpoint['model'])
    elif "TE" in task:
        model.load_state_dict(checkpoint)
    elif "LBGAT" in task:
        #model = nn.DataParallel(model).to(device)
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print('evaluate_wba model in '+filename+'.pt')
    result_10pgd=[]
    result_10if=[]
    result_fgsm=[]
    result_cwpgd=[]
    result_to_print = [task]
    
    #%% fgsm
    num_repeats=10  
    for noise_norm in noise_norm_list:
        start = time.time()
        result_fgsm.append(test_adv(model, device, loader, 10, noise_norm=noise_norm, norm_type=norm_type,
                                      max_iter=1, step=noise_norm, method='ifgsm', num_repeats=num_repeats))
        end = time.time()
        print('time cost:', end - start)
    noise=[0]
    acc=[result_fgsm[0]['acc_clean']]
    for k in range(0, len(result_fgsm)):
        noise.append(result_fgsm[k]['noise_norm'])
        acc.append(result_fgsm[k]['acc_noisy'])
    
    result_to_print.append(acc[-1])
    #%% ifgsm10
    num_repeats=10  
    for noise_norm in noise_norm_list:
        start = time.time()
        result_10if.append(test_adv(model, device, loader, 10, noise_norm=noise_norm, norm_type=norm_type,
                                      max_iter=10, step=noise_norm/4, method='ifgsm', num_repeats=num_repeats))
        end = time.time()
        print('time cost:', end - start)
    noise=[0]
    acc=[result_10if[0]['acc_clean']]
    for k in range(0, len(result_10if)):
        noise.append(result_10if[k]['noise_norm'])
        acc.append(result_10if[k]['acc_noisy'])
    result_to_print.append(acc[-1])
    
    #%% 10pgd
    num_repeats=10  
    for noise_norm in noise_norm_list:
        start = time.time()
        result_10pgd.append(test_adv(model, device, loader, 10, noise_norm=noise_norm, norm_type=norm_type,
                                      max_iter=10, step=noise_norm/4, method='pgd', num_repeats=num_repeats))
        end = time.time()
        print('time cost:', end - start)
    noise=[0]
    acc=[result_10pgd[0]['acc_clean']]
    for k in range(0, len(result_10pgd)):
        noise.append(result_10pgd[k]['noise_norm'])
        acc.append(result_10pgd[k]['acc_noisy'])
    #auc=cal_AUC_robustness(acc, noise)
    result_to_print.append(acc[-1])
#%%pgd cw
    num_repeats=10  
    for noise_norm in noise_norm_list:
        start = time.time()
        result_cwpgd.append(test_adv(model, device, loader, 10, noise_norm=noise_norm, norm_type=norm_type,
                                      max_iter=10, step=noise_norm/4, method='pgd_ce_cw', num_repeats=num_repeats))
        end = time.time()
        print('time cost:', end - start)
    noise=[0]
    acc=[result_cwpgd[0]['acc_clean']]
    for k in range(0, len(result_cwpgd)):
        noise.append(result_cwpgd[k]['noise_norm'])
        acc.append(result_cwpgd[k]['acc_noisy'])
    #auc=cal_AUC_robustness(acc, noise)
    #print('pgdcw auc is ', auc) 
    result_to_print.append(acc[-1])
    #%%
    with open(savedcsv, 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(result_to_print)    
    savename= filename +'_result_non_auto_wba_L'+str(norm_type)+'_r'+str(num_repeats)+'_'+data_name+'.pt'
    torch.save({'result_cwpgd':result_cwpgd,
                'result_10pgd':result_10pgd,
                'result_10if':result_10if,
                'result_fgsm':result_fgsm}, savename)
    print('saved:', savename)

#%%
if __name__ == "__main__":
    #main_evaluate()
    model = 0
    device = 1
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=str(device)
    noise_norm_lists = [[2/255], [0.3]]
    types = [np.inf, 2]
    for i, norm_type in enumerate(types):
        if i == 0:
            continue
        main(norm_type, noise_norm_lists[i])