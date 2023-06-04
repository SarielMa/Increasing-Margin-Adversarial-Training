import torch
from torch.utils.data import DataLoader as torch_dataloader
from torch.utils.data import Dataset as torch_dataset
import numpy as np
import matplotlib.pyplot as plt
import skimage
import skimage.io as io
import glob
import pandas as pd
#%%
class MyDataset(torch_dataset):
    def __init__(self, path, filenamelist, labellist, return_idx=False):
        self.path=path
        self.filenamelist=filenamelist
        self.labellist=labellist
        self.return_idx=return_idx
    def __len__(self):
        #return the number of data points
        return len(self.filenamelist)
    def __getitem__(self, idx):
        I=io.imread(self.path+self.filenamelist[idx])
        I=skimage.util.img_as_float32(I)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        #if len(I.shape) ==3:
        #    I = I.transpose((2, 0, 1))
        #
        I = I.reshape(1,I.shape[0],I.shape[1])
        I = torch.tensor(I, dtype=torch.float32)
        label=torch.tensor(self.labellist[idx], dtype=torch.int64)
        if self.return_idx == True:
            return I, label, idx
        else:
            return I, label
#%%
def get_dataloader(batch_size=32, num_workers=2, return_idx=(False, False, False),
                   shuffle=(True, False, False), image_size=224,
                   path='../../data/COVID19a/SARS-Cov-2/'):
    if image_size == 224:
        folder='S224/'
    elif image_size == 256:
        folder='S256/'
    path=path+folder
    df_train=pd.read_csv(path+'train.csv')
    df_val=pd.read_csv(path+'val.csv')
    df_test=pd.read_csv(path+'test.csv')

    dataset_train = MyDataset(path, df_train['filename'].values, df_train['label'].values, return_idx=return_idx[0])
    dataset_val = MyDataset(path, df_val['filename'].values, df_val['label'].values, return_idx=return_idx[1])
    dataset_test = MyDataset(path, df_test['filename'].values, df_test['label'].values, return_idx=return_idx[2])
    loader_train = torch_dataloader(dataset_train, batch_size=batch_size, num_workers=num_workers,
                                    shuffle=shuffle[0], pin_memory=True)
    loader_val = torch_dataloader(dataset_val, batch_size=batch_size, num_workers=num_workers,
                                  shuffle=shuffle[1], pin_memory=True)
    loader_test = torch_dataloader(dataset_test, batch_size=batch_size, num_workers=num_workers,
                                   shuffle=shuffle[2], pin_memory=True)
    return loader_train, loader_val, loader_test

