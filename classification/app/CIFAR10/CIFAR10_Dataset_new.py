import torch
from torch.utils.data import DataLoader as torch_dataloader
from torch.utils.data import Dataset as torch_dataset
from torchvision import transforms as tv_transforms
import numpy as np
#from ClassBalancedSampler import ClassBalancedSampler
#%%
'''
# this is how to get the .pt dataset file
from torchvision import datasets, transforms
train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('../data', train=True, download=True),
                                           batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('../data', train=False, download=False),  
                                          batch_size=64, shuffle=False)
#
data_train=train_loader.dataset
X_train=[]
Y_train=[]
for n in range(0, len(data_train)):
    X_train.append(data_train[n][0])
    Y_train.append(data_train[n][1])
X_train=np.array(X_train, dtype='object')
Y_train=np.array(Y_train, dtype='object')
#
data_test=test_loader.dataset
X_test=[]
Y_test=[]
for n in range(0, len(data_test)):
    X_test.append(data_test[n][0])
    Y_test.append(data_test[n][1])
X_test=np.array(X_test, dtype='object')
Y_test=np.array(Y_test, dtype='object')
#
rng=np.random.RandomState(0)
idxlist=np.arange(0, len(X_train))
rng.shuffle(idxlist)
idxlist_train=idxlist[0:45000] # 90% for training
idxlist_val=idxlist[45000:] # 10% for val
data={}
data['X_train']=X_train[idxlist_train]
data['Y_train']=Y_train[idxlist_train]
data['X_val']=X_train[idxlist_val]
data['Y_val']=Y_train[idxlist_val]
data['X_test']=X_test
data['Y_test']=Y_test
torch.save(data, 'cifar10_data_new.pt')
'''
#%%
class MyDataset(torch_dataset):
    def __init__(self, X, Y, x_shape, return_idx=False, transform=None):
        self.X=X
        self.Y=Y
        self.x_shape=x_shape
        self.return_idx=return_idx
        self.transform=transform
    def __len__(self):
        #return the number of data points
        return self.X.shape[0]        
    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform is not None:
            x=self.transform(x)        
        x=x.view(self.x_shape)
        y = self.Y[idx]
        if self.return_idx == False:
            return x, y
        else:
            return x, y, idx
#%%
def get_dataloader(batch_size=128, num_workers=2, x_shape=(3,32,32), return_idx=(False, False, False), data_aug=True):
    data = torch.load('../../data/CIFAR10/cifar10_data_new.pt')
    transform=tv_transforms.ToTensor()
    if data_aug == True:
        transform=tv_transforms.Compose([tv_transforms.Pad(4, padding_mode='reflect'),
                                         tv_transforms.RandomCrop(32),
                                         tv_transforms.RandomHorizontalFlip(),
                                         tv_transforms.ToTensor()])
    #--------
    #debug
    #data['X_train']=data['X_train'][::10]
    #data['Y_train']=data['Y_train'][::10]
    #--------
    dataset_train = MyDataset(data['X_train'], data['Y_train'], x_shape, return_idx=return_idx[0], transform=transform)
    dataset_val = MyDataset(data['X_val'], data['Y_val'], x_shape, return_idx=return_idx[1], 
                            transform=tv_transforms.ToTensor())
    dataset_test = MyDataset(data['X_test'], data['Y_test'], x_shape, return_idx=return_idx[2], 
                             transform=tv_transforms.ToTensor())
    #sampler_train = ClassBalancedSampler(data['Y_train'], True)
    #loader_train = torch_dataloader(dataset_train, batch_size=batch_size, sampler=sampler_train,
    #                                num_workers=num_workers, pin_memory=True)
    loader_train = torch_dataloader(dataset_train, batch_size=batch_size, shuffle=True, 
                                    num_workers=num_workers, pin_memory=True)
    loader_val = torch_dataloader(dataset_val, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=True)
    loader_test = torch_dataloader(dataset_test, batch_size=batch_size, shuffle=False,
                                   num_workers=num_workers, pin_memory=True)
    return loader_train, loader_val, loader_test
