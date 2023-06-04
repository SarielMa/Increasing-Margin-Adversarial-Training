import medmnist
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as torch_dataloader
import torchvision.transforms as transforms
from medmnist import INFO, Evaluator
from medmnist.dataset import PathMNIST
import os
from PIL import Image
# medminst https://medmnist.com/
# pathmnist is used in this case, 3x28x28
#%% inherit the 

class myDataset(PathMNIST):
    def __init__(self,
                 split,
                 transform=None,
                 target_transform=None,
                 download=False,
                 as_rgb=False,
                 root="",
                 return_idx = False):
        ''' dataset
        :param split: 'train', 'val' or 'test', select subset
        :param transform: data transformation
        :param target_transform: target transformation
        '''
        self.return_idx = return_idx
        self.info = INFO[self.flag]

        if root is not None and os.path.exists(root):
            self.root = root
        else:
            raise RuntimeError("Failed to setup the default `root` directory. " +
                               "Please specify and create the `root` directory manually.")

        if download:
            self.download()

        if not os.path.exists(
                os.path.join(self.root, "{}.npz".format(self.flag))):
            raise RuntimeError('Dataset not found. ' +
                               ' You can set `download=True` to download it')

        npz_file = np.load(os.path.join(self.root, "{}.npz".format(self.flag)))

        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.as_rgb = as_rgb

        if self.split == 'train':
            self.imgs = npz_file['train_images']
            self.labels = npz_file['train_labels'].reshape(-1)
        elif self.split == 'val':
            self.imgs = npz_file['val_images']
            self.labels = npz_file['val_labels'].reshape(-1)
        elif self.split == 'test':
            self.imgs = npz_file['test_images']
            self.labels = npz_file['test_labels'].reshape(-1)
        else:
            raise ValueError
            
    def __getitem__(self, index):
        '''
        return: (without transform/target_transofrm)
            img: PIL.Image
            target: np.array of `L` (L=1 for single-label)
        '''
        #print("here")
        img, target = self.imgs[index], self.labels[index]
        img = Image.fromarray(img)

        if self.as_rgb:
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        target = torch.tensor(target, dtype=torch.int64)
        
        if self.return_idx == True:
            return img, target, index
        else:
            return img, target 
    
#%%
def get_dataloader(batch_size=128, num_workers=2, return_idx=(False, False, False),
                   shuffle=(True, False, False), image_size=None,
                   path=r"../../data/MedMNIST"):

    # initialization
    data_flag = 'pathmnist'
    download = True
    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    data_transform = transforms.Compose([transforms.ToTensor()])
    train_set = myDataset(split='train', transform = data_transform, root = path, return_idx = return_idx[0])
    val_set = myDataset(split='val', transform = data_transform, root = path, return_idx = return_idx[1])
    test_set = myDataset(split='test', transform = data_transform, root = path, return_idx = return_idx[2])
    # get the data loader
    loader_train = torch_dataloader(train_set, batch_size=batch_size, num_workers=num_workers,
                                    shuffle=shuffle[0], pin_memory=True)
    loader_val = torch_dataloader(val_set, batch_size=batch_size, num_workers=num_workers,
                                  shuffle=shuffle[1], pin_memory=True)
    loader_test = torch_dataloader(test_set, batch_size=batch_size, num_workers=num_workers,
                                   shuffle=shuffle[2], pin_memory=True)
    
    return loader_train, loader_val, loader_test

