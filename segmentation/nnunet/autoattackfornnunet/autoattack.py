import math
import time

import numpy as np
import torch
import torch.nn as nn
from .other_utils import Logger
from autoattack import checks
from nnunet.training.loss_functions.dice_loss import SoftDiceLoss, DiceIndex, MyDiceIndex
import torch.nn.functional as F
from nnunet.utilities.nd_softmax import softmax_helper


def one_hot_old(gt, categories):
    # Check the new function in PyTorch!!!
    size = [*gt.shape] + [categories]
    y = gt.view(-1, 1)
    gt = torch.FloatTensor(y.nelement(), categories).zero_().cuda()
    gt.scatter_(1, y, 1)
    gt = gt.view(size).permute(0, 3, 1, 2).contiguous()
    return gt

def one_hot(src, shape):
    onehot = torch.zeros(shape)
    src = src.long()
    if src.device.type == "cuda":
        onehot = onehot.cuda(src.device.index)
    onehot.scatter_(1, src, 1)
    return onehot

class Dice_metric(nn.Module):
    def __init__(self, eps=1e-5):
        super(Dice_metric, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets, logits=True):    
        targets = one_hot(targets, inputs.shape)     
        if logits:
            inputs2 = torch.argmax(softmax_helper(inputs) , dim = 1, keepdim = True)
        inputs2 = one_hot(inputs2, inputs.shape)
        dims = tuple(range(2, targets.ndimension()))
        tps = torch.sum(inputs2 * targets, dims)
        fps = torch.sum(inputs2 * (1 - targets), dims)
        fns = torch.sum((1 - inputs2) * targets, dims)
        loss = (2 * tps) / (2 * tps + fps + fns + self.eps)
        return loss[:, 1:].mean(dim=1)


class AutoAttack():
    def __init__(self, model, norm_type='Linf', eps=.3, seed=None, verbose=True,
                 attacks_to_run=[], version='standard', is_tf_model=False,
                 device='cuda', log_path=None, loss_fn=None):
        self.model = model
        self.norm = norm_type
        assert self.norm in ['Linf', 'L2', 'L1']
        self.epsilon = eps
        self.seed = seed
        self.verbose = verbose
        self.attacks_to_run = attacks_to_run
        self.version = version
        self.is_tf_model = is_tf_model
        self.device = device
        self.logger = Logger(log_path)
        self.threshold = 0
        self.dice = Dice_metric()
        self.loss_fn = loss_fn
        self.prev = None

        if version in ['standard', 'plus', 'rand'] and attacks_to_run != []:
            raise ValueError("attacks_to_run will be overridden unless you use version='custom'")
        
        if not self.is_tf_model:
            from .autopgd import APGDAttack
            self.apgd = APGDAttack(
                self.model, dice_thresh =self.threshold, n_restarts=5, n_iter=100,
                verbose=False, eps=self.epsilon, eot_iter=1,
                rho=.75, seed=self.seed, device=self.device, loss_fn = self.loss_fn, norm_type = self.norm)
            """
            from .fab import FABAttack
            self.fab = FABAttack(
                self.model, dice_thresh=self.threshold, n_target_classes=n_target_classes,
                n_restarts=5, n_iter=100, eps=self.epsilon, seed=self.seed,
                verbose=False, device=self.device)
            """
            from .square import SquareAttack
            self.square = SquareAttack(
                self.model, dice_thresh=self.threshold, p_init=0.8, n_queries=100,
                eps=self.epsilon, n_restarts=1, seed=self.seed,
                verbose=False, device=self.device, resc_schedule=False,  norm_type = self.norm)
                
    
        else:
            from .autopgd_base import APGDAttack
            self.apgd = APGDAttack(self.model, n_restarts=5, n_iter=100, verbose=False,
                eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed, device=self.device,
                is_tf_model=False, logger=self.logger)
            
            from .fab_tf import FABAttack_TF
            self.fab = FABAttack_TF(self.model, n_restarts=5, n_iter=100, eps=self.epsilon, seed=self.seed,
                norm=self.norm, verbose=False, device=self.device)
        
            from .square import SquareAttack
            self.square = SquareAttack(self.model.predict, p_init=.8, n_queries=5000, eps=self.epsilon, norm=self.norm,
                n_restarts=1, seed=self.seed, verbose=False, device=self.device, resc_schedule=False)
                
            from .autopgd_base import APGDAttack_targeted
            self.apgd_targeted = APGDAttack_targeted(self.model, n_restarts=1, n_iter=100, verbose=False,
                eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed, device=self.device,
                is_tf_model=True, logger=self.logger)
    
        #if version in ['standard', 'plus', 'rand']:
        #    self.set_version(version)
        
    def get_logits_classification(self, x):
        if not self.is_tf_model:
            return self.model(x)
        else:
            return self.model.predict(x)
        
    def get_logits(self, x):
        assert not self.is_tf_model
        rawoutput = self.model(x)[0]
        output_softmax = softmax_helper(rawoutput)
        return output_softmax.argmax(1) # in fact this is not the logit
    
    def get_seed(self):
        return time.time() if self.seed is None else self.seed
    
    def run_standard_evaluation(self, x_orig, y_orig, bs=250, return_labels=False):
        if self.verbose:
            print('using {} version including {}'.format(self.version,
                ', '.join(self.attacks_to_run)))
        
        with torch.no_grad():
            # calculate accuracy
            n_batches = int(np.ceil(x_orig.shape[0] / bs))                    
            x_adv = x_orig.clone().detach()
            startt = time.time()
            # start runing all the attacks   
            self.prev = torch.ones(x_orig.shape[0]).to(self.device)
            
            for attack in self.attacks_to_run:
                # item() is super important as pytorch int division uses floor rounding
                n_batches = int(np.ceil( x_orig.shape[0]/ bs))               
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * bs
                    end_idx = min((batch_idx + 1) * bs, x_orig.shape[0])
                        
                        
                    x = x_orig.clone().to(self.device)
                    y = [y_orig[i].clone().to(self.device) for i in range(len(y_orig))]

                    # make sure that x is a 4d tensor even if there is only a single datapoint left
                    if len(x.shape) == 3:
                        x.unsqueeze_(dim=0)
                    
                    # run attack
                    if attack == 'apgd-ce':
                        # apgd on cross-entropy loss
                        print("apgd is running")
                        self.apgd.loss = 'dice'
                        self.apgd.threshold = self.threshold
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y) #cheap=True    
                        
                    elif attack == 'apgd-dlr':
                        print ("apgd dlr is running")
                        self.apgd.loss = 'dlr'
                        self.apgd.threshold = self.threshold
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y)
                    elif attack == 'square':
                        # square
                        self.square.seed = self.get_seed()
                        self.apgd.threshold = self.threshold
                        adv_curr = self.square.perturb(x, y)                                  
                    else:
                        raise ValueError('Attack not supported')
                
                    output = self.model(adv_curr)                   
                    dices = self.dice(output[0], y[0]).to(self.device)
                    false_batch = (dices < self.prev).to(x_adv.device)
                    self.prev = dices
                    #  only record those samples with lower dice scores than the last ones
                    x_adv[false_batch] = adv_curr[false_batch].detach().to(x_adv.device)
        return x_adv
        



