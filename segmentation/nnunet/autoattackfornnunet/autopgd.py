import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.training.loss_functions.dice_loss import SoftDiceLoss, MySoftDiceLoss
#from libs.utilities.utils import one_hot
#from libs.utilities.losses import Dice_metric

def L2_norm(x, keepdim=False):
    z = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
    if keepdim:
        z = z.view(-1, *[1]*(len(x.shape) - 1))
    return z 

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

class APGDAttack():
    """
    Auto Projected Gradient Descent (Linf)
    """
    def __init__(self, model, dice_thresh, n_iter=20, n_restarts=1, eps=None,
                 seed=0, loss='bce', eot_iter=1, rho=.75, verbose=False,
                 device='cuda', norm='Linf', loss_fn = None, norm_type = None):
        self.model = model
        self.n_iter = n_iter
        self.eps = eps
        self.n_restarts = n_restarts
        self.seed = seed
        self.loss = loss
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.verbose = verbose
        self.device = device
        # Dice loss
        self.dice_thresh = dice_thresh
        self.dice = Dice_metric(eps=1e-5)
        #self.dice = dice
        #self.loss_fn = MySoftDiceLoss()
        self.loss_fn = loss_fn
        self.norm = norm_type

    def init_hyperparam(self, x):

        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)
        if self.seed is None:
            self.seed = time.time()      

    def normalize(self, x):
        if self.norm == 'Linf':
            t = x.abs().view(x.shape[0], -1).max(1)[0]

        elif self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
            
        return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)
            
    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = np.zeros(x.shape[1])
        for counter5 in range(k):
            t += x[j - counter5] > x[j - counter5 - 1]

        return t <= (k * k3 * np.ones(t.shape))

    def check_shape(self, x):
        return x if len(x.shape) > 0 else np.expand_dims(x, 0)

    def dlr_loss(self, x, y):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        net_numpool = len(y)
        weights = np.array([1 / (2 ** i) for i in range(len(y))])
        mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
        weights[~mask] = 0
        weights = weights / weights.sum()
        #weights = [1] * len(x)
        l = weights[0] * self.dlr_loss_single(x[0], y[0])
        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.dlr_loss_single(x[i], y[i])
        return l 
    
    def dlr_loss_single(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)     
        output_softmax = softmax_helper(x_sorted)
        output_seg = torch.argmax(output_softmax, dim = 1, keepdim = True)
        
        ind = (ind_sorted[:,-1] == y[:,0]).float()
        # numerator
        corr_logits = torch.gather(x, 1, index=y)
        numerator = corr_logits - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)
        # denominator
        denominator = x_sorted[:, -1] - x_sorted[:, -3] + 1e-12
        loss = -numerator / denominator
        loss = loss.mean(dim=(1, 2, 3))  # collapse volumetric dims
        return loss
 
    def attack_single_run(self, x_in, y_in):
        x = x_in.clone() 
        y = y_in
        y = [y[i].long() for i in range(len(y))]
        #y = y.long()
        self.n_iter_2 = max(int(0.22 * self.n_iter), 1)
        self.n_iter_min = max(int(0.06 * self.n_iter), 1)
        self.size_decr = max(int(0.03 * self.n_iter), 1)
        if self.verbose:
            print(
                'parameters: ', self.n_iter, self.n_iter_2,
                self.n_iter_min, self.size_decr)
            
        if self.norm == "Linf":
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach()* t / (t.reshape([t.shape[0], -1]).abs().max(
                    dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1]))
        elif self.norm == "L2":
            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x + self.eps * torch.ones_like(x
                ).detach() * self.normalize(t)
        else:
            raise ValueError('norm not supported')
            
        x_adv = x_adv.clamp(0., 1.)        
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.n_iter, x.shape[0]])
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]])

        if self.loss == 'ce':
            xent = nn.CrossEntropyLoss(reduce=False, reduction='none')
            criterion_indiv = lambda x, y: xent(x, y).mean(dim=(1, 2, 3))
            #criterion_indiv = nn.CrossEntropyLoss(reduce=False, reduction='none')
        elif self.loss == 'dice':
            criterion_indiv = self.loss_fn
        elif self.loss == 'dlr':
            criterion_indiv = self.dlr_loss
        else:
            raise ValueError('unknowkn loss')

        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                logits = self.model(x_adv)  # 1 forward pass (eot_iter = 1)
                #loss_indiv = criterion_indiv(logits[0].view(-1,3), y.view(-1))
                #loss_indiv = loss_indiv.view(logits[0].shape[0], -1).mean(1)
                loss_indiv = criterion_indiv(logits, y)
                loss = loss_indiv.sum()

            # 1 backward pass (eot_iter = 1)
            grad += torch.autograd.grad(loss, [x_adv])[0].detach()

        grad /= float(self.eot_iter)
        grad_best = grad.clone()

        dice_best =  self.dice(logits[0].detach(), y[0])
        loss_best = loss_indiv.detach().clone()

        step_size = self.eps * torch.ones(
            [x.shape[0], 1, 1, 1]).to(self.device).detach()\
            * torch.Tensor([2.0]).to(self.device).detach().reshape(
                [1, 1, 1, 1])
        x_adv_old = x_adv.clone()
        k = self.n_iter_2 + 0
        u = np.arange(x.shape[0])
        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)

        for i in range(self.n_iter):
            # # # gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0
                if self.norm == "Linf":
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - self.eps), x + self.eps), 0.0, 1.0)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a), x - self.eps), x + self.eps),0.0, 1.0)
                elif self.norm == "L2":
                    x_adv_1 = x_adv + step_size * self.normalize(grad)
                    x_adv_1 = torch.clamp(x + self.normalize(x_adv_1 - x
                        ) * torch.min(self.eps * torch.ones_like(x).detach(),
                        L2_norm(x_adv_1 - x, keepdim=True)), 0.0, 1.0)
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    x_adv_1 = torch.clamp(x + self.normalize(x_adv_1 - x
                        ) * torch.min(self.eps * torch.ones_like(x).detach(),
                        L2_norm(x_adv_1 - x, keepdim=True)), 0.0, 1.0)

                else:
                    raise ValueError('norm not supported')
                    
                x_adv = x_adv_1 + 0.
            # # # get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    logits = self.model(x_adv)  # 1 forward pass (eot_iter = 1)
                    loss_indiv = criterion_indiv(logits, y)
                    loss = loss_indiv.sum()

                # 1 backward pass (eot_iter = 1)
                grad += torch.autograd.grad(loss, [x_adv])[0].detach()

            grad /= float(self.eot_iter)

            new_dice = self.dice(logits[0].detach(), y[0])
            index = (dice_best > new_dice).nonzero().squeeze()
            dice_best = torch.min(dice_best, new_dice)
            x_best_adv[index] = x_adv[index] + 0.
            if self.verbose:
                print('iteration: {} - Best loss: {:.6f}'.format(
                    i, loss_best.sum()))

            # # # check step size
            with torch.no_grad():
                y1 = loss_indiv.detach().clone()
                loss_steps[i] = y1.cpu() + 0
                ind = (y1 > loss_best).nonzero().squeeze()
                x_best[ind] = x_adv[ind].clone()
                grad_best[ind] = grad[ind].clone()
                loss_best[ind] = y1[ind] + 0
                loss_best_steps[i + 1] = loss_best + 0

                counter3 += 1

                if counter3 == k:
                    fl_oscillation = self.check_oscillation(
                        loss_steps.detach().cpu().numpy(), i, k,
                        loss_best.detach().cpu().numpy(), k3=self.thr_decr)
                    fl_reduce_no_impr = (~reduced_last_check) * (loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy())
                    fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                    reduced_last_check = np.copy(fl_oscillation)
                    loss_best_last_check = loss_best.clone()

                    if np.sum(fl_oscillation) > 0:
                        step_size[u[fl_oscillation]] /= 2.0

                        fl_oscillation = np.where(fl_oscillation)

                        x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                        grad[fl_oscillation] = grad_best[fl_oscillation].clone()

                    counter3 = 0
                    k = np.maximum(k - self.size_decr, self.n_iter_min)


        
        return x_best_adv

    def perturb(self, x_in, y_in):# y_in is a list
       
        x = x_in.clone() 
        self.init_hyperparam(x)
        y = y_in
        adv = x.clone()
        results = self.dice(self.model(x)[0], y[0])
        if self.verbose:
            print('---- running {}-attack with epsilon {:.4f} ----'.format(
                self.norm, self.eps))
            print('initial accuracy: {:.2%}'.format(results.float().mean()))
        startt = time.time()

        torch.random.manual_seed(self.seed)
        torch.cuda.random.manual_seed(self.seed)
        prev_dices = results
        for counter in range(self.n_restarts):

            x_to_fool = x.clone()

            y_to_fool = [y[i].clone() for i in range(len(y))]

            adv_curr = self.attack_single_run(
                x_to_fool, y_to_fool)
            results_curr = self.dice(self.model(adv_curr)[0], y_to_fool[0])
            ind_curr = (results > results_curr).nonzero().squeeze() # those sucessfully attacked
            results[ind_curr] = results_curr[ind_curr]
            adv[ind_curr] = adv_curr[ind_curr].clone()

        return adv
