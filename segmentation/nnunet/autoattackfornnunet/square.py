# Copyright (c) 2020-present, Francesco Croce
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time
import math
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from nnunet.utilities.nd_softmax import softmax_helper

#from libs.utilities.losses import Dice_metric
"""
def one_hot(gt, categories):
    # Check the new function in PyTorch!!!
    size = [*gt.shape] + [categories]
    y = gt.view(-1, 1)
    gt = torch.FloatTensor(y.nelement(), categories).zero_().cuda()
    gt.scatter_(1, y, 1)
    gt = gt.view(size).permute(0, 4, 1, 2, 3).contiguous()
    return gt


class Dice_metric(nn.Module):
    def __init__(self, eps=1e-5):
        super(Dice_metric, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets, logits=True):
        categories = inputs.shape[1]
        targets = targets.contiguous()
        targets = one_hot(targets, categories)
        if logits:
            inputs = torch.argmax(F.softmax(inputs, dim=1), dim=1)
        inputs = one_hot(inputs, categories)

        dims = tuple(range(2, targets.ndimension()))
        tps = torch.sum(inputs * targets, dims)
        fps = torch.sum(inputs * (1 - targets), dims)
        fns = torch.sum((1 - inputs) * targets, dims)
        loss = (2 * tps) / (2 * tps + fps + fns + self.eps)
        return loss[:, 1:].mean(dim=1)
"""  
    
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
    
    
class SquareAttack():
    """
    Square Attack
    https://arxiv.org/abs/1912.00049

    :param predict:       forward pass function
    :param n_restarts:    number of random restarts
    :param n_queries:     max number of queries (each restart)
    :param eps:           bound on the norm of perturbations
    :param seed:          random seed for the starting point
    :param p_init:        parameter to control size of squares
    :param loss:          loss function optimized ('margin', 'ce' supported)
    :param resc_schedule  adapt schedule of p to n_queries
    """

    def __init__(
            self,
            predict,
            dice_thresh,
            n_queries=5000,
            eps=None,
            p_init=.8,
            n_restarts=1,
            seed=0,
            verbose=False,
            loss='margin',
            resc_schedule=True,
            device=None,
            norm_type=None):
        """
        Square Attack implementation in PyTorch
        """

        self.predict = predict
        self.n_queries = n_queries
        self.eps = eps
        self.p_init = p_init
        self.n_restarts = n_restarts
        self.seed = seed
        self.verbose = verbose
        self.loss = loss
        self.rescale_schedule = resc_schedule
        self.device = device
        # Dice loss
        self.dice_thresh = dice_thresh
        self.dice = Dice_metric(eps=1e-5)
        self.norm = norm_type

    def margin_and_loss(self, x, y):
        """
        :param y:        correct labels
        """

        logits = self.predict(x)[0]
        dice = self.dice(logits.detach(), y)
        return dice, dice # margin, margin

    def init_hyperparam(self, x):
        assert self.eps is not None
        assert self.loss in ['ce', 'margin']

        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)
        if self.seed is None:
            self.seed = time.time()

    def random_target_classes(self, y_pred, n_classes):
        y = torch.zeros_like(y_pred)
        for counter in range(y_pred.shape[0]):
            lst = list(range(n_classes))
            lst.remove(y_pred[counter])
            t = self.random_int(0, len(lst))
            y[counter] = lst[t]

        return y.long().to(self.device)

    def check_shape(self, x):
        return x if len(x.shape) == (self.ndims + 1) else x.unsqueeze(0)

    def random_choice(self, shape):
        t = 2 * torch.rand(shape).to(self.device) - 1
        return torch.sign(t)

    def random_int(self, low=0, high=1, shape=[1]):
        t = low + (high - low) * torch.rand(shape).to(self.device)
        return t.long()
    
    
    def normalize(self, x):
        if self.norm == 'Linf':
            t = x.abs().view(x.shape[0], -1).max(1)[0]
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

        elif self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)
        
    def lp_norm(self, x):
        if self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
            return t.view(-1, *([1] * self.ndims))

        elif self.norm == 'L1':
            t = x.abs().view(x.shape[0], -1).sum(dim=-1)
            return t.view(-1, *([1] * self.ndims))                         

    def normalize(self, x):
        if self.norm == 'Linf':
            t = x.abs().view(x.shape[0], -1).max(1)[0]
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

        elif self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

        elif self.norm == 'L1':
            t = x.abs().view(x.shape[0], -1).sum(dim=-1)
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)
    
    def lp_norm(self, x):
        if self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
            return t.view(-1, *([1] * self.ndims))

        elif self.norm == 'L1':
            t = x.abs().view(x.shape[0], -1).sum(dim=-1)
            return t.view(-1, *([1] * self.ndims))
    
    def eta_rectangles(self, x, y):
        delta = torch.zeros([x, y]).to(self.device)
        x_c, y_c = x // 2 + 1, y // 2 + 1

        counter2 = [x_c - 1, y_c - 1]
        if self.norm == 'L2':
            for counter in range(0, max(x_c, y_c)):
              delta[max(counter2[0], 0):min(counter2[0] + (2*counter + 1), x),
                  max(0, counter2[1]):min(counter2[1] + (2*counter + 1), y)
                  ] += 1.0/(torch.Tensor([counter + 1]).view(1, 1).to(
                  self.device) ** 2)
              counter2[0] -= 1
              counter2[1] -= 1
    
            delta /= (delta ** 2).sum(dim=(0, 1), keepdim=True).sqrt()
        
        elif self.norm == 'L1':
            for counter in range(0, max(x_c, y_c)):
              delta[max(counter2[0], 0):min(counter2[0] + (2*counter + 1), x),
                  max(0, counter2[1]):min(counter2[1] + (2*counter + 1), y)
                  ] += 1.0/(torch.Tensor([counter + 1]).view(1, 1).to(
                  self.device) ** 4)
              counter2[0] -= 1
              counter2[1] -= 1
    
            delta /= delta.abs().sum(dim=(), keepdim=True)
        
        return delta

    def eta(self, s):
        if self.norm == 'L2':
            delta = torch.zeros([s, s]).to(self.device)
            delta[:s // 2] = self.eta_rectangles(s // 2, s)
            delta[s // 2:] = -1. * self.eta_rectangles(s - s // 2, s)
            delta /= (delta ** 2).sum(dim=(0, 1), keepdim=True).sqrt()
        
        elif self.norm == 'L1':
            delta = torch.zeros([s, s]).to(self.device)
            delta[:s // 2] = self.eta_rectangles(s // 2, s)
            delta[s // 2:] = -1. * self.eta_rectangles(s - s // 2, s)
            #delta = self.eta_rectangles(s, s)
            delta /= delta.abs().sum(dim=(), keepdim=True)
            #delta *= (torch.rand([1]) - .5).sign().to(self.device)
        
        if torch.rand([1]) > 0.5:
            delta = delta.permute([1, 0])

        return delta
    


    def p_selection(self, it):
        """ schedule to decrease the parameter p """

        if self.rescale_schedule:
            it = int(it / self.n_queries * 10000)

        if 10 < it <= 50:
            p = self.p_init / 2
        elif 50 < it <= 200:
            p = self.p_init / 4
        elif 200 < it <= 500:
            p = self.p_init / 8
        elif 500 < it <= 1000:
            p = self.p_init / 16
        elif 1000 < it <= 2000:
            p = self.p_init / 32
        elif 2000 < it <= 4000:
            p = self.p_init / 64
        elif 4000 < it <= 6000:
            p = self.p_init / 128
        elif 6000 < it <= 8000:
            p = self.p_init / 256
        elif 8000 < it:
            p = self.p_init / 512
        else:
            p = self.p_init
        return p

    def attack_single_run(self, x, y):
        with torch.no_grad():
            adv = x.clone()
            c, h, w = x.shape[1:]
            n_features = c * h * w
            n_ex_total = x.shape[0]
            
            if self.norm == "Linf":
                root_cube = lambda x: x**(1./3.) if 0 <= x else -(-x)**(1./3.)                               
                x_best = torch.clamp(x + self.eps * self.random_choice([x.shape[0], c, h, w]), 0., 1.)
            elif self.norm == "L2":
                delta_init = torch.zeros_like(x)
                s = h // 5
                sp_init = (h - s * 5) // 2
                vh = sp_init + 0
                for _ in range(h // s):
                    vw = sp_init + 0
                    for _ in range(w // s):
                        delta_init[:, :, vh:vh + s, vw:vw + s] += self.eta(
                            s).view(1, 1, s, s) * self.random_choice(
                            [x.shape[0], c, 1, 1])
                        vw += s
                    vh += s
                    
                x_best = torch.clamp(x + self.normalize(delta_init) * self.eps, 0., 1.)
            else: 
                raise ValueError('norm not supported')
            
            
            margin_min, loss_min = self.margin_and_loss(x_best, y)
            n_queries = torch.ones(x.shape[0]).to(self.device)
            for i_iter in range(self.n_queries):
                if i_iter % 100 == 0:
                    print(i_iter)
                idx_to_fool = (margin_min > self.dice_thresh).nonzero().squeeze()

                x_curr = self.check_shape(x[idx_to_fool])
                x_best_curr = self.check_shape(x_best[idx_to_fool])
                y_curr = y[idx_to_fool]
                margin_min_curr = margin_min[idx_to_fool]
                loss_min_curr = loss_min[idx_to_fool]


                
                
                if self.norm == "Linf":
                    p = self.p_selection(i_iter)
                    s = max(int(round(root_cube(p * n_features / c))), 1)
                    vh = self.random_int(0, h - s)
                    vw = self.random_int(0, w - s)
                    new_deltas = torch.zeros([c, h, w]).to(self.device)
                    new_deltas[:, vh:vh + s, vw:vw + s] = 2. * self.eps * self.random_choice([c, 1, 1])
                    x_new = x_best_curr + new_deltas
                    x_new = torch.min(torch.max(x_new, x_curr - self.eps),x_curr + self.eps)
                    x_new = torch.clamp(x_new, 0., 1.)
                elif self.norm == "L2":
                    delta_curr = x_best_curr - x_curr
                    p = self.p_selection(i_iter)
                    s = max(int(round(math.sqrt(p * n_features / c))), 3)
                    if s % 2 == 0:
                        s += 1
                    s = min(s, min(h, w))

                    vh = self.random_int(0, h - s)
                    vw = self.random_int(0, w - s)
                    new_deltas_mask = torch.zeros_like(x_curr)
                    new_deltas_mask[:, :, vh:vh + s, vw:vw + s] = 1.0
                    norms_window_1 = (delta_curr[:, :, vh:vh + s, vw:vw + s
                        ] ** 2).sum(dim=(-2, -1), keepdim=True).sqrt()

                    vh2 = self.random_int(0, h - s)
                    vw2 = self.random_int(0, w - s)
                    new_deltas_mask_2 = torch.zeros_like(x_curr)
                    new_deltas_mask_2[:, :, vh2:vh2 + s, vw2:vw2 + s] = 1.

                    norms_image = self.lp_norm(x_best_curr - x_curr)
                    mask_image = torch.max(new_deltas_mask, new_deltas_mask_2)
                    norms_windows = ((delta_curr * mask_image) ** 2).sum(dim=(
                        -2, -1), keepdim=True).sqrt()

                    new_deltas = torch.ones([x_curr.shape[0], c, s, s]
                        ).to(self.device)
                    new_deltas *= (self.eta(s).view(1, 1, s, s) *
                        self.random_choice([x_curr.shape[0], c, 1, 1]))
                    old_deltas = delta_curr[:, :, vh:vh + s, vw:vw + s] / (
                        1e-12 + norms_window_1)
                    new_deltas += old_deltas
                    new_deltas = new_deltas / (1e-12 + (new_deltas ** 2).sum(
                        dim=(-2, -1), keepdim=True).sqrt()) * (torch.max(
                        (self.eps * torch.ones_like(new_deltas)) ** 2 -
                        norms_image ** 2, torch.zeros_like(new_deltas)) /
                        c + norms_windows ** 2).sqrt()
                    delta_curr[:, :, vh2:vh2 + s, vw2:vw2 + s] = 0.
                    delta_curr[:, :, vh:vh + s, vw:vw + s] = new_deltas + 0

                    x_new = torch.clamp(x_curr + self.normalize(delta_curr) * self.eps, 0. ,1.)
                else: 
                    raise ValueError('norm not supported')
                
                
                
                x_new = self.check_shape(x_new)
                norms_image = self.lp_norm(x_new - x_curr)

                margin, loss = self.margin_and_loss(x_new, y_curr)
                # update loss if new loss is better
                idx_improved = (loss < loss_min_curr).half()

                loss_min[idx_to_fool] = idx_improved * loss + (
                    1. - idx_improved) * loss_min_curr

                # update margin and x_best if new loss is better
                # or misclassification
                idx_miscl = (margin < self.dice_thresh).half()
                idx_improved = torch.max(idx_improved, idx_miscl)

                margin_min[idx_to_fool] = idx_improved * margin + (
                    1. - idx_improved) * margin_min_curr
                idx_improved = idx_improved.reshape(
                    [-1, *[1]*len(x.shape[:-1])])
                x_best[idx_to_fool] = idx_improved * x_new + (
                    1. - idx_improved) * x_best_curr
                n_queries[idx_to_fool] += 1.

                curr_dice, _ = self.margin_and_loss(x_best, y)
                ind_succ = (self.dice(
                    self.predict(x_best)[0].detach(),
                    y) < self.dice_thresh).nonzero().squeeze()
                if self.verbose and ind_succ.numel() != 0:
                    print(
                        '{}'.format(i_iter + 1),
                        '- success rate={}/{} ({:.2%})'.format(
                            ind_succ.numel(), n_ex_total,
                            float(ind_succ.numel()) / n_ex_total),
                        '- avg # queries={:.1f}'.format(
                            n_queries[ind_succ].mean().item()),
                        '- med # queries={:.1f}'.format(
                            n_queries[ind_succ].median().item()),
                        '- loss={:.3f}'.format(loss_min.mean()))

                if ind_succ.numel() == n_ex_total:
                    break

        return x_best

    def perturb(self, x_in, y_in):# y_in is a list
        print ("square testing...")
        
        x = x_in.clone() 
        self.init_hyperparam(x)
        y = y_in
        adv = x.clone()
        results = self.dice(self.predict(x)[0], y[0])
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
                x_to_fool, y_to_fool[0])
            
            results_curr = self.dice(self.predict(adv_curr)[0], y_to_fool[0])
            ind_curr = (results > results_curr).nonzero().squeeze() # those sucessfully attacked
            results[ind_curr] = results_curr[ind_curr]
            adv[ind_curr] = adv_curr[ind_curr].clone()

        return adv
