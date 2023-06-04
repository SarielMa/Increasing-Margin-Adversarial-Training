import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import math

class TRADES_TE():
    def __init__(self, num_samples=50000, num_classes=10, momentum=0.9, es=90, step_size=0.003, epsilon=0.031, perturb_steps=10, norm='linf', beta=6.0):
        # initialize soft labels to onthot vectors
        print('number samples: ', num_samples, 'num_classes: ', num_classes)
        self.soft_labels = torch.zeros(num_samples, num_classes, dtype=torch.float).cuda(non_blocking=True)
        self.momentum = momentum
        self.es = es
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.norm = norm
        self.beta = beta

    def __call__(self, x_natural, y, index, epoch, model, optimizer, weight):
        criterion_kl = nn.KLDivLoss(size_average=False)
        model.eval()
        batch_size = len(x_natural)
        logits = model(x_natural)
        
        if epoch >= self.es:
            prob = F.softmax(logits.detach(), dim=1)
            self.soft_labels[index] = self.momentum * self.soft_labels[index] + (1 - self.momentum) * prob
            soft_labels_batch = self.soft_labels[index] / self.soft_labels[index].sum(1, keepdim=True)

        # generate adversarial example
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
        for _ in range(self.perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                logits_adv = model(x_adv)
                loss_kl = criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits, dim=1))
                if epoch >= self.es:
                    loss = (self.beta / batch_size) * loss_kl + weight * ((F.softmax(logits_adv, dim=1) - soft_labels_batch) ** 2).mean()
                else:
                    loss = loss_kl
            grad = torch.autograd.grad(loss, [x_adv])[0]
            if self.norm == 'linf':
                x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
            elif self.norm == 'l2':
                g_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1,1,1,1)
                scaled_grad = grad.detach() / (g_norm.detach() + 1e-10)
                x_adv = x_natural + (x_adv.detach() + self.step_size * scaled_grad - x_natural).view(x_natural.size(0), -1).renorm(p=2, dim=0, maxnorm=self.epsilon).view_as(x_natural)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
        # compute loss
        model.train()
        optimizer.zero_grad()
        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
        
        # calculate robust loss
        logits = model(x_natural)
        logits_adv = model(x_adv)
        loss_natural = F.cross_entropy(logits, y)
        loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits, dim=1))
        if epoch >= self.es:
            loss = loss_natural + self.beta * loss_robust + weight * ((F.softmax(logits, dim=1) - soft_labels_batch) ** 2).mean()
        else:
            loss = loss_natural + self.beta * loss_robust
        return loss

class PGD_TE():
    def __init__(self, loss_fn, multioutput_weights, num_samples, momentum, es, step_size, epsilon, perturb_steps=10, norm='l2'):
        # initialize soft labels to onthot vectors
        print('number samples: ', num_samples)
        #self.soft_labels = torch.zeros(num_samples, num_output, num_classes, dtype=torch.float).cuda(non_blocking=True)       
        self.soft_labels = []       
        self.momentum = momentum
        self.es = es
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.norm = norm
        self.loss_fn = loss_fn
        self.num_output = None
        self.num_samples = num_samples
        self.ds_loss_weights = multioutput_weights

    def multipleOutputReg(self, Z, Y):
        assert isinstance(Z, (tuple, list)), "x must be either tuple or list"
        assert isinstance(Y, (tuple, list)), "y must be either tuple or list"
        weights = self.ds_loss_weights
        l = weights[0] * ((F.softmax(Z[0].view(Z[0].shape[0], -1), dim=1) - Y[0]) ** 2).mean()
        for i in range(1, len(Z)):
            if weights[i] != 0:
                l += weights[i] * ((F.softmax(Z[i].view(Z[i].shape[0], -1), dim=1) - Y[i]) ** 2).mean()
        return l        

    def __call__(self, x_natural, y, index, epoch, model, optimizer, weight):
        model.eval()
        batch_size = len(x_natural)
        logits = model(x_natural)# the output of this is a tuple
        self.num_output = len(logits)
        if len(self.soft_labels) == 0:
            print ("creating the soft labels........ ")
            for i in range(self.num_output):
                self.soft_labels.append(torch.zeros(self.num_samples, logits[i].shape[1]*logits[i].shape[2]*logits[i].shape[3], 
                                                    dtype=torch.float).cuda(non_blocking=True))
        
        soft_labels_batch = []
        if epoch >= self.es:
            for i in range(self.num_output):
                prob = F.softmax(logits[i].view(logits[i].shape[0], -1).detach(), dim=1)
                self.soft_labels[i][index] = self.momentum * self.soft_labels[i][index] + (1 - self.momentum) * prob
                soft_labels_batch.append(self.soft_labels[i][index] / self.soft_labels[i][index].sum(1, keepdim=True))

        # generate adversarial example
        if self.norm == 'linf':
            x_adv = x_natural.detach() + torch.FloatTensor(*x_natural.shape).uniform_(-self.epsilon, self.epsilon).cuda()
        else:
            x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
        for _ in range(self.perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                logits_adv = model(x_adv)
                if epoch >= self.es:
                    loss = self.loss_fn(logits_adv, y) + weight * self.multipleOutputReg(logits_adv, soft_labels_batch)
                else:
                    loss = self.loss_fn(logits_adv, y)
            grad = torch.autograd.grad(loss, [x_adv])[0]
            if self.norm == 'linf':
                x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
            elif self.norm == 'l2':
                g_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1,1,1,1)
                scaled_grad = grad.detach() / (g_norm.detach() + 1e-10)
                x_adv = x_natural + (x_adv.detach() + self.step_size * scaled_grad - x_natural).view(x_natural.size(0), -1).renorm(p=2, dim=0, maxnorm=self.epsilon).view_as(x_natural)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
        # compute loss
        model.train()
        optimizer.zero_grad()
        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
        
        # calculate robust loss
        logits = model(x_adv)
        if epoch >= self.es:
            loss = self.loss_fn(logits, y) + weight * self.multipleOutputReg(logits, soft_labels_batch)
        else:
            loss = self.loss_fn(logits, y)
        return loss
