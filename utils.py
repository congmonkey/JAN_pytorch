import argparse
import pdb
import os
import shutil
import time
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import math

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class GRLayer(torch.autograd.Function):
    def __init__(self, max_iter=10000, alpha=10., high=1.):
        super(GRLayer, self).__init__()
        self.total = float(max_iter)
        self.alpha = alpha
        self.high = high
        
    def forward(self, input):
        return input
    
    def backward(self, gradOutput):
        global global_iter
        prog = global_iter/self.total
        lr = 2.*self.high / (1 + math.exp(-self.alpha * prog)) - self.high
        return (-lr) * gradOutput


def save_checkpoint(state, is_best):
    if is_best:
        torch.save(state, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, iter_num, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (1 + args.gamma * iter_num) ** (-args.power)
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr * args.SGD_param[i]['lr']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
