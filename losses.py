import argparse
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

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target])
    k1 = torch.sum(torch.pow(total, 2), 1).repeat(1,n_samples)
    k2 = torch.sum(torch.pow(total, 2), 1).resize(1, n_samples).repeat(n_samples, 1)
    k3 = torch.mm(total, torch.t(total))
    L2_distance = k1 + k2 - 2*k3
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth *= kernel_mul**(float(-kernel_num)/2)
    bandwidth_list = [bandwidth * (kernel_mul**kernel_num) for i in xrange(kernel_num)]
    kernel_val = [torch.exp(L2_distance/-bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

def MMDLoss(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1)%batch_size
        t1, t2 = s1+batch_size, s2+batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)

def JMMDLoss(source_list, target_list, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source_list[0].size()[0])
    layer_num = len(source_list)
    joint_kernels = None
    for i in range(layer_num):
        source = source_list[i]
        target = target_list[i]
        kernels = guassian_kernel(source, target,
            kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        if joint_kernels:
            joint_kernels = joint_kernels * kernels
        else:
            joint_kernels = kernels
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1)%batch_size
        t1, t2 = s1+batch_size, s2+batch_size
        loss += joint_kernels[s1, s2] + joint_kernels[t1, t2]
        loss -= joint_kernels[s1, t2] + joint_kernels[s2, t1]
    return loss / float(batch_size)


def CrossEntropyLoss(logits, target):
    loss = -logits * torch.log(torch.clamp(target.detach(), 0.001, 1))
    loss = torch.mean(torch.sum(loss, 1))
    return loss

def L2Distance(source, target):
    loss = torch.pow(logits - target, 2)
    loss = torch.mean(torch.sum(loss, 1))
    return loss
