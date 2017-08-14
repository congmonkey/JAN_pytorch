import argparse

import os
import shutil
import time
import itertools
import numpy as np

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
    L2_distance = ((total.unsqueeze(1)-total.unsqueeze(0))**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.detach()) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)


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


def JMMDLoss(source_list, target_list, kernel_muls=[2.0, 2.0], kernel_nums=[5, 1], fix_sigma_list=[None, 1.3]):
    batch_size = int(source_list[0].size()[0])
    layer_num = len(source_list)
    joint_kernels = None
    for i in range(layer_num):
        source = source_list[i]
        target = target_list[i]
        kernel_mul = kernel_muls[i]
        kernel_num = kernel_nums[i]
        fix_sigma = fix_sigma_list[i]
        kernels = guassian_kernel(source, target,
            kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        if joint_kernels is not None:
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
    loss = torch.pow(source - target, 2)
    loss = torch.mean(torch.sum(loss, 1))
    return loss

def x_Cy(x, y, C, b):
    loss = torch.pow(x - F.linear(y, C, b), 2)
    loss = torch.mean(torch.sum(loss, 1))
    return loss

class RevLayer(torch.autograd.Function):
    def forward(self, input):
        return input    
    def backward(self, gradOutput):
        return -1.*gradOutput


def Wasserstein_loss(source, target, source_l=None, target_l=None, kernel_type='gaussian'):
    def kernel(s, t, type=kernel_type):
        if type == 'linear':
            s_ = s.unsqueeze(1).repeat(1, t.size()[0], 1)
            t_ = t.unsqueeze(0).repeat(s.size()[0], 1, 1)
            return torch.sum(torch.mul(s_, t_), 2).squeeze()
        elif type == 'poly':
            return torch.pow(torch.ger(s, t), 2)
        elif type == 'gaussian':
            ip = torch.ger(s.squeeze(), t.squeeze())
            euclidean = torch.pow(s, 2).expand_as(ip) +\
                        torch.pow(t, 2).expand_as(ip).t() -\
                        2 * ip
            return (torch.exp(-euclidean / .01) + torch.exp(-euclidean / .02) + torch.exp(-euclidean / .04))/3.
    if source_l and target_l:
        # loss = kernel(source, source) * kernel(source_l, source_l, 'linear') +\
        #        kernel(target, target) * kernel(target_l, target_l, 'linear') -\
        #        2 * kernel(source, target) * kernel(target_l, target_l, 'linear')
        # softmax = nn.Softmax()
        rev = RevLayer()
        loss = JMMDLoss([source, rev(source_l)], [target, rev(target_l)])
    else:
        ### loss = kernel(source, source) + kernel(target, target) - 2*kernel(source, target)
        loss = MMDLoss(source, target)
    loss = -loss

    loss = torch.mean(loss)
    return loss

def Domain_loss(source, target, source_l=None, target_l=None):
    source_l = torch.autograd.Variable(torch.zeros(source.size()).cuda())
    target_l = torch.autograd.Variable(torch.ones(target.size()).cuda())
    output = torch.cat([source, target], 0)
    label = torch.cat([source_l, target_l], 0)

    loss = nn.BCELoss()(output, torch.autograd.Variable(torch.from_numpy(np.array([1] * 32 + [0] * 32)).float().cuda()))
    return loss
