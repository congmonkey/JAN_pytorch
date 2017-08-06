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

from losses import *
from utils import *

global_iter = 0

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


### Convert back-bone model
class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        # create model
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch]()

        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            self.feature_dim = model.classifier[6].in_features
            model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        elif args.arch.startswith('densenet'):
            self.feature_dim = model.classifier.in_features
            model = nn.Sequential(*list(model.children())[:-1])
        else:
            self.feature_dim = model.fc.in_features
            model = nn.Sequential(*list(model.children())[:-1])
        self.origin_feature = torch.nn.DataParallel(model)
        self.model = args.model
        self.arch = args.arch

        self.fcb = nn.Linear(self.feature_dim, args.bottleneck)
        self.fcb.weight.data.normal_(0, 0.005)
        self.fcb.bias.data.fill_(0.1)
        self.fc = nn.Linear(args.bottleneck, args.classes)
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.fill_(0.0)

        self.grl7 = GRLayer()
        dc_ip1 = nn.Linear(args.bottleneck, 1024)
        dc_ip1.weight.data.normal_(0, 0.01)
        dc_ip1.bias.data.fill_(0.0)
        dc_ip2 = nn.Linear(1024, 1024)
        dc_ip2.weight.data.normal_(0, 0.01)
        dc_ip2.bias.data.fill_(0.0)
        dc_ip3 = nn.Linear(1024, 1)
        dc_ip3.weight.data.normal_(0, 0.03)
        dc_ip3.bias.data.fill_(0.0)
        self.dc7 = nn.Sequential(
            dc_ip1,
            nn.ReLU(),
            nn.Dropout(0.5),
            dc_ip2,
            nn.ReLU(),
            nn.Dropout(0.5),
            dc_ip3,
            nn.Sigmoid(),
        )
        
        args.SGD_param = [
            {'params': model.origin_feature.parameters(), 'lr': 1,},
            {'params': model.fcb.parameters(), 'lr': 10,},
            {'params': model.fc.parameters(), 'lr': 10},
            {'params': model.dc7.parameters(), 'lr': 10},
        ]
            
    def forward(self, x, train_dc=False):
        x = self.origin_feature(x)
        if self.arch.startswith('densenet'):
            x = F.relu(x, inplace=True)
            x = F.avg_pool2d(x, kernel_size=7)
        x = x.view(x.size(0), -1)
        x = self.fcb(x)
        y = self.fc(x)
        if train_dc:
            dc7 = x.detach()
        else:
            dc7 = self.grl7(x)
        dc7 = self.dc7(dc7)
        return y, x, dc7


def train_val(source_loader, target_loader, val_loader, model, criterion, optimizer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    entropy_loss = AverageMeter()

    source_cycle = itertools.cycle(source_loader)
    target_cycle = itertools.cycle(target_loader)

    end = time.time()
    model.train()
    for i in range(args.train_iter):
        global global_iter
        global_iter = i
        adjust_learning_rate(optimizer, i, args)
        data_time.update(time.time() - end)
        source_input, label = source_cycle.next()
        if source_input.size()[0] < args.batch_size:
            source_input, label = source_cycle.next()
        target_input, _ = target_cycle.next()
        if target_input.size()[0] < args.batch_size:
            target_input, _ = target_cycle.next()
        label = label.cuda(async=True)
        source_var = torch.autograd.Variable(source_input)
        target_var = torch.autograd.Variable(target_input)
        label_var = torch.autograd.Variable(label)

###         if args.model == 'jan':
###             for _ in range(0):
###                 source_output, source_feature, source_dc = model(source_var, train_dc=True)
###                 target_output, target_feature, target_dc = model(target_var, train_dc=True)
###                 loss = args.alpha * Wasserstein_loss(source_dc, target_dc)
###                 optimizer.zero_grad()
###                 loss.backward()
###                 optimizer.step()
###             for _ in range(0):
###                 source_output, source_feature, source_dc = model(source_var)
###                 target_output, target_feature, target_dc = model(target_var)
###                 loss = args.alpha * Wasserstein_loss(source_dc, target_dc)
###                 optimizer.zero_grad()
###                 loss.backward()
###                 for p in optimizer.param_groups[-1]['params']:
###                     p.grad.data.zero_()
###                 optimizer.step()

        
        inputs = torch.cat([source_var, target_var], 0)
        outputs, features, dcs = model(inputs)

        source_output, target_output = outputs.chunk(2, 0)
        sourde_feature, target_feature = features.chunk(2, 0)
        source_dc, target_dc = dcs.chunk(2, 0)
     
        acc_loss = criterion(source_output, label_var)

        W_loss = Domain_loss(source_dc, target_dc)
        ### softmax = nn.Softmax()
        ### W_loss = JMMDLoss([source_feature, softmax(source_output).detach()], [target_feature, softmax(target_output).detach()])
            
        loss = acc_loss + args.alpha * W_loss

        prec1, _ = accuracy(source_output.data, label, topk=(1, 5))

        losses.update(loss.data[0], args.batch_size)
        loss1 = W_loss.data[0] if args.model == 'jan' else 0
        loss2 = 0
        top1.update(prec1[0], args.batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Iter: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss1:.4f} {loss2:.4f}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   i, args.train_iter, batch_time=batch_time,
                      loss=losses, top1=top1, loss1=loss1, loss2=loss2))

        if i % args.test_iter == 0 and i != 0:
            validate(val_loader, model, criterion, args)
            model.train()
            batch_time.reset()
            data_time.reset()
            losses.reset()
            top1.reset()



def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output, _, _ = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg