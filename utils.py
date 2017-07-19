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

from losses import *

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
        if args.model == 'dan':
            self.fc= nn.Linear(self.feature_dim, args.classes)
        elif args.model == 'jan':
            self.fcb = nn.Linear(self.feature_dim, args.bottleneck)
            self.fc = nn.Linear(args.bottleneck, args.classes)
            self.fcr = nn.Linear(args.bottleneck, self.feature_dim)
    def forward(self, x):
        x = self.origin_feature(x)
        xo = x
        if self.arch.startswith('densenet'):
            x = F.relu(x, inplace=True)
            x = F.avg_pool2d(x, kernel_size=7)
        x = x.view(x.size(0), -1)
        if self.model == 'jan':
            x = self.fcb(x)
            xr = self.fcr(x)
            dxr = L2Distance(xr, xo)
        y = self.fc(x)
        return y, x, dxr


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

        if args.model == 'jan':
            source_output, source_feature, _ = model(source_var)
            target_output, target_feature, distan_recons = model(target_var)
        elif args.model == 'dan':
            source_output, source_feature = model(source_var)
            target_output, target_feature = model(target_var)

        acc_loss = criterion(source_output, label_var)
        if args.model == 'dan':
            loss = acc_loss + args.alpha * MMDLoss(source_output, target_output) +\
                   MMDLoss(source_feature, target_feature)
        if args.model == 'jan':
            softmax = nn.Softmax()
            loss = acc_loss + args.alpha * JMMDLoss(
                [source_feature, softmax(source_output)],
                [target_feature, softmax(target_output)]) +\
                args.beta * distan_recons

        prec1, _ = accuracy(source_output.data, label, topk=(1, 5))

        losses.update(loss.data[0], args.batch_size)
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
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   i, args.train_iter, batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))

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

        # if i % args.print_freq == 0:
        #     print('Test: [{0}/{1}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
        #           'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
        #            i, len(val_loader), batch_time=batch_time, loss=losses,
        #            top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best):
    if is_best:
        torch.save(state, 'model_best.pth.tar')


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


def adjust_learning_rate(optimizer, iter_num, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (1 + args.gamma * iter_num) ** (-args.power)
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr * (10 if i > 0 else 1)


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
