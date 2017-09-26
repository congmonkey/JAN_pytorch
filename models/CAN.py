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
    def __init__(self, max_iter=2000, alpha=10., high=1.):
        super(GRLayer, self).__init__()
        self.total = float(max_iter)
        self.alpha = alpha
        self.high = high

    def forward(self, input):
        return input.view_as(input)

    def backward(self, gradOutput):
        prog = global_iter / self.total
        lr = 2.*self.high / (1 + math.exp(-self.alpha * prog)) - self.high
        return (-lr) * gradOutput


def create_W(dimensions):
    nets = []
    for in_dim, out_dim in zip(dimensions[:-1], dimensions[1:]):
        nets += [
            nn.Linear(in_dim, out_dim),
            nn.ReLU()
        ]
    return nn.Sequential(*nets[:-1]) # Remove last relu


def create_D(dimensions):
    nets = []
    for in_dim, out_dim in zip(dimensions[:-1], dimensions[1:]):
        nets += [
            nn.Linear(in_dim, out_dim),
            nn.ReLU()
        ]
    nets[-1] = nn.Sigmoid()
    nets[-2].weight.data.normal_(0, 0.03)
    nets[-2].bias.data.fill_(0.0)
    return nn.Sequential(*nets)


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        # create model
        if args.fromcaffe:
            print("=> using pre-trained model from caffe '{}'".format(args.arch))
            import models.caffe_resnet as resnet
            model = resnet.__dict__[args.arch]()
            state_dict = torch.load("models/"+args.arch+".pth")
            model.load_state_dict(state_dict)
        elif args.pretrained:
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

        self.fcb_s = nn.Linear(self.feature_dim, args.bottleneck)
        self.fcb_s.weight.data.normal_(0, 0.005)
        self.fcb_s.bias.data.fill_(0.1)
        self.fcb_t = nn.Linear(self.feature_dim, args.bottleneck)
        self.fcb_t.weight.data.normal_(0, 0.005)
        self.fcb_t.bias.data.fill_(0.1)
        self.fc_s = nn.Linear(args.bottleneck, args.classes)
        self.fc_s.weight.data.normal_(0, 0.01)
        self.fc_s.bias.data.fill_(0.0)
        self.fc_t = nn.Linear(args.bottleneck, args.classes)
        self.fc_t.weight.data.normal_(0, 0.01)
        self.fc_t.bias.data.fill_(0.0)

        self.W_st = create_W([args.bottleneck, args.bottleneck])
        self.W_ts = create_W([args.bottleneck, args.bottleneck])

        self.D_s = create_D([args.bottleneck, 1024, 1024, 1])
        self.D_t = create_D([args.bottleneck, 1024, 1024, 1])

        self.grl_ss = GRLayer()
        self.grl_tt = GRLayer()
        self.grl_st = GRLayer()
        self.grl_ts = GRLayer()

        args.SGD_param = [
            {'params': self.origin_feature.parameters(), 'lr': 1,},
            {
                'params': itertools.chain(self.fcb_s.parameters(),
                                          self.fcb_t.parameters()),
                'lr': 10
            },
            {
                'params': itertools.chain(self.fc_s.parameters(),
                                          self.fc_t.parameters()),
                'lr': 10
            },
            {
                'params': itertools.chain(self.W_ts.parameters(),
                                          self.W_st.parameters()),
                'lr': 10
            },
            {
                'params': itertools.chain(self.D_s.parameters(),
                                          self.D_t.parameters()),
                'lr': 10
            }
        ]

    def forward(self, x, train=True):
        x = self.origin_feature(x)
        if self.arch.startswith('densenet'):
            x = F.relu(x, inplace=True)
            x = F.avg_pool2d(x, kernel_size=7)
        x = x.view(x.size(0), -1)
        x = torch.autograd.Variable(x.data)
        if train:
            orign_feature_s, orign_feature_t = x.chunk(2, 0)
            feature_s = self.fcb_s(orign_feature_s)
            feature_t = self.fcb_t(orign_feature_t)
            output_s = self.fc_s(feature_s)
            output_t = self.fc_t(feature_t)
            fake_feature_t = self.W_st(feature_s) + feature_s
            fake_feature_s = self.W_ts(feature_t) + feature_t
            cycle_s = self.W_ts(fake_feature_t)
            cycle_t = self.W_st(fake_feature_s)
            fake_output_t = self.fc_t(fake_feature_t)
            discriminate_s = self.D_s(torch.cat([self.grl_ss(fake_feature_s),
                                                 self.grl_st(feature_s)], 0))
            discriminate_t = self.D_t(torch.cat([self.grl_ts(fake_feature_t),
                                                 self.grl_tt(feature_t)], 0))
            return (feature_s, feature_t), \
                   (cycle_s, cycle_t), \
                   (output_s, output_t),\
                   (fake_output_t,), \
                   (discriminate_s, discriminate_t)
        else:
            return self.fc_t(self.fcb_t(x))


def L2loss(source, target):
    return torch.sum((source - target) ** 2)


def train_val(source_loader, target_loader, val_loader, model, criterion, optimizer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    source_cycle = itertools.cycle(source_loader)
    target_cycle = itertools.cycle(target_loader)

    end = time.time()
    model.eval()
    cycle_criterion = L2loss
    discriminate_criterion = nn.BCELoss()
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
        domain_label = torch.autograd.Variable(
            torch.cat([torch.zeros(source_input.size()[0]),
                       torch.ones(source_input.size()[0])], 0)).cuda()
        label = label.cuda(async=True)
        source_var = torch.autograd.Variable(source_input)
        target_var = torch.autograd.Variable(target_input)
        label_var = torch.autograd.Variable(label)

        inputs = torch.cat([source_var, target_var], 0)
        (feature_s, feature_t), (cycle_s, cycle_t), \
            (output_s, output_t), (fake_output_t,), \
            (discriminate_s, discriminate_t) = model(inputs)

        acc_loss = criterion(output_s, label_var) \
            + criterion(fake_output_t, label_var)
        cycle_loss = cycle_criterion(feature_s, cycle_s) \
            + cycle_criterion(feature_t, cycle_t)
        discriminate_loss = discriminate_criterion(discriminate_s, domain_label) \
            + discriminate_criterion(discriminate_t, domain_label)

        loss = acc_loss + args.alpha * cycle_loss + args.beta * discriminate_loss
        # loss = discriminate_loss

        prec1, _ = accuracy(output_s.data, label, topk=(1, 5))

        losses.update(loss.data[0], args.batch_size)
        loss1 = acc_loss.data[0]
        loss2 = cycle_loss.data[0]
        loss3 = discriminate_loss.data[0]
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
                  'Loss {loss1:.4f} {loss2:.4f} {loss3:.4f}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, args.train_iter, batch_time=batch_time,
                loss=losses, top1=top1, loss1=loss1, loss2=loss2, loss3=loss3))

        if i % args.test_iter == 0 and i != 0:
            validate(val_loader, model, criterion, args)
            model.eval()
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
        output = model(input_var, train=False)
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
