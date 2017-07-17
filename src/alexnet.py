#from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
#import matplotlib.pyplot as plt
import time
import copy
import os
from PIL import Image, ImageOps

class ResizeImage():
    def __init__(self, size):
      if isinstance(size, int):
        self.size = (int(size), int(size))
      else:
        self.size = size
    def __call__(self, img):
      th, tw = self.size
      return img.resize((th, tw))

class PlaceCrop(object):
    """Crops the given PIL.Image at the particular index.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))

class ForceFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        return img.transpose(Image.FLIP_LEFT_RIGHT)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
data_transforms = {
    'train': transforms.Compose([
 ResizeImage(256),
       transforms.RandomSizedCrop(227),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])}

data_transforms['val'] = transforms.Compose([
    ResizeImage(256),
    transforms.Scale(256),
    transforms.CenterCrop(227),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

batch_size = {"train":64, 'val':5}
data_dir = "/home/caozhangjie/run-czj/dataset/imagenet_1000"
dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
         for x in ['train',"val"]}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size[x],
                                               shuffle="train" in x, num_workers=4)
                for x in ['train', "val"]}
dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
dset_classes = dsets["train"].classes
use_gpu = torch.cuda.is_available()
model_alexnet = models.alexnet(pretrained=True)
if use_gpu:
  model_alexnet = model_alexnet.cuda()
all_output = []
all_label = []
model_alexnet.train(False)
#for i, data in enumerate(dset_loaders['val']):
iter_val = iter(dset_loaders['val'])
for i in xrange(len(dset_loaders['val'])):
  data = iter_val.next()
  inputs, labels = data
  if use_gpu:
    inputs = Variable(inputs.cuda())
    labels = Variable(labels.cuda())
  else:
    inputs = Variable(inputs)
    labels = Variable(labels)
  outputs = model_alexnet(inputs)
  if i == 0:
    all_output = outputs.data.float()
    all_label = labels.data.float()
  else:
    all_output = torch.cat((all_output, outputs.data.float()), 0)
    all_label = torch.cat((all_label, labels.data.float()), 0)
  print i,"/",(50000.0/batch_size["val"])
_, predict = torch.max(all_output, 1)
torch.save(predict, "predict.pt")
torch.save(all_label, "label.pt")
print(torch.sum(torch.squeeze(predict).float() == all_label))
