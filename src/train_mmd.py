#from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, transforms
#import matplotlib.pyplot as plt
import time
import copy
import os
from PIL import Image, ImageOps
import numbers
import argparse
import pickle

import model_no_class as model_no
import caffe_transform as caffe_t
import mmd_loss as mmd
import adversarial1 as ad
from data import ImageList

#
def inv_lr_scheduler(optimizer, iter_num, gamma, power, init_lr=0.001):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (1 + gamma * iter_num) ** (-power)

    i=0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i+=1

    return optimizer

# train function
def train_source(model_list, model_all, model_predict, modelcriterion, optimizer, lr_scheduler, batch_size_train=64, num_epochs=25, num_classes=31):
    since = time.time()
    test_iter = 1
    best_model = model_all
    best_acc = 0.0
    iter_num = 1
    criterion1 = nn.BCELoss()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
          if phase == 'train':
                model_all.train(True)  # Set model to training mode
            else:
                model_all.train(False)  # Set model to evaluate mode
            
            if phase == "train":
                for iter_i in xrange(test_iter):
                  iter_source = iter(dset_loaders["train"])
                  iter_target = iter(dset_loaders["val"])
              # Iterate over data.
              running_transfer_loss = 0.0
              running_cross_loss = 0.0
              for i in xrange(len(dset_loaders["val"])-1):
                data_source = iter_source.next()
                data_target = iter_target.next()
                optimizer = lr_scheduler(optimizer, iter_num, init_lr=0.0003, gamma=0.0003, power=0.75)
                 # get the inputs
                inputs_source, labels_source = data_source
                inputs_target, labels_target = data_target
                inputs = torch.cat((inputs_source, inputs_target), dim=0)
                dc_target = torch.from_numpy(np.array([1] * batch_size_train + [0] * batch_size_train)).float()
                # wrap them in Variable
                if use_gpu:
                    inputs, labels, dc_target = Variable(inputs.cuda()), \
                        Variable(labels_source.cuda()), Variable(dc_target.cuda())
                else:
                    inputs, labels, dc_target = Variable(inputs), Variable(labels_source), Variable(dc_target)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model_list[0](inputs)
                outputs_bottleneck = model_list[1](outputs)
                outputs_fc8 = model_list[2](outputs_bottleneck)
                loss = criterion(outputs_fc8[0:(batch_size_train), :], labels)

                softmax_output = nn.Softmax()(outputs_fc8)
                # transfer rman
                softmax_output.require_grad = False
                rman_out_list = rman_layer.forward([outputs_bottleneck, softmax_output])
                rman_out = rman_out_list[0]
                for rman_single_out in rman_out_list[1:]:
                  rman_out = torch.mul(rman_out, rman_single_out)
                ad_out = model_list[3](model_list[4](rman_out.view(-1, 1024)))
                transfer_loss = criterion1(ad_out, dc_target)               
                # transfer op
                '''
                softmax_output.require_grad = False
                op_out = torch.bmm(softmax_output.unsqueeze(2), outputs_bottleneck.unsqueeze(1))
                ad_out = model_list[3](model_list[4](op_out.view(-1, 256*num_classes)))
                transfer_loss = criterion1(ad_out, dc_target)
                '''
                # transfer dan
                #transfer_loss = mmd.MMDLoss(outputs_bottleneck[0:batch_size_train, :], outputs_bottleneck[batch_size_train:2*batch_size_train]) + mmd.MMDLoss(outputs_fc8[0:(batch_size_train),:], outputs_fc8[batch_size_train:(batch_size_train*2),:])
                # transfer jan
                #transfer_loss = mmd.JMMDLoss([outputs_bottleneck[0:batch_size_train, :], softmax_output[0:batch_size_train, :]], [outputs_bottleneck[batch_size_train:batch_size_train*2, :], softmax_output[batch_size_train:batch_size_train*2, :]], kernel_mul_list=[2.0, 2.0], kernel_num_list=[5.0, 1.0], fix_sigma_list=[None, 1.3])


                total_loss = transfer_loss + loss
                total_loss.backward()
                optimizer.step()
                running_transfer_loss += transfer_loss.data[0]
                running_cross_loss += loss.data[0]
                iter_num += 1
              epoch_transfer_loss = running_transfer_loss / (len(dset_loaders["val"])-1)
              epoch_cross_loss = running_cross_loss / (len(dset_loaders["val"])-1)

              print('{} Transfer Loss: {}, Cross Entropy Loss: {:.4f}'.format(
                  phase, epoch_transfer_loss, epoch_cross_loss))
            else:
              epoch_acc = test_target(dset_loaders, model_predict)
              print('{} Acc: {:.4f}'.format(
                  phase, epoch_acc))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model_all)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return best_model

def test_target(loader, model):
  iter_val = [iter(loader['val'+str(i)]) for i in xrange(10)]
  for i in xrange(len(loader['val0'])):
    data = [iter_val[j].next() for j in xrange(10)]
    inputs = [data[j][0] for j in xrange(10)]
    labels = data[0][1]
    if use_gpu:
      for j in xrange(10):
        inputs[j] = Variable(inputs[j].cuda())
      labels = Variable(labels.cuda())
    else:
      for j in xrange(10):
        inputs[j] = Variable(inputs[j])
      labels = Variable(labels)
    outputs = []
    for j in xrange(10):
      outputs.append(model(inputs[j]))
    outputs = sum(outputs)
    if i == 0:
      all_output = outputs.data.float()
      all_label = labels.data.float()
    else:
      all_output = torch.cat((all_output, outputs.data.float()), 0)
      all_label = torch.cat((all_label, labels.data.float()), 0)
  _, predict = torch.max(all_output, 1)
  #torch.save(predict, "predict.pt")
  #torch.save(all_label, "label.pt")
  return torch.sum(torch.squeeze(predict).float() == all_label) / float(all_label.size()[0])



if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Transfer Learning')
  parser.add_argument('gpu_id', type=str, nargs='?', default='0', help="device id to run")
  args = parser.parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
  #set data_transforms
  data_transforms = {
      'train': caffe_t.transform_train(resize_size=256, crop_size=224),
      'val': caffe_t.transform_train(resize_size=256, crop_size=224),
  }
  data_transforms = caffe_t.transform_test(data_transforms=data_transforms, resize_size=256, crop_size=224)
  
  #set dataset
  batch_size = {"train":16, "val":16}
  for i in xrange(10):
    batch_size["val"+str(i)] = 4
  '''
  source_dir = "/home/caozhangjie/caffe-grl/data/office/domain_adaptation_images/webcam/images/"
  target_dir = "/home/caozhangjie/caffe-grl/data/office/domain_adaptation_images/amazon/images/"
  data_dir = {}
  data_dir["train"] = source_dir
  data_dir["val"] = target_dir
  dsets = {x: datasets.ImageFolder(data_dir["train"] if "train" in x else data_dir["val"], data_transforms[x])
           for x in ['train',"val"]+["val"+str(i) for i in xrange(10)]}
  dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size[x],
                                                 shuffle=True if x=="train" or x=="val" else False, num_workers=4)
                  for x in ['train','val']+["val"+str(i) for i in xrange(10)]}
  dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']+["val"+str(i) for i in xrange(10)]}
  dset_classes = dsets["train"].classes
  '''

  dsets = {"train": ImageList(open("./iccv_challenge/train_list.txt").readlines(), transform=data_transforms["train"]), "val":ImageList(open("./iccv_challenge/validation_list.txt").readlines(), transform=data_transforms["val"])}
  dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size[x],
                                                  shuffle=True, num_workers=4) 
                                                  for x in ['train','val']}

  for i in xrange(10):
    dsets["val"+str(i)] = ImageList(open("./iccv_challenge/validation_list.txt").readlines(), transform=data_transforms["val"+str(i)])
    dset_loaders["val"+str(i)] = torch.utils.data.DataLoader(dsets["val"+str(i)], batch_size=batch_size["val"+str(i)], shuffle=False, num_workers=4) 

  dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']+["val"+str(i) for i in xrange(10)]}
  dset_classes = range(12)
  
  #construct model
  use_gpu = torch.cuda.is_available()
  model_fc = model_no.Resnet50Fc()
  num_features = model_fc.output_num()
  bottleneck_layer = nn.Linear(num_features, 256)
  classifier_layer = nn.Linear(256, len(dset_classes))
  bottleneck_layer.weight.data.normal_(0, 0.005)
  classifier_layer.weight.data.normal_(0, 0.01)
  bottleneck_layer.bias.data.fill_(0.1)
  classifier_layer.bias.data.fill_(0.0)

  ad_layer1 = nn.Linear(1024, 1024)
  ad_layer2 = nn.Linear(1024,1024)
  ad_layer3 = nn.Linear(1024, 1)
  ad_layer1.weight.data.normal_(0, 0.01)
  ad_layer2.weight.data.normal_(0, 0.01)
  ad_layer3.weight.data.normal_(0, 0.3)
  ad_layer1.bias.data.fill_(0.0)
  ad_layer2.bias.data.fill_(0.0)
  ad_layer3.bias.data.fill_(0.0)

  ad_net = nn.Sequential(ad_layer1, nn.ReLU(), nn.Dropout(0.5), ad_layer2, nn.ReLU(), nn.Dropout(0.5), ad_layer3, nn.Sigmoid())
  rman_layer = ad.RMANLayer([256, len(dset_classes)], 1024)
  gradient_reverse_layer = ad.AdversarialLayer()
  model_all = nn.Sequential(model_fc, bottleneck_layer, classifier_layer, ad_net)
  layers_train = nn.Sequential(bottleneck_layer, classifier_layer, ad_net)
  if use_gpu:
    #model_fc = model_fc.cuda()
    #classifier_layer = classifier_layer.cuda()
    model_all = model_all.cuda()
    rman_layer.cuda()
  
  criterion = nn.CrossEntropyLoss()
  optimizer_ft = optim.SGD([{"params":model_fc.parameters(), "lr":1},{"params":layers_train.parameters(), "lr":10}], lr=1, momentum=0.9, weight_decay=0.0005)
  #optimizer_ft = optim.SGD(layers_train.parameters(), lr=10, momentum=0.9, weight_decay=0.0005)
  param_lr = []
  for param_group in optimizer_ft.param_groups:
    param_lr.append(param_group["lr"])
  print param_lr
  
  #start train
  print "start train"
  model_all = train_source([model_fc, bottleneck_layer, classifier_layer, ad_net, gradient_reverse_layer, rman_layer], model_all, nn.Sequential(model_fc, bottleneck_layer, classifier_layer), criterion, optimizer_ft, inv_lr_scheduler, batch_size_train = batch_size["train"],num_epochs=300, num_classes=len(dset_classes))
  #save model
  torch.save(model_fc, "../data/save_model.pth")
