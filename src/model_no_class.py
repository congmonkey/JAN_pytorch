import torch
import torch.nn as nn
import torchvision
from torchvision import models

# convnet without the last layer
class AlexnetFc(nn.Module):
  def __init__(self):
    super(AlexnetFc, self).__init__()
    model_alexnet = models.alexnet(pretrained=True)
    self.features = model_alexnet.features
    self.classifier = nn.Sequential()
    for i in xrange(6):
      self.classifier.add_module("classifier"+str(i), model_alexnet.classifier[i])
    self.__in_features = model_alexnet.classifier[6].in_features
  
  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), 256*6*6)
    x = self.classifier(x)
    return x

  def output_num(self):
    return self.__in_features

class Resnet50Fc(nn.Module):
  def __init__(self):
    super(Resnet50Fc, self).__init__()
    model_resnet50 = models.resnet50(pretrained=True)
    self.conv1 = model_resnet50.conv1
    self.bn1 = model_resnet50.bn1
    self.relu = model_resnet50.relu
    self.maxpool = model_resnet50.maxpool
    self.layer1 = model_resnet50.layer1
    self.layer2 = model_resnet50.layer2
    self.layer3 = model_resnet50.layer3
    self.layer4 = model_resnet50.layer4
    self.avgpool = model_resnet50.avgpool
    self.__in_features = model_resnet50.fc.in_features

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    return x

  def output_num(self):
    return self.__in_features

class Resnet152Fc(nn.Module):
  def __init__(self):
    super(Resnet152Fc, self).__init__()
    model_resnet152 = models.resnet152(pretrained=True)
    self.conv1 = model_resnet152.conv1
    self.bn1 = model_resnet152.bn1
    self.relu = model_resnet152.relu
    self.maxpool = model_resnet152.maxpool
    self.layer1 = model_resnet152.layer1
    self.layer2 = model_resnet152.layer2
    self.layer3 = model_resnet152.layer3
    self.layer4 = model_resnet152.layer4
    self.avgpool = model_resnet152.avgpool
    self.__in_features = model_resnet152.fc.in_features

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    return x

  def output_num(self):
    return self.__in_features
