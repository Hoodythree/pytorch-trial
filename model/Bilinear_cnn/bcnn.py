import torch
import torch.nn as nn
from torchvision.models import resnet50


class BCNN(nn.Module):
   def __init__(self):
       super(BCNN, self).__init__()
       self.features = nn.Sequential(resnet50().conv1, 

                              resnet50().bn1, 

                              resnet50().relu, 

                              resnet50().maxpool, 

                              resnet50().layer1,

                              resnet50().layer2,


                              resnet50().layer3,

                              resnet50().layer4)
      # the output of layer4 of ResNet50 is 2048,so bilinear vector is 2048**2
       self.classifiers = nn.Sequential(nn.Linear(2048 ** 2, 200))

   def forward(self, x):
       x = self.features(x)
       batch_size = x.size(0)
       x = x.view(batch_size, 2048, x.size(2) ** 2)
       # Q: Why divided by 28*28?
       # A: Average pooling
       x = (torch.bmm(x, torch.transpose(x, 1, 2)) / 28 ** 2).view(batch_size, -1)
       x = torch.nn.functional.normalize(torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-10))
       x = self.classifiers(x)
       return x
