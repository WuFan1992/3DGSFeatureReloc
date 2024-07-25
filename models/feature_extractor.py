import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms



class FeatureExtractor(nn.Module):
     def __init__(self):
         super(FeatureExtractor,self).__init__()
         self.extractor = nn.Sequential(
             nn.Conv2d(3, 64, kernel_size=(3,3)),
             nn.ReLU(True),
             nn.Conv2d(64, 64,kernel_size=(3,3)),
             nn.ReLU(True),
             nn.MaxPool2d(kernel_size=(2,2)), 
             nn.Conv2d(64, 64, kernel_size=(3,3)),
             nn.ReLU(True),  
             nn.Conv2d(64, 64, kernel_size=(3,3)),
             nn.ReLU(True),
             nn.MaxPool2d(kernel_size=(2,2)), 
             nn.Conv2d(64, 128, kernel_size=(3,3)),
             nn.ReLU(True),
             nn.Conv2d(128, 128, kernel_size=(3,3)),
             nn.ReLU(True),
             nn.Conv2d(128, 128,kernel_size=(3,3)),
             nn.ReLU(True),
             nn.Conv2d(128,32, kernel_size=(3,3) ), 
             nn.ReLU(True)
             )

     def forward(self, x):
         x = transforms.ToTensor()(x)
         return  self.extractor(x)       