"""
Adapted from https://github.com/GuHongyang/VaDE-pytorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck, BasicBlock, conv1x1
from torchvision.models import resnet50


def block(in_c, out_c):
    layers = [nn.Linear(in_c, out_c), nn.ReLU()]
    return layers

class Encoder(nn.Module):
    def __init__(self, input_dim=784, inter_dims=[500, 500], hid_dim=2000):
        super(Encoder, self).__init__()
        self.hid_dim = hid_dim
        self.encoder = nn.Sequential(
            *block(input_dim, inter_dims[0]),
            *block(inter_dims[0], inter_dims[1]),
            *block(inter_dims[1], hid_dim),
        )

    def forward(self, x):
        e = self.encoder(x)

        return e


class Decoder(nn.Module):
    def __init__(self, input_dim=784, inter_dims=[500, 500, 2000], hid_dim=10, sigmoid=True):
        super(Decoder, self).__init__()

        layers = [
            *block(hid_dim, inter_dims[-1]),
            *block(inter_dims[-1], inter_dims[-2]),
            *block(inter_dims[-2], inter_dims[-3]),
            nn.Linear(inter_dims[-3], input_dim),
        ]
        if sigmoid:
            layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(
            *layers
        )

    def forward(self, z):
        x_pro = self.decoder(z)

        return x_pro
    
class ResnetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResnetFeatureExtractor, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.resnet.eval()
        
    def forward(self, x):
        with torch.no_grad():
            x = self.resnet.conv1(x)
            x = self.resnet.bn1(x)
            x = self.resnet.relu(x)
            x = self.resnet.maxpool(x)

            x = self.resnet.layer1(x)
            x = self.resnet.layer2(x)
            x = self.resnet.layer3(x)
            x = self.resnet.layer4(x)

            x = self.resnet.avgpool(x)
            return x.reshape(x.shape[0],-1)