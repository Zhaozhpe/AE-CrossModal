from __future__ import absolute_import

'''SphereNet

WResnet is 4x wider
'''
import torch.nn as nn
import math
from sphereModel.SphereConv import Sphere_Conv2d


__all__ = ['sphere_resnet','sphere_resnet110','sphere_resnet56','sphere_resnet44','sphere_resnet32','sphere_resnet20',
            'sphere_wresnet20','sphere_wresnet32','sphere_wresnet44','sphere_wresnet110']

def sphereconv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return Sphere_Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SphereBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SphereBasicBlock, self).__init__()
        self.conv1 = sphereconv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = sphereconv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SphereBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SphereBottleneck, self).__init__()
        self.conv1 = Sphere_Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Sphere_Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = Sphere_Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SphereResNet(nn.Module):

    def __init__(self, depth, nfilter = [64,128,256], num_classes=10):
        super(SphereResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        # [16,32,64]
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = SphereBottleneck if depth >=44 else SphereBasicBlock

        self.inplanes = nfilter[0]
        self.conv1 = Sphere_Conv2d(3, self.inplanes, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, nfilter[0], n)
        self.layer2 = self._make_layer(block, nfilter[1], n, stride=2)
        self.layer3 = self._make_layer(block, nfilter[2], n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(nfilter[2] * block.expansion, num_classes)

        self.features = nn.Sequential(
                self.conv1,
                self.bn1,
                self.relu,    # 32x32
                self.layer1,  # 32x32
                self.layer2,  # 16x16
                self.layer3,  # 8x8
                self.avgpool
        )

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Sphere_Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x

    def project(self):
        for m in self.modules():
            if isinstance(m, Sphere_Conv2d):
                m.project()

    def showOrthInfo(self):
        for m in self.modules():
            if isinstance(m, Sphere_Conv2d):
                m.showOrthInfo()


def sphere_resnet(**kwargs):
    """
    Constructs a SphereResNet model.
    """
    return SphereResNet(**kwargs)

def sphere_resnet110(**kwargs):
    """Constructs a SphereResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SphereResNet(110, **kwargs)
    return model

def sphere_resnet56(**kwargs):
    """Constructs a SphereResNet-56 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SphereResNet(56, **kwargs)
    return model

def sphere_resnet44(**kwargs):
    """Constructs a SphereResNet-44 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SphereResNet(44, **kwargs)
    return model

def sphere_resnet32(**kwargs):
    """Constructs a SphereResNet-32 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SphereResNet(32, **kwargs)
    return model

def sphere_resnet20(**kwargs):
    """Constructs a SphereResNet-20 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SphereResNet(20, nfilter = [64,128,256],**kwargs)  # roc edit add nfilter
    return model

def sphere_wresnet20(**kwargs):
    """Constructs a Wide SphereResNet-20 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SphereResNet(20, nfilter = [64,128,256], **kwargs)
    return model

def sphere_wresnet32(**kwargs):
    """Constructs a Wide SphereResNet-32 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SphereResNet(32, nfilter = [64,128,256], **kwargs)
    return model

def sphere_wresnet44(**kwargs):
    """Constructs a Wide SphereResNet-44 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SphereResNet(44, nfilter = [64,128,256], **kwargs)
    return model

def sphere_wresnet110(**kwargs):
    """Constructs a Wide SphereResNet-110 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SphereResNet(110, nfilter = [64,128,256], **kwargs)
    return model
