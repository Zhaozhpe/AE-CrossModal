'''
Usage
'''
import torch
import torch.nn as nn
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.modules.conv import _ConvNd
from torch.nn import functional as F
from torch.autograd import Variable
import math
from torch.nn.parameter import Parameter

__all__ = ['Sphere_Conv2d']

class Sphere_Conv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1, bias=False, doini = 2, padding_mode: str = 'zeros', norm=True):

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Sphere_Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        self.scale = Parameter(torch.Tensor(1))
        self.scale.data.fill_(1)

        self.eps = 1e-8

        if norm:
            self.register_buffer('input_norm_wei',torch.ones(1, in_channels // groups, *kernel_size))

        if doini == 1:
            n = self.kernel_size[0] * self.kernel_size[1] * self.out_channels
            self.weight.data.normal_(0, math.sqrt(2. / n))
        elif doini == 2:
            n = self.kernel_size[0] * self.kernel_size[1] * self.out_channels
            nn.init.orthogonal(self.weight,math.sqrt(2. / n))

    def forward(self, input):

        _input = input
        self.project()
        _weight = self.weight
        _weight = _weight/ torch.norm(_weight.view(self.out_channels,-1),2,1).clamp(min = self.eps).view(-1,1,1,1)
        _output = F.conv2d(input, _weight*self.scale, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

        input_norm = torch.sqrt(F.conv2d(_input**2, Variable(self.input_norm_wei), None,
                            self.stride, self.padding, self.dilation, self.groups).clamp(min = self.eps))
        _output = _output/input_norm

        return _output

    def project(self, manifold_grad = False):
        '''
        Project weight to l2 ball
        '''
        originSize = self.weight.data.size()
        outputSize = self.weight.data.size()[0]
        self.weight.data =  self.weight.data/ torch.norm(self.weight.data.view(outputSize,-1),2,1).clamp(min = 1e-8).view(-1,1,1,1)

    def showOrthInfo(self):
        originSize = self.weight.data.size()
        outputSize = self.weight.data.size()[0]
        W = self.weight.data.view(outputSize,-1)
        _, s, _ = torch.svd(W.t())
        print('Singular Value Summary: ')
        print('max :',s.max())
        print('mean:',s.mean())
        print('min :',s.min())
        print('var :',s.var())
        print('penalty :', (W.mm(W.t())-torch.eye(outputSize)).norm()**2  )
