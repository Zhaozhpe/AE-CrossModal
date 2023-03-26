import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from attention.SEAttention import SEAttention

#add
class GatingContext(nn.Module):
    def __init__(self, dim, add_batch_norm=True):
        super(GatingContext, self).__init__()
        self.dim = dim
        self.add_batch_norm = add_batch_norm
        self.gating_weights = nn.Parameter(
            torch.randn(dim, dim) * 1 / math.sqrt(dim))
        self.sigmoid = nn.Sigmoid()

        if add_batch_norm:
            self.gating_biases = None
            self.bn1 = nn.BatchNorm1d(dim)
        else:
            self.gating_biases = nn.Parameter(
                torch.randn(dim) * 1 / math.sqrt(dim))
            self.bn1 = None

    def forward(self, x):
        gates = torch.matmul(x, self.gating_weights)

        if self.add_batch_norm:
            gates = self.bn1(gates)
        else:
            gates = gates + self.gating_biases

        gates = self.sigmoid(gates)

        activation = x * gates

        return activation

class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, alpha=100.0,
                 normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
#add
        self.hidden1_weights = nn.Parameter(
            torch.randn(num_clusters * dim, 256) * 1 / math.sqrt(dim))
        self.bn2 = nn.BatchNorm1d(256)
        self.context_gating = GatingContext(
            256, add_batch_norm=True)

        # self._init_params()

    # def _init_params(self):
    #     self.conv.weight = nn.Parameter(
    #         (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
    #     )
    #     self.conv.bias = nn.Parameter(
    #         - self.alpha * self.centroids.norm(dim=1)
    #     )
# add cluster initial
    def init_params(self, clsts, traindescs):
        clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
        dots = np.dot(clstsAssign, traindescs.T)
        dots.sort(0)
        dots = dots[::-1, :]  # sort, descending

        self.alpha = (-np.log(0.01) / np.mean(dots[0, :] - dots[1, :])).item()
        # noinspection PyArgumentList
        self.centroids = nn.Parameter(torch.from_numpy(clsts))
        # noinspection PyArgumentList
        self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha * clstsAssign).unsqueeze(2).unsqueeze(3))
        self.conv.bias = None

    def forward(self, x):
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)
        
        # calculate residuals to each clusters
        '''residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)'''
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for C in range(self.num_clusters):  # slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                       self.centroids[C:C + 1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:, C:C + 1, :].unsqueeze(2)
            vlad[:, C:C + 1, :] = residual.sum(dim=-1)
        # print('vlad.size')
        # print(vlad.shape)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        # # test
        # vlad_t = vlad

        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize


# add
        vlad = torch.matmul(vlad, self.hidden1_weights)
        # print("vald1.size")
        # print(vlad.shape)

        vlad = self.bn2(vlad)
        # print("vald2.size")
        # print(vlad.shape)

        vlad = self.context_gating(vlad)

        return vlad
#, vlad_t

class EmbedNet(nn.Module):
    def __init__(self, base_model, net_vlad):
        super(EmbedNet, self).__init__()
        self.base_model = base_model
        self.net_vlad = net_vlad

    def forward(self, x):
        x = self.base_model(x)
        embedded_x = self.net_vlad(x)
        return embedded_x


class TripletNet(nn.Module):
    def __init__(self, embed_net):
        super(TripletNet, self).__init__()
        self.embed_net = embed_net

    def forward(self, a, p, n):
        embedded_a = self.embed_net(a)
        embedded_p = self.embed_net(p)
        embedded_n = self.embed_net(n)
        return embedded_a, embedded_p, embedded_n

    def feature_extract(self, x):
        return self.embed_net(x)

def get_model_netvlad(encoder, encoder_dim, config, attention):
    nn_model = nn.Module()
    nn_model.add_module('encoder', encoder)
    if attention:
        seaAtten = SEAttention(channel=512, reduction=8)
        seaAtten.init_weights()
        nn_model.add_module('attention', seaAtten)
    net_vlad = NetVLAD(num_clusters=int(config['num_clusters']), dim=encoder_dim, alpha=1.0)
    nn_model.add_module('pool', net_vlad)
    return nn_model
