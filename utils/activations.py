# -*- coding: utf-8 -*-
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Activation functions
"""
import math
from abc import ABCMeta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Hard_Swish(nn.Module):  # export-friendly version of nn.SiLU()
    @staticmethod
    def Hard_Swish(x):
        f = x + 3
        relu6 = np.where(np.where(f < 0, 0, f) > 6, 6, np.where(f < 0, 0, f))
        return x * relu6 / 6


class leaky_relu(nn.Module):  # export-friendly version of nn.SiLU()
    @staticmethod
    def leaky_relu(x, a=0.01):
        return np.maximum(a * x, x)


class linspace(nn.Module):  # export-friendly version of nn.SiLU()
    @staticmethod
    def linspace(x, a=1, b=0):
        return x * a + b


class gelu(nn.Module):  # export-friendly version of nn.SiLU()
    @staticmethod
    def gelu(x):
        return x * nn.sigmod(1.702 * x)


class lrelu(nn.Module):  # export-friendly version of nn.SiLU()
    @staticmethod
    def lrelu(x):
        return np.maximum(0.01 * x, x)


class Hard_Sigmoid(nn.Module):  # export-friendly version of nn.SiLU()
    @staticmethod
    def Hard_Sigmoid(x):
        f = (2 * x + 5) / 10
        return np.where(np.where(f > 1, 1, f) < 0, 0, np.where(f > 1, 1, f))


class relu(nn.Module):  # export-friendly version of nn.SiLU()
    @staticmethod
    def relu(x):
        return np.maximum(0, x)


class Squareplus(nn.Module):  # export-friendly version of nn.SiLU()
    @staticmethod
    def Squareplus(x, b=0.2):
        x = 0.5 * (x + np.sqrt(x ** 2 + b))
        return x


class selu(nn.Module):  # export-friendly version of nn.SiLU()
    @staticmethod
    def selu(x, a=1, b=1):
        return a * np.where(x > 0, x, b * ((math.e ** x) - 1))


class elu(nn.Module):  # export-friendly version of nn.SiLU()
    @staticmethod
    def elu(x, alpha=1):
        a = x[x > 0]
        b = alpha * (np.exp(x[x < 0]) - 1)
        result = np.concatenate((b, a), axis=0)
        return result


class sigmoid(nn.Module):  # export-friendly version of nn.SiLU()
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


class softmax(nn.Module):  # export-friendly version of nn.SiLU()
    @staticmethod
    def softmax(x):
        x = np.exp(x) / np.sum(np.exp(x))
        return x


class softplus(nn.Module):  # export-friendly version of nn.SiLU()
    @staticmethod
    def softplus(x):
        return math.log(1 + pow(math.e, x))


class softsign(nn.Module):  # export-friendly version of nn.SiLU()

    @staticmethod
    def softsign(x):
        return x / (1 + np.abs(x))


class step_function(nn.Module):  # export-friendly version of nn.SiLU()

    @staticmethod
    def step_function(x):
        return np.array(x > 0, dtype=np.int)


class tanh(nn.Module):  # export-friendly version of nn.SiLU()
    @staticmethod
    def tanh(x):
        return 2 / (1 + np.exp(-2 * x)) - 1


# SiLU https://arxiv.org/pdf/1606.08415.pdf ----------------------------------------------------------------------------
class SiLU(nn.Module):  # export-friendly version of nn.SiLU()
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Hardswish(nn.Module):  # export-friendly version of nn.Hardswish()
    @staticmethod
    def forward(x):
        # return x * F.hardsigmoid(x)  # for TorchScript and CoreML
        return x * F.hardtanh(x + 3, 0.0, 6.0) / 6.0  # for TorchScript, CoreML and ONNX


# Mish https://github.com/digantamisra98/Mish --------------------------------------------------------------------------
class Mish(nn.Module):
    @staticmethod
    def forward(x):
        return x * F.softplus(x).tanh()


class MemoryEfficientMish(nn.Module):
    class F(torch.autograd.Function, metaclass=ABCMeta):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            fx = F.softplus(x).tanh()
            return grad_output * (fx + x * sx * (1 - fx * fx))

    def forward(self, x):
        return self.F.apply(x)


# FReLU https://arxiv.org/abs/2007.11824 -------------------------------------------------------------------------------
class FReLU(nn.Module):
    def __init__(self, c1, k=3):  # ch_in, kernel
        super().__init__()
        self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1, bias=False)
        self.bn = nn.BatchNorm2d(c1)

    def forward(self, x):
        return torch.max(x, self.bn(self.conv(x)))


# ACON https://arxiv.org/pdf/2009.04759.pdf ----------------------------------------------------------------------------
class AconC(nn.Module):
    r""" ACON activation (activate or not).
    AconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is a learnable parameter
    according to "Activate or Not: Learning Customized Activation" <https://arxiv.org/pdf/2009.04759.pdf>.
    """

    def __init__(self, c1):
        super().__init__()
        self.p1 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, c1, 1, 1))

    def forward(self, x):
        dpx = (self.p1 - self.p2) * x
        return dpx * torch.sigmoid(self.beta * dpx) + self.p2 * x


class MetaAconC(nn.Module):
    r""" ACON activation (activate or not).
    MetaAconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is generated by a small network
    according to "Activate or Not: Learning Customized Activation" <https://arxiv.org/pdf/2009.04759.pdf>.
    """

    def __init__(self, c1, k=1, s=1, r=16):  # ch_in, kernel, stride, r
        super().__init__()
        c2 = max(r, c1 // r)
        self.p1 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.fc1 = nn.Conv2d(c1, c2, k, s)
        self.fc2 = nn.Conv2d(c2, c1, k, s)
        # self.bn1 = nn.BatchNorm2d(c2)
        # self.bn2 = nn.BatchNorm2d(c1)

    def forward(self, x):
        y = x.mean(dim=2, keepdims=True).mean(dim=3, keepdims=True)
        # batch-size 1 bug/instabilities https://github.com/ultralytics/yolov5/issues/2891
        # beta = torch.sigmoid(self.bn2(self.fc2(self.bn1(self.fc1(y)))))  # bug/unstable
        beta = torch.sigmoid(self.fc2(self.fc1(y)))  # bug patch BN layers removed
        dpx = (self.p1 - self.p2) * x
        return dpx * torch.sigmoid(beta * dpx) + self.p2 * x


class celu(nn.Module):
    @staticmethod
    def forward(x):
        return nn.functional.celu(x)


class glu(nn.Module):
    @staticmethod
    def forward(x):
        return nn.functional.glu(x)


class hardshrink(nn.Module):
    @staticmethod
    def forward(x):
        return nn.functional.hardshrink(x)


class hardtanh(nn.Module):
    @staticmethod
    def forward(x):
        return nn.functional.hardtanh(x)


class prelu(nn.Module):
    @staticmethod
    def forward(x):
        return nn.functional.prelu(x)


class rrelu(nn.Module):
    @staticmethod
    def forward(x):
        return nn.functional.rrelu(x)


class softmin(nn.Module):
    @staticmethod
    def forward(x):
        return nn.functional.softmin(x)


class softsign(nn.Module):
    @staticmethod
    def forward(x):
        return nn.functional.softsign(x)


class tanhshrink(nn.Module):
    @staticmethod
    def forward(x):
        return nn.functional.tanhshrink(x)
