import os
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch import Tensor
from .quantize_utils import QConv2d

__all__ = ['qresnet18_cifar', 'qresnet50_cifar', 'qresnet18', 'qresnet50']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, wbit=32, abit=32):
    """3x3 convolution with padding"""

    return QConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=dilation, groups=groups, bias=False, dilation=dilation, w_bit=wbit, a_bit=abit)


def conv1x1(in_planes, out_planes, stride=1, wbit=32, abit=32):
    """1x1 convolution"""

    return QConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, w_bit=wbit, a_bit=abit)


class BasicBlock(nn.Module):
    expansion = 1
    num_layers = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    num_layers = 3

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class QResNet_cifar(nn.Module):

    def __init__(
            self,
            block,
            layers,
            num_classes=1000,
            zero_init_residual=False,
            groups=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None
    ):
        super(QResNet_cifar, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_last = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():

                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks,
                    stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, mode=None, TS=None, grad_out=None):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        f0 = self.layer4(x)

        if mode == "eval":
            pass
        elif mode == 'swa':
            f0.retain_grad()

        out = self.avgpool(f0)
        out = torch.flatten(out, 1)
        out = self.fc_last(out)

        if mode == 'swa':
            if not isinstance(grad_out, Variable):
                ind = out.data.max(1)[1]
                grad_out = out.data.clone().fill_(0.0).scatter_(1, ind.unsqueeze(0).t(), 1.0)

            swa = self.cal_grad(out, grad_out, TS, [f0])
            return out, swa

        return out

    def cal_grad(self, out, grad_out, TS, feature):
        attributions = []
        if TS == 'Full':
            out.backward(grad_out, retain_graph=True)
            feat = feature[0].clone().detach()
            grad = feature[0].grad.clone().detach()
            linear = torch.sum(torch.sum(grad, 3, keepdim=True), 2, keepdim=True)
            channel = linear * feat
            swa = torch.sum(channel, 1, keepdim=True)
            attributions.append(F.relu(swa))

            return attributions

        elif TS == 'Quant':
            out.backward(grad_out, create_graph=True)

            linear = torch.sum(torch.sum(feature[0].grad, 3, keepdim=True), 2, keepdim=True)
            channel = linear * feature[0]
            swa = torch.sum(channel, 1, keepdim=True)
            attributions.append(F.relu(swa))

        return attributions


class QResNet(nn.Module):

    def __init__(
            self,
            block,
            layers,
            num_classes=1000,
            zero_init_residual=False,
            groups=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None
    ):
        super(QResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks,
                    stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

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
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def qresnet18(pretrained=None, num_classes=1000):
    model = QResNet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes)

    if pretrained:
        ch = torch.load(pretrained)
        ch = {n.replace('module.', ''): v for n, v in ch['state_dict'].items()}
        model.load_state_dict(ch)
        print('loading pretrained')
    return model


def qresnet50(pretrained=None, num_classes=1000):
    model = QResNet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes)

    if pretrained:
        ch = torch.load(pretrained)
        ch = {n.replace('module.', ''): v for n, v in ch['state_dict'].items()}
        model.load_state_dict(ch)
        print('loading pretrained')
    return model


def qresnet18_cifar(pretrained=None, num_classes=1000):
    model = QResNet_cifar(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes)

    if pretrained:
        ch = torch.load(pretrained)
        ch = {n.replace('module.', ''): v for n, v in ch['state_dict'].items()}
        model.load_state_dict(ch)
        print('loading pretrained')
    return model


def qresnet50_cifar(pretrained=None, num_classes=1000):
    model = QResNet_cifar(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=num_classes)

    if pretrained:
        ch = torch.load(pretrained)
        ch = {n.replace('module.', ''): v for n, v in ch['state_dict'].items()}
        model.load_state_dict(ch)
        print('loading pretrained')
    return model


