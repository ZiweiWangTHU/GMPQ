import torch.nn as nn
import math
from . import quant_module as qm
import torch.nn.functional as F
from torch.autograd import Variable
from torch import Tensor
import torch

__all__ = [
    'mixres18_w2346a2346', 'mixres50_w234a234','mixres18_w234a234'
]


def conv3x3(conv_func, in_planes, out_planes, stride=1, **kwargs):
    "3x3 convolution with padding"
    return conv_func(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, **kwargs)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, conv_func, inplanes, planes, stride=1,
                 downsample=None, bnaff=True, **kwargs):
        super(BasicBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes, affine=bnaff)
        self.conv1 = conv3x3(conv_func, inplanes, planes, stride, **kwargs)
        self.bn1 = nn.BatchNorm2d(planes, affine=bnaff)
        self.conv2 = conv3x3(conv_func, planes, planes, **kwargs)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        out = self.bn0(x)
        if self.downsample is not None:
            residual = out
        else:
            residual = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, conv_func, inplanes, planes, stride=1,
                 downsample=None, bnaff=True, **kwargs):
        super(Bottleneck, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes, affine=bnaff)
        self.conv1 = conv_func(
            inplanes, planes, kernel_size=1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(planes, affine=bnaff)
        self.conv2 = conv_func(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(planes, affine=bnaff)
        self.conv3 = conv_func(
            planes, planes * self.expansion, kernel_size=1, bias=False, **kwargs)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample

    def forward(self, x):
        out = self.bn0(x)
        if self.downsample is not None:
            residual = out
        else:
            residual = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual

        return out


class ResNet(nn.Module):

    def __init__(self, block, conv_func, layers, num_classes=1000,
                 bnaff=True, **kwargs):
        if 'abits' in kwargs:
            print('abits: {}'.format(kwargs['abits']))
        if 'wbits' in kwargs:
            print('wbits: {}'.format(kwargs['wbits']))
        self.inplanes = 64
        self.conv_func = conv_func
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(
            block, conv_func, 64, layers[0], bnaff=bnaff, **kwargs)
        self.layer2 = self._make_layer(
            block, conv_func, 128, layers[1], stride=2, bnaff=bnaff, **kwargs)
        self.layer3 = self._make_layer(
            block, conv_func, 256, layers[2], stride=2, bnaff=bnaff, **kwargs)
        self.layer4 = self._make_layer(
            block, conv_func, 512, layers[3], stride=2, bnaff=bnaff, **kwargs)
        self.avgpool = nn.AvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_layer(self, block, conv_func, planes, blocks, stride=1, bnaff=True, **kwargs):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv_func(self.inplanes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False, **kwargs),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(conv_func, self.inplanes, planes, stride, downsample, bnaff=bnaff, **kwargs))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(conv_func, self.inplanes, planes, bnaff=bnaff, **kwargs))

        return nn.Sequential(*layers)

    def forward(self, x,mode=None,TS=None, grad_out=None):
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

        out = F.avg_pool2d(f0, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

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

    def complexity_loss(self):
        size_product = []
        loss = 0
        for m in self.modules():
            if isinstance(m, self.conv_func):
                loss += m.complexity_loss()
                size_product += [m.size_product]
        normalizer = size_product[0].item()
        loss /= normalizer
        return loss

    def fetch_best_arch(self):
        sum_bitops, sum_bita, sum_bitw = 0, 0, 0
        sum_mixbitops, sum_mixbita, sum_mixbitw = 0, 0, 0
        layer_idx = 0
        best_arch = None
        for m in self.modules():
            if isinstance(m, self.conv_func):
                layer_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw = m.fetch_best_arch(layer_idx)
                if best_arch is None:
                    best_arch = layer_arch
                else:
                    for key in layer_arch.keys():
                        if key not in best_arch:
                            best_arch[key] = layer_arch[key]
                        else:
                            best_arch[key].append(layer_arch[key][0])
                sum_bitops += bitops
                sum_bita += bita
                sum_bitw += bitw
                sum_mixbitops += mixbitops
                sum_mixbita += mixbita
                sum_mixbitw += mixbitw
                layer_idx += 1
        return best_arch, sum_bitops, sum_bita, sum_bitw, sum_mixbitops, sum_mixbita, sum_mixbitw

    def fetch_bit(self):
        sum_mix_bit=0.
        layer_idx = 0
        for m in self.modules():
            if isinstance(m, self.conv_func):
                mixbitw, mixbita = m.fetch_mix_bit(layer_idx)
                sum_mix_bit += 1./mixbitw * 1./mixbita
                layer_idx += 1

        return sum_mix_bit/layer_idx



def mixres18_w234a234(**kwargs):
    return ResNet(BasicBlock, qm.MixActivConv2d, [2, 2, 2, 2], wbits=[ 2, 3, 4], abits=[2, 3, 4],
                  share_weight=True, **kwargs)
def mixres18_w2346a2346(**kwargs):
    return ResNet(BasicBlock, qm.MixActivConv2d, [2, 2, 2, 2], wbits=[ 2, 3, 4, 6], abits=[2, 3, 4, 6],
                  share_weight=True, **kwargs)

def mixres50_w234a234(**kwargs):
    return ResNet(Bottleneck, qm.MixActivConv2d, [3, 4, 6, 3], wbits=[2, 3, 4], abits=[2, 3, 4],
                  share_weight=True, **kwargs)