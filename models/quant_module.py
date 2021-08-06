from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .quantize_utils import QModule


class QuantActiv(QModule):
    def __init__(self, bit=2):
        super(QuantActiv, self).__init__(w_bit=-1, a_bit=bit, half_wave=True)

    def forward(self, x):
        x = self._quantize_activation(inputs=x)
        return x


class QuantConv(QModule):
    def __init__(self, bit=2):
        super(QuantConv, self).__init__(w_bit=bit, a_bit=-1, half_wave=True)

    def forward(self, x):
        x = self._quantize_weight(weight=x)
        return x


class MixQuantActiv(nn.Module):

    def __init__(self, bits):
        super(MixQuantActiv, self).__init__()
        self.bits = bits
        self.alpha_activ = Parameter(torch.Tensor(len(self.bits)))
        self.alpha_activ.data.fill_(0.01)
        self.mix_activ = nn.ModuleList()
        for bit in self.bits:
            self.mix_activ.append(QuantActiv(bit=bit))

    def forward(self, input):
        outs = []
        sw = F.softmax(self.alpha_activ, dim=0)
        for i, branch in enumerate(self.mix_activ):
            outs.append(branch(input) * sw[i])
        activ = sum(outs)
        return activ


class MixQuantConv2d(nn.Module):

    def __init__(self, inplane, outplane, bits, **kwargs):
        super(MixQuantConv2d, self).__init__()
        assert not kwargs['bias']
        self.bits = bits
        self.alpha_weight = Parameter(torch.Tensor(len(self.bits)))
        self.alpha_weight.data.fill_(0.01)
        self.conv_list = nn.ModuleList()
        self.qconv_list = nn.ModuleList()
        for bit in self.bits:
            assert 0 < bit < 32
            self.conv_list.append(nn.Conv2d(inplane, outplane, **kwargs))
            self.qconv_list.append(QuantConv(bit))

    def forward(self, input):
        mix_quant_weight = []
        sw = F.softmax(self.alpha_weight, dim=0)
        for i, bit in enumerate(self.bits):
            weight = self.conv_list[i].weight
            quant_weight = self.qconv_list[i](weight)
            scaled_quant_weight = quant_weight * sw[i]
            mix_quant_weight.append(scaled_quant_weight)
        mix_quant_weight = sum(mix_quant_weight)
        conv = self.conv_list[0]
        out = F.conv2d(
            input, mix_quant_weight, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)
        return out


class SharedMixQuantConv2d(nn.Module):

    def __init__(self, inplane, outplane, bits, **kwargs):
        super(SharedMixQuantConv2d, self).__init__()
        assert not kwargs['bias']
        self.bits = bits
        self.alpha_weight = Parameter(torch.Tensor(len(self.bits)))
        self.alpha_weight.data.fill_(0.01)
        self.conv = nn.Conv2d(inplane, outplane, **kwargs)
        self.qconv_list = nn.ModuleList()
        for bit in self.bits:
            assert 0 < bit < 32
            self.qconv_list.append(QuantConv(bit))

    def forward(self, input):
        mix_quant_weight = []
        sw = F.softmax(self.alpha_weight, dim=0)
        conv = self.conv
        weight = conv.weight
        for i, bit in enumerate(self.bits):
            quant_weight = self.qconv_list[i](weight)

            scaled_quant_weight = quant_weight * sw[i]
            mix_quant_weight.append(scaled_quant_weight)
        mix_quant_weight = sum(mix_quant_weight)
        out = F.conv2d(
            input, mix_quant_weight, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)
        return out


class MixActivConv2d(nn.Module):

    def __init__(self, inplane, outplane, wbits=None, abits=None, share_weight=False, **kwargs):
        super(MixActivConv2d, self).__init__()
        if wbits is None:
            self.wbits = [1, 2]
        else:
            self.wbits = wbits
        if abits is None:
            self.abits = [1, 2]
        else:
            self.abits = abits

        self.mix_activ = MixQuantActiv(self.abits)
        self.share_weight = share_weight
        if share_weight:
            self.mix_weight = SharedMixQuantConv2d(inplane, outplane, self.wbits, **kwargs)
        else:
            self.mix_weight = MixQuantConv2d(inplane, outplane, self.wbits, **kwargs)

        stride = kwargs['stride'] if 'stride' in kwargs else 1
        if isinstance(kwargs['kernel_size'], tuple):
            kernel_size = kwargs['kernel_size'][0] * kwargs['kernel_size'][1]
        else:
            kernel_size = kwargs['kernel_size'] * kwargs['kernel_size']
        self.param_size = inplane * outplane * kernel_size * 1e-6
        self.filter_size = self.param_size / float(stride ** 2.0)
        self.register_buffer('size_product', torch.tensor(0, dtype=torch.float))
        self.register_buffer('memory_size', torch.tensor(0, dtype=torch.float))

    def forward(self, input):
        in_shape = input.shape
        tmp = torch.tensor(in_shape[1] * in_shape[2] * in_shape[3] * 1e-3, dtype=torch.float)
        self.memory_size.copy_(tmp)
        tmp = torch.tensor(self.filter_size * in_shape[-1] * in_shape[-2], dtype=torch.float)
        self.size_product.copy_(tmp)
        out = self.mix_activ(input)
        out = self.mix_weight(out)
        return out

    def complexity_loss(self):
        sw = F.softmax(self.mix_activ.alpha_activ, dim=0)
        mix_abit = 0
        abits = self.mix_activ.bits
        for i in range(len(abits)):
            mix_abit += sw[i] * abits[i]
        sw = F.softmax(self.mix_weight.alpha_weight, dim=0)
        mix_wbit = 0
        wbits = self.mix_weight.bits
        for i in range(len(wbits)):
            mix_wbit += sw[i] * wbits[i]
        complexity = self.size_product.item() * mix_abit * mix_wbit
        return complexity

    def fetch_best_arch(self, layer_idx):
        size_product = float(self.size_product.cpu().numpy())
        memory_size = float(self.memory_size.cpu().numpy())
        prob_activ = F.softmax(self.mix_activ.alpha_activ, dim=0)
        prob_activ = prob_activ.detach().cpu().numpy()
        best_activ = prob_activ.argmax()
        mix_abit = 0
        abits = self.mix_activ.bits
        for i in range(len(abits)):
            mix_abit += prob_activ[i] * abits[i]
        prob_weight = F.softmax(self.mix_weight.alpha_weight, dim=0)
        prob_weight = prob_weight.detach().cpu().numpy()
        best_weight = prob_weight.argmax()
        mix_wbit = 0
        wbits = self.mix_weight.bits
        for i in range(len(wbits)):
            mix_wbit += prob_weight[i] * wbits[i]
        if self.share_weight:
            weight_shape = list(self.mix_weight.conv.weight.shape)
        else:
            weight_shape = list(self.mix_weight.conv_list[0].weight.shape)

        best_arch = {'best_activ': [best_activ], 'best_weight': [best_weight]}
        bitops = size_product * abits[best_activ] * wbits[best_weight]
        bita = memory_size * abits[best_activ]
        bitw = self.param_size * wbits[best_weight]

        mixbitops = size_product * mix_abit * mix_wbit
        mixbita = memory_size * mix_abit
        mixbitw = self.param_size * mix_wbit
        return best_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw

    def fetch_mix_bit(self, layer_idx):

        prob_activ = F.softmax(self.mix_activ.alpha_activ, dim=0)
        prob_activ = prob_activ.detach().cpu().numpy()

        mix_abit = 0.
        abits = self.mix_activ.bits
        for i in range(len(abits)):
            mix_abit += prob_activ[i] * abits[i]
        prob_weight = F.softmax(self.mix_weight.alpha_weight, dim=0)
        prob_weight = prob_weight.detach().cpu().numpy()

        mix_wbit = 0.
        wbits = self.mix_weight.bits
        for i in range(len(wbits)):
            mix_wbit += prob_weight[i] * wbits[i]

        return mix_wbit, mix_abit


