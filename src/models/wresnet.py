import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import init_param, make_batchnorm, loss_fn, kld_loss
from config import cfg


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equal_inout = (in_planes == out_planes)
        self.shortcut = (not self.equal_inout) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                             padding=0, bias=False) or None

    def forward(self, x):
        if not self.equal_inout:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equal_inout else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        out = torch.add(x if self.equal_inout else self.shortcut(x), out)
        return out


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate):
        super().__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer(x)
        return out


class WideResNet(nn.Module):
    def __init__(self, data_shape, num_classes, depth, widen_factor, drop_rate):
        super().__init__()
        num_down = int(min(math.log2(data_shape[1]), math.log2(data_shape[2]))) - 3
        hidden_size = [16]
        for i in range(num_down + 1):
            hidden_size.append(16 * (2 ** i) * widen_factor)
        n = ((depth - 1) / (num_down + 1) - 1) / 2
        block = BasicBlock
        blocks = []
        blocks.append(nn.Conv2d(data_shape[0], hidden_size[0], kernel_size=3, stride=1, padding=1, bias=False))
        blocks.append(NetworkBlock(n, hidden_size[0], hidden_size[1], block, 1, drop_rate))
        for i in range(num_down):
            blocks.append(NetworkBlock(n, hidden_size[i + 1], hidden_size[i + 2], block, 2, drop_rate))
        blocks.append(nn.BatchNorm2d(hidden_size[-1]))
        blocks.append(nn.ReLU(inplace=True))
        blocks.append(nn.AdaptiveAvgPool2d(1))
        blocks.append(nn.Flatten())
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Linear(hidden_size[-1], num_classes)
        self.num_classes = num_classes

    def f(self, x):
        x = self.blocks(x)
        x = self.classifier(x)
        return x
        
    def entropy(self,p):
        return -torch.sum(p * torch.log(p + 1e-10), dim=1) 
        
    def forward(self, input):
        output = {}
        output['target'] = self.f(input['data'])
        if 'loss_mode' in input:
            if 'sup' in input['loss_mode']:
                output['loss'] = loss_fn(output['target'], input['target'])
            elif 'fix' in input['loss_mode'] and 'mix' not in input['loss_mode']:
                aug_output = self.f(input['aug'])
                output['loss'] = loss_fn(aug_output, input['target'].detach())
            elif 'fix' in input['loss_mode'] and 'mix' in input['loss_mode']:
                aug_output = self.f(input['aug'])
                output['loss'] = loss_fn(aug_output, input['target'].detach())
                if 'ent' in input['loss_mode']:
                    entropy_weight = 0.5
                    prob = F.softmax(aug_output, dim=1)
                    entropy_loss = self.entropy(prob).mean()
                    output['loss'] -= (entropy_weight * entropy_loss)
                mix_output = self.f(input['mix_data'])
                if 'KLDmix' in input['loss_mode']:
                    mix_target1 = F.one_hot(input['mix_target'][:, 0].detach(), num_classes=self.num_classes).float()
                    mix_target2 = F.one_hot(input['mix_target'][:, 1].detach(), num_classes=self.num_classes).float()
                    mix_target = input['lam']*mix_target1 + (1 - input['lam']) *mix_target2
                    _kld_loss = kld_loss(mix_output, mix_target)
                    output['loss'] +=  _kld_loss
                else:
                    target1 = input['mix_target'][:, 0].detach()
                    target2 = input['mix_target'][:, 1].detach()
                    output['loss'] += input['lam'] * loss_fn(mix_output, target1) \
                                   + (1 - input['lam']) * loss_fn(mix_output, target2)
            else:
                raise ValueError('Not valid loss mode')
        else:
            if not torch.any(input['target'] == -1):
                output['loss'] = loss_fn(output['target'], input['target'])
        return output

def wresnet28x2(momentum=None, track=False):
    data_shape = cfg['data_shape']
    target_size = cfg['target_size']
    depth = cfg['wresnet28x2']['depth']
    widen_factor = cfg['wresnet28x2']['widen_factor']
    drop_rate = cfg['wresnet28x2']['drop_rate']
    model = WideResNet(data_shape, target_size, depth, widen_factor, drop_rate)
    model.apply(init_param)
    model.apply(lambda m: make_batchnorm(m, momentum=momentum, track_running_stats=track))
    return model


def wresnet28x8(momentum=None, track=False):
    data_shape = cfg['data_shape']
    target_size = cfg['target_size']
    depth = cfg['wresnet28x8']['depth']
    widen_factor = cfg['wresnet28x8']['widen_factor']
    drop_rate = cfg['wresnet28x8']['drop_rate']
    model = WideResNet(data_shape, target_size, depth, widen_factor, drop_rate)
    model.apply(init_param)
    model.apply(lambda m: make_batchnorm(m, momentum=momentum, track_running_stats=track))
    return model


def wresnet37x2(momentum=None, track=False):
    data_shape = cfg['data_shape']
    target_size = cfg['target_size']
    depth = cfg['wresnet37x2']['depth']
    widen_factor = cfg['wresnet37x2']['widen_factor']
    drop_rate = cfg['wresnet37x2']['drop_rate']
    model = WideResNet(data_shape, target_size, depth, widen_factor, drop_rate)
    model.apply(init_param)
    model.apply(lambda m: make_batchnorm(m, momentum=momentum, track_running_stats=track))
    return model