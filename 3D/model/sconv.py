'''
    Source: Learning Convolutions from Scratch: https://arxiv.org/pdf/2007.13657.pdf
'''

import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.parameter import Parameter
from typing import List
from typing import Any

__all__ = ['SConv', 'SConvB', 'HHN_SConv', 'HHN_SConvB', 'sconv', 'sconvb', 'hhn_sconv', 'hhn_sconvb']


######## SConv no bias
class SConv(nn.Module):
    def __init__(self, n_layers, n_units, n_channels, num_classes=10):
        super(SConv, self).__init__()

        mid_layers = []
        mid_layers.extend([
            nn.Conv2d(n_channels, n_units, kernel_size=9, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True)
            ])

        for _ in range(n_layers-2):
            mid_layers.extend([
                nn.Conv2d(n_units, n_units, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(n_units, momentum=0.9),
                nn.ReLU(inplace=True),
            ])

        mid_layers.extend([
            nn.Conv2d(n_units, n_units, kernel_size=13, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(n_units, num_classes, bias=False)
        ])

        self.linear_stack = nn.Sequential(*mid_layers)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()

    def forward(self, x):
        logits = self.linear_stack(x)
        return logits


######## SConv with bias
class SConvB(nn.Module):
    def __init__(self, n_layers, n_units, n_channels, num_classes=10):
        super(SConvB, self).__init__()

        mid_layers = []
        mid_layers.extend([
            nn.Conv2d(n_channels, n_units, kernel_size=9, stride=2, padding=1),
            nn.ReLU(inplace=True)
            ])

        for _ in range(n_layers-2):
            mid_layers.extend([
                nn.Conv2d(n_units, n_units, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(n_units, momentum=0.9),
                nn.ReLU(inplace=True),
            ])

        mid_layers.extend([
            nn.Conv2d(n_units, n_units, kernel_size=13, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(n_units, num_classes)
        ])

        self.linear_stack = nn.Sequential(*mid_layers)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()

    def forward(self, x):
        logits = self.linear_stack(x)
        return logits


######## HHN-SConv no bias
class HHN_SConv(nn.Module):
    def __init__(self, dimensions, n_layers, n_units, n_channels, num_classes=10):
        super(HHN_SConv, self).__init__()
        self.hyper_stack = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, dimensions),
            nn.Softmax(dim=0)
        )

        self.dimensions = dimensions
        self.n_layers = n_layers
        self.n_units = n_units
        self.n_channels = n_channels
        self.num_classes = num_classes

        # self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.weight_list_conv1 = self.create_param_combination_conv(in_channels=n_channels,
                                                                    out_channels=n_units, kernel=9)

        self.running_mu = torch.zeros(self.n_units)#.to(self.device)  # zeros are fine for first training iter
        self.running_std = torch.ones(self.n_units)#.to(self.device)  # ones are fine for first training iter

        self.weight_and_biases = nn.ParameterList()
        for _ in range(n_layers - 2):
            w = self.create_param_combination_conv(in_channels=n_units,
                                                   out_channels=n_units, kernel=3)
            self.weight_and_biases += w

        self.weight_list_conv2 = self.create_param_combination_conv(in_channels=n_units,
                                                                    out_channels=n_units, kernel=13)
        self.weight_list_fc3 = self.create_param_combination_linear(in_features=n_units,
                                                                    out_features=num_classes)

    def create_param_combination_conv(self, in_channels, out_channels, kernel):
        weight_list = nn.ParameterList()
        for _ in range(self.dimensions):
            weight = Parameter(torch.empty((out_channels, in_channels, kernel, kernel)))
            init.kaiming_uniform_(weight, a=math.sqrt(5))
            weight_list.append(weight)
        return weight_list

    def create_param_combination_linear(self, in_features, out_features):
        weight_list = nn.ParameterList()
        for _ in range(self.dimensions):
            weight = Parameter(torch.empty((out_features, in_features)))
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            weight_list.append(weight)
        return weight_list

    def calculate_weighted_sum(self, param_list: List, factors: Tensor):
        weighted_list = [a * b for a, b in zip(param_list, factors)]
        return torch.sum(torch.stack(weighted_list), dim=0)

    def forward(self, x, hyper_x):
        hyper_output = self.hyper_stack(hyper_x)

        w_conv1 = self.calculate_weighted_sum(self.weight_list_conv1, hyper_output)
        w_conv2 = self.calculate_weighted_sum(self.weight_list_conv2, hyper_output)
        w_fc3 = self.calculate_weighted_sum(self.weight_list_fc3, hyper_output)

        logits = F.conv2d(x, weight=w_conv1, stride=2, padding=1, bias=None)
        logits = torch.relu(logits)

        it = iter(self.weight_and_biases)
        for w in zip(*[it] * self.dimensions):
            w = nn.ParameterList(w).to(w[0].device)
            # w = self.calculate_weighted_sum(w.to(self.device), hyper_output)
            w = self.calculate_weighted_sum(w, hyper_output)
            logits = F.conv2d(logits, weight=w, stride=1, padding=1, bias=None)
            logits = F.batch_norm(logits, self.running_mu.to(w[0].device), self.running_std.to(w[0].device), training=True, momentum=0.9)
            logits = torch.relu(logits)

        logits = F.conv2d(logits, weight=w_conv2, stride=1, padding=0, bias=None)
        logits = torch.relu(logits)
        logits = torch.flatten(logits, start_dim=1)
        logits = F.linear(logits, weight=w_fc3, bias=None)
        return logits


######## HHN-SConv with bias
class HHN_SConvB(nn.Module):
    def __init__(self, dimensions, n_layers, n_units, n_channels, num_classes=10):
        super(HHN_SConvB, self).__init__()
        self.hyper_stack = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, dimensions),
            nn.Softmax(dim=0)
        )

        self.dimensions = dimensions
        self.n_layers = n_layers
        self.n_units = n_units
        self.n_channels = n_channels
        self.num_classes = num_classes

        # self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.running_mu = torch.zeros(self.n_units)#.to(self.device)  # zeros are fine for first training iter
        self.running_std = torch.ones(self.n_units)#.to(self.device)  # ones are fine for first training iter

        self.weight_list_conv1, self.bias_list_conv1 = \
            self.create_param_combination_conv(in_channels=n_channels,
                                               out_channels=n_units, kernel=9)

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for _ in range(n_layers - 2):
            w, b = self.create_param_combination_conv(in_channels=n_units,
                                                      out_channels=n_units, kernel=3)
            self.weights += w
            self.biases += b

        self.weight_list_conv2, self.bias_list_conv2 = \
            self.create_param_combination_conv(in_channels=n_units,
                                               out_channels=n_units, kernel=13)
        self.weight_list_fc3, self.bias_list_fc3 = \
            self.create_param_combination_linear(in_features=n_units, out_features=num_classes)

    def create_param_combination_conv(self, in_channels, out_channels, kernel):
        weight_list = nn.ParameterList()
        bias_list = nn.ParameterList()
        for _ in range(self.dimensions):
            weight = Parameter(torch.empty((out_channels, in_channels, kernel, kernel)))
            init.kaiming_uniform_(weight, a=math.sqrt(5))
            weight_list.append(weight)

            bias = Parameter(torch.empty(out_channels))
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(bias, -bound, bound)
            bias_list.append(bias)
        return weight_list, bias_list

    def create_param_combination_linear(self, in_features, out_features):
        weight_list = nn.ParameterList()
        bias_list = nn.ParameterList()
        for _ in range(self.dimensions):
            weight = Parameter(torch.empty((out_features, in_features)))
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            weight_list.append(weight)

            bias = Parameter(torch.empty(out_features))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(bias, -bound, bound)
            bias_list.append(bias)
        return weight_list, bias_list

    def calculate_weighted_sum(self, param_list: List, factors: Tensor):
        weighted_list = [a * b for a, b in zip(param_list, factors)]
        return torch.sum(torch.stack(weighted_list), dim=0)

    def forward(self, x, hyper_x):
        hyper_output = self.hyper_stack(hyper_x)

        w_conv1 = self.calculate_weighted_sum(self.weight_list_conv1, hyper_output)
        w_conv2 = self.calculate_weighted_sum(self.weight_list_conv2, hyper_output)
        w_fc3 = self.calculate_weighted_sum(self.weight_list_fc3, hyper_output)

        b_conv1 = self.calculate_weighted_sum(self.bias_list_conv1, hyper_output)
        b_conv2 = self.calculate_weighted_sum(self.bias_list_conv2, hyper_output)
        b_fc3 = self.calculate_weighted_sum(self.bias_list_fc3, hyper_output)

        logits = F.conv2d(x, weight=w_conv1, bias=b_conv1, stride=2, padding=1)
        logits = torch.relu(logits)

        it_w = iter(self.weights)
        it_b = iter(self.biases)
        for (w, b) in zip(zip(*[it_w] * self.dimensions), zip(*[it_b] * self.dimensions)):
            w = nn.ParameterList(w).to(w[0].device)
            b = nn.ParameterList(b).to(b[0].device)
            # w = self.calculate_weighted_sum(w.to(self.device), hyper_output)
            # b = self.calculate_weighted_sum(b.to(self.device), hyper_output)
            w = self.calculate_weighted_sum(w, hyper_output)
            b = self.calculate_weighted_sum(b, hyper_output)
            logits = F.conv2d(logits, weight=w, bias=b, stride=1, padding=1)
            logits = F.batch_norm(logits, self.running_mu.to(w[0].device), self.running_std.to(w[0].device), training=True, momentum=0.9)
            logits = torch.relu(logits)

        logits = F.conv2d(logits, weight=w_conv2, bias=b_conv2, stride=1, padding=0)
        logits = torch.relu(logits)
        logits = torch.flatten(logits, start_dim=1)
        logits = F.linear(logits, weight=w_fc3, bias=b_fc3)
        return logits


def sconv(**kwargs: Any) -> SConv:
    model = SConv(**kwargs)
    return model

def sconvb(**kwargs: Any) -> SConvB:
    model = SConvB(**kwargs)
    return model

def hhn_sconv(**kwargs: Any) -> HHN_SConv:
    model = HHN_SConv(**kwargs)
    return model

def hhn_sconvb(**kwargs: Any) -> HHN_SConvB:
    model = HHN_SConvB(**kwargs)
    return model