####### no bias
'''
    Two SCN architectures: Conv+Dense and Conv+ShiftInvariant
'''
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.parameter import Parameter
from typing import List


######## SCN Conv+Dense
class HHNConvDense(nn.Module):
    def __init__(self, dimensions, n_layers, n_units, n_channels, n_classes=10):
        super(HHNConvDense, self).__init__()
        self.hyper_stack = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, dimensions),
            nn.Softmax(dim=0)
        )

        self.dimensions = dimensions
        self.n_layers = n_layers
        self.n_units = n_units
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.weight_list_conv1, self.bias_list_conv1 = \
            self.create_param_combination_conv(in_channels=1, out_channels=16, kernel=4)
        self.weight_list_conv2, self.bias_list_conv2 = \
            self.create_param_combination_conv(in_channels=16, out_channels=8, kernel=4)
        self.weight_list_fc1, self.bias_list_fc1 = \
            self.create_param_combination_linear(in_features=128, out_features=32)
        self.weight_list_fc2, self.bias_list_fc2 = \
            self.create_param_combination_linear(in_features=32, out_features=10)

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
        b_conv1 = self.calculate_weighted_sum(self.bias_list_conv1, hyper_output)

        w_conv2 = self.calculate_weighted_sum(self.weight_list_conv2, hyper_output)
        b_conv2 = self.calculate_weighted_sum(self.bias_list_conv2, hyper_output)

        w_fc1 = self.calculate_weighted_sum(self.weight_list_fc1, hyper_output)
        b_fc1 = self.calculate_weighted_sum(self.bias_list_fc1, hyper_output)

        w_fc2 = self.calculate_weighted_sum(self.weight_list_fc2, hyper_output)
        b_fc2 = self.calculate_weighted_sum(self.bias_list_fc2, hyper_output)

        logits = F.conv2d(x, weight=w_conv1, bias=b_conv1)
        logits = F.max_pool2d(logits, kernel_size=4)

        logits = F.conv2d(logits, weight=w_conv2, bias=b_conv2)
        # logits = F.max_pool2d(logits, kernel_size=4)

        logits = torch.flatten(logits, start_dim=1)
        logits = F.linear(logits, weight=w_fc1, bias=b_fc1)
        logits = torch.relu(logits)
        logits = F.linear(logits, weight=w_fc2, bias=b_fc2)
        return logits



######## SCN Conv+ShiftInvariant
class HHNShiftInv(nn.Module):
    def __init__(self, dimensions, n_layers, n_units, n_channels, n_classes=10):
        super(HHNShiftInv, self).__init__()
        self.hyper_stack = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, dimensions),
            nn.Softmax(dim=0)
        )

        self.dimensions = dimensions
        self.n_layers = n_layers
        self.n_units = n_units
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.weight_list_conv1, self.bias_list_conv1 = \
            self.create_param_combination_conv(in_channels=1, out_channels=16, kernel=4)
        self.weight_list_conv2, self.bias_list_conv2 = \
            self.create_param_combination_conv(in_channels=16, out_channels=10, kernel=4)

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

    def calculate_weighted_sum(self, param_list: List, factors: Tensor):
        weighted_list = [a * b for a, b in zip(param_list, factors)]
        return torch.sum(torch.stack(weighted_list), dim=0)

    def forward(self, x, hyper_x):
        hyper_output = self.hyper_stack(hyper_x)

        w_conv1 = self.calculate_weighted_sum(self.weight_list_conv1, hyper_output)
        b_conv1 = self.calculate_weighted_sum(self.bias_list_conv1, hyper_output)

        w_conv2 = self.calculate_weighted_sum(self.weight_list_conv2, hyper_output)
        b_conv2 = self.calculate_weighted_sum(self.bias_list_conv2, hyper_output)

        logits = F.conv2d(x, weight=w_conv1, bias=b_conv1)
        logits = F.max_pool2d(logits, kernel_size=4)

        logits = F.conv2d(logits, weight=w_conv2, bias=b_conv2)
        logits = F.max_pool2d(logits, kernel_size=4)

        logits = torch.flatten(logits, start_dim=1)
        return logits
