import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.parameter import Parameter
from typing import List


########################### MLPs no bias
class MLP(nn.Module):
    def __init__(self, n_layers, n_units, n_channels, n_classes=10):
        super(MLP, self).__init__()
        mid_layers = []
        mid_layers.extend([nn.Flatten(), nn.Linear(32 * 32 * n_channels, n_units, bias=False), nn.ReLU()])
        for _ in range(n_layers-1):
            mid_layers.extend([
                nn.Linear(n_units, n_units, bias=False),
                nn.BatchNorm1d(n_units, momentum=0.9),
                nn.ReLU(),
            ])
        mid_layers.extend([nn.Linear(n_units, n_classes, bias=False)])
        self.linear_relu_stack = nn.Sequential(*mid_layers)

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


########################### MLPs with bias
class MLPB(nn.Module):
    def __init__(self, n_layers, n_units, n_channels, n_classes=10):
        super(MLPB, self).__init__()
        mid_layers = []
        mid_layers.extend([nn.Flatten(), nn.Linear(32 * 32 * n_channels, n_units), nn.ReLU()])
        for _ in range(n_layers-1):
            mid_layers.extend([
                nn.Linear(n_units, n_units),
                nn.BatchNorm1d(n_units, momentum=0.9),
                nn.ReLU(),
            ])
        mid_layers.extend([nn.Linear(n_units, n_classes)])
        self.linear_relu_stack = nn.Sequential(*mid_layers)

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


########################### SCN-MLPs no bias
class HHN_MLP(nn.Module):
    def __init__(self, hin, dimensions, n_layers, n_units, n_channels, n_classes=10):
        super(SCN_MLP, self).__init__()
        self.hyper_stack = nn.Sequential(
            nn.Linear(hin, 64),
            nn.ReLU(),
            nn.Linear(64, dimensions),
            nn.Softmax(dim=0)
        )

        self.dimensions = dimensions
        self.n_layers = n_layers
        self.n_units = n_units
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.running_mu = torch.zeros(self.n_units).to(self.device)  # zeros are fine for first training iter
        self.running_std = torch.ones(self.n_units).to(self.device)  # ones are fine for first training iter


        self.weight_list_fc1 = \
            self.create_param_combination_linear(in_features=32 * 32 * n_channels, out_features=n_units)
        self.weight_and_biases = nn.ParameterList()
        for _ in range(n_layers - 1):
            w = self.create_param_combination_linear(in_features=n_units, out_features=n_units)
            self.weight_and_biases += w

        self.weight_list_fc2 = self.create_param_combination_linear(in_features=n_units, out_features=n_classes)

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

        weight_fc1 = self.calculate_weighted_sum(self.weight_list_fc1, hyper_output)
        weight_fc2 = self.calculate_weighted_sum(self.weight_list_fc2, hyper_output)

        logits = torch.flatten(x, start_dim=1)
        logits = F.linear(logits, weight=weight_fc1, bias=None)
        logits = torch.relu(logits)

        it = iter(self.weight_and_biases)
        for w in zip(*[it] * self.dimensions):
            w = nn.ParameterList(w)
            w = self.calculate_weighted_sum(w.to(self.device), hyper_output)
            logits = F.linear(logits, weight=w, bias=None)
            logits = F.batch_norm(logits, self.running_mu, self.running_std, training=True, momentum=0.9)
            logits = torch.relu(logits)

        logits = F.linear(logits, weight=weight_fc2, bias=None)
        return logits


########################### SCN-MLPs with bias
class HHN_MLPB(nn.Module):
    def __init__(self, hin, dimensions, n_layers, n_units, n_channels, n_classes=10):
        super(SCN_MLPB, self).__init__()
        self.hyper_stack = nn.Sequential(
            nn.Linear(hin, 64),
            nn.ReLU(),
            nn.Linear(64, dimensions),
            nn.Softmax(dim=0)
        )

        self.dimensions = dimensions
        self.n_layers = n_layers
        self.n_units = n_units
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.running_mu = torch.zeros(self.n_units).to(self.device)  # zeros are fine for first training iter
        self.running_std = torch.ones(self.n_units).to(self.device)  # ones are fine for first training iter

        self.weight_list_fc1, self.bias_list_fc1 = \
            self.create_param_combination_linear(in_features=32 * 32 * n_channels, out_features=n_units)
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for _ in range(n_layers - 1):
            w, b = self.create_param_combination_linear(in_features=n_units, out_features=n_units)
            self.weights += w
            self.biases += b
        self.weight_list_fc2, self.bias_list_fc2 = self.create_param_combination_linear(in_features=n_units,
                                                                                        out_features=n_classes)

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

        weight_fc1 = self.calculate_weighted_sum(self.weight_list_fc1, hyper_output)
        weight_fc2 = self.calculate_weighted_sum(self.weight_list_fc2, hyper_output)

        bias_fc1 = self.calculate_weighted_sum(self.bias_list_fc1, hyper_output)
        bias_fc2 = self.calculate_weighted_sum(self.bias_list_fc2, hyper_output)

        logits = torch.flatten(x, start_dim=1)
        logits = F.linear(logits, weight=weight_fc1, bias=bias_fc1)
        logits = torch.relu(logits)

        it_w = iter(self.weights)
        it_b = iter(self.biases)
        for (w, b) in zip(zip(*[it_w] * self.dimensions), zip(*[it_b] * self.dimensions)):
            w = nn.ParameterList(w)
            b = nn.ParameterList(b)
            w = self.calculate_weighted_sum(w.to(self.device), hyper_output)
            b = self.calculate_weighted_sum(b.to(self.device), hyper_output)
            logits = F.linear(logits, weight=w, bias=b)
            logits = F.batch_norm(logits, self.running_mu, self.running_std, training=True, momentum=0.9)
            logits = torch.relu(logits)

        logits = F.linear(logits, weight=weight_fc2, bias=bias_fc2)
        return logits
