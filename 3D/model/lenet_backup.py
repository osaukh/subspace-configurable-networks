
from torch import nn
import torch
from typing import List
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math
from typing import Any


__all__ = ['One4All', 'one4all', 'HyperNetwork', 'hypernetwork']

class One4All(nn.Module):
    def __init__(self, num_classes=10):
        super(One4All, self).__init__()
        self.con_block = nn.Sequential(
            # 1x32x32
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1),
            # 4x28x28
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 4x14x14
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=5, stride=1),
            # 16x10x10
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 16x5x5

        )

        self.fc = nn.Sequential(
            # nn.Flatten(1),
            nn.Linear(in_features=16 * 5 * 5, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_classes),
        )

        self._init_weight()

    def forward(self, x):
        x = self.con_block(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


class HyperNetwork(nn.Module):
    def __init__(self, dimensions = 6, hidden_units=64, num_classes=10) -> None:
        super(HyperNetwork,self).__init__()
        self.hyper_stack = nn.Sequential(
            nn.Linear(6, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, dimensions),
            nn.Softmax(dim=0)
        )


        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        self.dimensions = dimensions
        # self.num_classes = num_classes

        self.w_conv1_list, self.bias_conv1_list = self.__construct_conv2D_parameters_list(in_features=1, out_features=4, kernel_size=5, dimensions=self.dimensions)
        self.w_conv2_list, self.bias_conv2_list = self.__construct_conv2D_parameters_list(in_features=4, out_features=16, kernel_size=5, dimensions=self.dimensions)


        self.w_linear1_list, self.bias_linear1_list = self.__construct_linear_parameters_list(in_features=16 * 5 * 5, out_features=256,  dimensions=self.dimensions)
        self.w_linear2_list, self.bias_linear2_list = self.__construct_linear_parameters_list(in_features=256, out_features=128,  dimensions=self.dimensions)
        self.w_linear3_list, self.bias_linear3_list = self.__construct_linear_parameters_list(in_features=128, out_features=num_classes,  dimensions=self.dimensions)
        

    def __construct_conv2D_parameters_list(self, in_features, out_features, kernel_size, dimensions):
        weight_list = nn.ParameterList()
        bias_list = nn.ParameterList()
        for _ in range(dimensions):
            weight = Parameter(torch.empty((out_features, in_features, kernel_size, kernel_size)))
            nn.init.kaiming_normal_(weight, mode='fan_in', nonlinearity='relu')
            weight_list.append(weight)

            bias = Parameter(torch.empty(out_features))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(bias, -bound, bound)
#             nn.init.zeros_(bias)
            bias_list.append(bias)
        return weight_list, bias_list

    def __construct_linear_parameters_list(self, in_features, out_features, dimensions):
        weight_list = nn.ParameterList()
        bias_list = nn.ParameterList()
        for _ in range(dimensions):
            weight = Parameter(torch.empty((out_features, in_features)))
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            weight_list.append(weight)

            bias = Parameter(torch.empty(out_features))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(bias, -bound, bound)
#             nn.init.zeros_(bias)
            bias_list.append(bias)
        return weight_list, bias_list
        
    
    def __calculate_weighteed_sum(self, parameter_list: List, factors: torch.Tensor):
        weighted_list = [a * b for a, b in zip(parameter_list, factors)]
        return torch.sum(torch.stack(weighted_list), dim=0)
    
    def forward(self, x, hyper_x):

        hyper_output = self.hyper_stack(hyper_x)
        

        w_conv1 = self.__calculate_weighteed_sum(self.w_conv1_list, hyper_output)
        bias_conv1 = self.__calculate_weighteed_sum(self.bias_conv1_list, hyper_output)
        
        w_conv2 = self.__calculate_weighteed_sum(self.w_conv2_list, hyper_output)
        bias_conv2 = self.__calculate_weighteed_sum(self.bias_conv2_list, hyper_output)
        
        w_linear1 = self.__calculate_weighteed_sum(self.w_linear1_list, hyper_output)
        bias_linear1 = self.__calculate_weighteed_sum(self.bias_linear1_list, hyper_output)

        w_linear2 = self.__calculate_weighteed_sum(self.w_linear2_list, hyper_output)
        bias_linear2 = self.__calculate_weighteed_sum(self.bias_linear2_list, hyper_output)

        w_linear3 = self.__calculate_weighteed_sum(self.w_linear3_list, hyper_output)
        bias_linear3 = self.__calculate_weighteed_sum(self.bias_linear3_list, hyper_output)

        logits = F.conv2d(x, weight=w_conv1, bias=bias_conv1,stride=1)
        logits = torch.relu(logits)
        logits = F.max_pool2d(logits, 2)

        logits = F.conv2d(logits, weight=w_conv2, bias=bias_conv2,stride=1)
        logits = torch.relu(logits)
        logits = F.max_pool2d(logits, 2)

        logits = torch.flatten(logits, 1)

        logits = F.linear(logits, weight=w_linear1, bias=bias_linear1)
        logits = torch.relu(logits)
        logits = F.linear(logits, weight=w_linear2, bias=bias_linear2)
        logits = torch.relu(logits)
        logits = F.linear(logits, weight=w_linear3, bias=bias_linear3)

        return logits


def one4all(**kwargs: Any) -> One4All:
    model = One4All(**kwargs)
    return model

def hypernetwork(**kwargs: Any) -> HyperNetwork:
    model = HyperNetwork(**kwargs)
    return model