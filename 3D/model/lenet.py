
from torch import nn
import torch
from typing import List
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math
from typing import Union, List, Dict, Any

__all__ = ['LeNet', 'HHN_LeNet', 'lenet', 'hhn_lenet']

class LeNet(nn.Module):
    def __init__(self, num_classes=10, in_channels = 1, num_units=4, num_layers=2):
        super(LeNet, self).__init__()

        layers: List[nn.Module] = []

        if num_layers<2:
            raise NotImplementedError("There should be at least 2 Conv layers.")

        layers.extend([
            nn.Conv2d(in_channels=in_channels, out_channels=num_units, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        ])
        
        for _ in range(num_layers-2):
            layers.extend([
                nn.Conv2d(in_channels=4, out_channels=num_units, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            ])
        
        layers.extend([
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        ])

        self.conv_block=nn.Sequential(*layers)

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
        x = self.conv_block(x)
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


class HHN_LeNet(nn.Module):
    def __init__(self, dimensions = 6, hyper_hidden_units=64, num_classes=10, in_channels = 1, num_units=4, num_layers=2) -> None:
        super(HHN_LeNet,self).__init__()

        self.hyper_stack = nn.Sequential(
            nn.Linear(6, hyper_hidden_units),
            nn.ReLU(),
            nn.Linear(hyper_hidden_units, dimensions),
            nn.Softmax(dim=0)
        )


        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        if num_layers<2:
            raise NotImplementedError("There should be at least 2 Conv layers.")

        self.dimensions = dimensions
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.num_units = num_units

        self.running_mu = torch.zeros(self.num_units)#.to(self.device)  # zeros are fine for first training iter
        self.running_std = torch.ones(self.num_units)#.to(self.device)  # ones are fine for first training iter

        self.w_conv1_list, self.bias_conv1_list = self.__construct_conv2D_parameters_list(in_features=in_channels, out_features=num_units, kernel_size=5, dimensions=self.dimensions)

        self.w_conv_mid_list = nn.ParameterList()
        self.bias_conv_mid_list = nn.ParameterList()
        for _ in range(self.num_layers - 2):
            w, b = self.__construct_conv2D_parameters_list(in_features=num_units, out_features=num_units, kernel_size=3, dimensions=self.dimensions)
            self.w_conv_mid_list += w
            self.bias_conv_mid_list += b

        self.w_conv2_list, self.bias_conv2_list = self.__construct_conv2D_parameters_list(in_features=num_units, out_features=16, kernel_size=5, dimensions=self.dimensions)


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
        
    
    def __calculate_weighted_sum(self, parameter_list: List, factors: torch.Tensor):
        weighted_list = [a * b for a, b in zip(parameter_list, factors)]
        return torch.sum(torch.stack(weighted_list), dim=0)
    
    def forward(self, x, hyper_x):

        hyper_output = self.hyper_stack(hyper_x)
        

        w_conv1 = self.__calculate_weighted_sum(self.w_conv1_list, hyper_output)
        bias_conv1 = self.__calculate_weighted_sum(self.bias_conv1_list, hyper_output)
        
        w_conv2 = self.__calculate_weighted_sum(self.w_conv2_list, hyper_output)
        bias_conv2 = self.__calculate_weighted_sum(self.bias_conv2_list, hyper_output)
        
        w_linear1 = self.__calculate_weighted_sum(self.w_linear1_list, hyper_output)
        bias_linear1 = self.__calculate_weighted_sum(self.bias_linear1_list, hyper_output)

        w_linear2 = self.__calculate_weighted_sum(self.w_linear2_list, hyper_output)
        bias_linear2 = self.__calculate_weighted_sum(self.bias_linear2_list, hyper_output)

        w_linear3 = self.__calculate_weighted_sum(self.w_linear3_list, hyper_output)
        bias_linear3 = self.__calculate_weighted_sum(self.bias_linear3_list, hyper_output)

        logits = F.conv2d(x, weight=w_conv1, bias=bias_conv1,stride=1)
        logits = torch.relu(logits)
        logits = F.max_pool2d(logits, 2)


        #it_w = iter(self.w_conv_mid_list)
        #it_b = iter(self.bias_conv_mid_list)
        # for (w, b) in zip(zip(*[it_w] * self.dimensions), zip(*[it_b] * self.dimensions)):
        for i in range(self.num_layers-2):
            w_conv_list = self.w_conv_mid_list[i*self.dimensions : (i+1)*self.dimensions] #nn.ParameterList(w).to(w[0].device)
            bias_conv_list = self.bias_conv_mid_list[i*self.dimensions : (i+1)*self.dimensions] #nn.ParameterList(b).to(b[0].device)

            w = self.__calculate_weighted_sum(w_conv_list, hyper_output)
            b = self.__calculate_weighted_sum(bias_conv_list, hyper_output)

            logits = F.conv2d(logits, weight=w, bias=b, stride=1, padding=1)

            logits = F.batch_norm(logits, self.running_mu.to(w[0].device), self.running_std.to(w[0].device), training=True, momentum=0.9)
            logits = torch.relu(logits)



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


def lenet(**kwargs: Any) -> LeNet:
    model = LeNet(**kwargs)
    return model

def hhn_lenet(**kwargs: Any) -> HHN_LeNet:
    model = HHN_LeNet(**kwargs)
    return model
