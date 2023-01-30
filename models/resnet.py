import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union
from torch import nn, Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import init

'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])



###############

class HHN_ResNet18(nn.Module):
    def __init__(self, D, hin, num_classes=10):
        super(HHN_ResNet18, self).__init__()

        self.D = D
        self.num_classes = num_classes

        # Declare the hyper part
        self.hyper_stack = nn.Sequential(
            nn.Linear(hin, 64),
            # Change here the hyper-x (hyper input) dimensions. For example, in 2D rotation, the hyper input is cos(alpha), sin(alpha)
            nn.ReLU(),
            nn.Linear(64, self.D),
            nn.Softmax(dim=0)
        )

        # Declare layers and/or create weights

        # Pre-Conv
        self.conv1_weight_list = self.create_param_combination_conv2d(3, 64, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual blocks with indices as "layer-block-idx"

        # Layer 1
        self.conv1_1_1_weight_list = self.create_param_combination_conv2d(64, 64, kernel_size=3)
        self.bn1_1_1 = nn.BatchNorm2d(64)
        self.conv1_1_2_weight_list = self.create_param_combination_conv2d(64, 64, kernel_size=3)
        self.bn1_1_2 = nn.BatchNorm2d(64)

        self.conv1_2_1_weight_list = self.create_param_combination_conv2d(64, 64, kernel_size=3)
        self.bn1_2_1 = nn.BatchNorm2d(64)
        self.conv1_2_2_weight_list = self.create_param_combination_conv2d(64, 64, kernel_size=3)
        self.bn1_2_2 = nn.BatchNorm2d(64)

        # Layer 2
        self.conv2_1_1_weight_list = self.create_param_combination_conv2d(64, 128, kernel_size=3)
        self.bn2_1_1 = nn.BatchNorm2d(128)
        self.conv2_1_2_weight_list = self.create_param_combination_conv2d(128, 128, kernel_size=3)
        self.bn2_1_2 = nn.BatchNorm2d(128)
        self.shortcut2_1_weight_list = self.create_param_combination_conv2d(64, 128, kernel_size=1)
        self.shortcut2_2 = nn.BatchNorm2d(128)

        self.conv2_2_1_weight_list = self.create_param_combination_conv2d(128, 128, kernel_size=3)
        self.bn2_2_1 = nn.BatchNorm2d(128)
        self.conv2_2_2_weight_list = self.create_param_combination_conv2d(128, 128, kernel_size=3)
        self.bn2_2_2 = nn.BatchNorm2d(128)

        # Layer 3
        self.conv3_1_1_weight_list = self.create_param_combination_conv2d(128, 256, kernel_size=3)
        self.bn3_1_1 = nn.BatchNorm2d(256)
        self.conv3_1_2_weight_list = self.create_param_combination_conv2d(256, 256, kernel_size=3)
        self.bn3_1_2 = nn.BatchNorm2d(256)
        self.shortcut3_1_weight_list = self.create_param_combination_conv2d(128, 256, kernel_size=1)
        self.shortcut3_2 = nn.BatchNorm2d(256)

        self.conv3_2_1_weight_list = self.create_param_combination_conv2d(256, 256, kernel_size=3)
        self.bn3_2_1 = nn.BatchNorm2d(256)
        self.conv3_2_2_weight_list = self.create_param_combination_conv2d(256, 256, kernel_size=3)
        self.bn3_2_2 = nn.BatchNorm2d(256)

        # Layer 4
        self.conv4_1_1_weight_list = self.create_param_combination_conv2d(256, 512, kernel_size=3)
        self.bn4_1_1 = nn.BatchNorm2d(512)
        self.conv4_1_2_weight_list = self.create_param_combination_conv2d(512, 512, kernel_size=3)
        self.bn4_1_2 = nn.BatchNorm2d(512)
        self.shortcut4_1_weight_list = self.create_param_combination_conv2d(256, 512, kernel_size=1)
        self.shortcut4_2 = nn.BatchNorm2d(512)

        self.conv4_2_1_weight_list = self.create_param_combination_conv2d(512, 512, kernel_size=3)
        self.bn4_2_1 = nn.BatchNorm2d(512)
        self.conv4_2_2_weight_list = self.create_param_combination_conv2d(512, 512, kernel_size=3)
        self.bn4_2_2 = nn.BatchNorm2d(512)

        # Output Layer
        self.linear_weight_list, self.linear_bias_list = self.create_param_combination_linear(512, 10)

        # Create a parameter list for calculating the cosine similarity which is then used during training to force the vectors to be orthogonal to each other
        self.param_list = [self.conv1_weight_list,

                           self.conv1_1_1_weight_list, self.conv1_1_2_weight_list,
                           self.conv1_2_1_weight_list, self.conv1_2_2_weight_list,

                           self.conv2_1_1_weight_list, self.conv2_1_2_weight_list, self.shortcut2_1_weight_list,
                           self.conv2_2_1_weight_list, self.conv2_2_2_weight_list,

                           self.conv3_1_1_weight_list, self.conv3_1_2_weight_list, self.shortcut3_1_weight_list,
                           self.conv3_2_1_weight_list, self.conv3_2_2_weight_list,

                           self.conv4_1_1_weight_list, self.conv4_1_2_weight_list, self.shortcut4_1_weight_list,
                           self.conv4_2_1_weight_list, self.conv4_2_2_weight_list,

                           self.linear_weight_list, self.linear_bias_list
                           ]

    def create_param_combination_conv2d(self, in_channels, out_channels, kernel_size=3):
        """
        This function is used to create weight tensor list for a single conv2d layer without biases.
        The weight tensors are meant to be used for calculate the final weight of the layer via linear combination
        """

        weight_list = nn.ParameterList()
        bias_list = nn.ParameterList()
        for _ in range(self.D):
            weight = Parameter(torch.empty((out_channels, in_channels, kernel_size, kernel_size)))
            init.kaiming_uniform_(weight, a=math.sqrt(5))
            weight_list.append(weight)

            # bias = Parameter(torch.empty(out_channels))
            # fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            # bound = 1 / math.sqrt(fan_in)
            # init.uniform_(bias, -bound, bound)
            # bias_list.append(bias)

        return weight_list

    def create_param_combination_linear(self, in_features, out_features):
        """
        This function is used to create weight tensor list for a single linear layer with biases.
        The weight tensors are meant to be used for calculate the final weight of the layer via linear combination
        """

        weight_list = nn.ParameterList()
        bias_list = nn.ParameterList()
        for _ in range(self.D):
            weight = Parameter(torch.empty((out_features, in_features)))
            init.kaiming_uniform_(weight, a=math.sqrt(5))
            weight_list.append(weight)

            bias = Parameter(torch.empty(out_features))
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(bias, -bound, bound)
            bias_list.append(bias)

        return weight_list, bias_list

    def calculate_weighted_sum(self, param_list: List, coefficients: Tensor):
        """
        Calculate the weighted sum (linear combination) which is the final weight used during inference
        """
        weighted_list = [a * b for a, b in zip(param_list, coefficients)]
        return torch.sum(torch.stack(weighted_list), dim=0)

    def execute_hyper_conv2d(self, x, weight_list: List, stride, padding=0):
        """
        Execute one hyper-conv2d layer
        """
        weights = self.calculate_weighted_sum(weight_list, self.coefficients)

        return F.conv2d(x, weight=weights, stride=stride, padding=padding)

    def execute_hyper_linear(self, x, weight_list: List, bias_list: List):
        """
        Execute one hyper-linear layer
        """
        weights = self.calculate_weighted_sum(weight_list, self.coefficients)
        biases = self.calculate_weighted_sum(bias_list, self.coefficients)

        return F.linear(x, weight=weights, bias=biases)

    def forward(self, x, hyper_x):
        """
        Feedforward of the SCN-ResNet
        x: inference-input
        hyper_x: hyper-input
        """

        self.coefficients = self.hyper_stack(hyper_x)  # Calculate the betas (hyper output)

        # Pre-Conv
        x = self.execute_hyper_conv2d(x, self.conv1_weight_list, stride=1, padding=1)
        x = self.bn1(x)
        x = F.relu(x)

        # Layer 1
        res = self.execute_hyper_conv2d(x, self.conv1_1_1_weight_list, stride=1, padding=1)
        res = self.bn1_1_1(res)
        res = F.relu(res)
        res = self.execute_hyper_conv2d(res, self.conv1_1_2_weight_list, stride=1, padding=1)
        res = self.bn1_1_2(res)
        x = res + x
        x = F.relu(res)

        res = self.execute_hyper_conv2d(x, self.conv1_2_1_weight_list, stride=1, padding=1)
        res = self.bn1_2_1(res)
        res = F.relu(res)
        res = self.execute_hyper_conv2d(res, self.conv1_2_2_weight_list, stride=1, padding=1)
        res = self.bn1_2_2(res)
        x = res + x
        x = F.relu(x)

        # Layer 2
        res = self.execute_hyper_conv2d(x, self.conv2_1_1_weight_list, stride=2, padding=1)
        res = self.bn2_1_1(res)
        res = F.relu(res)
        res = self.execute_hyper_conv2d(res, self.conv2_1_2_weight_list, stride=1, padding=1)
        res = self.bn2_1_2(res)
        x = self.execute_hyper_conv2d(x, self.shortcut2_1_weight_list, stride=2)
        x = self.shortcut2_2(x)
        x = res + x
        x = F.relu(x)

        res = self.execute_hyper_conv2d(x, self.conv2_2_1_weight_list, stride=1, padding=1)
        res = self.bn2_2_1(res)
        res = F.relu(res)
        res = self.execute_hyper_conv2d(res, self.conv2_2_2_weight_list, stride=1, padding=1)
        res = self.bn2_2_2(res)
        x = res + x
        x = F.relu(x)

        # Layer 3
        res = self.execute_hyper_conv2d(x, self.conv3_1_1_weight_list, stride=2, padding=1)
        res = self.bn3_1_1(res)
        res = F.relu(res)
        res = self.execute_hyper_conv2d(res, self.conv3_1_2_weight_list, stride=1, padding=1)
        res = self.bn3_1_2(res)
        x = self.execute_hyper_conv2d(x, self.shortcut3_1_weight_list, stride=2)
        x = self.shortcut3_2(x)
        x = res + x
        x = F.relu(x)

        res = self.execute_hyper_conv2d(x, self.conv3_2_1_weight_list, stride=1, padding=1)
        res = self.bn3_2_1(res)
        res = F.relu(res)
        res = self.execute_hyper_conv2d(res, self.conv3_2_2_weight_list, stride=1, padding=1)
        res = self.bn3_2_2(res)
        x = res + x
        x = F.relu(x)

        # Layer 4
        res = self.execute_hyper_conv2d(x, self.conv4_1_1_weight_list, stride=2, padding=1)
        res = self.bn4_1_1(res)
        res = F.relu(res)
        res = self.execute_hyper_conv2d(res, self.conv4_1_2_weight_list, stride=1, padding=1)
        res = self.bn4_1_2(res)
        x = self.execute_hyper_conv2d(x, self.shortcut4_1_weight_list, stride=2)
        x = self.shortcut4_2(x)
        x = res + x
        x = F.relu(x)

        res = self.execute_hyper_conv2d(x, self.conv4_2_1_weight_list, stride=1, padding=1)
        res = self.bn4_2_1(res)
        res = F.relu(res)
        res = self.execute_hyper_conv2d(res, self.conv4_2_2_weight_list, stride=1, padding=1)
        res = self.bn4_2_2(res)
        x = res + x
        x = F.relu(x)

        # Output Layer
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)

        x = self.execute_hyper_linear(x, self.linear_weight_list, self.linear_bias_list)

        return x



