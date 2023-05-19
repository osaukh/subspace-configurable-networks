import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.parameter import Parameter
from typing import List
from typing import Any, Optional
from collections import OrderedDict
from models.scn import SCN

__all__ = ['scn_m5', 'SCN_M5', 'm5', 'M5']


# ----------------------------- SCNM5 with bias ------------------------#
class SCN_M5(SCN):
    def __init__(self, num_alpha: int, dimensions: int, device, n_layers=5, n_units=32, n_channels=1,
                 num_classes=35) -> None:
        base_model = M5(n_units=n_units, n_channels=n_channels, num_classes=num_classes)  # todo: support n_layers
        super(SCN_M5, self).__init__(num_alpha, dimensions, base_model, device)
        # self.n_layers = n_layers
        self.n_units = n_units
        self.n_channels = n_channels
        self.num_classes = num_classes

        # todo: clean the following dirty code!
        # self.register_buffer("running_mu1", torch.zeros(self.n_units))
        # self.register_buffer("running_std1", torch.ones(self.n_units))

        # self.register_buffer("running_mu2", torch.zeros(self.n_units))
        # self.register_buffer("running_std2", torch.ones(self.n_units))

        # self.register_buffer("running_mu3", torch.zeros(self.n_units * 2))
        # self.register_buffer("running_std3", torch.ones(self.n_units * 2))

        # self.register_buffer("running_mu4", torch.zeros(self.n_units * 2))
        # self.register_buffer("running_std4", torch.ones(self.n_units * 2))
        self.bn1 = nn.BatchNorm1d(n_units,track_running_stats=False)
        self.bn2 = nn.BatchNorm1d(n_units,track_running_stats=False)
        self.bn3 = nn.BatchNorm1d(2 * n_units,track_running_stats=False)
        self.bn4 = nn.BatchNorm1d(2 * n_units,track_running_stats=False)

        # self.bn1 = nn.InstanceNorm1d(n_units)
        # self.bn2 = nn.InstanceNorm1d(n_units)
        # self.bn3 = nn.InstanceNorm1d(2 * n_units)
        # self.bn4 = nn.InstanceNorm1d(2 * n_units)

    def forward(self, x, hyper_x):
        hyper_output = self.hyper_stack(hyper_x)
        para_name = iter(self.parameter_name_list)

        # 1
        logits = F.conv1d(x, weight=self.calculate_weighted_sum(hyper_output, next(para_name)),
                          bias=self.calculate_weighted_sum(hyper_output, next(para_name)),
                          stride=16)
        # logits = F.batch_norm(logits, self.running_mu1, self.running_std1,
        #                       weight=self.calculate_weighted_sum(hyper_output, next(para_name)),
        #                       bias=self.calculate_weighted_sum(hyper_output, next(para_name)), training=self.training,
        #                       momentum=0.1)

        logits = self.bn1(logits)        
        logits = torch.relu(logits)
        logits = F.max_pool1d(logits, 4)

        # 2
        logits = F.conv1d(logits, weight=self.calculate_weighted_sum(hyper_output, next(para_name)),
                          bias=self.calculate_weighted_sum(hyper_output, next(para_name)))
        # logits = F.batch_norm(logits, self.running_mu2, self.running_std2,
        #                       weight=self.calculate_weighted_sum(hyper_output, next(para_name)),
        #                       bias=self.calculate_weighted_sum(hyper_output, next(para_name)), training=self.training,
        #                       momentum=0.1)
        logits = self.bn2(logits)        
        logits = torch.relu(logits)
        logits = F.max_pool1d(logits, 4)

        # 3
        logits = F.conv1d(logits, weight=self.calculate_weighted_sum(hyper_output, next(para_name)),
                          bias=self.calculate_weighted_sum(hyper_output, next(para_name)))
        # logits = F.batch_norm(logits, self.running_mu3, self.running_std3,
        #                       weight=self.calculate_weighted_sum(hyper_output, next(para_name)),
        #                       bias=self.calculate_weighted_sum(hyper_output, next(para_name)), training=self.training,
        #                       momentum=0.1)
        logits = self.bn3(logits)        
        logits = torch.relu(logits)
        logits = F.max_pool1d(logits, 4)

        # 4
        logits = F.conv1d(logits, weight=self.calculate_weighted_sum(hyper_output, next(para_name)),
                          bias=self.calculate_weighted_sum(hyper_output, next(para_name)))
        # logits = F.batch_norm(logits, self.running_mu4, self.running_std4,
        #                       weight=self.calculate_weighted_sum(hyper_output, next(para_name)),
        #                       bias=self.calculate_weighted_sum(hyper_output, next(para_name)), training=self.training,
        #                       momentum=0.1)
        logits = self.bn4(logits)        
        logits = torch.relu(logits)
        logits = F.max_pool1d(logits, 4)

        logits = F.avg_pool1d(logits, logits.shape[-1])
        logits = logits.permute(0, 2, 1)
        logits = F.linear(logits, weight=self.calculate_weighted_sum(hyper_output, next(para_name)),
                          bias=self.calculate_weighted_sum(hyper_output, next(para_name)))

        logits = F.log_softmax(logits, dim=2)
        return logits


# ----------------------------- M5 with bias ------------------------#
class M5(nn.Module):
    #  paper: https://arxiv.org/pdf/1610.00087.pdf
    def __init__(self, n_channels=1, n_units=32, num_classes=35):
        super().__init__()

        self.conv1 = nn.Conv1d(n_channels, n_units, kernel_size=80, stride=16)
        self.bn1 = nn.BatchNorm1d(n_units,track_running_stats=False)
        # self.bn1 = nn.InstanceNorm1d(n_units)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_units, n_units, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_units,track_running_stats=False)
        # self.bn2 = nn.InstanceNorm1d(n_units)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_units, 2 * n_units, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_units,track_running_stats=False)
        # self.bn3 = nn.InstanceNorm1d(2* n_units)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_units, 2 * n_units, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_units,track_running_stats=False)
        # self.bn4 = nn.InstanceNorm1d(2 * n_units)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_units, num_classes)

        self._init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)


def m5(**kwargs: Any) -> M5:
    model = M5(**kwargs)
    return model


def scn_m5(**kwargs: Any) -> SCN:
    scn_model = SCN_M5(**kwargs)
    return scn_model


def test():
    scn_model = SCN_M5(num_alpha=1, dimensions=2, device="cuda:0", n_layers=4, n_units=32, n_channels=3, num_classes=35)
    # scn_model._modules


if __name__ == '__main__':
    test()