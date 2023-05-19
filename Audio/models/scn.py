import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.parameter import Parameter
from typing import List
from typing import Any, Optional, Mapping
from collections import OrderedDict

__all__ = ['SCN']

class SCN(nn.Module):
    def __init__(self, num_alpha:int, dimensions:int, base_model:nn.Module, device) -> None:
        super(SCN, self).__init__()
        self.dimensions=dimensions
        self.num_alpha = num_alpha
        self.device = device
        self.hyper_stack  = nn.Sequential(
            nn.Linear(self.num_alpha, 64),
            nn.ReLU(),
            nn.Linear(64, self.dimensions),
            nn.Softmax(dim=0)
        )

        self.base_model = base_model
        # self.base_model will not be updated and only its parameter names will be used to create parameter list for multi-dimensional inference models.
        # For example, self.base_model.conv1.weight is used to create self.inference_model1.conv1,..., and  self.inference_model1.convD (D == number of dimensions).
        self.__create_parameter_list(dimensions)


    def __create_parameter_list(self, dimensions):
        self.parameter_name_list = []
        # assume base_model has been initlized
        for k,v in self.base_model.named_parameters():
            # print(f'name: {k} | shape: {v.shape} | type: {type(v)} | requires_grad: {v.requires_grad}')
            if not v.requires_grad:
                continue
            r_k = k.replace('.', '-')

            if r_k.find('bn')>=0:
                print(f'{r_k} will not be processed')
                continue
            self.add_module(f'base_model-{r_k}-list', nn.ParameterList([Parameter(v.clone().detach().data) for i in range(dimensions)]))
            self.parameter_name_list.append(f'base_model-{r_k}-list')


    def calculate_weighted_sum(self, factors, para_name) -> Parameter:
        """
        A non-recursion algorithm to find the final Parameter child in a nn.module/model
        """
        # para_name, in self.__parameter_names:
        module_names = para_name.split("-")[1:-1]
        module = self.base_model
        for mn in module_names:
            if len(module._modules.items())==0:
                paralist = self._modules[para_name]
                weight_list = [a * b for a, b in zip(paralist, factors)]
                w = torch.sum(torch.stack(weight_list), dim=0)
                return w
            elif len(module._modules[mn]._modules.items())>0:
                module = module._modules[mn]
            else:
                # stupid conditions. Todo: remove it.
                module = module._modules[mn]
    
    def extend_inference_models(self, add_dimensions, init_parameter_index=0):
        with torch.no_grad():
            for name in self.parameter_name_list:
                for d in range(add_dimensions):
                    self._modules[name].append(self._modules[name][init_parameter_index].clone().detach())
            
            # add new parameters to hypernet
            # You may want to re-use some hypernet parameters from the original SCN
            self.hyper_stack._modules['2'] = nn.Linear(in_features=64, out_features=self.dimensions+add_dimensions, bias=True).to(self.device) # todo, move all host-device copying operations out of model's function.
            # otherwise, you just init the hypernet's weights
            # old_last_layer_of_hypernet =  self.hyper_stack._modules['2']
            # self.hyper_stack._modules['2'].weight[:self.dimensions, :64] =  old_last_layer_of_hypernet.weight.detach().clone() 
            nn.init.normal_(self.hyper_stack._modules['2'].weight, 0, 0.01)
            nn.init.zeros_(self.hyper_stack._modules['2'].bias)
            self.dimensions=self.dimensions+add_dimensions
        print("After extending dimension, Please reload optimizer for new parameters")
    
    def freeze_inference_models(self, ids:List[int]):
        for i in ids:
            assert i<self.dimensions
        with torch.no_grad():
            for name in self.parameter_name_list:
                for d in range(self.dimensions):
                    if d in ids:
                        self._modules[name][d].requires_grad=False
        print(f"Freeze the {ids} inference models")

    def unfreeze_inference_models(self, ids:List[int]):
        for i in ids:
            assert i<self.dimensions
        with torch.no_grad():
            for name in self.parameter_name_list:
                for d in range(self.dimensions):
                    if d in ids:
                        self._modules[name][d].requires_grad=True
        print(f"Unfreeze the {ids} inference models")

    def load_base_model_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        keys =  self.base_model.load_state_dict(state_dict, strict)
        self.__create_parameter_list(self.dimensions)
        print("Loaded weights to SCN's base model and re-create SCN's parameter list. You must init optimizer.")
        return keys
        