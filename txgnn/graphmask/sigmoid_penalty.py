import torch
from torch.nn.parameter import Parameter
import numpy as np
from torch import sigmoid
import copy

class SoftConcrete(torch.nn.Module):

    def __init__(self):
        super(SoftConcrete, self).__init__()
        self.loc_bias = 3
        
    def forward(self, input_element, summarize_penalty=True):  
        input_element = input_element + self.loc_bias
        
        penalty = sigmoid(input_element)
        penalty_not_sum = copy.deepcopy(penalty.detach())

        clipped_s = self.clip(penalty)

        soft_concrete = (clipped_s > 0.5).float()
        clipped_s_ = clipped_s + (soft_concrete - clipped_s).detach()
        
        if summarize_penalty:
            penalty = penalty.mean()
        
        return clipped_s_, penalty, clipped_s, penalty_not_sum
        
    def clip(self, x, min_val=0, max_val=1):
        return x.clamp(min_val, max_val)