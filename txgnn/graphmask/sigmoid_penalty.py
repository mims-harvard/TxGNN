import torch
from torch.nn.parameter import Parameter
import numpy as np
from torch import sigmoid
import copy

class SoftConcrete(torch.nn.Module):

    def __init__(self, threshold = 0.5, remove_key_parts = False, use_top_k = False, k = 0.05):
        super(SoftConcrete, self).__init__()
        self.loc_bias = 3
        self.threshold = threshold
        self.remove_key_parts = remove_key_parts
        
        self.use_top_k = use_top_k
        self.k = k
        #if self.remove_key_parts:
            #print('Remove key parts...')
        #if self.threshold != 0.5:
        #    print('Using threshold: ', self.threshold)
            
    def forward(self, input_element, summarize_penalty=True):  
        input_element = input_element + self.loc_bias
        
        penalty = sigmoid(input_element)
        penalty_not_sum = copy.deepcopy(penalty.detach())

        clipped_s = self.clip(penalty)
        
        if self.use_top_k:
            #threshold = torch.quantile(clipped_s, 1-self.k)
            
            # Calculate the number of elements to include
            num_elements_to_include = int(len(clipped_s) * self.k)
            num_elements_to_include = max(num_elements_to_include, 1)
            threshold = torch.topk(clipped_s, num_elements_to_include).values[-1]
        else:
            threshold = self.threshold
            
        if self.remove_key_parts:
            soft_concrete = (clipped_s < threshold).float()
        else:
            soft_concrete = (clipped_s > threshold).float()
            
        clipped_s_ = clipped_s + (soft_concrete - clipped_s).detach()
        
        if summarize_penalty:
            penalty = penalty.mean()
        
        return clipped_s_, penalty, sigmoid(input_element), penalty_not_sum
        
    def clip(self, x, min_val=0, max_val=1):
        return x.clamp(min_val, max_val)