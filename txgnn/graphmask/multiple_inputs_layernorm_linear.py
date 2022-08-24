import math

import torch
from torch.nn import Linear, LayerNorm
from torch.nn import Parameter
from torch.nn import init


class MultipleInputsLayernormLinear(torch.nn.Module):

    """
    Properly applies layernorm to a list of inputs, allowing for separate rescaling of potentially unnormalized components.
    This is inspired by the implementation of layer norm for LSTM from the original paper.
    """

    components = None

    def __init__(self, input_dims, output_dim, init_type="xavier", force_output_dim=None, requires_grad=True):
        super(MultipleInputsLayernormLinear, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim if force_output_dim is None else force_output_dim
        self.init_type = init_type
        self.components = len(input_dims)

        self.transforms = []
        self.layer_norms = []
        for i, input_dim in enumerate(input_dims):
            self.transforms.append(Linear(input_dim, output_dim, bias=False))
            self.layer_norms.append(LayerNorm(output_dim))

        self.transforms = torch.nn.ModuleList(self.transforms)
        self.layer_norms = torch.nn.ModuleList(self.layer_norms)

        self.full_bias = Parameter(torch.Tensor(output_dim))

        self.reset_parameters()

        if not requires_grad:
            for parameter in self.parameters():
                parameter.requires_grad = False

    def reset_parameters(self):
        fan_in = sum(self.input_dims)

        std = math.sqrt(2.0 / float(fan_in + self.output_dim))
        a = math.sqrt(3.0) * std # Calculate uniform bounds from standard deviation

        for transform in self.transforms:
            if self.init_type == "xavier":
                init._no_grad_uniform_(transform.weight, -a, a)
            else:
                print("did not implement he init")
                exit()

        init.zeros_(self.full_bias)

        for layer_norm in self.layer_norms:
            layer_norm.reset_parameters()

    def forward(self, input_tensors):
        output = self.full_bias

        for component in range(self.components):
            transform = self.transforms[component]
            norm = self.layer_norms[component]
            output = output + norm(transform(input_tensors[component]))

        return output / self.components
