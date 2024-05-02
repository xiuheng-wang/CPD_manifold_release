import torch 
from torch import nn
import math

# Xavier Initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, 0, math.sqrt(2.0/(m.weight.size()[0] + m.weight.size()[1]))) 
        m.bias.data.fill_(0.5)

class Net(nn.Module):
    def __init__(self, input_shape, in_and_hi_layer):
        super(Net, self).__init__()
        self.sequential = nn.Sequential()
        self.sequential.add_module('fc_in', nn.Linear(input_shape, in_and_hi_layer[0]))
        self.sequential.add_module('af_in', nn.ReLU()) # nn.Tanh()
        for i in range(len(in_and_hi_layer)-1):
            self.sequential.add_module('fc_hi_%d'%(i+1), nn.Linear(in_and_hi_layer[i], in_and_hi_layer[i+1]))
            self.sequential.add_module('af_hi_%d'%(i+1), nn.ReLU()) # nn.Tanh()
        self.sequential.add_module('fc_out', nn.Linear(in_and_hi_layer[-1],1))
        self.sequential.add_module('af_out', nn.Sigmoid())

    def forward(self, x):
        x = self.sequential(x)
        return x