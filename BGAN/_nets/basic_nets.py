import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


from IPython.core.debugger import set_trace


class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, xdim, bn_last=True, device="cuda"):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bn_last = bn_last
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, xdim)
        self.norm = nn.BatchNorm1d(xdim, momentum=1.0, affine=False)
        self.device = device
        self.to(device)

    def forward(self, x):
        '''
        input x: (batch_size x T x dim_x)
        '''
        # Set initial states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) # 2 for bidirection
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        if self.bn_last:
            return self.norm(out)
        return out

class DeepSets(nn.Module):
    def __init__(self, dim_x, dim_ss, factor=16, 
                 num_layers=2, device="cuda", 
                 bn_last=True):
        super(DeepSets, self).__init__()

        self.common_feature_net = MLP(device=device,
                                      dim=dim_x,
                                      z_dim = dim_ss,
                                      dropout=0,#.5,
                                      #positive=True,
                                      factor=64,n_layers=3) 
        #important: should have an enough capacity
        #important: therefore, drop out is bad because it decreases the feature capacity too much-!
        #this self.common_feature_net has been found as the most important thing: when n_sample is increasing, we should have enough capacity for this. Probably because the generator should be a contraction for each conditional input. It may need to be a complex function.

        self.next_net = MLP(device=device,
                            dim=dim_ss,
                            z_dim = dim_ss,
                            factor=factor,
                            n_layers=num_layers)

        self.to(device)
        self.device = device
        self.bn_last = bn_last
        self.norm = nn.BatchNorm1d(dim_ss, momentum=1.0, affine=False)

    def forward(self, x):
        shape = x.shape
        assert len(shape)==3
        phi = self.common_feature_net(x.view(-1,shape[-1])).view(x.shape[0],x.shape[1],-1).mean(1)
        #phi = self.common_feature_net(x.view(-1,shape[-1])).view(x.shape[0],x.shape[1],-1).sum(1)
        out = self.next_net(phi)
        if self.bn_last:
            return self.norm(out)
        return out

def get_layer(in_d, out_d, lip=False):
    if lip: return nn.utils.spectral_norm(nn.Linear(in_d, out_d))
    else: return nn.Linear(in_d, out_d)

class MLP_batchnorm(nn.Module):
    def __init__(self, device="cuda", dim=2, z_dim=1,
                 leaky=0.1, factor=64, n_layers=2, lip=False, dropout=0, positive=False):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.non_linear = nn.LeakyReLU(leaky) if leaky > 0 else nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()  # List to hold batch norm layers

        # First layer
        self.layers.append(get_layer(dim, factor, lip=lip))
        self.batch_norms.append(nn.BatchNorm1d(factor))  # Batch norm for the first layer

        # Hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(get_layer(factor, factor, lip=lip))
            self.batch_norms.append(nn.BatchNorm1d(factor))  # Batch norm for hidden layers

        # Last layer
        self.layers.append(get_layer(factor, z_dim, lip=lip))

        self.to(device)
        self.device = device
        self.positive = positive
        
    def forward(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:  # Apply batch norm and non-linearity on all but last layer
                h = self.batch_norms[i](h)
                h = self.non_linear(h)
            h = self.dropout(h)
            
        if self.positive:
            return h.abs()
        else:
            return h




    
class MLP(nn.Module):
    def __init__(self, device="cuda", dim=2, z_dim=1,
                 leaky=0.1, factor=64, n_layers=2, lip=False,dropout=0, positive=False):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.non_linear = nn.LeakyReLU(leaky) if leaky > 0 else nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # First layer
        self.layers = nn.ModuleList([get_layer(dim, factor, lip=lip)])
        for _ in range(n_layers):
            self.layers.append(get_layer(factor, factor, lip=lip))

        # Last layer
        self.layers.append(get_layer(factor, z_dim,lip=lip))

        self.to(device)
        self.device = device
        self.positive = positive
            
    def forward(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            h = self.dropout(layer(h))
            if i < len(self.layers) - 1:  # Apply non-linearity on all but last layer
                h = self.dropout(self.non_linear(h))
        if self.positive:
            return h.abs()
        else:
            return h