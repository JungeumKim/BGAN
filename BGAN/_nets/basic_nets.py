import numpy as np
import torch
import torch.nn as nn


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
        input x: (batch_size x T x input_size)
        outpt (batch_size x xdim)
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
def get_layer(in_d, out_d, lip=False):
    if lip: return nn.utils.spectral_norm(nn.Linear(in_d, out_d))
    else: return nn.Linear(in_d, out_d)
    
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
