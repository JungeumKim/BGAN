from _nets.basic_nets import  MLP,BiRNN
import torch
import torch.nn as nn

class LSTM_DeepSets(nn.Module):
    def __init__(self, dim_x, dim_ss,
                 nextnet_factor=16, nextnet_layers=2,
                 common_factor=512,common_layers = 1,
                 device="cuda"
                 ):
        super().__init__()
        '''
            apply LSTM to batch_size*n_rep x lx x dx -> output: batch_size*n_rep x f2
            then reshape to : batch_size x n_rep x dim_ss 
            then aggregate to batch_size x dim_ss 
            
            X.shape[0] = batch_size
            X.shape[1] = n_rep
            X.shape[2] = lx (x_length)
            X.shape[3] = dx (dim_x)
        '''
        #self.n_rep = n_rep
        self.f2 = BiRNN(input_size=dim_x,
            hidden_size=common_factor,
            num_layers=common_layers,
            xdim=dim_ss,
            bn_last=False,
            device=device)

        self.next_net = MLP(device=device,
                            dim=dim_ss,
                            z_dim = dim_ss,
                            factor= nextnet_factor,
                            n_layers=nextnet_layers)

        self.to(device)
        self.device = device

    def forward(self, x):
        shape = x.shape
        assert len(shape)==4

        X_reshaped = x.view(-1, x.shape[-2], x.shape[-1]) #  batch_size*n_rep x lx x dx
        out_f2 = self.f2(X_reshaped) #batch_size*n_rep x dim_ss
        out_f2 = out_f2.view(x.shape[0], x.shape[1], -1) #batch_size x n_rep x dim_ss
        out_f2 = out_f2.mean(1) # aggregated to batch_size x dim_ss

        out = self.next_net(out_f2)
        return out #batch_size x dim_ss
