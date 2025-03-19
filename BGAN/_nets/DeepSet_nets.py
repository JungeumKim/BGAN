from _nets.basic_nets import  MLP,BiRNN
import torch
import torch.nn as nn
from IPython.core.debugger import set_trace

class DeepSets(nn.Module):
    def __init__(self, dim_x, dim_ss, x_length=None, nextnet_factor=16, nextnet_layers=2,
                 common_factor=64,common_layers = 3, aggregation=True, #use_nextnet=True, ##[mean, colbind]
                 device="cuda",
                 bn_last=True):
        super(DeepSets, self).__init__()
        '''
            original version:
                input: x: [batch, length, dim_x]
                common_feature_net output: phi: [batch * length, dim_ss] -> aggregated to [batch, dim_ss]
                next_net output: out: [batch, dim_ss]
        '''
        self.common_feature_net = MLP(device=device,
                                      dim=dim_x,
                                      z_dim = dim_ss,
                                      dropout=0,#.5,
                                      factor = common_factor,
                                      n_layers=common_layers)
        #important: should have an enough capacity
        #important: therefore, drop out is bad because it decreases the feature capacity too much-!
        #this self.common_feature_net has been found as the most important thing: when n_sample is increasing, we should have enough capacity for this. Probably because the generator should be a contraction for each conditional input. It may need to be a complex function.
        #self.use_nextnet  = use_nextnet
        #if self.use_nextnet:
        self.next_net = MLP(device=device,
                            dim=dim_ss if aggregation else dim_ss * x_length ,
                            z_dim = dim_ss,
                            factor= nextnet_factor,
                            n_layers=nextnet_layers)

        self.to(device)
        self.device = device
        self.aggregation  = aggregation
        #self.bn_last = bn_last
        #self.norm = nn.BatchNorm1d(dim_ss, momentum=1.0, affine=False)

    def forward(self, x):
        shape = x.shape
        assert len(shape)==3
        #set_trace()
        phi = self.common_feature_net(x.view(-1,shape[-1]))
        if self.aggregation: # == "mean":
            phi = phi.view(x.shape[0],x.shape[1],-1).mean(1)
        else:
            phi = phi.view(x.shape[0],-1)
        # ELSE: just side by side output.
        #if self.use_nextnet:
        out = self.next_net(phi)
        #else:
        #    out = phi
        #if self.bn_last:
        #    return self.norm(out)
        return out

class Auto_ss(nn.Module):
    def __init__(self, x_length=None, f1_dim=2, f2_dim=2, device="cuda", x_dim=2,
                 factor=64, f1_layers =3,
                 common_factor=64,common_layers = 3, aggregation=True , *args, **kwargs):
        super().__init__()
        self.f1_dim = f1_dim
        self.f2_dim= f2_dim
        if not aggregation:
            assert x_length is not None
        if  self.f1_dim>0:
            self.f1 = DeepSets(dim_x=x_dim,
                               dim_ss=self.f1_dim,
                               nextnet_factor=factor,
                               nextnet_layers=f1_layers,
                               common_factor= common_factor,
                               common_layers = common_layers,aggregation=aggregation,
                               x_length = x_length,
                               bn_last=False,# this bn_last=False is a super important argument!!
                               device=device)
        if  self.f2_dim>0:
            self.f2 = BiRNN(input_size=x_dim,
                            hidden_size=512,
                            num_layers=1,
                            xdim=self.f2_dim,
                            bn_last=False,
                            device=device# this bn_last=False is a super important argument!!
                            )

        self.device=device
        self.to(device)

    def forward(self, x):
        if self.f1_dim>0 and  self.f2_dim ==0:
            return self.f1(x)
        elif self.f1_dim==0 and  self.f2_dim >0:
            return self.f2(x)
        else:
            x= torch.cat([self.f1(x),self.f2(x)], 1)
            return x
