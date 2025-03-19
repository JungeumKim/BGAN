from _nets.basic_nets import  MLP
import torch 
from torch import nn
from IPython.core.debugger import set_trace

class basic_DeepSet(nn.Module):
    def __init__(self, input_dim, ss_dim, 
                  c_factor = 64,c_n_layers=3, 
                 device="cuda"):
        super(basic_DeepSet,self).__init__()
        self.common_feature_net = MLP(device=device,
                                      dim=input_dim,
                                      z_dim = ss_dim,
                                      dropout=0,
                                      factor=c_factor,
                                      n_layers=c_n_layers) 
        self.to(device)
        self.device = device
        
    def forward(self, x):
        shape = x.shape
        assert len(shape)==3
        phi = self.common_feature_net(x.view(-1,shape[-1])).view(x.shape[0],x.shape[1],-1).mean(1)
        return phi
'''  
class Net(nn.Module):
    def __init__(self,input_shape,output_shape, factor=64, depth = 2):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(input_shape,factor)
        self.linears = nn.ModuleList([nn.Linear(factor,factor) for i in range(depth)])
        self.fc4 = nn.Linear(factor,output_shape)
        
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        for md in self.linears:
            x = torch.relu(md(x))
        x = self.fc4(x)
        return x
'''
    
class DeepSet2(nn.Module):
    def __init__(self, input_dim, ss_dim=2, factor=64, depth=2, ss_dim_med=16, device="cuda"):

        #super(DeepSet2, self).__init__(ss_dim, factor, depth)
        super(DeepSet2, self).__init__()
        
        self.suff_net1 = basic_DeepSet(device=device,
                                       input_dim = input_dim,
                                       ss_dim = ss_dim_med)
        
        self.suff_net2 = basic_DeepSet(device=device,
                                       input_dim = 1,#ss_dim_med,
                                       ss_dim = ss_dim)
        
    def forward(self,x):
        med = self.suff_net1(x).unsqueeze(-1)
        #set_trace()
        med2= self.suff_net2(med)
        return med2 #super().forward(med2)
    
    
