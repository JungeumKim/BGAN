"""
This is code for  Adversarial Bayesian Simulation (Yuexi Wang, Veronika Rockova), developed and implemented by Jungeum Kim to adopt DeepSet with a regularizer.
"""

from IPython.core.debugger import set_trace
from _nets.basic_nets import  MLP,DeepSets,BiRNN
from BGAN import Critic as BGAN_CRITIC
from BGAN import Generator as BGAN_GENERATOR
from BGAN import BGAN
import torch
import torch.nn as nn

class Critic(BGAN_CRITIC):


    def __init__(self,theta_dim=2,x_dim=2,  f1_dim=2,f2_dim=2,
                 d_hidden = [128,128,128], leaky=0.1):
        super().__init__(input_dim = theta_dim,
                         cond_dim = f1_dim + f2_dim,
                         d_hidden = d_hidden
                         )
        self.ss = Auto_ss(f1_dim=f1_dim, f2_dim=f2_dim, x_dim=x_dim)

    def forward(self, x, context):

        f_context = self.ss(context)# .unsqueeze(-1))
        return super().forward(x, f_context)

class Generator(BGAN_GENERATOR):

    def __init__(self, x_dim=2, theta_dim = 2,
                f1_dim=2,f2_dim=2, leaky=0.1,
                 d_hidden = [128,128,128]):
        
        super().__init__(d_hidden=d_hidden,
                         theta_dim=theta_dim,
                         z_dim = theta_dim,
                         cond_dim = f1_dim + f2_dim,
                         leaky=leaky)
        
        self.d_cond = f1_dim + f2_dim
        
        self.ss = Auto_ss(f1_dim=f1_dim, f2_dim=f2_dim, x_dim=x_dim)


    def forward(self, context, noise = None):
        f_context = self.ss(context)
        return super().forward(f_context, noise)


class Auto_ss(nn.Module):
    def __init__(self,  f1_dim=2, f2_dim=2, device="cuda", x_dim=2,
                 factor=64, f1_layers =3,*args, **kwargs):
        super().__init__()
        self.f1_dim = f1_dim
        self.f2_dim= f2_dim
        if  self.f1_dim>0:
            self.f1 = DeepSets(dim_x=x_dim,
                               dim_ss=self.f1_dim,
                               factor=factor, 
                               num_layers=f1_layers, 
                               bn_last=False,# this bn_last=False is a super important argument!!
                               device=device)
        if  self.f2_dim>0:
            self.f2 = BiRNN(input_size=x_dim,
                            hidden_size=512,
                            num_layers=1,
                            xdim=self.f2_dim, 
                            bn_last=False # this bn_last=False is a super important argument!!
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


class DBGAN(BGAN):

    def __init__(self, simulator, theta_dim,  x_dim, x_length,
                 f1_dim=2, f2_dim=5, 
                 device="cuda",epoch=150, batch_size = 200, 
                 seed=1234, d_hidden = 128,
                 critic_lr = 0.001, generator_lr = 0.001,
                 *args, **kwargs):
        super().__init__(simulator, theta_dim, x_dim, x_length,
                 device=device,epoch=epoch, batch_size = batch_size, d_hidden=d_hidden,
                 critic_lr=critic_lr, generator_lr = generator_lr,
                 seed=seed)

        self.generator = Generator(x_dim, x_length,
                                   theta_dim = theta_dim,
                                   f1_dim=f1_dim, 
                                   f2_dim=f2_dim,
                                   d_hidden = [d_hidden,d_hidden,d_hidden]
                                   )
        
        self.critic = Critic(
                             theta_dim=theta_dim,
                             f1_dim=f1_dim,
                             f2_dim=f2_dim,
                             x_dim = x_dim,
                             d_hidden = [d_hidden,d_hidden,d_hidden])
        
