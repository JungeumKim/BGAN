"""
This is code for  Adversarial Bayesian Simulation (Yuexi Wang, Veronika Rockova), developed and implemented by Jungeum Kim to adopt DeepSet with a regularizer.
"""

from IPython.core.debugger import set_trace
from _nets.basic_nets import  MLP,DeepSets,BiRNN
#from BGAN import Critic
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from time import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

def mmd(X, Y, gamma=1.0):
    """MMD using RBF kernel (k(x,y) = exp(-gamma * ||x-y||^2 / 2))"""
    XX = rbf_kernel(X, X, gamma)
    YY = rbf_kernel(Y, Y, gamma)
    XY = rbf_kernel(X, Y, gamma)
    #set_trace()
    return XX.mean() + YY.mean() - 2 * XY.mean()

def gradient_penalty(net, x, x_hat):

        alpha = torch.rand(x.size(0)).unsqueeze(1).unsqueeze(2).to(x.device)
        #set_trace()
        interpolated = x * alpha + x_hat * (1 - alpha)
        interpolated = torch.autograd.Variable(interpolated.detach(), requires_grad=True)
        output = net(interpolated )
        gradients = torch.autograd.grad(output, 
                                        interpolated, 
                                        torch.ones_like(output),
                                        retain_graph=True, 
                                        create_graph=True, 
                                        only_inputs=True)[0]
        penalty = F.relu(gradients.norm(2, dim=1) - 1).mean()             # one-sided
        # penalty = (gradients.norm(2, dim=1) - 1).pow(2).mean()          # two-sided
        return penalty


class Critic(nn.Module):

    def __init__(self,dropout = 0,theta_dim=2,x_dim=2,  f1_dim=2,f2_dim=2,
                 d_hidden = [128,128,128], leaky=0.1):

        super().__init__()
        self.d_cond = f1_dim + f2_dim
        d_in = [theta_dim+self.d_cond] + d_hidden
        d_out = d_hidden + [1]

        self.ss = Auto_ss(f1_dim=f1_dim, f2_dim=f2_dim, x_dim=x_dim)

        self.layers = nn.ModuleList([nn.Linear(i, o) for i, o in zip(d_in, d_out)])
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(leaky)

    def forward(self, x, context):

        f_context = self.ss(context)# .unsqueeze(-1))
        x = torch.cat([x, f_context], -1)
        #set_trace()
        #x = torch.cat([x, context.squeeze(-1)], -1)

        for layer in self.layers[:-1]:
            #set_trace()
            x = self.dropout(self.activation(layer(x)))
        return self.layers[-1](x)

    def gradient_penalty(self, x, x_hat, context):

        alpha = torch.rand(x.size(0)).unsqueeze(1).to(x.device)
        interpolated = x * alpha + x_hat * (1 - alpha)
        interpolated = torch.autograd.Variable(interpolated.detach(), requires_grad=True)
        critic = self(interpolated, context)
        gradients = torch.autograd.grad(critic, interpolated, torch.ones_like(critic),
                                        retain_graph=True, create_graph=True, only_inputs=True)[0]
        penalty = F.relu(gradients.norm(2, dim=1) - 1).mean()             # one-sided
        # penalty = (gradients.norm(2, dim=1) - 1).pow(2).mean()          # two-sided
        return penalty

class Generator(nn.Module):

    def __init__(self, x_dim=2,x_length=1, theta_dim = 2,  dropout = 0.1,
                 activation = "relu", f1_dim=2,f2_dim=2, leaky=0.1,
                 d_hidden = [128,128,128]):
        
        super().__init__()
        
        self.d_cond = f1_dim + f2_dim
        
        self.ss = Auto_ss(f1_dim=f1_dim, f2_dim=f2_dim, x_dim=x_dim)

        self.d_noise = theta_dim
        self.theta_dim =theta_dim
        
        d_in = [self.d_noise + self.d_cond] + d_hidden
        d_out = d_hidden + [theta_dim]
        self.layers = nn.ModuleList([nn.Linear(i, o) for i, o in zip(d_in, d_out)])
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(leaky)

    def forward(self, context, noise = None):
        # context: conditioning variable.
        if noise is None:
            noise = torch.randn(context.size(0), self.d_noise).to(context.device)
        
        f_context = self.ss(context)
        
        x = torch.cat([noise, f_context], -1)

        for layer in self.layers:
            x = self.dropout(self.activation(layer(x)))
        
        return x


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


class DBGAN():

    def __init__(self, simulator, theta_dim,  x_dim, x_length,
                 f1_dim=2, f2_dim=5, 
                 device="cuda",epoch=150, batch_size = 200, 
                 seed=1234, d_hidden = 128,
                 critic_lr = 0.001, generator_lr = 0.001, #inner_lr = 0.001,
                 lr_decay= 0.99, w_regul = 1.0,Q_freq = 1,
                 *args, **kwargs):

        self.generator = Generator(x_dim, x_length,
                                   theta_dim = theta_dim,
                                   dropout = 0.1,
                                   f1_dim=f1_dim, 
                                   f2_dim=f2_dim,
                                   d_hidden = [d_hidden,d_hidden,d_hidden]
                                   )
        
        self.critic = Critic(dropout = 0,
                             theta_dim=theta_dim,
                             f1_dim=f1_dim,
                             f2_dim=f2_dim,
                             x_dim = x_dim,
                             d_hidden = [d_hidden,d_hidden,d_hidden])
        
        #self.Q = MLP(dim=f1_dim+ f2_dim, z_dim=theta_dim)
        
        self.np_random = np.random.RandomState(seed)
        self.generator.to(device), self.critic.to(device)
        #self.Q.to(device)
        self.simulator = simulator
        self.device = device
        self.epoch = epoch
        self.batch_size = batch_size
        self.x_dim=x_dim
        self.x_length=x_length
       
        self.critic.to(device)
        self.generator.to(device)
        
        self.critic_lr = critic_lr
        self.generator_lr = generator_lr
        #self.inner_lr = inner_lr
        self.lr_decay=lr_decay
        self.w_regul = w_regul
        self.Q_freq = Q_freq
        self.qualities = []
        
    def train(self,true_x=None, true_thetas=None,
              start_epoch=1, end_epoch=None, critic_gp_factor = 5, critic_steps = 5, n_iter=100):
        
        lr_decay = self.lr_decay
        
        if end_epoch ==None:
            end_epoch = self.epoch+1
            
        for epoch in range(start_epoch, end_epoch):
            running_loss = 0.0
            
            print(f"Epoch {epoch}")
            opt_generator = optim.Adam(self.generator.parameters(), lr=self.generator_lr*(lr_decay**epoch))
            #opt_ss = optim.Adam(self.generator.ss.parameters(), lr=self.generator_lr*(lr_decay**epoch))

            #opt_ss = optim.Adam(self.Q.parameters(), lr=self.generator_lr*(lr_decay**epoch))
            opt_critic = optim.Adam(self.critic.parameters(), lr=self.critic_lr*(lr_decay**epoch))
            
            #opt_ss2 = optim.Adam(self.generator.ss.parameters(), lr=self.inner_lr*(lr_decay**epoch))
            #opt_Q2 = optim.Adam(self.Q.parameters(), lr=self.inner_lr*(lr_decay**epoch))
            
            
            # train loop
            WD_train, WD_test= 0, 0
            n_critic = 0
            #critic_update = True

            for iter in range(n_iter):
                theta, X = self.simulator(batch_size = self.batch_size, np_random = self.np_random)
                theta, X = theta.to(self.device), X.to(self.device)
                self.generator.zero_grad()
                self.critic.zero_grad()
                #self.Q.zero_grad()
                
                theta_hat = self.generator(X)
                SS = self.generator.ss(X)
                
                critic_theta_hat = self.critic(theta_hat, X).mean()
                #loss_regul = (self.Q(SS)-theta).pow(2).sum(1).mean()
                

                if n_critic < critic_steps:
                    critic_x = self.critic(theta, X).mean()
                    WD = critic_x-critic_theta_hat
                    loss = -WD #+ self.w_regul * loss_regul
                    loss += critic_gp_factor * self.critic.gradient_penalty(theta, theta_hat, X)
                    #loss += critic_gp_factor * gradient_penalty(self.generator.ss, X, X2)
                    loss.backward()
                    
                    opt_critic.step()
                    
                    #if n_critic<self.Q_freq:
                    #    opt_ss2.step()
                    #    opt_Q2.step()
                    
                    WD_train += WD.item()
                    n_critic += 1

                else: #generator 1 step.
                    loss = - critic_theta_hat #+ self.w_regul * loss_regul

                    loss.backward()
                    
                    opt_generator.step()
                    #opt_ss.step()
                    #opt_Q.step()
            
                    running_loss += loss.item()
                    n_critic = 0 # now, the critic will again be trained.
                    
                    if true_thetas is not None:
                        with torch.no_grad():
                            dist_quality = mmd(true_thetas,self.generator(true_x).cpu())
                        self.qualities.append({"epoch": epoch, 
                                               "iter":iter, 
                                               "mmd":round(dist_quality,3)})
                        
            WD_train /= n_iter
            self.loss_cum = WD_train
            

    def sampler(self, X, sample_size, shaper = None):
        
        if shaper is None:
            X = torch.from_numpy(X).float().view(1, -1).repeat(sample_size, 1)
        else: 
            X = shaper(X)
            
        X = X.to(self.device)
        with torch.no_grad():
            return self.generator(X).to("cpu")

    def save(self, path):

        torch.save({
        'generator': self.generator.state_dict(),
        'critic': self.critic.state_dict()
        }, path)

    def load(self, path):
        saved = torch.load(path)
        self.generator.load_state_dict(saved['generator'])
        self.critic.load_state_dict(saved['critic'])


