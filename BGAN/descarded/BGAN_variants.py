"""
This is code for  Adversarial Bayesian Simulation (Yuexi Wang, Veronika Rockova), developed and implemented by Jungeum Kim to adopt DeepSet with a regularizer.
"""

from IPython.core.debugger import set_trace
from _nets.basic_nets import  MLP,DeepSets,BiRNN
from BGAN import Critic
from BGAN_deep_critic import DBGAN
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

class hyb_critic(Critic):
    def __init__(self, *args, **kwargs):  # Added 'self' as the first argument
        super().__init__(*args, **kwargs)

    def forward(self, x, context):  # Added 'self' as the first argument
        return super().forward(x, context.squeeze(-1))  # Fixed 'foward' typo and added '()' after 'super'
        
class BGAN_hybrid(DBGAN):
    def __init__(self, simulator, theta_dim, x_dim, x_length,
                 f1_dim=2, f2_dim=5, 
                 device="cuda", epoch=150, batch_size=200, 
                 seed=1234, d_hidden=128,
                 critic_lr=0.001, generator_lr=0.001, 
                 lr_decay=0.99, w_regul=1.0, Q_freq=1,
                 *args, **kwargs):
        super().__init__(simulator, theta_dim, x_dim, x_length,
                 f1_dim=f1_dim, f2_dim=f2_dim, 
                 device=(device), epoch=epoch, batch_size=batch_size, 
                 seed= seed, d_hidden=d_hidden,
                 critic_lr=critic_lr, generator_lr=generator_lr, 
                 lr_decay=lr_decay, w_regul=w_regul, Q_freq=Q_freq)
        
        self.critic = hyb_critic(input_dim=theta_dim,
                             cond_dim = x_dim*x_length,
                             d_hidden = [d_hidden,d_hidden,d_hidden])

        self.critic.to(device)
