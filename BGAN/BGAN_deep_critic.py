"""
This is code for  Adversarial Bayesian Simulation (Yuexi Wang, Veronika Rockova), developed and implemented by Jungeum Kim to adopt DeepSet with a regularizer.
"""
from IPython.core.debugger import set_trace
from BGAN import Critic as BGAN_CRITIC
from BGAN import Generator as BGAN_GENERATOR
from _nets.DeepSet_nets import Auto_ss
from _nets.LSTM_Deepsets import LSTM_DeepSets
import torch
from BGAN import BGAN


class Critic(BGAN_CRITIC):

    def __init__(self,theta_dim=2,x_dim=2,  f1_dim=2,f2_dim=2,
                 d_hidden = [128,128,128], leaky=0.1, aggregation=True,x_length=None, 
                 factor=64, f1_layers =3,
                 common_factor=64,common_layers = 3,device ="cuda"
                 ):
        super().__init__(input_dim = theta_dim,
                         cond_dim = f1_dim + f2_dim,
                         d_hidden = d_hidden)
        self.ss = Auto_ss(f1_dim=f1_dim, f2_dim=f2_dim, x_dim=x_dim, aggregation=aggregation, x_length=x_length,
                                   factor=factor, f1_layers =f1_layers,
                                   common_factor=common_factor,common_layers = common_layers,
                                    device = device)
    def forward(self, x, context):
        f_context = self.ss(context)# .unsqueeze(-1))
        return super().forward(x, f_context)

class Generator(BGAN_GENERATOR):

    def __init__(self, x_dim=2, theta_dim = 2,z_dim = 2,
                f1_dim=2,f2_dim=2, leaky=0.1,x_length=None, 
                 d_hidden = [128,128,128], aggregation=True,
                factor=64, f1_layers =3,
                 common_factor=64,common_layers = 3, device="cuda"):
        
        super().__init__(d_hidden=d_hidden,
                         theta_dim=theta_dim,
                         z_dim = z_dim,
                         cond_dim = f1_dim + f2_dim,
                         leaky=leaky)
        #self.d_cond = f1_dim + f2_dim
        self.ss = Auto_ss(f1_dim=f1_dim, f2_dim=f2_dim, x_dim=x_dim, aggregation=aggregation, x_length=x_length,
                           factor=factor, f1_layers =f1_layers,
                           common_factor=common_factor,common_layers = common_layers,device=device)


    def forward(self, context, noise = None):
        f_context = self.ss(context)
        return super().forward(f_context, noise)

class DBGAN(BGAN):
    '''
        Assumption:
            The data X: batch_size x x_length x x_dim :
                It will not be reshaped before the network operation
            The parameter: batch_size x theta_dim
    '''
    def __init__(self, simulator, theta_dim,  x_dim, x_length,z_dim=None,
                 f1_dim=2, f2_dim=5, 
                 device="cuda",epoch=150, batch_size = 200, 
                 seed=1234, d_hidden = 128,
                 critic_lr = 0.001, generator_lr = 0.001,aggregation=True,
                 factor=64, f1_layers =3,
                 common_factor=64,common_layers = 3,hidden_layers=3,
                 *args, **kwargs):

        if z_dim is None:
            z_dim = theta_dim
        super().__init__(simulator, theta_dim, x_dim, x_length,z_dim=z_dim,
                 device=device,epoch=epoch, batch_size = batch_size, d_hidden=d_hidden,
                 critic_lr=critic_lr, generator_lr = generator_lr,
                 seed=seed, LSTM = (f2_dim >0))

        self.generator = Generator(x_dim,
                                   theta_dim = theta_dim,
                                   z_dim = z_dim,
                                   f1_dim=f1_dim, 
                                   f2_dim=f2_dim,
                                   d_hidden = [d_hidden for _ in range(hidden_layers)],
                                   aggregation=aggregation,
                                   x_length=x_length,
                                   factor=factor, f1_layers =f1_layers,
                                   common_factor=common_factor,common_layers = common_layers,
                                   device = device
                                   )
        self.critic = Critic(
                             theta_dim=theta_dim,
                             f1_dim=f1_dim,
                             f2_dim=f2_dim,
                             x_dim = x_dim,
                             d_hidden = [d_hidden  for _ in range(hidden_layers)],
                             aggregation=aggregation,
                             x_length=x_length,
                             factor=factor, f1_layers =f1_layers,
                             common_factor=common_factor,common_layers = common_layers,
                            device = device)
        self.generator.to(device)        
        self.critic.to(device)

#Pickup here

class DBGAN_mix(DBGAN):
    '''
        Assumption:
            The data X: batch_size x x_length x x_dim :
                It will not be reshaped before the network operation
            The parameter: batch_size x theta_dim
    '''

    def __init__(self, simulator, theta_dim, x_dim, x_length, z_dim=None,
                 f1_dim=2, f2_dim=5,
                 device="cuda", epoch=150, batch_size=200,
                 seed=1234, d_hidden=128,
                 critic_lr=0.001, generator_lr=0.001, aggregation=True,
                 factor=64, f1_layers=3,
                 common_factor=64, common_layers=3, hidden_layers=3,
                 *args, **kwargs):
        # Properly initialize the parent class (DBGAN)
        if z_dim is None:
            z_dim = theta_dim

        super().__init__(simulator, theta_dim, x_dim, x_length,z_dim = z_dim,
                         f1_dim=f1_dim, f2_dim=f2_dim, device=device, epoch=epoch,
                         batch_size=batch_size, seed=seed, d_hidden=d_hidden,
                         critic_lr=critic_lr, generator_lr=generator_lr,
                         aggregation=aggregation, factor=factor, f1_layers=f1_layers,
                         common_factor=common_factor, common_layers=common_layers,
                         hidden_layers=hidden_layers, *args, **kwargs)
        self.critic.ss =  Auto_ss(f1_dim=f1_dim, f2_dim=f2_dim,
                                  x_dim=x_dim + theta_dim, #---> This is the only change
                                  aggregation=aggregation, x_length=x_length,
                                   factor=factor, f1_layers =f1_layers,
                                   common_factor=common_factor,common_layers = common_layers,device=device)#
        self.generator.ss =  Auto_ss(f1_dim=f1_dim, f2_dim=f2_dim,
                                     x_dim=x_dim+z_dim, #---> This is the only change
                                     aggregation=aggregation, x_length=x_length,
                                    factor=factor, f1_layers =f1_layers,
                                   common_factor=common_factor,common_layers = common_layers,device=device)
        # Override the forward methods
        def critic_forward_closure(x, context):
            return self.mixed_forward_critic(x, context)

        def generator_forward_closure(context, noise=None):
            return self.mixed_forward_gen(context, noise)

        self.generator.forward = generator_forward_closure #self.mixed_forward_gen.__get__(self.generator, Generator)
        self.critic.forward = critic_forward_closure #self.mixed_forward_critic.__get__(self.critic, Critic)

    def mixed_forward_gen(self, context, noise=None):
        if noise is None:
            noise = torch.randn(context.size(0), self.generator.d_noise).to(context.device)

        reshaped_noise = noise.unsqueeze(1).expand(context.shape[0], context.shape[1], self.generator.d_noise)
        #set_trace()
        f_context = self.generator.ss(torch.cat([context, reshaped_noise], dim=-1))
        return super(type(self.generator), self.generator).forward(f_context, noise)
    
        #return super(DBGAN.generator, self.generator).forward(f_context, noise)

    def mixed_forward_critic(self, x, context):
        reshaped_x = x.unsqueeze(1).expand(context.shape[0], context.shape[1], -1)
        f_context = self.critic.ss(torch.cat([context, reshaped_x], dim=-1))

        return super(type(self.critic), self.critic).forward(x, f_context)


class DBGAN_mix_lotka(DBGAN):
    '''
        Assumption:
            The data X: batch_size x n_rep x x_length x x_dim
            The parameter: batch_size x theta_dim
        The reason I code this new class rather than using DBGAN_mix is because of the
            dimension of the input data.
            I will make a Deepset for LSTM.
    '''

    def __init__(self, simulator, theta_dim, x_dim, x_length, z_dim=None,
                 f2_dim=5,
                 device="cuda", epoch=150, batch_size=200,
                 seed=1234, d_hidden=128,
                 critic_lr=0.001, generator_lr=0.001, aggregation=True,
                 factor=64, f1_layers=3,
                 common_factor=64, common_layers=3, hidden_layers=3,
                 *args, **kwargs):
        # Properly initialize the parent class (DBGAN)
        if z_dim is None:
            z_dim = theta_dim

        super().__init__(simulator, theta_dim, x_dim, x_length, z_dim=z_dim,
                         f1_dim=0, f2_dim=f2_dim, device=device, epoch=epoch,
                         batch_size=batch_size, seed=seed, d_hidden=d_hidden,
                         critic_lr=critic_lr, generator_lr=generator_lr,
                         aggregation=aggregation, factor=factor, f1_layers=f1_layers,
                         common_factor=common_factor, common_layers=common_layers,
                         hidden_layers=hidden_layers, *args, **kwargs)

        self.critic.ss = LSTM_DeepSets(x_dim + theta_dim, dim_ss = f2_dim,
                                        nextnet_factor=16, nextnet_layers=2,
                                        common_factor=512, common_layers = 1,
                                        device=device)

        self.generator.ss =LSTM_DeepSets(x_dim + z_dim, dim_ss = f2_dim,
                                        nextnet_factor=16, nextnet_layers=2,
                                        common_factor=512, common_layers = 1,
                                        device=device)
        # Override the forward methods
        self.generator.forward = self.mixed_forward_gen.__get__(self.generator, Generator)
        self.critic.forward = self.mixed_forward_critic.__get__(self.critic, Critic)

    def mixed_forward_gen(self, context, noise=None):
        if noise is None:
            noise = torch.randn(context.size(0), self.generator.d_noise).to(context.device)

        #reshaped_noise = noise.unsqueeze(1).expand(context.shape[0], context.shape[1], self.generator.d_noise)
        reshaped_noise = noise.unsqueeze(1).unsqueeze(1).expand(context.shape[0],
                                                                context.shape[1],
                                                                context.shape[2],
                                                                self.generator.d_noise)
        f_context = self.generator.ss(torch.cat([context, reshaped_noise], dim=-1))
        return super(type(self.generator), self.generator).forward(f_context, noise)

    def mixed_forward_critic(self, x, context):
        #reshaped_x = x.unsqueeze(1).unsqueeze(1).expand(context.shape[0], context.shape[1], -1)
        reshaped_x = x.unsqueeze(1).unsqueeze(1).expand(context.shape[0],
                                                             context.shape[1],
                                                             context.shape[2], -1)
        f_context = self.critic.ss(torch.cat([context, reshaped_x], dim=-1))

        return super(type(self.critic), self.critic).forward(x, f_context)
