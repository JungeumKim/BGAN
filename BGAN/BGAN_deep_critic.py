"""
This is code for  Adversarial Bayesian Simulation (Yuexi Wang, Veronika Rockova), developed and implemented by Jungeum Kim to adopt DeepSet with a regularizer.
"""

from IPython.core.debugger import set_trace

from BGAN import Critic as BGAN_CRITIC
from BGAN import Generator as BGAN_GENERATOR
from _nets.DeepSet_nets import Auto_ss
from BGAN import BGAN


class Critic(BGAN_CRITIC):


    def __init__(self,theta_dim=2,x_dim=2,  f1_dim=2,f2_dim=2,
                 d_hidden = [128,128,128], leaky=0.1, aggregation=True,x_length=None, 
                 factor=64, f1_layers =3,
                 common_factor=64,common_layers = 3,
                 ):
        super().__init__(input_dim = theta_dim,
                         cond_dim = f1_dim + f2_dim,
                         d_hidden = d_hidden
                         )
        self.ss = Auto_ss(f1_dim=f1_dim, f2_dim=f2_dim, x_dim=x_dim, aggregation=aggregation, x_length=x_length,
                                   factor=factor, f1_layers =f1_layers,
                                   common_factor=common_factor,common_layers = common_layers
                                   )

    def forward(self, x, context):

        f_context = self.ss(context)# .unsqueeze(-1))
        return super().forward(x, f_context)

class Generator(BGAN_GENERATOR):

    def __init__(self, x_dim=2, theta_dim = 2,
                f1_dim=2,f2_dim=2, leaky=0.1,x_length=None, 
                 d_hidden = [128,128,128], aggregation=True,
                factor=64, f1_layers =3,
                 common_factor=64,common_layers = 3
                 ):
        
        super().__init__(d_hidden=d_hidden,
                         theta_dim=theta_dim,
                         z_dim = theta_dim,
                         cond_dim = f1_dim + f2_dim,
                         leaky=leaky)
        
        self.d_cond = f1_dim + f2_dim
        
        self.ss = Auto_ss(f1_dim=f1_dim, f2_dim=f2_dim, x_dim=x_dim, aggregation=aggregation, x_length=x_length,
                           factor=factor, f1_layers =f1_layers,
                                   common_factor=common_factor,common_layers = common_layers
                                    )


    def forward(self, context, noise = None):
        f_context = self.ss(context)
        return super().forward(f_context, noise)




class DBGAN(BGAN):

    def __init__(self, simulator, theta_dim,  x_dim, x_length,
                 f1_dim=2, f2_dim=5, 
                 device="cuda",epoch=150, batch_size = 200, 
                 seed=1234, d_hidden = 128,
                 critic_lr = 0.001, generator_lr = 0.001,aggregation=True,
                 factor=64, f1_layers =3,
                 common_factor=64,common_layers = 3,hidden_layers=3,
                 *args, **kwargs):
        super().__init__(simulator, theta_dim, x_dim, x_length,
                 device=device,epoch=epoch, batch_size = batch_size, d_hidden=d_hidden,
                 critic_lr=critic_lr, generator_lr = generator_lr,
                 seed=seed)

        self.generator = Generator(x_dim,
                                   theta_dim = theta_dim,
                                   f1_dim=f1_dim, 
                                   f2_dim=f2_dim,
                                   d_hidden = [d_hidden for _ in (hidden_layers)],
                                   aggregation=aggregation,
                                   x_length=x_length,
                                   factor=factor, f1_layers =f1_layers,
                                   common_factor=common_factor,common_layers = common_layers
                                   )
        
        self.critic = Critic(
                             theta_dim=theta_dim,
                             f1_dim=f1_dim,
                             f2_dim=f2_dim,
                             x_dim = x_dim,
                             d_hidden = [d_hidden  for _ in (hidden_layers)],
                             aggregation=aggregation,
                             x_length=x_length,
                             factor=factor, f1_layers =f1_layers,
                             common_factor=common_factor,common_layers = common_layers)
        self.generator.to(device)        
        self.critic.to(device)
