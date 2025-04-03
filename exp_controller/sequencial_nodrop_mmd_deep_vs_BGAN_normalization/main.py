from IPython.core.debugger import set_trace
import pandas as pd
import pathlib
from os.path import join
import argparse
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning)

def parse_one():
    #Arguments:
    parser = argparse.ArgumentParser(description='Arguments.')
    parser.add_argument('--exp_dir', default ="../configs/nat0.json")
    exp_dir, lest = parser.parse_known_args()
    return exp_dir, lest

def plot(self, path, true, observed_data, deepset_model=True):
    n_row,n_col = 3,1
    
    fig,axis = plt.subplots(n_row, n_col, figsize=(6*n_col,4*n_row))#, sharex=True, sharey=True)

    if not deepset_model: 
        sample = self.sampler(observed_data.reshape(1,-1), 300, 
                            shaper = lambda X: torch.from_numpy(X).float().repeat(300, 1))
    else:    
        sample = self.sampler(observed_data.reshape(1,-1,1), 300, 
                            shaper = lambda X: torch.from_numpy(X).float().repeat(300, 1,1))
    
    sample += torch.randn(size=sample.shape)*0.0001 

    ax = axis[0] 
    sns.kdeplot(x=true[:,0], y=true[:,1], ax=ax, fill=True, 
                color="orange", alpha= 0.3) 
    sns.kdeplot(x=sample[:,0], y=sample[:,1], ax=ax, fill=False)
    
    ax = axis[1] 
    pd.DataFrame(self.qualities).plot(y="mmd",ax = ax)
    
    ax = axis[2] 
    pd.DataFrame(self.qualities).plot(y="loss",ax = ax)

    fig.savefig(path)
    

def main(args):
    print("Input arguments:")
    for key, val in vars(args).items(): print("{:16} {}".format(key, val))

    ID_ = F"{args.method}_n_{args.n_sample}_lr_{args.lr}"
    
    net_path = join(args.exp_dir,F"results/trained_nets/{ID_}")
    
    pathlib.Path(net_path).mkdir(parents=True, exist_ok=True)
    
    HPARAM = {"nu":args.nu, "sigma0_sq":args.sigma0_sq, 
              "mu0":args.mu0,"kappa":args.kappa}


    x_obs=1 
    
    


    
    def simulator(batch_size, np_random=None):
        n_samples=args.n_sample
        h_param = HPARAM
        Theta,X =  forward_sampler(n = n_samples, 
                        batch_size=batch_size,
                        h_param=h_param, as_torch = True, np_random=np_random)

        return Theta, X.unsqueeze(-1)
    
    t,x = simulator(30000)
    
    normalization={"x_mean":x.mean(0).to(args.device),
                   "x_std":x.std(0).to(args.device)+10**(-20),
                   "theta_mean":t.mean(0).to(args.device),
                   "theta_std":t.std(0).to(args.device)+10**(-20)}


    if args.n_sample ==2:
        n_round = 32
    elif args.n_sample ==4:
        n_round = 16
    else:
        n_round =8

    chup_observed_data=np.array([x_obs for _ in range(args.n_sample)])
    chup_observed_data= np.expand_dims(chup_observed_data, axis=(0,-1))
    observed_data=np.array([x_obs for _ in range(args.n_sample)])
    
    for j in range(n_round):

        method = BGAN(simulator=simulator, 
                       epoch=args.epoch,
                       x_dim = 1,
                       x_length = args.n_sample,
                       theta_dim=2,n_iter=100,
                       batch_size = args.batch_size,
                       critic_lr = args.lr, generator_lr = args.lr,
                       f1_dim =2, 
                       f2_dim=0,
                       normalize=normalization)
        new_observed_data=np.array([x_obs for _ in range(args.n_sample*(j+1))])
        theta, sigma_sq = posterior_sampler(X = new_observed_data,
                                                batch_size=300, 
                                                h_param = HPARAM)
        true = np.stack([theta,sigma_sq],1)
        x_true = torch.tensor(np.repeat(observed_data.reshape(1, args.n_sample*(1), 1), 300, axis=0), 
                              dtype=torch.float).cuda()


        if "deep" not in args.method:
            method.train(true_x=x_true.squeeze(-1), true_thetas=true)
        else:
            method.train(true_x=x_true, true_thetas=true)

        method.save(net_path+f"/n{args.n_sample*(j+1)}.net")
        torch.save(method.qualities, net_path+f"/n{args.n_sample*(j+1)}.qualities")

        print("model saved at")
        print(net_path+f"/n{args.n_sample*(j+1)}.net")

        plot(method,net_path+f"/n{args.n_sample*(j+1)}.png",true, 
             observed_data,  deepset_model=("deep" in args.method))

        previous_method = method
        #previous_observed_data = torch.tensor(observed_data).unsqueeze(-1)

        if "deep" in args.method:
            def shaper(x):
                return torch.from_numpy(x).float().view(1, -1,1).repeat(300, 1,1).cuda()
        else: #regular
            def shaper(x):
                return torch.from_numpy(x).float().view(1, -1).repeat(300, 1).cuda()

        def simulator(batch_size, np_random=None):
            h_param = HPARAM
            thetas = previous_method.sampler(chup_observed_data,300,
                                             shaper = shaper)
            X = X_forward_sampler(thetas, 
                          n=args.n_sample, 
                          np_random = np_random,
                          as_torch = True)

            return thetas, X.unsqueeze(-1)

        t,x = simulator(500)
        normalization={"x_mean":x.mean(0).to(args.device),
                       "x_std":x.std(0).to(args.device)+10**(-20),
                       "theta_mean":t.mean(0).to(args.device),
                       "theta_std":t.std(0).to(args.device)+10**(-20)}
            
if __name__ == '__main__':

    args_one, args_lest = parse_one()
    sys.path.append(args_one.exp_dir)

    from config import parse_args

    args = parse_args(args_lest, name_space=args_one)

    sys.path.insert(0, args.worktree)
        
    if args.method=="deepset": 
        from BGAN_deep_critic import DBGAN_mix as BGAN
        
    else:
        from BGAN import BGAN

    from _data.gaussian_conjugate import posterior_sampler, forward_sampler, X_forward_sampler

    
    main(args)

