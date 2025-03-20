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

def plot(self, X0_rep, Theta0, save_path=None, show=False):
    n_row, n_col = 2, 4
    fig, axis = plt.subplots(n_row, n_col, figsize=(4 * n_col, 4 * n_row))#, sharex=True, sharey=True)
    # set_trace()
    with torch.no_grad():
        sample = self.generator(X0_rep.to(args.device)).to("cpu").detach().numpy()
    # sample += torch.randn(size=sample.shape)*0.00001
    for i in range(4):
        ax = axis[0,i]
        sns.kdeplot(x=sample[:, i], ax=ax, fill=True, color="blue")
        ax.axvline(x=Theta0[i], color='red', linestyle='--', linewidth=2)
    #set_trace()
    pd.DataFrame(self.qualities).plot(y="mse", ax=axis[1,0])
    pd.DataFrame(self.qualities).plot(y="loss", ax=axis[1,1])
    if show:
        plt.show()
    else:
        plt.savefig(save_path)  # Save the plot to the specified path
    

def main(args):
    print("Input arguments:")
    for key, val in vars(args).items(): print("{:16} {}".format(key, val)) 
    
    Theta0 = [1,0.01,0.5,0.01]
    X0 = integrate(a0 = [1],
                 b0 = [0.01],
                 c0 = [0.5],
                 d0 = [0.01],
                 np_random=None, seed=1234, 
                 n_steps=args.n_steps)

    X0 = X0.transpose(0, 2, 1)  # Rearrange the dimensions

    X0_rep = torch.from_numpy(np.repeat(X0, repeats=150, 
                                        axis=0)).float().to(args.device)
    true_thetas = np.repeat([Theta0], repeats=150, axis=0)


    def simulator(batch_size, np_random=None, device = args.device, 
                  n_steps = args.n_steps):
        Thetas, x = forward_sampler(batch_size,np_random=np_random, 
                                    n_steps = n_steps)
        Thetas = torch.from_numpy(Thetas).float().to(device)
        X = torch.from_numpy(x).float().to(device)
        X = X.clip(0,10**7)
        X = X.permute(0, 3, 2,1).squeeze(-1)
        return Thetas, X
    
    #for rp in range(args.random_repeat):
    rp = args.repeat_id
    np.random.seed(args.seed *(rp+1))
    torch.manual_seed(args.seed *(rp+1))

    net_path = join(args.exp_dir,F"results/{args.method}/net_id{rp}/")

    pathlib.Path(net_path).mkdir(parents=True, exist_ok=True)

    method = BGAN(simulator=simulator,
          epoch=args.epoch,
          x_length = args.n_steps+1,
          x_dim = 2,
          theta_dim=4,
          z_dim=4,
          device=args.device,
                  
          critic_lr=args.lr, generator_lr = args.lr,

          batch_size = args.batch_size,
          f1_dim =3, f2_dim=4,
          factor=128, f1_layers =2,
          common_factor=128,
          common_layers = 2,hidden_layers=2,
          d_hidden=128

          )

    BY = 10
    for j in range(int(args.epoch/BY)):
        method.train(true_x=X0_rep, true_thetas=true_thetas, 
                     msr = "mse", n_iter = args.n_iter,
                     start_epoch=j*BY+1, end_epoch=BY*(j+1)
                     ) 
        plot(method,X0_rep, Theta0,
             net_path+f"epoch{BY*(j+1)}.png")
        if j % 10 == 0 and j != 0:
            method.save(net_path+f"epoch{BY*(j+1)}.net")
    method.save(net_path+f"net.net")
    print("model saved at")
    print(net_path+f"net.net")

    torch.save(method.qualities, net_path+f"qualities.dat")
    
if __name__ == '__main__':

    args_one, args_lest = parse_one()
    sys.path.append(args_one.exp_dir)

    from config import parse_args

    args = parse_args(args_lest, name_space=args_one)

    sys.path.insert(0, args.worktree)
    
    if args.method=="deepset": 
        from BGAN_deep_critic import DBGAN_mix as BGAN
        #torch.backends.cudnn.enabled = False #since f2_dim>0: i.e., LSTM is used.
        
    else:
        from BGAN import BGAN

    from _data.lotka_voltera import simulate as forward_sampler
    from _data.lotka_voltera import integrate
    
    main(args)

