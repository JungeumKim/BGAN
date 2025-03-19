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

    ID_ = F"{args.method}"
    
    net_path = join(args.exp_dir,F"results/trained_nets/{ID_}")
    
    pathlib.Path(net_path).mkdir(parents=True, exist_ok=True)

    Theta0 = [1, 0.01, 0.5, 0.01]
    X0 = integrate(a0=np.repeat(Theta0[0], args.n_iid),
                   # aaa,a'a'a', a''a''a'', a'''a'''a''' if n_iid = 3 and batch_size=4
                   b0=np.repeat(Theta0[1], args.n_iid),
                   c0=np.repeat(Theta0[2], args.n_iid),
                   d0=np.repeat(Theta0[3], args.n_iid),
                   x0=100, y0=50,
                   timestep=0.1, np_random=None, u_up=1.2, u_down=0.8, n_steps=args.n_steps)

    X0 = X0.transpose(0, 2, 1)
    X0 = np.expand_dims(X0, axis=0)

    X0_rep = torch.from_numpy(np.repeat(X0, repeats=150, axis=0)).float().to(args.device)
    true_thetas = np.repeat([Theta0], repeats=150, axis=0)

    def simulator(batch_size, np_random=None, device=args.device, n_steps=args.n_steps,
                  n_iid=args.n_iid):
        Thetas, x = forward_sampler(batch_size, np_random=np_random, n_steps=n_steps, n_iid=n_iid)
        Thetas = torch.from_numpy(Thetas).float().to(device)
        X = torch.from_numpy(x).float().to(device)  # X: batch_size x n_iid x 2 x (n_steps+1)
        X = X.clip(0, 10 ** 7)
        X = X.squeeze(1).permute(0, 1, -1, -2)  # X: batch_size x n_iid x (n_steps+1) x 2
        return Thetas, X

    
    for j in range(args.random_repeat):
        method = BGAN(simulator=simulator,
                                 epoch=args.epoch,
                                 x_length=args.n_steps + 1,
                                 x_dim=2,
                                 theta_dim=4,
                                 device=args.device,
                                 batch_size=args.batch_size,
                                 f2_dim=4,
                                 factor=128, f1_layers=2,
                                 common_factor=128,
                                 common_layers=2, hidden_layers=2,
                                 d_hidden=128)
        BY = 10
        for j in range(int(args.epoch/BY)):
            method.train(true_x=X0_rep, true_thetas=true_thetas, 
                         msr = "mse", n_iter = args.n_iter,
                         start_epoch=j*BY+1, end_epoch=BY*(j+1)
                         ) 
            plot(method,X0_rep, Theta0,
                 net_path+f"/net_id{j}_epoch{BY*(j+1)}.png")

        method.save(net_path+f"/net_id{j}.net")
        print("model saved at")
        print(net_path+f"/net_id{j}.net")
        
        torch.save(method.qualities, net_path+f"/net_id{j}.qualities")
    
if __name__ == '__main__':

    args_one, args_lest = parse_one()
    sys.path.append(args_one.exp_dir)

    from config import parse_args

    args = parse_args(args_lest, name_space=args_one)

    sys.path.insert(0, args.worktree)
    
    if args.method=="deepset": 
        from BGAN_deep_critic import DBGAN_mix_lotka as BGAN
        
    else:
        from BGAN import BGAN

    from _data.lotka_voltera import simulate as forward_sampler
    from _data.lotka_voltera import integrate
    
    main(args)

