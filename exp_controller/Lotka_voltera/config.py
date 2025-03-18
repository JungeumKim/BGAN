import argparse
import numpy as np
import torch
import random

def parse_args(args, name_space):
    parser = argparse.ArgumentParser(description='Arguments.')
    parser.add_argument('--worktree',
                        #default= "/home/kim2712/Desktop/research/BGAN/BGAN",
                        default= "/home/kim2712/Desktop/research/BGAN/BGAN/worktrees/bgan_54d5dd0/BGAN",
                        help = "parent directory")
    parser.add_argument('--exp_dir', default = "./",  help= 'global path')
    parser.add_argument('--device', default = "cuda",  help= ' ')    
    parser.add_argument('--method', default = "deepset",  help= ' ')
    parser.add_argument('--random_repeat', type=int, default=5, help=' ')

    parser.add_argument('--epoch', type=int, default = 1000, help= ' ')
    parser.add_argument('--n_iter', type=int, default = 500, help= ' ')
    parser.add_argument('--batch_size', type=int, default = 128, help= ' ')

    parser.add_argument('--seed', type=int, default = 12345, help= ' ')
    parser.add_argument('--lr', type=float, default = 0.0001, help= ' ')
    parser.add_argument('--n_steps', type=int, default=200, help=' ')

    parser.add_argument('--factor', type=int, default = 16, help= ' ')
    parser.add_argument('--f1_layers', type=int, default = 1, help= ' ')
    parser.add_argument('--common_layers', type=int, default = 32, help= ' ')
    parser.add_argument('--common_factor', type=int, default = 2, help= ' ')    
    parser.add_argument('--hidden_layers', type=int, default = 1, help= ' ')
    parser.add_argument('--d_hidden', type=int, default = 16, help= ' ')
    

    #learning setting:
    args = parser.parse_args(args, namespace=name_space) #parser.parse_args()
    return args


