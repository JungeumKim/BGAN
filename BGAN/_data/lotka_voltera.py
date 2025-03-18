import numpy as np
import torch
from IPython.core.debugger import set_trace

def integrate(a0 = [1],
             b0 = [0.01],
             c0 = [0.5],
             d0 = [0.01],
             x0=100, y0=50,
             timestep=0.1, np_random=None, seed=1234, u_up=1.2, u_down=0.8, n_steps=200):


    if np_random is None:
        np_random = np.random.RandomState(seed)
    '''
    It considers: t = np.linspace(0,20,num=201) 
    
    x0: prey, y0: predator
    
    Birth rate of rabbits
    a0 = 1 #theta3: 1 U[0,2]
    
    Death rate of rabbits due to predation
    b0 = 0.01 #theta4: 0.01 U[0,0.1]
    
    Natural death rate of foxes
    c0 = 0.5 #theta2: 0.5 U[0,1]
    
    Factor that describes how many eaten rabbits give birth to a new fox
    d0 = 0.01 #theta1: 0.01 U[0,0.1]
    '''
    batch_size = len(a0)

    x_prev = np.ones(batch_size)*x0
    y_prev = np.ones(batch_size)*y0

    x=[x_prev] #Prey
    y=[y_prev] #Predator

    for _ in range(n_steps):
        a= a0*np_random.uniform(low=u_down, high=u_up, size=batch_size)
        b= b0*np_random.uniform(low=u_down, high=u_up, size=batch_size)
        c= c0*np_random.uniform(low=u_down, high=u_up, size=batch_size)
        d= d0*np_random.uniform(low=u_down, high=u_up, size=batch_size)

        # evaluate the current differentials
        xd = x_prev  * (a - b*y_prev)
        yd = -y_prev * (c - d*x_prev)

        # add the next value of x and y using differentials
        x_new = x_prev + xd * timestep
        y_new = y_prev + yd * timestep

        x.append(x_new)
        y.append(y_new)

        x_prev, y_prev = x_new,y_new

    x,y = np.column_stack(x),np.column_stack(y) 
    return np.stack((x,y), axis=1) # len(a0) x 2 x 201

def simulate(batch_size = 100,np_random=None, seed=1234,x0=100, y0=50,
             u_up=1.2, u_down=0.8,n_steps=200, n_iid=1, as_torch = False, device="cpu"):
    if np_random is None:
        np_random = np.random.RandomState(seed)

    a = np_random.uniform(low=0, high=2, size=batch_size)
    b = np_random.uniform(low=0, high=0.1, size=batch_size)
    c = np_random.uniform(low=0, high=1, size=batch_size)
    d = np_random.uniform(low=0, high=0.1, size=batch_size)

    x = integrate(a0 = np.repeat(a, n_iid), # aaa,a'a'a', a''a''a'', a'''a'''a''' if n_iid = 3 and batch_size=4
                    b0 = np.repeat(a, n_iid),
                    c0 = np.repeat(a, n_iid), 
                    d0 = np.repeat(a, n_iid), 
                    x0=x0, y0=y0,
                    timestep=0.1, np_random=np_random, u_up=u_up, u_down=u_down, n_steps=n_steps)
    
    thetas = np.column_stack([a,b,c,d])
    
    # x,y: from n_iid*batch_size x (n_steps+1) to batch_size x n_iid x 2 x (n_steps+1)
    # x.reshape(n_iid,batch_size,-1) makes it as n_iid x batch_size x 2 x (n_steps+1), and so need to rotate.
   
    x = x.reshape(batch_size,n_iid,2, -1)
    if as_torch: 
        x = torch.tensor(x).float().to(device)
    return thetas, x# data
