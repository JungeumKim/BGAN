import numpy as np
import torch
import random
from IPython.core.debugger import set_trace

import numpy as np


import numpy as np



def integrate(a0=[0.01],
              b0=[0.5],
              c0=[1],
              d0=[0.01],
              x0=50, y0=100,
              timestep=0.1, np_random=None, seed=1234, n_steps=200):
    # Ensure input consistency
    assert len(a0) == len(b0) == len(c0) == len(d0)
    b_size = len(a0)  # Number of independent simulations (batch size)

    # Setup random number generator
    if np_random is None:
        np_random = np.random.RandomState(seed)

    # Discrete time points
    time_discrete = np.arange(0, (n_steps + 1) * timestep, timestep)

    # Initialize variables
    X = [np.repeat(x0, b_size).astype(np.float32)]  # X populations
    Y = [np.repeat(y0, b_size).astype(np.float32)]  # Y populations
    times = [np.repeat(0.0, b_size).astype(np.float32)]  # Time tracker for each batch
    spoils = np.repeat(False, b_size)  # Spoiling conditions, per batch

    for i in range(20000):  # Allow up to 20000 steps
        # Stop when all batches are either spoiled or done
        if np.all(times[-1] >= time_discrete[-1]) or np.all(spoils):
            break

        # Compute reaction rates for all batches
        rates = np.column_stack([
            a0 * X[-1] * Y[-1],  # Reaction 1
            b0 * X[-1],  # Reaction 2
            c0 * Y[-1],  # Reaction 3
            d0 * X[-1] * Y[-1]  # Reaction 4
        ])
        rate = rates.sum(axis=1)  # Total rates for each batch

        # Mark spoiled batches (rate == 0)
        spoils = spoils | (rate == 0)

        # Compute next times (skip spoiled batches)
        next_times = times[-1] + np.where(~spoils, np_random.exponential(1 / rate, size=b_size), 0)
        times.append(next_times)

        # Compute reaction probabilities and sample actions
        probabilities = np.where(rate[:, None] > 0,
                                 rates / rate[:, None], 0)  # Normalize rates for valid batches
        actions = np.array([
            np_random.choice([1, 2, 3, 4], p=p) if r > 0 else 0
            for r, p in zip(rate, probabilities)
        ])  # Action = 0 for spoiled batches

        # Update populations X and Y based on actions
        x_new = X[-1] + (actions == 1) - (actions == 2)  # Increment for action 1, decrement for action 2
        y_new = Y[-1] + (actions == 3) - (actions == 4)  # Increment for action 3, decrement for action 4

        # Keep X and Y unchanged for spoiled batches
        x_new = np.where(spoils, X[-1], x_new)
        y_new = np.where(spoils, Y[-1], y_new)

        X.append(x_new)
        Y.append(y_new)

    # Stack results into 2D arrays
    X = np.stack(X, axis=0)  # (steps, batches)
    Y = np.stack(Y, axis=0)  # (steps, batches)
    times = np.stack(times, axis=0)  # (steps, batches)

    # Impute values to match `time_discrete`
    indices = []  # Collect indices for each batch
    for batch in range(times.shape[1]):  # Loop over each batch
        indices.append(np.searchsorted(times[:, batch], time_discrete, side="right") - 1)
    indices = np.array(indices)
    indices = np.clip(indices, 0, len(times) - 1)  # Clip to valid range

    # Extract final interpolated results for X, Y, and times
    X_final = np.array([X[indices[batch], batch] for batch in range(len(indices))])
    Y_final = np.array([Y[indices[batch], batch] for batch in range(len(indices))])
    T_final = np.array([times[indices[batch], batch] for batch in range(len(indices))])

    # Return result: (batch_size x 3 x len(time_discrete))
    return np.stack((X_final, Y_final, T_final), axis=1)

def simulate(batch_size = 100,np_random=None, seed=1234,x0=50, y0=100,
            n_steps=200, n_iid=1, as_torch = False, device="cpu"):
    if np_random is None:
        np_random = np.random.RandomState(seed)

    a = np_random.uniform(low=0, high=0.1, size=batch_size)
    b = np_random.uniform(low=0, high=1, size=batch_size)
    c = np_random.uniform(low=0, high=0.2, size=batch_size)
    d = np_random.uniform(low=0, high=0.1, size=batch_size)

    x = integrate(a0 = np.repeat(a, n_iid), # aaa,a'a'a', a''a''a'', a'''a'''a''' if n_iid = 3 and batch_size=4
                    b0 = np.repeat(b, n_iid),
                    c0 = np.repeat(c, n_iid),
                    d0 = np.repeat(d, n_iid),
                    x0=x0, y0=y0,
                    timestep=0.1, np_random=np_random,  n_steps=n_steps)
    
    thetas = np.column_stack([a,b,c,d])
    #set_trace()
    # x,y: from n_iid*batch_size x (n_steps+1) to batch_size x n_iid x 2 x (n_steps+1)
    # x.reshape(n_iid,batch_size,-1) makes it as n_iid x batch_size x 2 x (n_steps+1), and so need to rotate.
   
    x = x.reshape(batch_size,n_iid,3, -1)
    if as_torch: 
        x = torch.tensor(x).float().to(device)
    return thetas, x# data
