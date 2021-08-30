import h5py
import numpy as np

def one_hot_array(i, n):
    return list(map(int, [ix == i for ix in range(n)]))

def many_one_hot(indices, d):
    # (t,) - indices for n documents and t timesteps
    t = indices.shape[0]
    oh = np.zeros((t, d))
    oh[np.arange(t), indices] = 1
    return oh
