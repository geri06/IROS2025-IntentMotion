import torch
import numpy as np

def find_intentions_mode(x):
    """
    Given a tensor of shape ([256, 10]) returns the mode of each 10 intentions
    in a tensor of shape ([256])
    """
    batch_intentions = []
    for sample in range(x.shape[0]):
        vals,counts = np.unique(x[sample,:], return_counts=True)
        index = np.argmax(counts)
        intention = vals[index]
        batch_intentions.append(intention)
    return torch.tensor(batch_intentions)


def get_dct_matrix(N):
    """
    Compute DCT and IDCT matrix with dim NxN to transform data
    """
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m
