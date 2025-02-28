import numpy as np
from pygments.styles import vs
from scipy.special.cython_special import gdtrc
from scipy.stats import truncnorm
import torch
import glob
import pickle


def generate_task(n_parallel, num_steps, task="conditioning", return_full=False):
    assert task in [
        "dependent_continuous",
        "independent_continuous",
    ]
    if task == "dependent_continuous":
        m0 = 0.500
        v0 = 0.150 ** 2
        vs = 0.200 ** 2
        vd = 0.065 ** 2
        gd = 0.080
        mean_r = np.zeros([num_steps, n_parallel])
        a, b = (0 - m0) / np.sqrt(v0), (1 - m0) / np.sqrt(v0)
        mean_r[0] = truncnorm.rvs(a, b, loc=m0, scale=np.sqrt(v0))
        for i in range(1, num_steps):
            a, b = (0 - mean_r[i - 1]) / np.sqrt(vd), (1 - mean_r[i - 1]) / np.sqrt(vd)
            mean_r[i] = truncnorm.rvs(a, b, loc=mean_r[i - 1], scale=np.sqrt(vd))
            mean_r[i] = mean_r[i] * (1 - gd) + 0.5 * gd
        rewards = np.zeros([num_steps, n_parallel, 2])
        a, b = (0 - mean_r) / np.sqrt(vs), (1 - mean_r) / np.sqrt(vs)
        rewards[:, :, 0] = truncnorm.rvs(a, b, loc=mean_r, scale=np.sqrt(vs))
        rewards[:, :, 1] = 1 - rewards[:, :, 0]
        proba_r = None
    elif task == "independent_continuous":
        m0 = 0.500
        v0 = 0.250 ** 2
        vs = 0.200 ** 2
        vd = 0.100 ** 2
        gd = 0.050
        rewards = np.zeros([num_steps, n_parallel, 2])
        for i_arm in range(2):
            mean_r = np.zeros([num_steps, n_parallel])
            a, b = (0 - m0) / np.sqrt(v0), (1 - m0) / np.sqrt(v0)
            mean_r[0] = truncnorm.rvs(a, b, loc=m0, scale=np.sqrt(v0))
            for i in range(1, num_steps):
                a, b = (0 - mean_r[i-1]) / np.sqrt(vd), (1 - mean_r[i-1]) / np.sqrt(vd)
                mean_r[i] = truncnorm.rvs(a, b, loc=mean_r[i-1], scale=np.sqrt(vd))
                mean_r[i] = mean_r[i]*(1 - gd) + 0.5*gd
            a, b = (0 - mean_r) / np.sqrt(vs), (1 - mean_r) / np.sqrt(vs)
            rewards[:, :, i_arm] = truncnorm.rvs(a, b, loc=mean_r, scale=np.sqrt(vs))
        proba_r = None
    else:
        raise NotImplementedError
    if return_full:
        return (torch.from_numpy(rewards)) * 2.0 - 1.0, proba_r
    return (torch.from_numpy(rewards)) * 2.0 - 1.0