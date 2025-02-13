import numpy as np
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
        mean_r = np.zeros([num_steps, n_parallel])
        mean_r[:] = np.random.choice(
            [0.05, 0.95] * (int(n_parallel / 2) if n_parallel > 1 else 1),
            size=n_parallel,
            replace=False,
        )
        std_obs_noise = 0.153
        for i in range(1, num_steps):
            a, b = (0 - mean_r[i - 1]) / 0.108, (1 - mean_r[i - 1]) / 0.108
            mean_r[i] = truncnorm.rvs(a, b, loc=mean_r[i - 1], scale=0.108)
        a, b = (0 - mean_r) / std_obs_noise, (1 - mean_r) / std_obs_noise
        rewards = np.zeros([num_steps, n_parallel, 2])
        rewards[:, :, 0] = truncnorm.rvs(a, b, loc=mean_r, scale=std_obs_noise)
        rewards[:, :, 1] = 1 - rewards[:, :, 0]
        proba_r = None
    elif task == "independent_continuous":        
        rewards = np.zeros([num_steps, n_parallel, 2])
        for i_arm in range(2):
            mean_r = np.zeros([num_steps, n_parallel])
            mean_r[:] = np.random.choice(
                [0.05, 0.95] * (int(n_parallel / 2) if n_parallel > 1 else 1),
                size=n_parallel,
                replace=False,
            )
            std_obs_noise = 0.153
            for i in range(1, num_steps):
                a, b = (0 - mean_r[i - 1]) / 0.108, (1 - mean_r[i - 1]) / 0.108
                mean_r[i] = truncnorm.rvs(a, b, loc=mean_r[i - 1], scale=0.108)
            a, b = (0 - mean_r) / std_obs_noise, (1 - mean_r) / std_obs_noise
            rewards[:, :, i_arm] = truncnorm.rvs(a, b, loc=mean_r, scale=std_obs_noise)
    else:
        raise NotImplementedError
    if return_full:
        return (torch.from_numpy(rewards)) * 2.0 - 1.0, proba_r
    return (torch.from_numpy(rewards)) * 2.0 - 1.0