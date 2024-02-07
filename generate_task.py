import numpy as np
from scipy.stats import truncnorm
import torch


def generate_task(n_parallel, num_steps, task="conditioning", return_full=False):
    assert task in ["conditioning", "restless_continuous", "behrens", "behrens_legacy"]
    if task == "restless_continuous":
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
    elif task == "conditioning":
        k = 0.05
        rand_int = np.random.randint(2)
        proba_r = np.zeros(num_steps)
        proba_r[:] = (k) * (rand_int == 0) + (1.0 - k) * (rand_int == 1)
        rewards = np.zeros([num_steps, 2])
        while True:
            random_numb = np.random.rand(num_steps)
            rewards[:, 0] = (proba_r < random_numb) * 1.0
            rewards[:, 1] = (proba_r >= random_numb) * 1.0
            if (
                np.abs(
                    rewards[:, 1].mean()
                    - ((0.0 + k) * (rand_int == 0) + (1.0 - k) * (rand_int == 1))
                )
                < 0.01
            ):
                break
        rewards = rewards[:, None]
    elif task == "behrens_legacy":
        proba_r = np.zeros([num_steps, n_parallel])
        proba_r[:] = np.random.choice(
            [0.2, 0.8, -0.2, -0.8] * (int(n_parallel / 4) if n_parallel > 1 else 1),
            size=n_parallel,
            replace=False,
        )
        with_volatility = np.where(proba_r[0] < 0)[0]
        proba_r = np.abs(proba_r)
        for i in range(1, num_steps):
            if i % 25 == 0:
                proba_r[i, with_volatility] = 1 - proba_r[i - 1, with_volatility]
            else:
                proba_r[i, with_volatility] = proba_r[i - 1, with_volatility]

        while True:
            random_numb = np.random.rand(num_steps, n_parallel)
            rewards = np.zeros([num_steps, n_parallel, 2])
            rewards[:, :, 0] = (proba_r < random_numb) * 1.0
            rewards[:, :, 1] = (proba_r >= random_numb) * 1.0
            if (
                np.abs(
                    (
                        rewards[:, ~with_volatility, 0]
                        == (proba_r[:, ~with_volatility] > 0.5)
                    ).mean()
                )
                - 0.2
                < 0.001
            ) and (
                np.abs(
                    (
                        rewards[:, with_volatility, 0]
                        == (proba_r[:, with_volatility] > 0.5)
                    ).mean()
                )
                - 0.2
                < 0.001
            ):
                break
    elif task == "behrens":
        vol_at_trial0 = np.random.choice(
            [0, 0.04] * (int(n_parallel / 2) if n_parallel > 1 else 1),
            size=n_parallel,
            replace=False,
        )
        vols = np.vstack(
            (
                np.tile(vol_at_trial0[None], (int(num_steps / 2), 1)),
                np.tile(0.04 - vol_at_trial0[None], (int(num_steps / 2), 1)),
            )
        )
        proba_r = np.zeros([num_steps, n_parallel])
        proba_r[0] = np.random.choice(
            [0.2, 0.8] * (int(n_parallel / 2) if n_parallel > 1 else 1),
            size=n_parallel,
            replace=False,
        )
        for j in range(n_parallel):
            while True:
                for i in range(1, num_steps):
                    randn = np.random.rand()
                    proba_r[i, j] = proba_r[i - 1, j] * (randn > vols[i, j]) + (
                        1 - proba_r[i - 1, j]
                    ) * (randn <= vols[i, j])

                nb_expected_reversals = (vols[:, j] == 0.04).sum() * 0.04
                if np.all(
                    (proba_r[1:, j] != proba_r[:-1, j]).sum() == nb_expected_reversals
                ):
                    break

        rewards = np.zeros([num_steps, n_parallel, 2])
        for j in range(n_parallel):
            rew_ = np.zeros([num_steps, 2])
            while True:
                random_numb = np.random.rand(num_steps)
                rew_[:, 0] = (proba_r[:, j] < random_numb) * 1.0
                rew_[:, 1] = (proba_r[:, j] >= random_numb) * 1.0
                if np.all(
                    np.abs((rew_[:, 0] == (proba_r[:, j] > 0.5)).mean(axis=0) - 0.2)
                    < 0.01
                ):
                    break
            rewards[:, j] = rew_
    else:
        raise NotImplementedError
    if return_full:
        return (torch.from_numpy(rewards)) * 2.0 - 1.0, proba_r
    return (torch.from_numpy(rewards)) * 2.0 - 1.0


if __name__ == "__main__":
    n_parallel = 10
    num_steps = 100
    task = "behrens"
