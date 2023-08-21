import torch
import numpy as np
from torch.optim import Adam, RMSprop, LBFGS

def systematic_resampling(logweights):
    nb_particles = logweights.shape[0]
    unif = torch.rand(1) + torch.arange(nb_particles)
    weights = torch.exp(logweights - logweights.max())
    cumsum = weights.cumsum(0)
    V = cumsum/cumsum[-1] * nb_particles
    idx = (unif[None] > V[:, None]).sum(axis=0)
    if torch.any(idx >= nb_particles):
        raise AssertionError
    return idx

def evaluate_rl(a, r, alpha, beta, zeta):
    if len(a.shape) != 1 or len(r.shape) != 1:
        raise AssertionError('action or rewards do not have the correct shape')
    nb_particles = 1000
    nb_trials = a.shape[0]
    Q_values = torch.zeros([nb_particles, 2]) + 0.5
    loglkd = 0
    for i_trial in range(nb_trials):
        logpolicy = torch.nn.LogSoftmax(dim=-1)(Q_values * beta)
        loglkd += torch.logsumexp(logpolicy[:, a[i_trial]], 0) - np.log(nb_particles)
        if torch.isnan(loglkd):
            assert False
        idx = systematic_resampling(logpolicy[:, a[i_trial]])
        Q_values = Q_values[idx]
        all_r = r[i_trial] * (1 - a[i_trial]) + 0.5 * a[i_trial], 0.5 * (1 - a[i_trial]) + r[i_trial] * a[i_trial]
        PE = torch.FloatTensor(all_r).unsqueeze(0) - Q_values
        Q_values = torch.clip(Q_values + alpha * PE + zeta * torch.randn(size=(nb_particles, 2)) * PE.abs(),
                              -1e30, 1e30)
    return loglkd

def fit_rl_gradient_descent(a, r):
    alpha = torch.nn.Parameter(torch.zeros(1))
    beta = torch.nn.Parameter(torch.ones(1))
    zeta = torch.nn.Parameter(torch.ones(1) * 1e-6)

    optimizer = LBFGS([alpha, beta, zeta], lr=1e-1, line_search_fn="strong_wolfe")

    current_loss = np.inf
    counter, nb_iterations = 0, 0

    # L-BFGS
    def closure():
        optimizer.zero_grad()
        objective = -evaluate_rl(a, r, torch.sigmoid(alpha), beta.abs(), zeta.abs())
        objective.backward()
        return objective

    while True:
        loss = optimizer.step(closure)

        if torch.isnan(loss) or torch.isnan(torch.sigmoid(alpha.detach())) or torch.isnan(beta.abs().detach()) or torch.isnan(zeta.abs().detach()):
            raise AssertionError

        if loss < current_loss:
            current_loss = loss.detach()
            counter = 0
        else:
            counter += 1

        if counter == 5:
            break

        nb_iterations += 1

        print(f'nb_iterations are {nb_iterations}')
        print(f'loss is {current_loss}')
        print(f'fitted parameters are '
              f'alpha={torch.sigmoid(alpha.detach())}, '
              f'beta={beta.abs().detach()}, '
              f'zeta={zeta.abs().detach()}')

    return torch.sigmoid(alpha), beta.abs(), zeta.abs()
