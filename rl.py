import torch
import numpy as np
from scipy.stats import norm

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

def evaluate_rl(a, r, init):
    alpha, beta, zeta = scale_parameters(init)
    if len(a.shape) != 1 or len(r.shape) != 1:
        raise AssertionError('action or rewards do not have the correct shape')
    nb_particles = 1000
    nb_trials = a.shape[0]
    Q_values = torch.zeros([nb_particles, 2])
    loglkd = 0
    for i_trial in range(nb_trials):
        logpolicy = torch.nn.LogSoftmax(dim=-1)(Q_values * beta)
        loglkd += torch.logsumexp(logpolicy[:, a[i_trial]], 0) - np.log(nb_particles)
        if torch.isnan(loglkd):
            assert False
        idx = systematic_resampling(logpolicy[:, a[i_trial]])
        Q_values = Q_values[idx]
        all_r = r[i_trial] * (1 - a[i_trial]) + 0, r[i_trial] * a[i_trial] + 0.
        PE = torch.FloatTensor(all_r).unsqueeze(0) - Q_values
        Q_values = torch.clip(Q_values + alpha * PE + zeta * torch.randn(size=(nb_particles, 2)) * PE.abs(),
                              -1e30, 1e30)
    return loglkd


from botorch.models import SingleTaskGP
import sobol_seq
from botorch.acquisition import UpperConfidenceBound, qExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils import standardize

def scale_parameters(param):
    alpha, beta, zeta = param
    return alpha, beta * 50, zeta

def bo_rl(a, r, nb_init_pts=20, nb_BO_iterations=80):
    """
    ref : https://jonathan-guerne.medium.com/multi-objective-bayesian-optimization-with-botorch-3c5cf348c63b
    """
    # tensors
    bounds = torch.tensor([[0, 1e-4, 0], [1, 1, 1]]).double()
    X = torch.vstack((
                        torch.from_numpy(sobol_seq.i4_sobol_generate(3, nb_init_pts)),
                        torch.zeros([nb_BO_iterations, 3])
    ))
    realizations = torch.zeros(nb_init_pts + nb_BO_iterations).double()
    for i_init in range(nb_init_pts):
        realizations[i_init] = evaluate_rl(a, r, X[i_init])

    for i_cand in range(nb_BO_iterations):
        normalized_realizations = standardize(realizations[:(nb_init_pts + i_cand)])
        model = SingleTaskGP(X[:(nb_init_pts + i_cand)], normalized_realizations.unsqueeze(-1))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        acquisition = qExpectedImprovement(
            model=model,
            best_f=normalized_realizations.max()
        )

        candidate, _ = optimize_acqf(
            acq_function=acquisition,
            bounds=bounds,
            q=1,
            num_restarts=200,
            raw_samples=512,
        )

        realizations[nb_init_pts + i_cand] = evaluate_rl(a, r, candidate.squeeze())
        X[nb_init_pts + i_cand] = candidate.squeeze()

print(scale_parameters(X.T))
print(scale_parameters(X[realizations.argmax()]))