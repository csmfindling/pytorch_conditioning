import torch
import numpy as np
from scipy.stats import norm

import numpy as np
from matplotlib import pyplot as plt
import glob, sys, pickle, sobol_seq, numpy
from scipy.stats import norm, truncnorm
from scipy.special import logsumexp


def rl(a, r, alphas, softs, epsis=None):
    beta_ravel = 1.0 / softs

    nb_samples = 1
    S = np.zeros(len(a), dtype=int)  # Sequence of Stimuli
    action_num = 2  # Number of Actions possible
    state_num = 1  # Number of states or stimuli
    K = np.prod(np.arange(action_num + 1)[-state_num:])  # Number of possible Task Sets
    trial_num = len(a)  # Number of Trials

    A = numpy.array(a, dtype=int)
    R = numpy.array(r, dtype=int)
    inc_log_lkd = numpy.zeros([len(alphas)])

    p_A = numpy.zeros([len(alphas), nb_samples, action_num])
    Q_values = numpy.zeros([len(alphas), nb_samples, state_num, action_num]) + 0.5
    for T in numpy.arange(trial_num):
        if T == 1:
            Q_values[:, :, int(S[T - 1]), A[T - 1]] = R[T - 1] * 1
            Q_values[:, :, int(S[T - 1]), 1 - A[T - 1]] = 1 - R[T - 1] * 1
        elif T > 1:
            ancestor_Q = np.array(Q_values)
            pe = R[T - 1] * 1 - ancestor_Q[:, :, int(S[T - 1]), A[T - 1]]
            abs_pe = np.abs(pe)
            Q_values[:, :, int(S[T - 1]), A[T - 1]] = (
                ancestor_Q[:, :, int(S[T - 1]), A[T - 1]] + (alphas[:, np.newaxis]) * pe
            )

            pe = (1 - R[T - 1] * 1) - ancestor_Q[:, :, int(S[T - 1]), 1 - A[T - 1]]
            abs_pe = np.abs(pe)
            Q_values[:, :, int(S[T - 1]), 1 - A[T - 1]] = (
                ancestor_Q[:, :, int(S[T - 1]), 1 - A[T - 1]]
                + (alphas[:, np.newaxis]) * pe
            )

        log_unnormalized_softmax = np.log(Q_values[:, :, int(S[T])]) + np.log(
            beta_ravel[:, np.newaxis, np.newaxis]
        )

        log_pA = (
            log_unnormalized_softmax
            - logsumexp(log_unnormalized_softmax, axis=-1)[:, :, np.newaxis]
        )
        if epsis is not None:
            log_pA = np.logaddexp(
                log_pA + np.log(1 - epsis[:, np.newaxis, np.newaxis]),
                np.log(epsis[:, np.newaxis, np.newaxis] / 2.0),
            )

        if np.isnan(Q_values).sum() > 0:
            assert False

        inc_log_lkd_T = log_pA[:, :, A[T]]  # np.log(p_A[:,:,A[T]])
        inc_log_lkd += logsumexp(inc_log_lkd_T, axis=1) - np.log(nb_samples)

    return inc_log_lkd


def systematic_resampling(logweights):
    nb_params, nb_particles = logweights.shape
    unif = torch.rand(nb_params)[:, None] + torch.arange(nb_particles)[None]
    weights = torch.exp(logweights - logweights.max(axis=0, keepdims=True).values)
    cumsum = weights.cumsum(1)
    V = cumsum / cumsum[:, -1].unsqueeze(-1) * nb_particles
    idx = (unif[:, None] > V[:, :, None]).sum(axis=1)
    if torch.any(idx >= nb_particles):
        raise AssertionError
    return idx


def evaluate_rl(a, r, init, nb_particles=1000):
    alpha, beta, zeta = scale_parameters(init.T)
    if len(a.shape) != 1 or len(r.shape) != 1:
        raise AssertionError("action or rewards do not have the correct shape")
    nb_params = len(alpha)
    nb_trials = a.shape[0]
    Q_values = torch.zeros([nb_params, nb_particles, 2])
    loglkd = torch.zeros(nb_params)
    for i_trial in range(nb_trials):
        logpolicy = torch.nn.LogSoftmax(dim=-1)(Q_values * beta[:, None, None])
        loglkd += torch.logsumexp(logpolicy[:, :, a[i_trial]], 1) - np.log(nb_particles)
        if torch.any(torch.isnan(loglkd)):
            assert False
        idx = systematic_resampling(logpolicy[:, :, a[i_trial]])
        Q_values = torch.stack(
            (
                torch.gather(Q_values[:, :, 0], 1, idx),
                torch.gather(Q_values[:, :, 1], 1, idx),
            ),
            axis=-1,
        )
        all_r = r[i_trial] * (1 - a[i_trial]) + 0, r[i_trial] * a[i_trial] + 0.0
        PE = torch.FloatTensor(all_r).unsqueeze(0) - Q_values
        Q_values = torch.clip(
            Q_values
            + alpha[:, None, None] * PE
            + zeta[:, None, None] * torch.randn(size=(nb_particles, 2)) * PE.abs(),
            -1e30,
            1e30,
        )
    return loglkd


'''
from botorch.models import SingleTaskGP
import sobol_seq
from botorch.acquisition import (
    UpperConfidenceBound,
    qExpectedImprovement,
    qUpperConfidenceBound,
)
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils import standardize


def scale_parameters(param):
    alpha, beta, zeta = param
    return torch.vstack([alpha, beta * 100, zeta * 1.5])


from tqdm import tqdm


def bo_rl(a, r, nb_batch=3, nb_BO_iter=10, verbose=False):
    """
    ref : https://jonathan-guerne.medium.com/multi-objective-bayesian-optimization-with-botorch-3c5cf348c63b
    """
    # tensors
    nb_init = 20
    bounds = torch.tensor([[0, 1e-4, 0], [1, 1, 1]]).double()
    X = torch.vstack(
        (
            torch.from_numpy(sobol_seq.i4_sobol_generate(3, nb_init)),
            torch.zeros([nb_batch * nb_BO_iter, 3]),
        )
    )
    realizations = torch.zeros(nb_init + nb_batch * nb_BO_iter).double()
    realizations[:nb_init] = evaluate_rl(a, r, X[:nb_init])

    for i_cand in tqdm(range(nb_BO_iter), disable=not verbose):
        normalized_realizations = standardize(
            realizations[: (nb_init + nb_batch * i_cand)]
        )
        model = SingleTaskGP(
            X[: (nb_init + nb_batch * i_cand)], normalized_realizations.unsqueeze(-1)
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        acquisition = qExpectedImprovement(
            model=model, best_f=normalized_realizations.max()
        )

        candidate, _ = optimize_acqf(
            acq_function=acquisition,
            bounds=bounds,
            q=nb_batch,
            num_restarts=32,
            raw_samples=512,
        )

        realizations[
            (nb_init + nb_batch * i_cand) : (nb_init + nb_batch * (1 + i_cand))
        ] = evaluate_rl(a, r, candidate.squeeze())
        X[
            (nb_init + nb_batch * i_cand) : (nb_init + nb_batch * (1 + i_cand))
        ] = candidate.squeeze()

    return realizations.max(), scale_parameters(X[realizations.argmax()])

'''
