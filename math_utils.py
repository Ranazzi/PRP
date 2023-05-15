# math utilities: some algebra tools and the normalized data mismatch objective function
import numpy as np


def cov(M, D, large=None):
    """
    sample covariance matrix using two Nm X N and Nd x N arrays (better for large Nm)
    :param M:
    :param D:
    :param large: calculation for large ensemble size
    :return: CMD = covariance between M and D
    """
    N = M.shape[1]
    if large is None:
        # compute ensembles perturbation matrix
        Mp = M @ (np.eye(N) - np.ones((N, N)) * (1 / N))
        Dp = D @ (np.eye(N) - np.ones((N, N)) * (1 / N))

    else:
        Mp = M - np.tile(M.mean(), N)
        Dp = D - np.tile(M.mean(), N)

    # compute sample covariance (using N-1)
    CMD = (Mp @ Dp.T) / (N - 1)

    return CMD


def corcov(C):
    """
    compute correlation matrix from covariance matrix
    :param C: covariance matrix
    :return: Cor = correlation matrix
    """
    v = np.sqrt(np.diag(C))
    outer_v = np.outer(v, v)
    Cor = C / outer_v
    Cor[C == 0] = 0
    return Cor


def normalized_data_mismatch(D, Dobs, Cd):
    icd = np.diag(np.diag(Cd) ** -1)
    N = D.shape[1]
    ond = np.zeros(N)
    # Dobs = np.expand_dims(Dobs, axis=1)
    for n in range(N):
        ond[n] = 1/(2 * Dobs.shape[0]) * (D[:, [n]] - Dobs).transpose() @ icd @ (D[:, [n]] - Dobs)
    return ond
