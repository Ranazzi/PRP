import numpy as np
from math_utils import cov
from numpy.random import default_rng

rng = default_rng()


class Localization:
    def __init__(self, M, D):
        self.M = M
        self.D = D
        self.N = M.shape[1]

    def pseudo_optimal(self, type='std', random_shuffle=None):
        if type != 'std':
            if random_shuffle == 'times':  # shuffle Nr times
                n_r = min(self.N, 500)

                cmd_rs = np.zeros((self.M.shape[0], self.D.shape[0], n_r))
                for r in range(n_r):
                    m_shuffled = rng.choice(self.M, self.N, replace=False, axis=1)
                    cmd_rs[:, :, r] = cov(m_shuffled, self.D)

                sig = np.zeros((self.M.shape[0], self.D.shape[0]))
                threshold = np.zeros((self.M.shape[0], self.D.shape[0]))
                for i in range(self.M.shape[0]):
                    for j in range(self.D.shape[0]):
                        err = cmd_rs[i, j, :]
                        sig[i, j] = np.median(np.abs(err)) / 0.6745
                        threshold[i, j] = np.sqrt(2 * np.log(n_r)) * sig[i, j]

            elif random_shuffle == 'group':
                m_shuffled = rng.choice(self.M, self.N, replace=False, axis=1)
                cmd_rs = cov(m_shuffled, self.D)
                sig = np.median(np.abs(cmd_rs), axis=0) / 0.6745
                threshold = np.sqrt(2 * np.log(self.M.shape[0])) * sig
                threshold = np.tile(threshold, (self.M.shape[0], 1))

        Cmd = cov(self.M, self.D)

        varM = np.var(self.M, axis=1)
        varD = np.var(self.D, axis=1)
        if type == 'fixed':
            beta = threshold
        elif type == 'linear':
            lower = np.multiply(np.tile(varM, (self.D.shape[0], 1)).T,
                                np.tile(varD, (self.M.shape[0], 1)))
            beta = 1 - (np.divide(np.power(Cmd, 2), lower, out=np.zeros_like(Cmd), where=lower != 0))
            beta = np.multiply(beta, threshold)
        elif type == 'gc':  # compute using POL-GC
            lower = np.multiply(np.tile(varM, (self.D.shape[0], 1)).T,
                                np.tile(varD, (self.M.shape[0], 1)))
            z = 1 * np.divide(np.power(Cmd, 2), lower, out=np.zeros_like(Cmd), where=lower != 0)
            beta = np.zeros((self.M.shape[0], self.D.shape[0]))
            fil = z[np.logical_and(z <= 1, z >= 0)]
            beta[np.logical_and(z <= 1, z >= 0)] = (-1 / 4 * fil ** 5) + (1 / 2 * fil ** 4) + (
                    5 / 8 * fil ** 3) - (5 / 3 * fil ** 2) + 1
            fil = z[np.logical_and(z <= 2, z >= 1)]
            beta[np.logical_and(z <= 2,
                                z >= 1)] = 1 / 12 * fil ** 5 - 1 / 2 * fil ** 4 + 5 / 8 * fil ** 3 + 5 / 3 * fil ** 2 - 5 * fil + 4 - 2 / 3 * fil ** -1
            beta = np.multiply(beta, threshold)

        elif type == 'exp':  # compute using POL-EXP
            emp = - (6 * Cmd / (1.5 / np.sqrt(self.N)))
            beta = np.multiply(threshold, np.exp(emp))
        else:
            beta = 0

        lower = (np.power(Cmd, 2) + (np.multiply(np.tile(varM, (self.D.shape[0], 1)).T,
                                                 np.tile(varD, (self.M.shape[0],
                                                                1)))) / self.N + beta)
        rho_matrix = np.divide(np.power(Cmd, 2), lower, out=np.zeros_like(Cmd), where=lower != 0)
        return rho_matrix
