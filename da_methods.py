# Data assimilation methods
# Update performed row by row following (Emerick, 2016)

import numpy as np
from math_utils import cov, corcov
from numpy.linalg import inv, svd


class Esmda:
    def __init__(self, M, D, Dobs, Cd, loc=None):
        self.M = M
        self.D = D
        self.Cd = Cd
        self.N = M.shape[1]
        self.Dobs = Dobs
        self.loc = loc

    def explicit(self, alpha):
        if self.N > 10000:
            opt1 = "large"
        else:
            opt1 = None
        # compute kalman gain before inverse:
        K1 = (cov(self.D, self.D, opt1) + alpha * self.Cd)
        CMD = cov(self.M, self.D, opt1)
        dp = np.random.multivariate_normal(np.zeros(self.Dobs.shape)[:, 0], alpha * self.Cd, self.N).T  # perturb here
        if K1.ndim > 1:  # matrices
            Ma = self.M + CMD @ inv(K1) @ (self.Dobs + dp - self.D)
        else:  # scalar
            Ma = self.M + CMD @ (1 / K1) @ (self.Dobs + dp - self.D)
        return Ma

    def subspace(self, alpha, energy, loc=None):
        """truncated svd with rescaling based on Cd diagonal elements
            for more information, see Emerick (2016, appendix)
        """

        s = np.diag(np.diag(self.Cd) ** 0.5)  # square root of the diagonal components of Cd
        corr = corcov(self.Cd)  # correlation matrix of the measurement errors

        d_p = self.D @ (np.eye(self.N) - np.ones((self.N, self.N)) * (1 / self.N))  # data annomalies matrix

        # first svd here
        u, w, __ = svd(np.diag(1 / np.diag(s)) @ d_p)  # invert only diagonal elements because s is always diagonal

        # select g values based on retained energy
        e1 = w.cumsum() / w.cumsum()[-1]  # ratio between energy and total energy
        n_r = (e1 >= energy).nonzero()[0][0]  # in python index (true dimension is Nr +1)

        # slice elements in Nr
        u_r = u[:, 0:n_r]
        w_r = np.diag(w[0:n_r])

        w_i = np.diag(np.diag(w_r) ** -1)

        R = alpha * (self.N - 1) * (w_i @ u_r.T @ corr @ u_r @ w_i.T)

        # second svd
        z_r, h_r, __ = svd(R)

        s_i = np.diag(np.diag(s) ** -1)

        # final matrices
        x = s_i @ u_r @ w_i @ z_r
        ll = np.diag(np.diag(np.eye(n_r) + np.diag(h_r)) ** -1)

        x_1 = ll @ x.T
        x_2 = d_p.T @ x
        x_3 = x_2 @ x_1

        # row by row update
        Ma = np.zeros(self.M.shape)

        m_p = self.M @ (np.eye(self.N) - np.ones((self.N, self.N)) * (1 / self.N))  # data annomalies matrix

        n_m = self.M.shape[0]

        Dobs_p = np.zeros(self.D.shape)
        for j in range(Dobs_p.shape[-1]):
            Dobs_p[:, j] = np.random.normal(self.Dobs[:, -1], np.diag(self.Cd**0.5), self.Dobs.shape[0])

        for n in range(n_m):
            if self.M[n, :].std() > 0:  # ignore std 0 in M
                if loc is None:
                    k_gain = m_p[n:n + 1, :] @ x_3
                else:
                    k_gain = loc[n:n + 1, :] * (m_p[n:n + 1, :] @ x_3)  # Schur product
                x_4 = k_gain @ (Dobs_p - self.D)
                Ma[n:n + 1, :] = self.M[n:n + 1, :] + x_4

        return Ma
