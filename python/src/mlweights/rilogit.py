"""Weighted random-intercept logistic regression by pseudo-maximum likelihood.

log PL(beta, sigma) = sum_j w2_j log INT exp{ sum_i w1_ij log f(y_ij | x_ij'beta + sigma z) } phi(z) dz

w2_j is the cluster weight and w1_ij the within-cluster weight. The scale of w2
does not affect the estimates; the scale of w1 does, which is why w1 should be
rescaled within clusters. The integral uses adaptive Gauss-Hermite quadrature.
"""
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from .diagnostics import icc_logit, mor


class WeightedRILogit:
    """Two-level random-intercept logit model with weights at both levels.

    Parameters
    ----------
    y : array of 0/1 outcomes.
    X : design matrix (include a column of ones for the intercept).
    cluster : cluster identifiers.
    w1 : level-1 weights (default 1). They are used as given; rescale them first.
    w2 : level-2 weights, one per unit (constant within cluster) or one per cluster
         in sorted order of the cluster identifiers (default 1).
    n_quad : number of quadrature points.
    """

    def __init__(self, y, X, cluster, w1=None, w2=None, n_quad=15):
        y = np.asarray(y, float); X = np.asarray(X, float); cl = np.asarray(cluster)
        if X.ndim != 2 or len(y) != len(X) or len(cl) != len(y):
            raise ValueError("y, X and cluster must have matching lengths")
        if not np.all((y == 0) | (y == 1)):
            raise ValueError("y must be 0/1")
        order = np.argsort(cl, kind="stable")
        self._order = order
        y, X, cl = y[order], X[order], cl[order]
        ids, start, counts = np.unique(cl, return_index=True, return_counts=True)
        J, nmax, p = len(ids), counts.max(), X.shape[1]
        pos = np.arange(len(y)) - np.repeat(start, counts)
        row = np.repeat(np.arange(J), counts)
        self.ids, self.J, self.p = ids, J, p
        self._start = start
        self._row, self._pos = row, pos
        self.Y = np.zeros((J, nmax)); self.Y[row, pos] = y
        self.X = np.zeros((J, nmax, p)); self.X[row, pos] = X
        self.W1 = np.zeros((J, nmax))
        self.W1[row, pos] = 1.0 if w1 is None else np.asarray(w1, float)[order]
        if w2 is None:
            w2 = np.ones(J)
        else:
            w2 = np.asarray(w2, float)
            w2 = w2[order][start] if len(w2) == len(y) else w2
            if len(w2) != J:
                raise ValueError("w2 must have one value per unit or per cluster")
        if np.any(self.W1[row, pos] <= 0) or np.any(w2 <= 0):
            raise ValueError("weights must be positive")
        self.w2 = w2 / w2.mean()
        xq, wq = hermgauss(n_quad)
        self._xq, self._lwq = xq, np.log(wq) + xq ** 2
        self._zhat = np.zeros(J)
        self._scale = float((self.w2[:, None] * self.W1).sum())

    def _mode(self, xb, sigma):
        z = self._zhat.copy()
        for _ in range(100):
            pr = expit(xb + sigma * z[:, None])
            g = sigma * (self.W1 * (self.Y - pr)).sum(1) - z
            H = -sigma ** 2 * (self.W1 * pr * (1 - pr)).sum(1) - 1.0
            step = np.clip(g / H, -3, 3)
            z = z - step
            if np.max(np.abs(step)) < 1e-10:
                break
        pr = expit(xb + sigma * z[:, None])
        H = -sigma ** 2 * (self.W1 * pr * (1 - pr)).sum(1) - 1.0
        self._zhat = z
        return z, 1.0 / np.sqrt(-H)

    def _cluster_scores(self, theta):
        """Per-cluster score vectors of the log pseudo-likelihood (unnormalised), J x (p+1).
        Derivative with respect to (beta, log sigma); cluster weights included."""
        beta, sigma = theta[:self.p], np.exp(min(theta[self.p], 10.0))
        xb = self.X @ beta
        zh, tau = self._mode(xb, sigma)
        Z = zh[:, None] + np.sqrt(2) * tau[:, None] * self._xq[None, :]
        eta = xb[:, :, None] + sigma * Z[:, None, :]
        ll = self.W1[:, :, None] * (self.Y[:, :, None] * eta - np.logaddexp(0, eta))
        h = ll.sum(1) - 0.5 * Z ** 2 + self._lwq[None, :]
        r = np.exp(h - logsumexp(h, axis=1)[:, None])
        res = self.W1[:, :, None] * (self.Y[:, :, None] - expit(eta))
        gb = np.einsum("jn,jnp->jp", (res * r[:, None, :]).sum(2), self.X)
        gs = (r * Z * res.sum(1)).sum(1) * sigma
        return self.w2[:, None] * np.column_stack([gb, gs])

    def negloglik(self, theta, grad=True):
        """Negative log pseudo-likelihood (divided by the total weight) and its gradient.
        theta = (beta, log sigma)."""
        beta, sigma = theta[:self.p], np.exp(min(theta[self.p], 10.0))
        xb = self.X @ beta
        zh, tau = self._mode(xb, sigma)
        Z = zh[:, None] + np.sqrt(2) * tau[:, None] * self._xq[None, :]
        eta = xb[:, :, None] + sigma * Z[:, None, :]
        ll = self.W1[:, :, None] * (self.Y[:, :, None] * eta - np.logaddexp(0, eta))
        h = ll.sum(1) - 0.5 * Z ** 2 + self._lwq[None, :]
        lse = logsumexp(h, axis=1)
        logI = lse + np.log(np.sqrt(2) * tau) - 0.5 * np.log(2 * np.pi)
        val = -(self.w2 * logI).sum() / self._scale
        if not grad:
            return val
        r = np.exp(h - lse[:, None])
        res = self.W1[:, :, None] * (self.Y[:, :, None] - expit(eta))
        gb = np.einsum("j,jn,jnp->p", self.w2, (res * r[:, None, :]).sum(2), self.X)
        gs = (self.w2 * (r * Z * res.sum(1)).sum(1)).sum() * sigma
        return val, -np.append(gb, gs) / self._scale

    def fit(self, start=None, gtol=1e-6, maxiter=500):
        """Fit by BFGS. Returns self with beta, sigma2, icc, mor, converged."""
        start = np.zeros(self.p + 1) if start is None else np.asarray(start, float).copy()
        self._zhat[:] = 0
        opt = minimize(self.negloglik, start, jac=True, method="BFGS", options={"gtol": gtol, "maxiter": maxiter})
        if not opt.success:
            opt = minimize(self.negloglik, opt.x, jac=True, method="BFGS", options={"gtol": 10 * gtol, "maxiter": maxiter})
        self.theta = opt.x
        self.beta = opt.x[:self.p]
        self.sigma2 = float(np.exp(2 * min(opt.x[self.p], 10.0)))
        self.icc = float(icc_logit(self.sigma2))
        self.mor = float(mor(self.sigma2))
        self.converged = bool(opt.success or float(np.max(np.abs(opt.jac))) < 1e-4)
        return self

    def vcov(self, strata=None, eps=1e-5):
        """Design-based (sandwich) covariance of (beta, log sigma).

        Bread: numerical Hessian of the log pseudo-likelihood. Meat: between-cluster
        variation of the weighted cluster scores within strata, with the factor
        n_h / (n_h - 1) (first-stage, with-replacement linearisation as in survey
        software). strata: one stratum code per cluster in sorted order of the cluster
        identifiers, or one per unit (constant within cluster); None = one stratum.
        Strata with a single cluster are centred at the grand mean.
        """
        th = self.theta.copy()
        k = len(th)
        H = np.zeros((k, k))
        for i in range(k):
            e = np.zeros(k); e[i] = eps
            gp = self._cluster_scores(th + e).sum(0)
            gm = self._cluster_scores(th - e).sum(0)
            H[:, i] = (gp - gm) / (2 * eps)
        H = 0.5 * (H + H.T)
        U = self._cluster_scores(th)
        if strata is None:
            st = np.zeros(self.J)
        else:
            st = np.asarray(strata)
            if len(st) != self.J:
                st = st[self._order][self._start]
        meat = np.zeros((k, k))
        grand = U.mean(0)
        for h in np.unique(st):
            Uh = U[st == h]
            n = len(Uh)
            if n > 1:
                D = Uh - Uh.mean(0)
                meat += n / (n - 1) * D.T @ D
            else:
                D = Uh - grand
                meat += D.T @ D
        Hi = np.linalg.inv(-H)
        self.cov = Hi @ meat @ Hi
        return self.cov

    def summary(self, strata=None, level=0.95):
        """Estimates with design-based standard errors and Wald intervals.
        The interval for sigma2 is formed on the log sigma scale and transformed."""
        from scipy.stats import norm
        V = self.vcov(strata)
        z = norm.ppf(0.5 + level / 2)
        se = np.sqrt(np.diag(V))
        b = self.beta
        ls, sls = self.theta[self.p], se[self.p]
        out = {"beta": b, "se_beta": se[:self.p], "lo_beta": b - z * se[:self.p], "hi_beta": b + z * se[:self.p],
               "sigma2": self.sigma2, "se_sigma2": 2 * self.sigma2 * sls,
               "lo_sigma2": float(np.exp(2 * (ls - z * sls))), "hi_sigma2": float(np.exp(2 * (ls + z * sls)))}
        return out

    def cluster_effects(self):
        """Posterior modes of the cluster effects on the logit scale (one per cluster, sorted ids)."""
        return np.sqrt(self.sigma2) * self._zhat

    def cluster_info(self):
        """Per-cluster v_j = 1 / sum_i p_ij (1 - p_ij), evaluated at the fitted values
        with level-1 weights set to 1. Input for approx_inflated_variance."""
        p = expit(self.X @ self.beta + self.cluster_effects()[:, None])
        mask = np.zeros_like(self.Y); mask[self._row, self._pos] = 1.0
        return 1.0 / (mask * p * (1 - p)).sum(1)
