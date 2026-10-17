"""
ComBat batch correction with a fit / transform split.

This follows the parametric empirical Bayes model of Johnson et al. (2007), the
same model implemented in sva::ComBat and inmoose.pycombat_norm. We wrote it out
here because neither of those lets you estimate the batch parameters on one set
of samples (the training fold) and then apply them to new samples (the test fold).
That split is what we need to keep the test fold out of the batch correction.

Only the mean-only variant can be applied to unlabelled samples. For mean-only
ComBat the corrected value is

    y*_gj = y_gj - gamma*_gb * sigma_g

so the shift depends on the batch b of sample j, not on its disease label. Full
ComBat also rescales the residual around the covariate-predicted mean, which needs
the test sample's label, so we only allow `transform` for mean_only=True.
"""
import numpy as np
import pandas as pd


def _postmean(g_hat, g_bar, n, d_star, t2):
    return (t2 * n * g_hat + d_star * g_bar) / (t2 * n + d_star)


def _postvar(sum2, n, a, b):
    return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)


def _aprior(d):
    m, s2 = d.mean(), d.var(ddof=1)
    return (2 * s2 + m ** 2) / s2


def _bprior(d):
    m, s2 = d.mean(), d.var(ddof=1)
    return (m * s2 + m ** 3) / s2


class ComBat:
    """
    Parameters
    ----------
    mean_only : bool
        If True only the batch location is adjusted (primary analysis).
    conv : float
        Convergence tolerance for the iterative EB solution (full ComBat only).
    """

    def __init__(self, mean_only=True, conv=1e-4):
        self.mean_only = mean_only
        self.conv = conv

    def fit(self, X, batch, covar=None):
        """
        X     : samples x genes array/DataFrame (no missing values)
        batch : length-n array of batch labels
        covar : samples x k array/DataFrame of covariates to protect (dummy coded,
                no intercept), or None
        """
        X = np.asarray(X, float)
        batch = np.asarray(batch)
        n, p = X.shape
        self.batches_ = np.array(sorted(pd.unique(batch)))
        B = (batch[:, None] == self.batches_[None, :]).astype(float)   # n x nb
        n_b = B.sum(0)
        if covar is None:
            C = np.zeros((n, 0))
        else:
            C = np.asarray(covar, float)
        design = np.hstack([B, C])

        # drop covariate columns that are constant / collinear in this sample
        keep = list(range(B.shape[1]))
        for j in range(C.shape[1]):
            trial = design[:, keep + [B.shape[1] + j]]
            if np.linalg.matrix_rank(trial) == len(keep) + 1:
                keep.append(B.shape[1] + j)
        design = design[:, keep]

        beta = np.linalg.lstsq(design, X, rcond=None)[0]               # (nb+k) x p
        grand_mean = (n_b / n) @ beta[: len(self.batches_)]             # p
        resid = X - design @ beta
        var_pooled = (resid ** 2).mean(0)                                # p
        sd = np.sqrt(var_pooled)

        stand_mean = np.tile(grand_mean, (n, 1))
        if design.shape[1] > len(self.batches_):
            stand_mean = stand_mean + design[:, len(self.batches_):] @ beta[len(self.batches_):]
        Z = (X - stand_mean) / sd

        gamma_hat = np.linalg.solve(B.T @ B, B.T @ Z)                  # nb x p
        if self.mean_only:
            delta_hat = np.ones_like(gamma_hat)
        else:
            delta_hat = np.vstack([Z[batch == b].var(0, ddof=1) for b in self.batches_])

        gamma_bar = gamma_hat.mean(1)
        t2 = gamma_hat.var(1, ddof=1)

        gamma_star = np.zeros_like(gamma_hat)
        delta_star = np.ones_like(gamma_hat)
        for i, b in enumerate(self.batches_):
            if self.mean_only:
                gamma_star[i] = _postmean(gamma_hat[i], gamma_bar[i], 1.0, 1.0, t2[i])
            else:
                a_p, b_p = _aprior(delta_hat[i]), _bprior(delta_hat[i])
                Zb = Z[batch == b]
                nb = Zb.shape[0]
                g_old, d_old = gamma_hat[i].copy(), delta_hat[i].copy()
                for _ in range(1000):
                    g_new = _postmean(gamma_hat[i], gamma_bar[i], nb, d_old, t2[i])
                    sum2 = ((Zb - g_new) ** 2).sum(0)
                    d_new = _postvar(sum2, nb, a_p, b_p)
                    change = max(np.max(np.abs(g_new - g_old) / np.abs(g_old)),
                                 np.max(np.abs(d_new - d_old) / d_old))
                    g_old, d_old = g_new, d_new
                    if change < self.conv:
                        break
                gamma_star[i], delta_star[i] = g_old, d_old

        self.grand_mean_ = grand_mean
        self.sd_ = sd
        self.gamma_star_ = gamma_star
        self.delta_star_ = delta_star
        self._train = dict(Z=Z, stand_mean=stand_mean, batch=batch)
        return self

    def _batch_index(self, batch):
        idx = np.searchsorted(self.batches_, batch)
        bad = (idx >= len(self.batches_)) | (self.batches_[np.minimum(idx, len(self.batches_) - 1)] != batch)
        if bad.any():
            raise ValueError(f"Batches not seen during fit: {set(np.asarray(batch)[bad])}")
        return idx

    def fit_transform(self, X, batch, covar=None):
        self.fit(X, batch, covar)
        Z, stand_mean, batch = self._train["Z"], self._train["stand_mean"], self._train["batch"]
        idx = self._batch_index(batch)
        Zs = (Z - self.gamma_star_[idx]) / np.sqrt(self.delta_star_[idx])
        return Zs * self.sd_ + stand_mean

    def transform(self, X, batch):
        """Apply the fitted batch shift to new samples. Needs no labels."""
        if not self.mean_only:
            raise ValueError("Full ComBat needs the sample's covariates to transform; "
                             "only mean-only ComBat can be applied to unlabelled samples.")
        X = np.asarray(X, float)
        idx = self._batch_index(np.asarray(batch))
        return X - self.gamma_star_[idx] * self.sd_
