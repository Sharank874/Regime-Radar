"""Gaussian HMM written from scratch so every step is inspectable.

Design choices
--------------
* Diagonal covariance. This makes the emission log-likelihood a *sum over
  features*, which is what lets the UI attribute each belief update exactly
  to individual features (see `shapley_update`).
* Missing values are handled by marginalisation: a NaN feature simply drops
  out of the likelihood (exact for a diagonal Gaussian). News features are
  therefore usable from 2017 without discarding earlier price history.
* "Sticky" Dirichlet prior on the transition diagonal so regimes persist.
* Only the FORWARD filter is used for anything shown or evaluated. The
  backward pass (smoothing) is used inside EM fitting on the training window
  only, never to label the past for evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import factorial

import numpy as np

LOG2PI = np.log(2 * np.pi)


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


@dataclass
class HMMParams:
    pi: np.ndarray       # (S,)
    A: np.ndarray        # (S,S)
    mu: np.ndarray       # (S,K)
    var: np.ndarray      # (S,K)
    loglik: float = np.nan

    def to_dict(self) -> dict:
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}


def effective_temper(X: np.ndarray) -> float:
    """K_eff / K with K_eff = K^2 / sum(corr^2) (participation ratio of the correlation matrix)."""
    ok = ~np.isnan(X).any(1)
    K = X.shape[1]
    if ok.sum() < 50 or K < 2:
        return 1.0
    C = np.corrcoef(X[ok].T)
    return float(np.clip((K * K / np.sum(C * C)) / K, 0.3, 1.0))


class GaussianHMM:
    def __init__(self, n_states=3, n_iter=150, tol=1e-4, n_init=6, sticky=50.0, min_var=1e-3,
                 order_by: int = 1, seed=0, temper: float = 1.0):
        self.S, self.n_iter, self.tol, self.n_init = n_states, n_iter, tol, n_init
        # temper < 1 down-weights evidence to correct for correlated features being
        # treated as independent by the diagonal covariance (set from K_eff / K)
        self.temper = temper
        self.sticky, self.min_var, self.order_by, self.seed = sticky, min_var, order_by, seed
        self.params: HMMParams | None = None

    # ------------------------------------------------------------------ emissions
    @staticmethod
    def feature_logdens(X: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
        """Per-feature log density, shape (T,S,K). NaN features contribute 0."""
        d = X[:, None, :] - mu[None, :, :]
        ld = -0.5 * (LOG2PI + np.log(var)[None] + d * d / var[None])
        return np.where(np.isnan(X)[:, None, :], 0.0, ld)

    # ------------------------------------------------------------------ fitting
    def _init_params(self, X: np.ndarray, rng) -> HMMParams:
        T, K = X.shape
        Xf = np.where(np.isnan(X), 0.0, X)
        # k-means++ style seeding on the ordering feature + noise
        idx = rng.choice(T, self.S, replace=False)
        mu = Xf[idx] + rng.normal(0, 0.3, (self.S, K))
        var = np.ones((self.S, K))
        A = np.full((self.S, self.S), 0.05 / (self.S - 1))
        np.fill_diagonal(A, 0.95)
        pi = np.full(self.S, 1 / self.S)
        return HMMParams(pi, A, mu, var)

    def _forward_backward(self, logB: np.ndarray, p: HMMParams):
        """Scaled forward-backward (Rabiner). Emissions are rescaled per time step."""
        T, S = logB.shape
        mx = logB.max(1, keepdims=True)
        B = np.exp(logB - mx)
        A = p.A
        alpha = np.empty((T, S))
        c = np.empty(T)
        a = p.pi * B[0]
        c[0] = a.sum()
        alpha[0] = a / c[0]
        for t in range(1, T):
            a = (alpha[t - 1] @ A) * B[t]
            c[t] = a.sum()
            alpha[t] = a / c[t]
        beta = np.empty((T, S))
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
        gamma = alpha * beta
        gamma /= gamma.sum(1, keepdims=True)
        xi_sum = A * np.einsum("ti,tj->ij", alpha[:-1], (B[1:] * beta[1:]) / c[1:, None])
        ll = float(np.log(c).sum() + mx.sum())
        return gamma, xi_sum, ll

    def _em(self, X: np.ndarray, p: HMMParams) -> HMMParams:
        mask = ~np.isnan(X)
        Xf = np.where(mask, X, 0.0)
        prev = -np.inf
        for _ in range(self.n_iter):
            logB = self.temper * self.feature_logdens(X, p.mu, p.var).sum(-1)
            gamma, xi, ll = self._forward_backward(logB, p)
            # M-step (NaN aware)
            w = gamma[:, :, None] * mask[:, None, :]                  # (T,S,K)
            wsum = w.sum(0) + 1e-9
            mu = (w * Xf[:, None, :]).sum(0) / wsum
            d = Xf[:, None, :] - mu[None]
            var = np.maximum((w * d * d).sum(0) / wsum, self.min_var)
            counts = xi + self.sticky * np.eye(self.S) + 1e-3
            A = counts / counts.sum(1, keepdims=True)
            pi = gamma[0] + 1e-3
            p = HMMParams(pi / pi.sum(), A, mu, var, ll)
            if ll - prev < self.tol * abs(ll):
                break
            prev = ll
        return p

    def fit(self, X: np.ndarray) -> "GaussianHMM":
        rng = np.random.default_rng(self.seed)
        best = None
        for _ in range(self.n_init):
            p = self._em(X, self._init_params(X, rng))
            if best is None or p.loglik > best.loglik:
                best = p
        # order states by mean of the ordering feature (volatility) -> 0 = calm ... S-1 = stress
        order = np.argsort(best.mu[:, self.order_by])
        self.params = HMMParams(best.pi[order], best.A[np.ix_(order, order)],
                                best.mu[order], best.var[order], best.loglik)
        return self

    # ------------------------------------------------------------------ filtering
    def filter(self, X: np.ndarray, init: np.ndarray | None = None):
        """Forward filter. Returns prior (predicted) probs, posterior (filtered) probs and the
        per-feature log densities used at each step."""
        p = self.params
        L = self.temper * self.feature_logdens(X, p.mu, p.var)  # (T,S,K), tempered
        T, S = L.shape[:2]
        prior = np.empty((T, S))
        post = np.empty((T, S))
        belief = p.pi if init is None else init
        for t in range(T):
            pr = belief @ p.A if (t > 0 or init is not None) else belief
            prior[t] = pr
            lg = np.log(pr + 1e-300) + L[t].sum(-1)
            lg -= lg.max()
            b = np.exp(lg)
            belief = b / b.sum()
            post[t] = belief
        return prior, post, L


def shapley_update(prior: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Exact Shapley attribution of the posterior to each feature's evidence.

    prior: (T,S) predicted probabilities (after the transition step)
    L:     (T,S,K) per-feature log densities
    returns phi: (T,S,K) with  prior + phi.sum(-1) == posterior  (exactly)

    The 'game' value of a coalition M of features is the posterior you would get
    using only the evidence in M: softmax(log prior + sum_{k in M} L_k). With
    K <= 6 features all 2^K coalitions are enumerated.
    """
    T, S, K = L.shape
    subsets = [()]
    for r in range(1, K + 1):
        subsets += list(combinations(range(K), r))
    index = {s: i for i, s in enumerate(subsets)}
    masks = np.zeros((len(subsets), K))
    for i, s in enumerate(subsets):
        masks[i, list(s)] = 1
    logits = np.log(prior + 1e-300)[:, None, :] + np.einsum("mk,tsk->tms", masks, L)
    logits -= logits.max(-1, keepdims=True)
    P = np.exp(logits)
    P /= P.sum(-1, keepdims=True)                          # (T,M,S)
    phi = np.zeros((T, S, K))
    for s in subsets:
        n = len(s)
        for k in range(K):
            if k in s:
                continue
            w = factorial(n) * factorial(K - n - 1) / factorial(K)
            with_k = tuple(sorted(s + (k,)))
            phi[:, :, k] += w * (P[:, index[with_k], :] - P[:, index[s], :])
    return phi
