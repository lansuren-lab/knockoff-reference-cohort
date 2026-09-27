"""
knockoffs.py -- Model-X knockoffs for ROI-level neuroimaging.

Reference
---------
Candes, Fan, Janson & Lv (2018), "Panning for gold: 'model-X' knockoffs for
high dimensional controlled variable selection", JRSS-B 80(3):551-577.

What this file implements, and where each piece comes from in the paper
------------------------------------------------------------------------
1. `solve_s_diag`      -- knockoff parameter s with  s >= 0  and  diag(s) <= 2*Sigma
                          (paper Sec. 3.4.2, eq. 3.11-3.14; including the
                          approximate-SDP "ASDP" two-step construction)
2. `sample_knockoffs`  -- Xtilde | X ~ N(mu, V), the conditional sampler of
                          Sec. 3.1.1, which makes (X, Xtilde) exactly
                          swap-invariant when X is Gaussian
3. `lcd_statistic`     -- the LCD statistic W_j = |b_j| - |b_{j+p}| from
                          Sec. 3.2, eq. 3.5-3.6. Cross-validation is done on
                          the AUGMENTED design (X, Xtilde) so that the
                          flip-sign property (eq. 3.3) still holds exactly.
4. `knockoff_threshold`-- the knockoffs / knockoffs+ threshold inversion,
                          Sec. 3.3, eq. 3.9-3.10

Conventions
-----------
- X is standardized to mean 0 / variance 1 BEFORE knockoffs are built, so
  that Sigma is a correlation matrix and the knockoff columns are on the same
  scale.  This is the paper's convention ("we shall assume that the covariates
  have each been translated and rescaled to have mean 0 and variance 1").
  Do NOT standardize the augmented matrix afterwards: standardizing column j
  and column j+p by *different* factors would break the flip-sign property.
- Rows are subjects (iid), columns are ROI features.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky

__all__ = [
    "solve_s_diag",
    "sample_knockoffs",
    "lcd_statistic",
    "knockoff_threshold",
    "knockoff_filter",
]


# --------------------------------------------------------------------------- #
# 1. Choosing s  (Sec. 3.4.2)
# --------------------------------------------------------------------------- #
def _sdp_block(Sigma_bb: np.ndarray, use_cvxpy: bool = True) -> np.ndarray:
    """Solve, on one block:  min sum_j |1 - s_j|  s.t.  s_j >= 0, diag(s) <= 2 Sigma_bb.

    Falls back to the equicorrelated-within-block solution (always feasible)
    when cvxpy is unavailable.
    """
    p = Sigma_bb.shape[0]
    if use_cvxpy:
        try:
            import cvxpy as cp

            s = cp.Variable(p, nonneg=True)
            prob = cp.Problem(
                cp.Minimize(cp.sum(cp.abs(1 - s))),
                [cp.diag(s) << 2 * Sigma_bb],
            )
            prob.solve()
            if s.value is not None and np.all(np.isfinite(s.value)):
                return np.clip(np.asarray(s.value).ravel(), 0.0, None)
        except Exception:
            pass
    # Fallback: s_j = 2*lambda_min(block) for all j in the block.  Feasible
    # because c*I <= 2*Sigma_bb  <=>  c <= 2*lambda_min(Sigma_bb).
    lam = float(np.linalg.eigvalsh(Sigma_bb).min())
    return np.full(p, min(2.0 * lam, 1.0))


def _gamma_max(s_hat: np.ndarray, Sigma: np.ndarray) -> float:
    """Largest gamma with diag(gamma * s_hat) <= 2 Sigma.

    diag(g*s) <= 2*Sigma  <=>  gamma <= 2 * lambda_min(D^{-1/2} Sigma D^{-1/2})
    with D = diag(s_hat).  (The paper describes a bisection search for this
    step; the closed form above is exact and faster.)
    """
    d = np.sqrt(np.maximum(s_hat, 1e-12))
    M = Sigma / np.outer(d, d)
    return 2.0 * float(np.linalg.eigvalsh(M).min())


# --------------------------------------------------------------------------- #
# 1b. Minimum-reconstructability constructions (MRC): MVR and ME
#     Spector & Janson (2022), Ann. Statist. 50(1):252-276.
#
#     With G_S = [[Sigma, Sigma-S],[Sigma-S, Sigma]] the two losses are
#         L_MVR(S) = (1/2) [ Tr(S^{-1}) + Tr((2 Sigma - S)^{-1}) ] = (1/2) Tr(G_S^{-1})
#         L_ME(S)  = -log det(2 Sigma - S) - log det(S)            = log det(G_S^{-1})
#     Derivation of the MVR identity: the inverse of the 2x2 block matrix is
#     [[X,Y],[Y,X]] with X = (1/2)[(2 Sigma - S)^{-1} + S^{-1}], and
#     L_MVR = sum_{j=1..p} (G_S^{-1})_{jj} = Tr(X).  The ME identity uses
#     det [[A,B],[B,A]] = det(A+B) det(A-B).  Both are checked numerically in
#     mrc_study/test_mrc.py.  Both losses are convex in S and both blow up as
#     2 Sigma - S becomes singular, which acts as an implicit barrier.
# --------------------------------------------------------------------------- #
def _D_inverse(s: np.ndarray, Sigma: np.ndarray):
    """Return (D^{-1}, log det D) for D = 2 Sigma - diag(s); (None, None) if not PD."""
    D = 2.0 * Sigma - np.diag(s)
    try:
        L = cholesky(D, lower=True)
    except np.linalg.LinAlgError:
        return None, None
    Dinv = cho_solve((L, True), np.eye(D.shape[0]))
    logdetD = 2.0 * np.sum(np.log(np.diag(L)))
    return Dinv, logdetD


def mvr_objective(s, Sigma, return_grad=False):
    """L_MVR = (1/2)[ sum_j 1/s_j + Tr((2 Sigma - S)^{-1}) ],  S = diag(s)."""
    s = np.asarray(s, float)
    Dinv, _ = _D_inverse(s, Sigma)
    if Dinv is None:
        return (np.inf, np.zeros_like(s)) if return_grad else np.inf
    val = 0.5 * float(np.sum(1.0 / s) + np.trace(Dinv))
    if not return_grad:
        return val
    # d/ds_j Tr(D^{-1}) = (D^{-2})_jj = ||D^{-1} e_j||^2 = squared norm of column j
    grad = 0.5 * (-1.0 / s ** 2 + np.sum(Dinv ** 2, axis=0))
    return val, grad


def me_objective(s, Sigma, return_grad=False):
    """L_ME = -log det(2 Sigma - S) - log det(S)."""
    s = np.asarray(s, float)
    Dinv, logdetD = _D_inverse(s, Sigma)
    if Dinv is None:
        return (np.inf, np.zeros_like(s)) if return_grad else np.inf
    val = -logdetD - float(np.sum(np.log(s)))
    if not return_grad:
        return val
    grad = np.diag(Dinv) - 1.0 / s
    return val, grad


def _is_feasible(s, Sigma) -> bool:
    """True iff 2 Sigma - diag(s) is positive definite."""
    try:
        cholesky(2.0 * Sigma - np.diag(s), lower=True)
        return True
    except np.linalg.LinAlgError:
        return False


def _project_feasible(s: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """Shrink s uniformly until diag(s) <= 2 Sigma.  Never increases s."""
    s = np.maximum(np.asarray(s, float), 1e-14)
    for _ in range(400):
        if _is_feasible(s, Sigma):
            return s
        s = s * 0.98
    return s


def solve_s_mrc(Sigma: np.ndarray, loss: str = "mvr", s_init=None,
                maxiter: int = 4000, safety: float = 0.999,
                gtol: float = 1e-9) -> np.ndarray:
    """Minimise the MVR or ME reconstructability loss over the diagonal S.

    Both losses are convex and smooth on the feasible set {s >= 0,
    diag(s) <= 2 Sigma}, and both blow up on its boundary.  We therefore use
    projected gradient descent: a gradient step in log-coordinates (which keeps
    s > 0), a multiplicative projection back onto the feasible set, and
    backtracking on the true objective.

    L-BFGS-B was tried first and rejected: the feasible set is curved, so the
    optimiser repeatedly probes infeasible points, and any "return a huge value"
    fallback there creates a zero-gradient plateau that makes it stop early.
    On an equicorrelated Sigma the projected method recovers the analytic
    optimum (1 - rho) * 1 to within 0.4% at p = 50 and 0.03% at p = 200.
    """
    Sigma = np.asarray(Sigma, float)
    p = Sigma.shape[0]
    base = mvr_objective if loss == "mvr" else me_objective
    obj = lambda s, return_grad=False: base(s, Sigma, return_grad=return_grad)  # noqa: E731

    if s_init is None:
        s_init = np.full(p, max(float(np.linalg.eigvalsh(Sigma).min()), 1e-8))
    s = _projected_gradient(obj, s_init, Sigma, maxiter, gtol)
    return np.clip(_project_feasible(s, Sigma) * safety, 1e-12, None)


def _mrc_risk_terms(s, Sigma, loss):
    """Return (Q, dQ/ds, coef) with  Var[delta L] = coef * Q / n_ref.

    The loss is a function of Sigma, so when Sigma is ESTIMATED from n_ref
    reference samples the loss itself carries noise.  Writing the estimation
    error as delta Sigma = (1/n_ref) sum_r (z_r z_r^T - Sigma) and using
    Var(z^T A z) = 2 Tr((A Sigma)^2) for z ~ N(0, Sigma), we get

        L_MVR = (1/2)[Tr(S^{-1}) + Tr(D^{-1})],  delta L = -Tr(D^{-2} delta Sigma)
            => Var = (2/n_ref) Tr((D^{-2} Sigma)^2)
        L_ME  = -log det(D) - log det(S),        delta L = -2 Tr(D^{-1} delta Sigma)
            => Var = (8/n_ref) Tr((D^{-1} Sigma)^2)

    The gradient uses d(A^2)/ds_j = u u^T A + A u u^T with u = A e_j, which
    collapses to a diagonal of a matrix product because A = D^{-1} is symmetric.
    (The Var identity was checked against Monte Carlo for p = 1, 2, 5 in
    mrc_study; the constants are approximate for CORRELATION matrices because
    standardisation pins the diagonal, but kappa absorbs that.)
    """
    Dinv, _ = _D_inverse(s, Sigma)
    if Dinv is None:
        return None, None, None
    A = Dinv
    A2 = A @ A
    if loss == "mvr":
        Q = float(np.trace(A2 @ Sigma @ A2 @ Sigma))
        W = A2 @ Sigma @ A2 @ Sigma @ A + A @ Sigma @ A2 @ Sigma @ A2
        return Q, 2.0 * np.diag(W), 2.0
    Q = float(np.trace(A @ Sigma @ A @ Sigma))
    W = A @ Sigma @ A @ Sigma @ A
    return Q, 2.0 * np.diag(W), 8.0


def mrc_robust_objective(s, Sigma, n_ref, loss="mvr", kappa=1.0,
                         return_grad=False):
    """L(s; Sigma) + kappa * sd[ L(s; Sigmahat) ].

    A mean-plus-risk (robust optimisation) objective.  As n_ref -> infinity the
    penalty vanishes and this reduces to the plain MRC loss; when the reference
    cohort is small it automatically pulls the solution back towards
    low-degree-of-freedom constructions.  One formula, no interpolation
    coefficient to tune -- only kappa, the number of standard deviations.
    """
    base = mvr_objective if loss == "mvr" else me_objective
    L, gL = base(s, Sigma, return_grad=True)
    if not np.isfinite(L):
        return (np.inf, np.zeros_like(s)) if return_grad else np.inf
    Q, dQ, coef = _mrc_risk_terms(s, Sigma, loss)
    if Q is None or not np.isfinite(Q):
        return (np.inf, np.zeros_like(s)) if return_grad else np.inf
    var = coef * Q / float(n_ref)
    pen = kappa * np.sqrt(max(var, 0.0))
    val = L + pen
    if not return_grad:
        return val
    if var <= 1e-300:
        return val, gL
    grad = gL + kappa * coef * dQ / (2.0 * np.sqrt(var) * float(n_ref))
    return val, grad


def _projected_gradient(obj, s_init, Sigma, maxiter, gtol):
    """Projected gradient descent on the convex MRC losses.

    Steps in log-coordinates (so s > 0), projects back onto
    {s >= 0, diag(s) <= 2 Sigma}, and backtracks on the true objective.
    L-BFGS-B was tried first and rejected: the feasible set is curved, so the
    optimiser repeatedly probes infeasible points, and any "return a huge
    value" fallback there creates a zero-gradient plateau that makes it stop
    early (this actually happened -- ME stopped at L = 132 when the MVR
    solution achieved 111 on the ME loss).
    """
    s = _project_feasible(np.asarray(s_init, float), Sigma)
    val = obj(s)
    if not np.isfinite(val):
        return s

    step = 0.25
    for _ in range(maxiter):
        _, g = obj(s, return_grad=True)
        if not np.all(np.isfinite(g)):
            break
        d = g * s                                  # gradient w.r.t. log s
        gnorm = float(np.max(np.abs(d)))
        if gnorm < gtol:
            break

        improved = False
        for _ in range(60):                        # backtracking line search
            s_try = _project_feasible(np.exp(np.log(s) - step * d / gnorm), Sigma)
            v_try = obj(s_try)
            if np.isfinite(v_try) and v_try < val - 1e-15 * abs(val):
                improved = True
                break
            step *= 0.5
            if step < 1e-14:
                break
        if not improved:
            break

        s, val = s_try, v_try
        step = min(step * 1.5, 10.0)               # expand on success
    return s


def solve_s_mrc_robust(Sigma: np.ndarray, n_ref: float, loss: str = "mvr",
                       kappa: float = 1.0, s_init=None, maxiter: int = 4000,
                       safety: float = 0.999, gtol: float = 1e-9) -> np.ndarray:
    """Regularised MRC: minimise the mean-plus-risk objective of
    `mrc_robust_objective`.  Requires the reference-cohort size n_ref."""
    Sigma = np.asarray(Sigma, float)
    p = Sigma.shape[0]
    if s_init is None:
        s_init = np.full(p, max(float(np.linalg.eigvalsh(Sigma).min()), 1e-8))
    obj = lambda s, return_grad=False: mrc_robust_objective(  # noqa: E731
        s, Sigma, n_ref, loss=loss, kappa=kappa, return_grad=return_grad)
    s = _projected_gradient(obj, s_init, Sigma, maxiter, gtol)
    return np.clip(_project_feasible(s, Sigma) * safety, 1e-12, None)


def solve_s_mrc_cv(ref_data: np.ndarray, loss: str = "mvr", n_folds: int = 2,
                   n_grid: int = 11, seed: int = 0, verbose: bool = False):
    """Choose the MRC / equicorrelated interpolation by HELD-OUT validation on
    the reference cohort itself.

        s(lambda) = (1 - lambda) * s_MRC(Sigmahat) + lambda * s_eq(Sigmahat)

    Why this and not a risk penalty.  Optimising the MRC loss at an estimated
    Sigmahat over-fits: at a reference size of 200 the mean REGRET of the MRC
    solution when scored at the true Sigma is 28.6, while the standard deviation
    of that loss is only 11.7.  The damage is a bias, not a variance, so a
    "mean plus kappa * sd" penalty has the wrong functional form and its
    constant would have to be fitted.  Cross-validation targets the bias
    directly.

    Why the folds must be disjoint.  s(lambda) is built from Sigmahat_{-k} and
    scored on the independent Sigmahat_k.  Scoring on the SAME data used to
    build s is exactly what produces the over-fitting, so the split is
    essential, not cosmetic.

    The feasible set {s >= 0, diag(s) <= 2 Sigma} is convex in s for fixed
    Sigma, so the interpolation of two feasible points stays feasible -- but
    only for the Sigma they were built with; the held-out score therefore
    re-projects defensively.

    Returns (s, lambda_star, scores).
    """
    Xref = np.asarray(ref_data, float)
    n, p = Xref.shape
    mu, sd = Xref.mean(0), Xref.std(0, ddof=1)
    sd[sd == 0] = 1.0
    Z = (Xref - mu) / sd

    def corr(A):
        C = np.corrcoef(A, rowvar=False)
        C = 0.5 * (C + C.T)
        np.fill_diagonal(C, 1.0)
        return C

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = [f for f in np.array_split(idx, n_folds) if len(f) > 2]
    lambdas = np.linspace(0.0, 1.0, n_grid)
    scores = np.zeros(n_grid)
    loss_fn = mvr_objective if loss == "mvr" else me_objective

    for k, test_idx in enumerate(folds):
        tr_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != k])
        S_tr, S_te = corr(Z[tr_idx]), corr(Z[test_idx])
        s_mrc = solve_s_mrc(S_tr, loss=loss)
        s_eq = solve_s_diag(S_tr, method="equicorrelated")
        for i, lam in enumerate(lambdas):
            s = (1.0 - lam) * s_mrc + lam * s_eq
            s = _project_feasible(s, S_te)
            scores[i] += loss_fn(s, S_te) / len(folds)

    lam_star = float(lambdas[int(np.argmin(scores))])
    if verbose:
        for lam, sc in zip(lambdas, scores):
            print(f"      lambda={lam:.2f}  held-out L={sc:.3f}")

    S_full = corr(Z)
    s = ((1.0 - lam_star) * solve_s_mrc(S_full, loss=loss)
         + lam_star * solve_s_diag(S_full, method="equicorrelated"))
    return _project_feasible(s, S_full), lam_star, scores


def solve_s_diag(
    Sigma: np.ndarray,
    method: str = "asdp",
    blocks: list[np.ndarray] | None = None,
    block_size: int = 25,
    use_cvxpy: bool = True,
    safety: float = 0.99,
    n_ref: float | None = None,
    kappa: float = 1.0,
) -> np.ndarray:
    """Return the knockoff parameter s (diagonal), satisfying s >= 0, diag(s) <= 2 Sigma.

    Parameters
    ----------
    Sigma : (p, p) correlation matrix of the covariates.
    method : one of
        'equicorrelated' -- s_j = min(2*lambda_min(Sigma), 1).  Fastest, weakest.
        'sdp'            -- exact semidefinite programme on the full Sigma.
        'asdp'           -- approximate SDP (paper Sec. 3.4.2): solve blockwise
                            with Sigma_approx = block-diagonal, then rescale by
                            gamma.  RECOMMENDED for p of a few hundred.
        'identity'       -- Sigma_approx = I, i.e. s_hat = 1 then gamma-rescale.
                            Recovers the equicorrelated construction.
    blocks : optional list of index arrays defining the blocks used by 'asdp'.
        IMPORTANT: put strongly correlated ROIs in the same block (e.g. order
        columns by lobe / Yeo network / anatomical adjacency first).  If None,
        contiguous chunks of size `block_size` are used.
    safety : shrink s slightly below the constraint boundary for numerical
        safety when factorizing V = 2D - D Sigma^{-1} D.
    """
    p = Sigma.shape[0]

    if method == "equicorrelated":
        # s_j = min(2 lambda_min(Sigma), 1), shrunk by `safety` so that
        # 2 Sigma - diag(s) stays strictly positive definite.  Without the
        # shrink this sits exactly on the boundary whenever 2 lambda_min >= 1
        # (e.g. equicorrelated Sigma with rho = 0.5), making the knockoff
        # covariance singular and the sampler fail.
        lam = float(np.linalg.eigvalsh(Sigma).min())
        return np.full(p, min(2.0 * lam, 1.0) * safety)

    if method in ("mvr", "me"):
        # Minimum-reconstructability constructions: the optimiser returns the
        # final s directly -- no gamma rescaling is applied.
        return solve_s_mrc(Sigma, loss=method)

    if method in ("mvr_robust", "me_robust"):
        # Mean-plus-risk regularised MRC.  Needs the reference-cohort size.
        if n_ref is None:
            raise ValueError(
                f"method={method!r} requires n_ref (the reference-cohort size "
                "used to estimate Sigma). Pass n_ref=... or use method "
                "'mvr'/'me' if Sigma is known exactly.")
        return solve_s_mrc_robust(Sigma, n_ref=n_ref,
                                  loss=method.split("_")[0], kappa=kappa)

    if method == "identity":
        s_hat = np.ones(p)
        return _rescale(s_hat, Sigma, safety)

    if method == "sdp":
        s_hat = _sdp_block(Sigma, use_cvxpy=use_cvxpy)
        return _rescale(s_hat, Sigma, safety)

    if method == "asdp":
        if blocks is None:
            blocks = np.array_split(np.arange(p), max(1, int(np.ceil(p / block_size))))
        s_hat = np.zeros(p)
        for b in blocks:
            b = np.asarray(b)
            s_hat[b] = _sdp_block(Sigma[np.ix_(b, b)], use_cvxpy=use_cvxpy)
        return _rescale(s_hat, Sigma, safety)

    raise ValueError(f"unknown method: {method!r}")


def _rescale(s_hat: np.ndarray, Sigma: np.ndarray, safety: float) -> np.ndarray:
    """ASDP step 2: maximise gamma with diag(gamma*s_hat) <= 2 Sigma.

    gamma is capped at 1 so that s_j <= 1, i.e. corr(X_j, Xtilde_j) = 1 - s_j
    stays non-negative.  This reproduces the paper's statement that ASDP with
    Sigma_approx = Sigma gives gamma = 1.
    """
    gamma = min(_gamma_max(s_hat, Sigma), 1.0) * safety
    s = gamma * s_hat
    # numerical guard: enforce the constraint we actually rely on
    for _ in range(50):
        if np.linalg.eigvalsh(2.0 * Sigma - np.diag(s)).min() > 0:
            break
        s *= 0.98
    return np.clip(s, 0.0, None)


# --------------------------------------------------------------------------- #
# 2. Sampling the knockoffs  (Sec. 3.1.1)
# --------------------------------------------------------------------------- #
def _safe_pd(Sigma: np.ndarray, floor_rel: float = 1e-6) -> np.ndarray:
    """Clip eigenvalues from below so Sigma is usable as a covariance matrix.

    Reference cohorts routinely give a Sigmahat that is singular or numerically
    indefinite -- an in-sample correlation matrix has rank <= min(n-1, p) by
    construction, and shrinkage does not always rescue it.  Without this the
    call to cholesky() in the sampler raises "N-th leading minor is not
    positive definite", which is exactly what happened when a reference cohort
    was used for the first time.
    """
    S = 0.5 * (np.asarray(Sigma, float) + np.asarray(Sigma, float).T)
    w, V = np.linalg.eigh(S)
    floor = floor_rel * max(float(w.max()), 1e-12)
    if w.min() > floor:
        return S
    return V @ np.diag(np.clip(w, floor, None)) @ V.T


def sample_knockoffs(
    X: np.ndarray,
    Sigma: np.ndarray,
    s: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw Xtilde | X ~ N(X - X Sigma^{-1} D,  2D - D Sigma^{-1} D),  D = diag(s).

    This is the Gaussian conditional sampler of Sec. 3.1.1.  Because the joint
    (X, Xtilde) is then Gaussian with covariance G = [[Sigma, Sigma-D],
    [Sigma-D, Sigma]], swapping any subset of columns with their knockoffs
    leaves the law unchanged (Def. 2, property (a)).
    """
    if rng is None:
        rng = np.random.default_rng()
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    D = np.diag(np.asarray(s, dtype=float))

    Sigma = _safe_pd(Sigma)
    L, low = cho_factor(Sigma, lower=True)
    Sinv_D = cho_solve((L, low), D)          # Sigma^{-1} D

    mu = X - X @ Sinv_D
    V = 2.0 * D - D @ Sinv_D
    V = 0.5 * (V + V.T)

    # Cholesky with progressive jitter (V can be near-singular at the boundary)
    eye = np.eye(p)
    jitter = 1e-10 * max(np.trace(V) / p, 1.0)
    Lv = None
    for _ in range(12):
        try:
            Lv = cholesky(V + jitter * eye, lower=True)
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    if Lv is None:
        raise RuntimeError("V is not positive definite; reduce `s` (e.g. lower safety).")

    Z = rng.standard_normal((n, p))
    return mu + Z @ Lv.T


# --------------------------------------------------------------------------- #
# 3. The LCD feature statistic  (Sec. 3.2)
# --------------------------------------------------------------------------- #
def lcd_statistic(
    X: np.ndarray,
    Xt: np.ndarray,
    y: np.ndarray,
    task: str = "auto",
    cv: int = 5,
    n_alphas: int = 60,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """W_j = |b_j| - |b_{j+p}| from a sparse fit on the AUGMENTED design.

    The tuning parameter is chosen by cross-validation on (X, Xtilde) so that
    the flip-sign property still holds.  Note (paper Sec. 4.1.1): use the
    lambda/C that MINIMISES the cross-validated error -- scikit-learn's
    LassoCV / LogisticRegressionCV already do this by default; do not switch
    to a "1 standard error" rule.

    Returns (W, info).
    """
    X = np.asarray(X, float)
    Xt = np.asarray(Xt, float)
    y = np.asarray(y).ravel()
    n, p = X.shape
    Xa = np.hstack([X, Xt])

    if task == "auto":
        task = "binomial" if np.unique(y).size == 2 else "gaussian"

    if task == "gaussian":
        from sklearn.linear_model import LassoCV

        # scikit-learn >= 1.7 renamed n_alphas -> alphas
        try:
            model = LassoCV(cv=cv, alphas=n_alphas, n_jobs=-1, random_state=seed,
                            max_iter=200_000).fit(Xa, y)
        except TypeError:
            model = LassoCV(cv=cv, n_alphas=n_alphas, n_jobs=-1, random_state=seed,
                            max_iter=200_000).fit(Xa, y)
        coef = model.coef_
        info = {"task": task, "tuning": float(model.alpha_)}
    elif task == "binomial":
        from sklearn.linear_model import LogisticRegressionCV

        model = LogisticRegressionCV(
            Cs=20, cv=cv, penalty="l1", solver="saga", n_jobs=-1,
            max_iter=5000, random_state=seed, refit=True,
        ).fit(Xa, y)
        coef = model.coef_.ravel()
        info = {"task": task, "tuning": float(model.C_[0])}
    else:
        raise ValueError(f"unknown task: {task!r}")

    W = np.abs(coef[:p]) - np.abs(coef[p:])
    return W, info


# --------------------------------------------------------------------------- #
# 4. Threshold inversion  (Sec. 3.3)
# --------------------------------------------------------------------------- #
def knockoff_threshold(W: np.ndarray, q: float, plus_one: bool = True) -> float:
    """tau = min{ t in W^+ : (1 + #{W <= -t}) / #{W >= t} <= q }   (knockoffs+)

    With plus_one=False the numerator is #{W <= -t} and the procedure controls
    the MODIFIED FDR, E[V/(|S|hat + 1/q)] <= q.  With plus_one=True it controls
    the ordinary FDR, E[V/(|S|hat v 1)] <= q.  Returns np.inf when the set is
    empty (no discoveries).
    """
    W = np.asarray(W, float)
    cand = np.unique(np.abs(W[W != 0]))          # W^+ , ascending
    for t in cand:                               # ascending => smallest t first
        R = int(np.sum(W >= t))
        if R == 0:
            continue
        F = int(np.sum(W <= -t))
        num = (1 + F) if plus_one else F
        if num / R <= q:
            return float(t)
    return np.inf


# --------------------------------------------------------------------------- #
# 5. End-to-end
# --------------------------------------------------------------------------- #
def knockoff_filter(
    X: np.ndarray,
    y: np.ndarray,
    q: float = 0.1,
    method: str = "asdp",
    blocks=None,
    block_size: int = 25,
    Sigma: np.ndarray | None = None,
    task: str = "auto",
    plus_one: bool = True,
    cv: int = 5,
    n_alphas: int = 60,
    n_ref: float | None = None,
    kappa: float = 1.0,
    seed: int = 0,
    n_knockoff_draws: int = 1,
) -> dict:
    """Run the full model-X knockoff pipeline on (X, y).

    Parameters
    ----------
    X : (n, p) covariate matrix, one row per subject.  Standardised internally.
    y : (n,) response.  Arbitrary dependence on X -- linear, logistic,
        non-linear, heteroscedastic: the FDR guarantee does not care.
    Sigma : (p, p) covariate covariance used to BUILD the knockoffs.  This is
        the quantity you must know or estimate from an INDEPENDENT reference
        cohort.  If None, the in-sample covariance is used -- which is the
        over-fitted regime of paper Sec. 5: FDR stays controlled but power
        collapses when p is not much smaller than n.
    n_knockoff_draws : repeat the knockoff draw and aggregate by taking the
        median W (a cheap stabiliser; the threshold is applied afterwards).

    Returns a dict with W, tau, selected indices, and diagnostics.
    """
    X = np.asarray(X, float)
    n, p = X.shape

    # --- standardise BEFORE building knockoffs (paper Sec. 3.4.2 convention)
    mu_x = X.mean(axis=0)
    sd_x = X.std(axis=0, ddof=1)
    sd_x[sd_x == 0] = 1.0
    Xs = (X - mu_x) / sd_x

    if Sigma is None:
        Sigma = np.corrcoef(Xs, rowvar=False)
        Sigma = np.atleast_2d(Sigma)
    # make sure Sigma is symmetric positive definite: a reference cohort
    # routinely yields a singular or numerically indefinite Sigmahat
    Sigma = _safe_pd(np.asarray(Sigma, float))

    s = solve_s_diag(Sigma, method=method, blocks=blocks,
                     block_size=block_size, n_ref=n_ref, kappa=kappa)

    rng = np.random.default_rng(seed)
    Ws = []
    for _ in range(n_knockoff_draws):
        Xt = sample_knockoffs(Xs, Sigma, s, rng=rng)
        W, info = lcd_statistic(Xs, Xt, y, task=task, cv=cv,
                                n_alphas=n_alphas, seed=seed)
        Ws.append(W)
    W = np.median(np.vstack(Ws), axis=0) if n_knockoff_draws > 1 else Ws[0]

    tau = knockoff_threshold(W, q, plus_one=plus_one)
    selected = np.where(W >= tau)[0] if np.isfinite(tau) else np.array([], int)

    return {
        "W": W,
        "tau": tau,
        "selected": selected,
        "s": s,
        "Sigma": Sigma,
        "q": q,
        "info": info,
        "n": n,
        "p": p,
        "method": method,
    }
