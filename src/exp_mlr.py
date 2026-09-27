"""
exp_mlr.py -- 第2个替代手段：Masked Likelihood Ratio (MLR) 统计量

出处：Spector & Fithian, "Asymptotically Optimal Knockoff Statistics via the
Masked Likelihood Ratio"（arXiv:2212.08766, 2024）。

为什么值得试：摘要里点名批评了我们正在用的那一类统计量 ——
    "state-of-the-art lasso-based feature statistics often prioritize features
     that are unlikely to be discovered, leading to low power"

MLR **没有罚参数**、渐近最优、且对维度无限制。我们 §penalty 那一整节
（c 必须同时满足"不过罚"与"能收敛"）会直接消失。

定义（式 1.3）：
    MLR_j = log[ P_j^pi(X_j | D) / P_j^pi(X~_j | D) ],  D = (Y, {X_j, X~_j})

实现（论文 Algorithm 1，Gibbs 采样）：
    for i in 1..nsample:
        for j in 1..p:
            eta_j = log P_{Y|X}^{(θi)}(Y | [X_{-j}, X_j])
                  - log P_{Y|X}^{(θi)}(Y | [X_{-j}, X~_j])
            p_j = logit^{-1}(eta_j)
            X_j <- X_j 以概率 p_j，否则 <- X~_j
        从 θ* | Y, X 的后验重抽 θ
    MLR_j = log(sum_i p_j^(i)) - log(sum_i (1 - p_j^(i)))

【本实现的两个关键化简】
1. 线性高斯模型 + 已知 θ=β 时，似然比只剩残差平方和之差。而把第 j 列由
   X_j 换成 X~_j 只让残差改变 (X_j - X~_j) * beta_j，所以
       eta_j = [2 r·d_j beta_j + ||d_j||^2 beta_j^2] / (2 sigma^2),  d_j = X_j - X~_j
   是 O(n) 的，不需要重算回归。
2. 共轭先验下后验有闭式：beta | . ~ N(mu, V)，
   V = (X'X/sigma^2 + I/tau^2)^{-1}，mu = V X'y/sigma^2；
   sigma^2 | . ~ IG(a0 + n/2, b0 + RSS/2)。p=113 下一次性求逆即可。
   这比论文用的通用贝叶斯采样器简单，但符合它 Appendix E.3 允许的简化。

Run:  python exp_mlr.py --reps 200 --workers 14
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.refcohort import (C_FIXED, Q_DEFAULT, conditional_t_all,              # noqa: E402
                           knockoff_run, ledoit_sigma, prepare, provenance,
                           split_xy, write_csv)
from exp_positive_control_v3 import calibrate_amplitude                       # noqa: E402
from knockoff_adni.knockoffs import solve_s_diag, sample_knockoffs            # noqa: E402


def mlr_statistics(X, Xt, y, n_sample=60, seed=0, tau2=1.0, a0=1.0, b0=1.0):
    """Gibbs 采样算 MLR 统计量。X, Xt 已标准化；y 已中心化。

    【实现要点 —— 两处极易写错的地方】
    1. 论文 Algorithm 1 第 1 行要求 X_j^(0) ~ Unif({X_j, X~_j}) 的【随机】初始化。
       若初始化成原始 X，链会卡在不动点（残差由 X 拟合 -> 每个 j 的似然比都
       偏向原列 -> MLR_j 全为正 -> 零假设统计量完全不对称）。
    2. 式 (4) 里的 eta_j 必须【始终】是
           log P(Y | [X_{-j}, X_j]) - log P(Y | [X_{-j}, X~_j])
       即 "原列减 knockoff"，与链当前处于哪个状态【无关】。
       若按"另一列减当前列"来算，符号会随状态翻转，采样器收敛到错误的分布。
       本实现显式跟踪 state[j] 来避免这一点。
    """
    n, p = X.shape
    rng = np.random.default_rng(seed)
    beta = np.zeros(p)
    sigma2 = float(np.var(y)) if np.var(y) > 0 else 1.0
    sum_p = np.zeros(p)
    sum_1mp = np.zeros(p)
    X_orig = X.copy()
    D = X_orig - Xt                       # d_j = X_j - X~_j（始终是"原减 knockoff"）

    state = np.zeros(p, dtype=bool)       # True = 当前位置上是原始 X_j
    X = X_orig.copy()
    for j in range(p):
        if rng.random() < 0.5:
            state[j] = True
        else:
            X[:, j] = Xt[:, j]

    for _ in range(n_sample):
        r = y - X @ beta
        for j in range(p):
            dd = D[:, j] * beta[j]
            rss_c = float(r @ r)                      # 当前状态下的 RSS
            rss_a = float((r + dd) @ (r + dd))        # 换成另一列后的 RSS
            if state[j]:                              # 当前是原列 -> 原=rss_c, ko=rss_a
                eta = (rss_c - rss_a) / (2.0 * sigma2)
            else:                                     # 当前是 knockoff -> 原=rss_a
                eta = (rss_a - rss_c) / (2.0 * sigma2)
            pj = 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))
            sum_p[j] += pj
            sum_1mp[j] += 1.0 - pj
            new_state = bool(rng.random() < pj)
            if new_state != state[j]:
                X[:, j] = X_orig[:, j] if new_state else Xt[:, j]
                state[j] = new_state
                r = y - X @ beta
        # 共轭后验重抽 beta 与 sigma2
        V = np.linalg.inv(X.T @ X / sigma2 + np.eye(p) / tau2)
        mu = V @ (X.T @ y) / sigma2
        beta = rng.multivariate_normal(mu, V)
        r = y - X @ beta
        sigma2 = max(float((b0 + float(r @ r) / 2.0)
                           / rng.gamma(a0 + n / 2.0, 1.0)), 1e-8)

    eps = 1e-12
    return np.log(sum_p + eps) - np.log(sum_1mp + eps)


def mlr_threshold(M, q):
    """knockoff+ 的阈值形式（MLR 是合法的 knockoff 统计量，具有 flip-sign 性质）。"""
    cand = np.sort(np.unique(np.abs(M[M != 0])))
    for t in cand[::-1]:
        R = int((M >= t).sum())
        if R == 0:
            continue
        F = int((M <= -t).sum())
        if (1 + F) / R <= q:
            return float(t), np.where(M >= t)[0]
    return np.inf, np.array([], int)


def mlr_run(X, Xt, y, q=Q_DEFAULT, seed=0, n_sample=60):
    mu, sd = X.mean(0), X.std(0, ddof=1)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    yc = y - y.mean()
    M = mlr_statistics(Xs, Xt, yc, n_sample=n_sample, seed=seed)
    tau, sel = mlr_threshold(M, q)
    return {"sel": set(sel.tolist()), "tau": tau, "M": M}


_W = {}


def _init_w(Xm, ym, Xr, c, seed, ns):
    global _W
    _W = dict(Xm=Xm, ym=ym, Xr=Xr, c=c, seed=seed, ns=ns)


def _job(job):
    kind, r = job
    w = _W
    Xm = w["Xm"]
    p = Xm.shape[1]
    rng = np.random.default_rng(w["seed"] * 100000 + r)
    supp = tuple(sorted(rng.choice(p, 12, replace=False).tolist()))
    # ---- 响应（三种），与 knockoff 侧共用 ----
    if kind == "ctrl":
        noise = rng.standard_normal(len(Xm))
        sd = Xm.std(0, ddof=1)
        sd[sd == 0] = 1.0
        Xs = (Xm - Xm.mean(0)) / sd
        _, des = conditional_t_all(Xs, w["ym"])
        a = calibrate_amplitude(Xs, des, supp, 6.0, noise)
        beta = np.zeros(p)
        beta[list(supp)] = a
        y = Xs @ beta + noise
    elif kind == "null":
        y = rng.standard_normal(len(Xm))
    else:
        y = w["ym"]

    # ---- 同一份 knockoff 给两法共用 ----
    S = ledoit_sigma(w["Xr"])
    s = solve_s_diag(S, method="equicorrelated")
    mu, sd = Xm.mean(0), Xm.std(0, ddof=1)
    sd[sd == 0] = 1.0
    Xs = (Xm - mu) / sd
    Xt = sample_knockoffs(Xs, S, s, rng=np.random.default_rng(w["seed"] * 7 + r))

    m = mlr_run(Xm, Xt, y, q=Q_DEFAULT, seed=w["seed"] * 1000 + r,
                n_sample=w["ns"])
    ko = knockoff_run(Xm, y, S, q=Q_DEFAULT, seed=w["seed"] * 1000 + r, c=w["c"])

    out = {"kind": kind, "rep": r, "mlr_nsel": len(m["sel"]),
           "ko_nsel": len(ko["sel"]), "ko_conv": float(ko["converged"])}
    if kind == "ctrl":
        sup = set(supp)
        out["mlr_power"] = len(m["sel"] & sup) / 12.0
        out["ko_power"] = len(set(ko["sel"].tolist()) & sup) / 12.0
        out["mlr_fdp"] = (len(m["sel"] - sup) / len(m["sel"])) if m["sel"] else 0.0
        out["ko_fdp"] = ((len(set(ko["sel"].tolist()) - sup) / len(ko["sel"]))
                         if len(ko["sel"]) else 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--c", type=float, default=C_FIXED)
    ap.add_argument("--n-sample", type=int, default=60, dest="ns")
    ap.add_argument("--outdir", default="results_mlr")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(args.outdir, exist_ok=True)

    X, names, man, ism, ms = prepare("feat2_abide1_aal")
    Xm, ym, Xr = split_xy(X, man, ism)
    print(f"ABIDE1-AAL116 meanconn  p={X.shape[1]}  n_main={len(Xm)}"
          f"  MLR Gibbs 采样 {args.ns} 轮   对比: MLR vs 固定-λ LCD\n")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    rows = []
    for kind, label in (("ctrl", "正对照：注入已知信号（条件|t|≈6, k=12）"),
                        ("null", "全局零假设：y 与 X 独立 —— 任何发现都是假的"),
                        ("real", "真实 ASD/对照 响应")):
        print("=" * 78)
        print(label)
        print("=" * 78)
        g = []
        jobs = [(kind, r) for r in range(args.reps)]
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_init_w,
                                 initargs=(Xm, ym, Xr, args.c, args.seed,
                                           args.ns)) as ex:
            futs = [ex.submit(_job, j) for j in jobs]
            for i, f in enumerate(as_completed(futs), 1):
                g.append(f.result())
                if i % 50 == 0:
                    print(f"    {i}/{len(jobs)}  ({time.time()-t0:.0f}s)",
                          flush=True)
        row = {"kind": kind, "reps": len(g),
               "mlr_p_nonempty": float(np.mean([1.0 if x["mlr_nsel"] > 0 else 0.0
                                                for x in g])),
               "mlr_nsel_median": float(np.median([x["mlr_nsel"] for x in g])),
               "mlr_nsel_max": int(np.max([x["mlr_nsel"] for x in g])),
               "ko_p_nonempty": float(np.mean([1.0 if x["ko_nsel"] > 0 else 0.0
                                               for x in g])),
               "ko_nsel_median": float(np.median([x["ko_nsel"] for x in g])),
               "ko_nsel_max": int(np.max([x["ko_nsel"] for x in g]))}
        if kind == "ctrl":
            row.update({"mlr_power": float(np.mean([x["mlr_power"] for x in g])),
                        "ko_power": float(np.mean([x["ko_power"] for x in g])),
                        "mlr_fdp": float(np.mean([x["mlr_fdp"] for x in g])),
                        "ko_fdp": float(np.mean([x["ko_fdp"] for x in g]))})
        rows.append(row)
        print(f"    MLR: P(非空)={row['mlr_p_nonempty']:.3f}  "
              f"|S|中位={row['mlr_nsel_median']:.1f}  最大={row['mlr_nsel_max']}")
        print(f"    KO : P(非空)={row['ko_p_nonempty']:.3f}  "
              f"|S|中位={row['ko_nsel_median']:.1f}  最大={row['ko_nsel_max']}")
        if kind == "ctrl":
            print(f"    MLR power={row['mlr_power']:.3f} FDP={row['mlr_fdp']:.3f}"
                  f"   |  KO power={row['ko_power']:.3f} FDP={row['ko_fdp']:.3f}")
        print()

    path = write_csv(os.path.join(args.outdir, "mlr_vs_knockoff.csv"), rows)
    provenance(args.outdir, "exp_mlr", args)
    print(f"-> {path}\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
