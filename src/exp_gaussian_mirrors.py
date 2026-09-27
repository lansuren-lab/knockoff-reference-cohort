"""
exp_gaussian_mirrors.py -- 第1个替代手段：Gaussian Mirrors

出处：Xing, Zhao & Liu, "Controlling False Discovery Rate Using Gaussian
Mirrors", JASA 2023 (arXiv:1911.09761)。

为什么值得试：摘要里有一句几乎是针对我们写的 ——
    "more powerful than many existing methods ... especially for cases when
     the covariates are highly correlated and the influential variables are
     not overly sparse"

而且它与 knockoff 有一个本质区别：**完全不求 Sigma^{-1}**。我们整条
"退化 -> lambda_min 极小 -> s 塌缩" 的因果链根本不进入计算。

方法（论文 Algorithm 1，OLS 版）：
    对每个 j（可并行）：
      (a) 抽 z_j ~ N(0, I_n)
      (b) 由 Lemma 2 算 c_j = ||(I-H_{-j}) x_j|| / ||(I-H_{-j}) z_j||
          其中 H_{-j} 是 X_{-j} 的帽子矩阵。等价地
              x_j' (I-H_{-j}) x_j = 1 / [(X'X)^{-1}]_{jj}    （VIF 的倒数）
      (c) 在 [X_{-j}, x_j^+ = x_j + c_j z_j, x_j^- = x_j - c_j z_j] 上做 OLS
      (d) 镜像统计量 M_j = |b_j^+ + b_j^-| - |b_j^+ - b_j^-|
    阈值：tau_q = min{t : #{M_j <= -t} / (#{M_j >= t} v 1) <= q}
    选中：{j : M_j >= tau_q}

注意阈值形式与 knockoff+ 不同：**分子没有 +1**，因为零假设下 M_j 是
【严格对称】的（Lemma 1），不只需要符号对称。

本脚本在三种响应下与我们的等相关 knockoff 正面对比：
    (A) 正对照：注入已知信号（条件 |t| 标定）
    (B) 全局零假设：y 与 X 独立 -> 任何发现按构造都是假的
    (C) 真实 ASD/对照 响应

Run:  python exp_gaussian_mirrors.py --reps 200 --workers 14
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


def gm_statistics(X, y, rng, use_vif_identity=True):
    """Gaussian Mirrors 的 M_j 向量；X 已中心化、y 已中心化。"""
    n, p = X.shape
    M = np.empty(p)
    if use_vif_identity:
        XtX_inv = np.linalg.pinv(X.T @ X)
    for j in range(p):
        Xm = np.delete(X, j, axis=1)
        xj = X[:, j]
        zj = rng.standard_normal(n)
        Q, _ = np.linalg.qr(Xm, mode="reduced")
        rx = xj - Q @ (Q.T @ xj)
        rz = zj - Q @ (Q.T @ zj)
        nz = np.linalg.norm(rz)
        cj = np.linalg.norm(rx) / nz if nz > 1e-12 else 0.0
        A = np.column_stack([Xm, xj + cj * zj, xj - cj * zj])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        bp, bm = coef[-2], coef[-1]
        M[j] = abs(bp + bm) - abs(bp - bm)
    return M


def gm_threshold(M, q):
    """tau_q = min{t : #{M<=-t}/(#{M>=t} v 1) <= q}，无 +1。"""
    cand = np.sort(np.unique(M[M != 0]))
    for t in cand[::-1]:
        R = int((M >= t).sum())
        F = int((M <= -t).sum())
        if R == 0:
            continue
        if F / max(R, 1) <= q:
            return float(t), np.where(M >= t)[0]
    return np.inf, np.array([], int)


def gm_run(X, y, q=Q_DEFAULT, seed=0):
    mu, sd = X.mean(0), X.std(0, ddof=1)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    yc = y - y.mean()
    rng = np.random.default_rng(seed)
    M = gm_statistics(Xs, yc, rng)
    tau, sel = gm_threshold(M, q)
    return {"sel": set(sel.tolist()), "tau": tau,
            "n_bad": int((M <= -abs(tau)).sum()) if np.isfinite(tau) else 0}


_W = {}


def _init_w(Xm, ym, Xr, c, seed):
    global _W
    _W = dict(Xm=Xm, ym=ym, Xr=Xr, c=c, seed=seed)


def _make_y(kind, w, r, supp, target=6.0):
    """三种响应之一。注意 beta 与 y 的构造与 knockoff 侧完全共用，
    保证两法面对的是同一份数据。"""
    Xm = w["Xm"]
    rng = np.random.default_rng(w["seed"] * 100000 + r)
    if kind == "ctrl":
        noise = rng.standard_normal(len(Xm))
        sd = Xm.std(0, ddof=1)
        sd[sd == 0] = 1.0
        Xs = (Xm - Xm.mean(0)) / sd
        _, des = conditional_t_all(Xs, w["ym"])
        a = calibrate_amplitude(Xs, des, supp, target, noise)
        beta = np.zeros(Xm.shape[1])
        beta[list(supp)] = a
        return Xs @ beta + noise
    if kind == "null":
        return rng.standard_normal(len(Xm))          # 与 X 独立
    return w["ym"]                                    # real


def _job(job):
    kind, r = job
    w = _W
    p = w["Xm"].shape[1]
    rng = np.random.default_rng(w["seed"] * 100000 + r)
    supp = tuple(sorted(rng.choice(p, 12, replace=False).tolist()))
    y = _make_y(kind, w, r, supp)

    # --- Gaussian Mirrors ---
    g = gm_run(w["Xm"], y, q=Q_DEFAULT, seed=w["seed"] * 1000 + r)
    # --- 我们的等相关 knockoff（同一份 y，同一份数据）---
    S = ledoit_sigma(w["Xr"])
    res = knockoff_run(w["Xm"], y, S, q=Q_DEFAULT,
                       seed=w["seed"] * 1000 + r, c=w["c"])

    out = {"kind": kind, "rep": r, "gm_nsel": len(g["sel"]),
           "ko_nsel": len(res["sel"]), "ko_conv": float(res["converged"]),
           "gm_tau_finite": float(np.isfinite(g["tau"]))}
    if kind == "ctrl":
        sup = set(supp)
        out["gm_power"] = len(g["sel"] & sup) / 12.0
        out["ko_power"] = len(set(res["sel"].tolist()) & sup) / 12.0
        out["gm_fdp"] = (len(g["sel"] - sup) / len(g["sel"])) if g["sel"] else 0.0
        out["ko_fdp"] = ((len(set(res["sel"].tolist()) - sup)
                          / len(res["sel"])) if len(res["sel"]) else 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--c", type=float, default=C_FIXED)
    ap.add_argument("--outdir", default="results_gm")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(args.outdir, exist_ok=True)

    X, names, man, ism, ms = prepare("feat2_abide1_aal")
    Xm, ym, Xr = split_xy(X, man, ism)
    print(f"ABIDE1-AAL116 meanconn  p={X.shape[1]}  n_main={len(Xm)}"
          f"  n_ref={len(Xr)}   对比: Gaussian Mirrors vs 等相关 knockoff\n")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    from collections import defaultdict
    rows = []
    for kind, label in (("ctrl", "正对照：注入已知信号（条件|t|≈6, k=12）"),
                        ("null", "全局零假设：y 与 X 独立 —— 任何发现都是假的"),
                        ("real", "真实 ASD/对照 响应")):
        print("=" * 78)
        print(label)
        print("=" * 78)
        acc = defaultdict(list)
        jobs = [(kind, r) for r in range(args.reps)]
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_init_w,
                                 initargs=(Xm, ym, Xr, args.c,
                                           args.seed)) as ex:
            futs = [ex.submit(_job, j) for j in jobs]
            for i, f in enumerate(as_completed(futs), 1):
                rr = f.result()
                acc["all"].append(rr)
                if i % 50 == 0:
                    print(f"    {i}/{len(jobs)}  ({time.time()-t0:.0f}s)",
                          flush=True)
        g = acc["all"]
        row = {"kind": kind, "reps": len(g),
               "gm_p_nonempty": float(np.mean([1.0 if x["gm_nsel"] > 0 else 0.0
                                               for x in g])),
               "gm_nsel_median": float(np.median([x["gm_nsel"] for x in g])),
               "gm_nsel_max": int(np.max([x["gm_nsel"] for x in g])),
               "ko_p_nonempty": float(np.mean([1.0 if x["ko_nsel"] > 0 else 0.0
                                               for x in g])),
               "ko_nsel_median": float(np.median([x["ko_nsel"] for x in g])),
               "ko_nsel_max": int(np.max([x["ko_nsel"] for x in g])),
               "ko_frac_converged": float(np.mean([x["ko_conv"] for x in g]))}
        if kind == "ctrl":
            row.update({"gm_power": float(np.mean([x["gm_power"] for x in g])),
                        "ko_power": float(np.mean([x["ko_power"] for x in g])),
                        "gm_fdp": float(np.mean([x["gm_fdp"] for x in g])),
                        "ko_fdp": float(np.mean([x["ko_fdp"] for x in g]))})
        rows.append(row)
        print(f"    GM:  P(非空)={row['gm_p_nonempty']:.3f}  "
              f"|S|中位={row['gm_nsel_median']:.1f}  最大={row['gm_nsel_max']}")
        print(f"    KO:  P(非空)={row['ko_p_nonempty']:.3f}  "
              f"|S|中位={row['ko_nsel_median']:.1f}  最大={row['ko_nsel_max']}")
        if kind == "ctrl":
            print(f"    GM power={row['gm_power']:.3f} FDP={row['gm_fdp']:.3f}"
                  f"   |  KO power={row['ko_power']:.3f} FDP={row['ko_fdp']:.3f}")
        print()

    path = write_csv(os.path.join(args.outdir, "gm_vs_knockoff.csv"), rows)
    provenance(args.outdir, "exp_gaussian_mirrors", args)
    print(f"-> {path}\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
