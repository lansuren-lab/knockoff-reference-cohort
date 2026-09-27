"""
exp_supporting_numbers.py -- 为手稿里两个尚未有产物的数字补上产物

手稿 v2 引用了两类数字，但此前只在临时的交互式会话里算过，没有留下可核对
的文件。按本项目自己的规矩（见 scripts/verify_claims.py 的说明），任何进入
手稿的数字都必须能追到某个 CSV 的具体单元格，所以这里补上：

  1. results/pc1_dominance.csv
     「每个特征在 PC1 上的中位 R² = 0.694」以及各定义的 PC1 占比。
     这是"meanconn 为什么退化"的直达证据。

  2. results/penalty_sweep.csv
     「c 必须同时满足不过罚与能收敛」那张表：注入已知强度的信号，扫 c，
     记录找回个数、套索不收敛比例、中位迭代数。

Run:  python exp_supporting_numbers.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.refcohort import (DEF_DATASETS_FOR_DIAG, corr_of, knockoff_run,      # noqa: E402
                           prepare, provenance, split_xy, write_csv)
from exp_positive_control_v3 import calibrate_amplitude                     # noqa: E402
from src.refcohort import conditional_t_all                                 # noqa: E402

DEFS = ("meanconn", "gsr", "pcresid", "partial")
C_GRID = (0.02, 0.03, 0.05, 0.10, 0.15, 0.30)


def pc1_dominance(outdir):
    """每个特征在自身第一主成分上的 R²（PC1 主导程度）。"""
    rows = []
    for ds, d in DEF_DATASETS_FOR_DIAG.items():
        for deff in DEFS:
            try:
                X, names, man, ism, ms = prepare(d, fname=f"X_{deff}.csv")
            except Exception:
                continue
            _, _, Xr = split_xy(X, man, ism)
            if Xr.shape[0] < 50:
                continue
            Xs = (Xr - Xr.mean(0)) / Xr.std(0, ddof=1)
            U, S, _ = np.linalg.svd(Xs, full_matrices=False)
            pc1 = U[:, 0] * S[0]
            r2 = np.array([np.corrcoef(Xs[:, j], pc1)[0, 1] ** 2
                           for j in range(Xs.shape[1])])
            C = corr_of(Xr)
            w = np.linalg.eigvalsh(C)
            rows.append({"dataset": ds, "feature": deff, "p": Xr.shape[1],
                         "pc1_share": float(w.max() / w.sum()),
                         "r2_on_pc1_median": float(np.median(r2)),
                         "r2_on_pc1_min": float(r2.min()),
                         "r2_on_pc1_max": float(r2.max())})
    return write_csv(os.path.join(outdir, "pc1_dominance.csv"), rows), rows


def penalty_sweep(outdir, reps=60, seed=0):
    """扫 c：注入已知条件 |t| 的信号，记录找回数与收敛情况。"""
    X, names, man, ism, ms = prepare("feat2_abide1_aal")
    Xm, ym, Xr = split_xy(X, man, ism)
    p = X.shape[1]
    mu, sd = Xm.mean(0), Xm.std(0, ddof=1)
    sd[sd == 0] = 1.0
    Xs = (Xm - mu) / sd
    _, design = conditional_t_all(Xs, ym)
    from src.refcohort import ledoit_sigma
    S = ledoit_sigma(Xr)

    rows = []
    for c in C_GRID:
        rec, conv, iters = [], [], []
        sup_is_probe = True
        for r in range(reps):
            rr = np.random.default_rng(seed * 100000 + r)
            supp = tuple(sorted(rr.choice(p, 12, replace=False).tolist()))
            noise = rr.standard_normal(len(Xs))
            a = calibrate_amplitude(Xs, design, supp, 6.0, noise)
            beta = np.zeros(p)
            beta[list(supp)] = a
            y = Xs @ beta + noise
            res = knockoff_run(Xm, y, S, q=0.20, seed=seed * 1000 + r, c=c)
            sel = set(res["sel"].tolist())
            rec.append(len(sel & set(supp)))
            conv.append(float(res["converged"]))
            iters.append(res["n_iter"])
        rows.append({"c": c, "k_injected": 12, "target_cond_t": 6.0,
                     "mean_recovered": float(np.mean(rec)),
                     "frac_recovered_12of12": float(np.mean(
                         [1.0 if x == 12 else 0.0 for x in rec])),
                     "frac_nonconverged": 1.0 - float(np.mean(conv)),
                     "median_n_iter": float(np.median(iters)), "reps": reps})
        print(f"  c={c:4.2f}  找回 {rows[-1]['mean_recovered']:5.2f}/12  "
              f"未收敛 {rows[-1]['frac_nonconverged']:6.1%}  "
              f"中位迭代 {rows[-1]['median_n_iter']:.0f}", flush=True)
    del sup_is_probe
    return write_csv(os.path.join(outdir, "penalty_sweep.csv"), rows), rows


def real_response_convergence(outdir, reps=60, seed=0):
    """c 的第二半条件：在【真实(弱信号)响应】上套索必须收敛。

    penalty_sweep 用的是注入信号，lambda 大、解稀疏、收敛轻松，因此那里
    任何 c 都不出现不收敛。但真实 ASD/对照响应很弱 -> lambda 很小 -> 解
    稠密 -> 坐标下降可能在上限内不收敛。两个条件是分开测的，必须都报。

    这正是 §penalty 那张表里「未收敛」一列的真实来源。
    """
    X, names, man, ism, ms = prepare("feat2_abide1_aal")
    Xm, ym, Xr = split_xy(X, man, ism)
    from src.refcohort import ledoit_sigma
    arms = {"empirical": __import__("src.refcohort", fromlist=["x"]).pd_floor(
                corr_of(Xr)),
            "LW": ledoit_sigma(Xr),
            "in_sample": ledoit_sigma(Xm)}
    rows = []
    for c in C_GRID:
        for tag, S in arms.items():
            conv, iters, nsel = [], [], []
            for r in range(reps):
                res = knockoff_run(Xm, ym, S, q=0.20, seed=seed * 1000 + r, c=c)
                conv.append(float(res["converged"]))
                iters.append(res["n_iter"])
                nsel.append(len(res["sel"]))
            rows.append({"c": c, "Sigma": tag,
                         "frac_nonconverged": 1.0 - float(np.mean(conv)),
                         "median_n_iter": float(np.median(iters)),
                         "median_nsel": float(np.median(nsel)), "reps": reps})
        sub = [r for r in rows if r["c"] == c]
        print(f"  c={c:4.2f}  真实响应未收敛 "
              f"{max(r['frac_nonconverged'] for r in sub):6.1%} "
              f"(最差臂)  中位迭代 "
              f"{max(r['median_n_iter'] for r in sub):.0f}", flush=True)
    return write_csv(os.path.join(outdir, "penalty_convergence_real.csv"),
                     rows), rows


def main():
    outdir = "results_c0.05"
    p1, rows1 = pc1_dominance(outdir)
    print(f"-> {p1}")
    for r in rows1:
        if r["dataset"] == "ABIDE1-AAL116-CPAC":
            print(f"   {r['feature']:9s} PC1={r['pc1_share']:.3f}  "
                  f"中位 R2={r['r2_on_pc1_median']:.3f}")
    p2, rows2 = penalty_sweep(outdir)
    print(f"-> {p2}")
    p3, rows3 = real_response_convergence(outdir)
    print(f"-> {p3}")
    provenance(outdir, "exp_supporting_numbers", None)


if __name__ == "__main__":
    main()
