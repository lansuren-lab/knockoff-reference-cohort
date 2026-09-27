"""
exp_v3_supporting.py -- 为 v3 手稿里尚未有产物的数字补上产物

手稿里每一个数字都必须能追到某个 CSV 的具体单元格（见 scripts/verify_claims.py）。
v3 新增了四类此前只在交互式会话里算过的数字，本脚本把它们落盘：

 1. results_v3/gsr_rank_robustness.csv
    「GSR 在 ABIDE I 抬高有效秩、在 ABIDE II 降低」是否稳健？
    用 5 次随机参考队列切分，报告均值与标准差；并报告 ABIDE II 的 T 分布
    （用来排除"秩上限"这个解释）。

 2. results_v3/partial_rank_robustness.csv
    偏相关特征的高有效秩是否只是正则化造出来的？
    在 T > p 的受试者上，用收缩强度 0 / 0.001 / ... / 0.5 重算，
    报告每一步的有效秩。收缩强度 0 即未正则化。

 3. results_v3/gm_mechanism.csv
    Gaussian Mirrors 为什么功效低：每个变量被其余变量解释多少。
    报告每个 ROI 在其自身残差上的范数、||x_j||，以及 R^2。

 4. results_v3/pipeline_summary_copy.csv
    跨管线一致性（本应已由 cross_pipeline.py 产出，这里落一份到统一位置）。

Run:  python exp_v3_supporting.py
"""
from __future__ import annotations

import csv
import glob
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.refcohort import (corr_of, effective_rank, ledoit_sigma, prepare,
                           provenance, split_xy, write_csv)

OUT = "results_v3"
os.makedirs(OUT, exist_ok=True)


def gsr_rank_robustness(n_splits=5):
    """GSR 对有效秩的方向是否稳健？附带 T 分布以排除秩上限解释。"""
    rows = []
    for ds in ("feat2_abide1_aal", "feat2_abide2"):
        X, names, man, ism, ms = prepare(ds, fname="X_meanconn.csv")
        Xg, _, _, _, _ = prepare(ds, fname="X_gsr.csv")
        Ts = np.array([int(r["n_tr"]) for r in man])
        p = X.shape[1]
        mc, gs = [], []
        for seed in range(n_splits):
            rng = np.random.default_rng(seed)
            idx = rng.permutation(X.shape[0])[:min(393, X.shape[0])]
            mc.append(effective_rank(corr_of(X[idx])))
            gs.append(effective_rank(corr_of(Xg[idx])))
        mc, gs = np.array(mc), np.array(gs)
        rows.append({
            "dataset": ds, "p": p, "n": X.shape[0], "splits": n_splits,
            "meanconn_effrank_mean": float(mc.mean()),
            "meanconn_effrank_sd": float(mc.std()),
            "gsr_effrank_mean": float(gs.mean()),
            "gsr_effrank_sd": float(gs.std()),
            "direction": "raises" if gs.mean() > mc.mean() else "lowers",
            "T_min": int(Ts.min()), "T_median": float(np.median(Ts)),
            "T_max": int(Ts.max()),
            "n_subjects_T_le_p": int((Ts <= p).sum()),
        })
        print(f"  {ds:18s} meanconn {mc.mean():6.2f}±{mc.std():.2f} -> "
              f"gsr {gs.mean():6.2f}±{gs.std():.2f}  ({rows[-1]['direction']})"
              f"   T<=p: {rows[-1]['n_subjects_T_le_p']}/{len(Ts)}")
    return write_csv(os.path.join(OUT, "gsr_rank_robustness.csv"), rows)


def partial_rank_robustness(n_subjects=200, lams=(0.0, 0.001, 0.01, 0.05,
                                                   0.2, 0.5)):
    """在 T > p 的受试者上，扫描收缩强度，报告偏相关特征的有效秩。"""
    mats = []
    for fn in sorted(glob.glob("abide_rois_aal/*.1D")):
        L = [l for l in open(fn) if l.strip()]
        T = np.array([[float(x) for x in l.split()] for l in L[1:]])
        if (T.std(0, ddof=1) < 1e-8).any():
            continue
        if T.shape[0] <= T.shape[1]:
            continue
        mats.append(T)
        if len(mats) >= n_subjects:
            break

    def feat(lam):
        out = []
        for T in mats:
            C = corr_of(T)
            S = (1 - lam) * C + lam * np.eye(C.shape[0])
            P = np.linalg.inv(S)
            d = np.sqrt(np.abs(np.diag(P)))
            PC = -P / np.outer(d, d)
            np.fill_diagonal(PC, 0.0)
            out.append(np.arctanh(np.clip(PC, -0.999, 0.999)).mean(1))
        return np.vstack(out)

    def lw_feat():
        from sklearn.covariance import LedoitWolf
        out = []
        for T in mats:
            S = LedoitWolf().fit(T).covariance_
            d0 = np.sqrt(np.clip(np.diag(S), 1e-300, None))
            Sc = S / np.outer(d0, d0)
            P = np.linalg.pinv(Sc)
            d = np.sqrt(np.abs(np.diag(P)))
            PC = -P / np.outer(d, d)
            np.fill_diagonal(PC, 0.0)
            out.append(np.arctanh(np.clip(PC, -0.999, 0.999)).mean(1))
        return np.vstack(out)

    rows = []
    for lam in lams:
        er = effective_rank(corr_of(feat(lam)))
        rows.append({"shrinkage": lam, "effective_rank": er,
                     "n_subjects": len(mats), "estimator": "fixed"})
        print(f"  收缩强度 lam={lam:<6.3f}  偏相关特征有效秩={er:7.2f}")
    er = effective_rank(corr_of(lw_feat()))
    rows.append({"shrinkage": -1.0, "effective_rank": er,
                 "n_subjects": len(mats), "estimator": "LedoitWolf"})
    print(f"  Ledoit--Wolf 自适应        偏相关特征有效秩={er:7.2f}")
    return write_csv(os.path.join(OUT, "partial_rank_robustness.csv"), rows)


def gm_mechanism():
    """Gaussian Mirrors 的扰动尺度由每个变量对其余变量的残差决定。"""
    X, names, man, ism, ms = prepare("feat2_abide1_aal")
    Xm, ym, Xr = split_xy(X, man, ism)
    mu, sd = Xm.mean(0), Xm.std(0, ddof=1)
    sd[sd == 0] = 1.0
    Xs = (Xm - mu) / sd
    XtX_inv = np.linalg.pinv(Xs.T @ Xs)
    # 残差平方和 = 1/[(X'X)^{-1}]_jj（VIF 的倒数）
    rss = 1.0 / np.diag(XtX_inv)
    norm_x = np.sqrt(np.einsum("ij,ij->j", Xs, Xs))
    resid_norm = np.sqrt(rss)
    r2 = 1.0 - rss / (norm_x ** 2)
    rows = [{"roi": names[j], "norm_x": float(norm_x[j]),
             "resid_norm": float(resid_norm[j]), "r2_on_others": float(r2[j])}
            for j in range(len(names))]
    print(f"  ||x_j||: 中位 {np.median(norm_x):.2f}")
    print(f"  残差范数 ||P_-j x_j||: 中位 {np.median(resid_norm):.2f}")
    print(f"  被其余变量解释的比例 R^2 中位: {np.median(r2):.4f}")
    return write_csv(os.path.join(OUT, "gm_mechanism.csv"), rows)


def pipeline_copy():
    src = "knockoff-reference-cohort/results/pipeline_summary.csv"
    if not os.path.exists(src):
        print("  [skip] 找不到 pipeline_summary.csv")
        return None
    rows = list(csv.DictReader(open(src)))
    r = np.array([float(x["per_roi_corr"]) for x in rows])
    tc = np.array([float(x["t_cpac"]) for x in rows])
    tn = np.array([float(x["t_niak"]) for x in rows])
    out = [{"n_roi": len(r), "median_per_roi_corr": float(np.median(r)),
            "n_below_0.5": int((r < 0.5).sum()),
            "t_stat_r": float(np.corrcoef(tc, tn)[0, 1])}]
    print(f"  per-ROI 中位 r={out[0]['median_per_roi_corr']:.4f}  "
          f"<0.5 的 ROI={out[0]['n_below_0.5']}/{len(r)}  "
          f"t 统计量 r={out[0]['t_stat_r']:.4f}")
    return write_csv(os.path.join(OUT, "pipeline_agreement.csv"), out)


if __name__ == "__main__":
    print("=== 1. GSR 有效秩方向稳健性 ===")
    gsr_rank_robustness()
    print("\n=== 2. 偏相关有效秩 vs 收缩强度 ===")
    partial_rank_robustness()
    print("\n=== 3. Gaussian Mirrors 机制（每变量被其余解释多少）===")
    gm_mechanism()
    print("\n=== 4. 跨管线一致性 ===")
    pipeline_copy()
    provenance(OUT, "exp_v3_supporting", None)
    print(f"\n-> {OUT}/")
