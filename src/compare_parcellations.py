"""
compare_parcellations.py -- 跨图谱稳健性检验

问题：我们在 AAL116 上发现「参考队列经验 Sigma 近奇异 -> 构造塌陷 -> 零发现；
      适度收缩能救回」。这个结论是 AAL 特有的，还是换图谱也成立？

做法：对每个图谱独立跑一遍完整流程，比较
      1. 参考队列 Sigma 的谱（lam_min / 条件数）
      2. 经验 vs Ledoit-Wolf vs 固定收缩 下的 s
      3. 阈值比 (1+F)/R —— 决定有没有发现
      4. 发现数与 top-W 区域

图谱：rois_aal(116) / rois_cc200(200) / rois_ho(111)

Run:  python compare_parcellations.py
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, "knockoff_adni")
sys.path.insert(0, ".")
from knockoffs import solve_s_diag, sample_knockoffs, knockoff_threshold  # noqa: E402
from validate_shrinkage_fdr import lcd_fixed_lambda, pd_floor  # noqa: E402

MAIN_SITES = ["NYU", "UM_1", "USM", "UCLA_1", "YALE", "PITT"]
MAX_ROI_MISS = 20          # 与主分析保持一致的清洗阈值
C_FIXED = 0.15             # 固定 lambda = c * lam_max

PARCELLATIONS = [
    ("AAL116", "features", "rois_aal"),
    ("CC200", "features_rois_cc200", "rois_cc200"),
    ("HarvardOxford", "features_rois_ho", "rois_ho"),
]


def corr_of(A):
    C = np.corrcoef(A, rowvar=False)
    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 1.0)
    return C


def ledoit_sigma(Xr):
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf().fit(Xr).covariance_
    d = np.sqrt(np.clip(np.diag(lw), 1e-300, None))
    return pd_floor(lw / np.outer(d, d))


def load_clean(dirname):
    f = os.path.join(dirname, "X_meanconn.csv")
    X = np.loadtxt(f, delimiter=",", skiprows=1)
    names = open(f).readline().strip().split(",")
    man = list(csv.DictReader(open(os.path.join(dirname, "subject_manifest.csv"))))
    n, p = X.shape

    roi_miss = np.isnan(X).sum(axis=0)
    keep_roi = roi_miss <= MAX_ROI_MISS
    keep_sub = (~np.isnan(X[:, keep_roi])).all(axis=1)
    Xc = X[np.ix_(keep_sub, keep_roi)]
    names_c = [names[j] for j in range(p) if keep_roi[j]]
    man_c = [man[i] for i in range(n) if keep_sub[i]]
    return Xc, names_c, man_c


def run_one(X, names, man, q, seed=0):
    """返回 (发现数, 阈值比, s中位, lam_min, 选中区域, top-W)"""
    is_main = np.array([r["site"] in MAIN_SITES for r in man])
    y = np.array([int(r["y"]) for r in man])
    Xm, ym = X[is_main], y[is_main]
    Xr = X[~is_main]
    out = {}
    for tag, S in (("empirical", pd_floor(corr_of(Xr))),
                   ("LW", ledoit_sigma(Xr)), ("sh0.05", None)):
        if S is None:
            S = pd_floor(0.95 * corr_of(Xr) + 0.05 * np.eye(X.shape[1]))
        mu, sd = Xm.mean(0), Xm.std(0, ddof=1)
        sd[sd == 0] = 1.0
        Xs = (Xm - mu) / sd
        s = solve_s_diag(S, method="equicorrelated")
        Xt = sample_knockoffs(Xs, S, s, rng=np.random.default_rng(seed))
        W = lcd_fixed_lambda(np.hstack([Xs, Xt]), ym, c=C_FIXED)
        tau = knockoff_threshold(W, q, plus_one=True)
        sel = np.where(W >= tau)[0] if np.isfinite(tau) else np.array([], int)
        if np.isfinite(tau) and len(sel):
            R = int((W >= tau).sum()); F = int((W <= -tau).sum())
            ratio = (1 + F) / R
        else:
            ratio = np.inf
        out[tag] = dict(nsel=len(sel), ratio=ratio, s=np.median(s),
                        lam=np.linalg.eigvalsh(S).min(),
                        sel=[names[j] for j in sel],
                        top=[names[j] for j in np.argsort(-W)[:8]])
    return out


def main():
    q = 0.20
    print("=" * 86)
    print(f"跨图谱稳健性检验   q={q}   固定 lambda = {C_FIXED} * lam_max")
    print("=" * 86)
    summary = {}
    for tag, d, deriv in PARCELLATIONS:
        if not os.path.isdir(d):
            print(f"\n[{tag}] 目录 {d} 不存在，跳过")
            continue
        X, names, man = load_clean(d)
        n, p = X.shape
        n_ref = sum(1 for r in man if r["site"] not in MAIN_SITES)
        print(f"\n{'='*86}\n[{tag}]  矩阵 {n} x {p}   参考队列 n={n_ref}"
              f"   (需要 > p = {p} 才能估 Sigma: "
              f"{'满足' if n_ref > p else '不满足!'})")
        res = run_one(X, names, man, q)
        summary[tag] = (p, res)
        print(f"  {'Sigma':<11s}{'lam_min':>9s}{'s中位':>8s}{'发现数':>7s}"
              f"{'(1+F)/R':>10s}   判定")
        for k in ("empirical", "LW", "sh0.05"):
            r = res[k]
            verdict = "有发现" if r["nsel"] > 0 else "零发现"
            print(f"  {k:<11s}{r['lam']:9.5f}{r['s']:8.4f}{r['nsel']:7d}"
                  f"{r['ratio']:10.3f}   {verdict}")
        print(f"  LW 下 top-8: {', '.join(res['LW']['top'][:8])}")

    # 跨图谱一致性：经验 Sigma 是否都失效？LW 是否都能救回？
    print("\n" + "=" * 86)
    print("结论汇总")
    print("=" * 86)
    print(f"  {'图谱':<16s}{'p':>5s}{'经验Sigma发现数':>17s}{'LW发现数':>11s}")
    for tag, (p, res) in summary.items():
        print(f"  {tag:<16s}{p:>5d}{res['empirical']['nsel']:>17d}"
              f"{res['LW']['nsel']:>11d}")
    ok = all(res["LW"]["nsel"] > res["empirical"]["nsel"]
             for _, res in summary.values())
    print()
    print("  " + ("✓ 模式在所有图谱上一致：经验 Sigma 失效，收缩救回 -> 结论不依赖图谱选择"
                   if ok else
                   "✗ 模式在不同图谱间不一致 -> 需要讨论图谱依赖，不能笼统下结论"))


if __name__ == "__main__":
    main()
