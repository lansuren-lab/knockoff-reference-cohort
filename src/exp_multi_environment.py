"""
exp_multi_environment.py -- 第3个替代手段：多环境 knockoff 滤波器

出处：Li, Sesia, Romano, Candès & Sabatti, "Searching for consistent
associations with a multi-environment knockoff filter"（arXiv:2106.04118）。

动机（论文摘要原话）：
    "large data sets may contain numerous associations that are statistically
     significant and yet misleading, as they are induced by confounders or
     sampling imperfections. However, associations consistently replicated
     under different conditions may be more interesting."

**这正是我们的处境。** 我们最强的观测就是"跨特征定义/跨管线/跨站点什么都
不一致"。这个方法把它从阴性结果变成一个有 FDR 保证的推断目标：问的不是
"某个 ROI 与 ASD 有关吗"，而是"某个 ROI 在多少个站点上一致有关"。

方法（§4.2-4.3）：
  1. 每个环境 e 单独跑 knockoff 滤波器 -> W^e ∈ R^p（每个环境用自己的样本）
     Σ 用全局参考队列估计（特征在各站点间是可比量的；Σ 是总体量）
  2. 一致性统计量 |W_j^cst| = prod_{e=1..E} |W_j^e|
  3. 部分一致性参数 r：检验"至少在 r 个环境中有关联"（r=1 退化为合并分析）
     记 n_j^- 为第 j 列负号的个数，n_j^0 为 0 的个数，则
       p_j^{pcst,r} = Psi(n_j^- - 1, (E-r+1-n_j^0) v 0, 1/2)
                      + U_j * psi(n_j^-, (E-r+1-n_j^0) v 0, 1/2)
     Psi 是二项分布 CDF，psi 是 PMF，U_j ~ U[0,1]
     排序用 W_j^{pcst,r} = prod_{e=1..r} |W_j|_(E-e+1)（该列最大的 r 个绝对值之积）
  4. 在降序上跑 selective SeqStep+（单比特 p 值下截断 c 只能取 1/2）

环境选择：ABIDE I 按站点切分，取两类样本都存在的站点。
跨环境独立性由站点不同保证（同一位受试者只出现在一个站点）。

Run:  python exp_multi_environment.py --q 0.20
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter

import numpy as np
from scipy import stats as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.refcohort import (C_FIXED, Q_DEFAULT, ledoit_sigma, prepare,
                           provenance, write_csv)
from knockoff_adni.knockoffs import (knockoff_threshold, sample_knockoffs,   # noqa: E402
                                     solve_s_diag)
from src.refcohort import lcd_fixed                                         # noqa: E402


def knockoff_W(X, y, Sigma, s, seed, c=C_FIXED):
    """单个环境的 knockoff 统计量 W（不做阈值，只取 W）。"""
    mu, sd = X.mean(0), X.std(0, ddof=1)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    Xt = sample_knockoffs(Xs, Sigma, s, rng=np.random.default_rng(seed))
    return lcd_fixed(np.hstack([Xs, Xt]), y, c=c)


def seqstep_plus(pvals, order, q, c=0.5):
    """selective SeqStep+（单比特 p 值下 c 固定为 1/2）。

    按 order 降序排列后，取
        k_hat = max{ k : (1 + #{i<=k : p_i > c}) / k <= q }
    拒绝前 k_hat 个中 p_i <= c 的那些。
    """
    idx = np.argsort(-order)
    p = np.asarray(pvals)[idx]
    best_k = 0
    for k in range(1, len(p) + 1):
        n_above = int((p[:k] > c).sum())
        if (1 + n_above) / k <= q:
            best_k = k
    if best_k == 0:
        return np.array([], int)
    sel = [idx[i] for i in range(best_k) if p[i] <= c]
    return np.array(sorted(sel), int)


def multienv(X_list, y_list, q, r_values, seed=0, c=C_FIXED):
    """多环境滤波器。返回 {r: 选中的 ROI 下标集合}。

    Σ 采用【留一环境】估计：对第 e 个环境，用其余全部环境的数据估 Σ。
    这样每个环境的 knockoff 都不使用它自己的样本去估协方差，
    避免 in-sample 的乐观偏差。
    """
    E = len(X_list)
    W = np.empty((E, X_list[0].shape[1]))
    for e in range(E):
        others = np.vstack([X_list[k] for k in range(E) if k != e])
        Sig = ledoit_sigma(others)
        se = solve_s_diag(Sig, method="equicorrelated")
        W[e] = knockoff_W(X_list[e], y_list[e], Sig, se,
                          seed=seed * 100 + e, c=c)
    out = {}
    for r in r_values:
        n_neg = (W < 0).sum(0)
        n_zero = (W == 0).sum(0)
        m = np.maximum(E - r + 1 - n_zero, 0)
        pv = st.binom.cdf(n_neg - 1, m, 0.5) + \
            np.random.default_rng(seed).uniform(size=len(n_neg)) * \
            st.binom.pmf(n_neg, m, 0.5)
        absW = np.sort(np.abs(W), axis=0)[::-1]           # 每列降序
        order = np.prod(absW[:r], axis=0) if r <= E else np.zeros(len(n_neg))
        out[r] = {"sel": seqstep_plus(pv, order, q),
                  "pvals": pv, "order": order, "W": W}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=float, default=Q_DEFAULT)
    ap.add_argument("--c", type=float, default=C_FIXED)
    ap.add_argument("--min-n", type=int, default=40)
    ap.add_argument("--outdir", default="results_multienv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(args.outdir, exist_ok=True)

    X, names, man, ism, ms = prepare("feat2_abide1_aal")
    sites = np.array([r["site"] for r in man])
    y = np.array([float(r["y"]) if str(r["y"]).strip() != "" else np.nan
                  for r in man])
    ok = np.isfinite(y)
    X, y, sites = X[ok], y[ok], sites[ok]
    print(f"ABIDE1-AAL116 meanconn  p={X.shape[1]}  受试者={len(y)}  "
          f"站点={len(set(sites))}")

    # 环境 = 站点，要求 n >= min_n 且两类都有
    envs = []
    for s_ in sorted(set(sites)):
        m = sites == s_
        if m.sum() >= args.min_n and len(set(y[m])) == 2:
            envs.append((s_, X[m], y[m], int(m.sum()),
                         int((y[m] == 1).sum())))
    print(f"可用环境 {len(envs)} 个：")
    for s_, _, _, n, na in envs:
        print(f"    {s_:12s} n={n:4d}  ASD={na:3d}  对照={n-na:3d}")
    E = len(envs)

    r_values = [r for r in (1, 2, 3, 4, 5) if r <= E]
    print(f"\n部分一致性参数 r ∈ {r_values}  (r=1 退化为合并分析)")
    print("=" * 70)
    res = multienv([e[1] for e in envs], [e[2] for e in envs],
                   args.q, r_values, seed=args.seed, c=args.c)

    rows = []
    for r in r_values:
        sel = res[r]["sel"]
        rows.append({"r": r, "E": E, "q": args.q,
                     "n_selected": len(sel),
                     "selected": ";".join(names[j] for j in sel[:30]),
                     "min_pval": float(np.min(res[r]["pvals"])),
                     "p_median": float(np.median(res[r]["pvals"]))})
        print(f"  r={r}: 选中 {len(sel):3d} 个 ROI   "
              f"最小 p={np.min(res[r]['pvals']):.4f}  "
              f"p 中位={np.median(res[r]['pvals']):.4f}")
        if len(sel):
            print(f"        {', '.join(names[j] for j in sel[:12])}"
                  f"{' ...' if len(sel) > 12 else ''}")

    # 阴性对照：把 y 随机打乱后重跑（跨环境一致性应当消失）
    print("\n阴性对照：把 y 在每个环境内随机打乱（一致性应当消失）")
    rng = np.random.default_rng(args.seed + 999)
    yshuf = [rng.permutation(e[2]) for e in envs]
    res0 = multienv([e[1] for e in envs], yshuf, args.q, r_values,
                    seed=args.seed + 999, c=args.c)
    for r in r_values:
        n0 = len(res0[r]["sel"])
        print(f"  r={r}: 打乱后选中 {n0:3d} 个")
        rows.append({"r": r, "E": E, "q": args.q, "n_selected": n0,
                     "selected": "SHUFFLED-CONTROL",
                     "min_pval": float(np.min(res0[r]["pvals"])),
                     "p_median": float(np.median(res0[r]["pvals"]))})

    path = write_csv(os.path.join(args.outdir, "multienv.csv"), rows)
    provenance(args.outdir, "exp_multi_environment", args)
    print(f"\n-> {path}\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
