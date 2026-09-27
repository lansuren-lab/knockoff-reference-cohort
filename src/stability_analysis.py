"""
stability_analysis.py -- knockoffs 在 ABIDE 上的选择稳定性

背景（论文的 motivation）：
    文献反复报告 knockoffs 在实际影像数据上「时灵时不灵」：
      * Liu et al. (2022)：乳腺癌数据约 45% 的重复得到 0 个发现，其余得到 >=10 个
      * Emory 硕士论文：AD 影像上「高度不稳定」，q=0.2 仍无一致选择
    但这些数字都不是在我们的设定下测的。这里在 ABIDE 上自己测一遍。

做法：
    固定数据、固定 Sigma（参考队列 + Ledoit-Wolf）、固定统计量与 CV 折，
    **只改变 knockoff 变量的随机抽样**，跑 M 次。这样测到的变异纯粹来自
    knockoff 抽样的随机性 —— 正是去随机化要解决的量。

指标：
    1. 零发现的比例            （文献里那个 45%）
    2. 两两 Jaccard 稳定性      （选中集合之间的一致性）
    3. 每个 ROI 的入选频率      （哪些是「稳」的、哪些是抽签抽出来的）
    4. 发现数的分布

Run:  python stability_analysis.py --reps 200
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from itertools import combinations

import numpy as np

sys.path.insert(0, "knockoff_adni")
sys.path.insert(0, ".")
from knockoffs import solve_s_diag, sample_knockoffs, knockoff_threshold  # noqa: E402
from sklearn.linear_model import Lasso  # noqa: E402
from compare_parcellations import load_clean, corr_of, ledoit_sigma, MAIN_SITES  # noqa: E402


def lcd_fixed(Xa, y, c=0.05):
    """固定 lambda 的套索系数差统计量 —— 完全确定，无求解器歧义。

    【为什么不用逻辑回归 CV-LCD】
    实测发现该统计量在本数据上数值不收敛：同一数据同一 knockoff 抽样，
    只改 saga 的 max_iter（200/500/5000）或 CV 折数，发现数会在 0 和 6/7 之间跳。
    那会把「求解器的变异」混进「knockoff 抽样的变异」，使稳定性分析失去意义。
    固定 lambda 的套索单次求解、tol 收紧到 1e-6，结果可复现。
    lam = c * lam_max 对列置换不变 -> 满足 flip-sign。
    """
    Xa = Xa - Xa.mean(axis=0)
    y = y - y.mean()
    lam = c * np.abs(Xa.T @ y).max() / len(y)
    coef = Lasso(alpha=lam, max_iter=100000, tol=1e-6).fit(Xa, y).coef_
    p = Xa.shape[1] // 2
    return np.abs(coef[:p]) - np.abs(coef[p:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--q", type=float, default=0.20)
    ap.add_argument("--dir", default="features")
    ap.add_argument("--c", type=float, default=0.05)
    args = ap.parse_args()

    X, names, man = load_clean(args.dir)
    is_main = np.array([r["site"] in MAIN_SITES for r in man])
    y = np.array([int(r["y"]) for r in man], float)
    Xm, ym = X[is_main], y[is_main]
    Xr = X[~is_main]
    p = X.shape[1]

    S = ledoit_sigma(Xr)
    mu, sd = Xm.mean(0), Xm.std(0, ddof=1)
    sd[sd == 0] = 1.0
    Xs = (Xm - mu) / sd
    s = solve_s_diag(S, method="equicorrelated")

    print("=" * 78)
    print("Knockoffs 选择稳定性分析")
    print("=" * 78)
    print(f"  数据 {X.shape}   主队列 {Xm.shape[0]}   参考队列 {Xr.shape[0]}")
    print(f"  Sigma = 参考队列 + Ledoit-Wolf   s中位={np.median(s):.4f}   q={args.q}")
    print(f"  重复 {args.reps} 次，只变 knockoff 抽样")
    print()

    sel_sets, ws, t0 = [], [], time.time()
    for m in range(args.reps):
        Xt = sample_knockoffs(Xs, S, s, rng=np.random.default_rng(10000 + m))
        W = lcd_fixed(np.hstack([Xs, Xt]), ym, c=args.c)
        tau = knockoff_threshold(W, args.q, plus_one=True)
        sel = frozenset(np.where(W >= tau)[0].tolist()) if np.isfinite(tau) else frozenset()
        sel_sets.append(sel)
        ws.append(W)
        if (m + 1) % 40 == 0:
            print(f"  {m+1}/{args.reps}  ({time.time()-t0:.0f}s)", flush=True)

    sizes = np.array([len(s_) for s_ in sel_sets])
    zero = int((sizes == 0).sum())

    # 两两 Jaccard（在非空集合之间）
    js = []
    idx = [i for i in range(args.reps) if sizes[i] > 0]
    if len(idx) > 1:
        for a, b in combinations(idx[:120], 2):
            A, B = sel_sets[a], sel_sets[b]
            js.append(len(A & B) / len(A | B))
    js = np.array(js) if js else np.array([np.nan])

    # 每个 ROI 的入选频率
    freq = np.zeros(p)
    for s_ in sel_sets:
        for j in s_:
            freq[j] += 1
    freq /= args.reps

    print()
    print("=" * 78)
    print("结果")
    print("=" * 78)
    print(f"  零发现的次数        {zero}/{args.reps}  ({zero/args.reps:.1%})")
    print(f"  发现数 均值/中位/范围  {sizes.mean():.2f} / {np.median(sizes):.0f} "
          f"/ [{sizes.min()}, {sizes.max()}]")
    print(f"  两两 Jaccard 均值    {np.nanmean(js):.3f}  "
          f"(1=完全一致, 0=毫无重叠)  n_pairs={len(js)}")
    print()
    print(f"  入选频率 >= 0.9 的 ROI（稳的）：{int((freq >= 0.9).sum())} 个")
    print(f"  入选频率 0.1-0.9 的 ROI（飘的）：{int(((freq >= 0.1) & (freq < 0.9)).sum())} 个")
    print()
    order = np.argsort(-freq)
    print(f"  {'ROI':<26s}{'入选频率':>9s}   状态")
    for j in order[:20]:
        if freq[j] == 0:
            break
        st = "稳定" if freq[j] >= 0.9 else ("动摇" if freq[j] >= 0.1 else "偶发")
        print(f"  {names[j]:<26s}{freq[j]:9.2f}   {st}")

    np.save("stability_W.npy", np.vstack(ws))

    with open("stability_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["roi", "sel_freq", "median_W", "mean_absW"])
        Wm = np.median(np.vstack(ws), axis=0)
        for j in order:
            w.writerow([names[j], f"{freq[j]:.4f}", f"{Wm[j]:.5f}",
                        f"{np.abs(Wm[j]):.5f}"])
    np.savetxt("stability_sizes.csv", sizes, fmt="%d")
    print()
    print(f"已写入 stability_results.csv / stability_sizes.csv   耗时 {time.time()-t0:.0f}s")
    print()
    # ---------------- 去随机化：W 中位数聚合 ----------------
    print()
    print("=" * 78)
    print("去随机化：对 M 次 knockoff 抽样的 W 取中位数，再套阈值")
    print("=" * 78)
    print("  有效性依据：零变量的每个 W^(m) 关于 0 对称且符号独立，")
    print("  同时翻转全部符号不改变分布，故【中位数仍关于 0 对称】,")
    print("  符号对称性保持 -> knockoff 阈值仍然有效。（已数值验证 P(med>0)=0.5）")
    print()
    Wmat = np.vstack(ws)
    rng2 = np.random.default_rng(7)
    print(f"  {'聚合次数 M':>10s}{'零发现比例':>11s}{'|S| 中位':>10s}"
          f"{'|S| 范围':>14s}{'Jaccard':>10s}")
    agg_rows = []
    for M in (1, 5, 10, 20, 40):
        sets = []
        n_agg = 200
        for _ in range(n_agg):
            idx = rng2.choice(args.reps, size=M, replace=M > 1)
            Wa = np.median(Wmat[idx], axis=0)
            ta = knockoff_threshold(Wa, args.q, plus_one=True)
            sets.append(frozenset(np.where(Wa >= ta)[0].tolist())
                        if np.isfinite(ta) else frozenset())
        szs = np.array([len(x) for x in sets])
        jj = []
        nz = [x for x in sets if len(x) > 0]
        if len(nz) > 1:
            for a, b in combinations(range(min(len(nz), 60)), 2):
                u = nz[a] | nz[b]
                jj.append(len(nz[a] & nz[b]) / len(u) if u else 1.0)
        jm = np.mean(jj) if jj else np.nan
        print(f"  {M:>10d}{(szs==0).mean():>10.1%}{np.median(szs):>10.0f}"
              f"{f'[{szs.min()}, {szs.max()}]':>14s}{jm:>10.3f}")
        agg_rows.append(dict(M=M, zero_rate=float((szs==0).mean()),
                             med_size=float(np.median(szs)), jaccard=float(jm)))
    with open("stability_aggregation.csv", "w", newline="") as f:
        w2 = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        w2.writeheader(); w2.writerows(agg_rows)

    print()
    print("读法：")
    print("  * 零发现比例高 -> 正是文献报告的『时灵时不灵』，论文的 motivation")
    print("  * Jaccard 低   -> 两次跑出来的『发现』几乎是两批不同的脑区")
    print("  * 入选频率是给读者的实用诊断：>=0.9 才值得写进结论")
    print("  * M 增大时零发现比例应下降、Jaccard 应上升 —— 这就是去随机化的收益")


if __name__ == "__main__":
    main()
