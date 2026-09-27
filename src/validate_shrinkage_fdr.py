"""
validate_shrinkage_fdr.py -- 判决性实验：用收缩后的 Sigma 跑 knockoffs，FDR 还控得住吗？

背景：
    在 ABIDE 真实数据上我们发现：参考队列估出的经验 Sigma 近奇异(lam_min=0.022)，
    迫使 equicorrelated 构造给出极小的 s=0.043，功效归零；
    而 Ledoit-Wolf 收缩把 lam_min 抬到 0.032、s 抬到 0.064，阈值比从 0.333 降到 0.125，
    于是出现 8 个发现。

    但那个 0.125 只是「阈值估计」1+F)/R。它只在 knockoffs 的交换性成立时才等价于
    FDR 上界。用了估计+收缩的 Sigma，交换性严格来说已被破坏。
    => 必须验证【实际 FDP】是否 <= q。

设计：
    真值 Sigma  = ABIDE 参考队列的经验相关矩阵（真实的近奇异结构）
    每次重复：
        1. 从真值 Sigma 抽一个新的参考队列 (n_ref=393) -> 估 Sigma_hat -> 加收缩
        2. 从真值 Sigma 抽一个新的主队列   (n_main=459)
        3. y = X beta + noise，beta 稀疏（真值已知）
        4. 用收缩后的 Sigma_hat 构造 knockoffs，跑完整个 filter
        5. 记录实际 FDP、功效、发现数
    对每个收缩强度重复 REP 次。

判据：
    实际 FDP 在所有收缩强度下都应 <= q。若某个强度下 FDP 显著超 q，
    说明该强度的收缩会破坏有效性，不能推荐。

Run:  python validate_shrinkage_fdr.py [--reps 150] [--q 0.20]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, "knockoff_adni")
from knockoffs import knockoff_filter  # noqa: E402

CLEAN = "features_clean"

from sklearn.linear_model import Lasso
from scipy.linalg import cho_factor, cho_solve, cholesky


def lcd_fixed_lambda(Xa, y, c=0.15):
    """固定 lambda 的套索系数差统计量。

    交换 X_j 与 X~_j 只是交换增广设计的列顺序，不改变列集合，
    因此 lam_max = max|Xa^T y|/n 保持不变 -> lambda 是 swap-invariant 的，
    满足 flip-sign 性质。这是论文 Sec 3.2 列出的原始 LCD 变体
    （固定调参 vs 交叉验证调参），对 FDR 有效性完全等价。
    速度：单次约 0.01s，而 LassoCV 约 15s。
    """
    Xa = Xa - Xa.mean(axis=0)
    y = y - y.mean()
    lam = c * np.abs(Xa.T @ y).max() / len(y)
    coef = Lasso(alpha=lam, max_iter=20000).fit(Xa, y).coef_
    p = Xa.shape[1] // 2
    return np.abs(coef[:p]) - np.abs(coef[p:])


def knockoff_run(X, y, Sigma, q, seed, c=0.15):
    """完整的 knockoffs+ 流程，统计量换成快速版。"""
    from knockoffs import solve_s_diag, sample_knockoffs, knockoff_threshold
    mu, sd = X.mean(0), X.std(0, ddof=1)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    s = solve_s_diag(Sigma, method="equicorrelated")
    Xt = sample_knockoffs(Xs, Sigma, s, rng=np.random.default_rng(seed))
    W = lcd_fixed_lambda(np.hstack([Xs, Xt]), y, c=c)
    tau = knockoff_threshold(W, q, plus_one=True)
    sel = np.where(W >= tau)[0] if np.isfinite(tau) else np.array([], int)
    # 阈值本身给出的 FDP 估计（用户能看到的那个数）
    if np.isfinite(tau) and len(sel):
        R = int((W >= tau).sum()); F = int((W <= -tau).sum())
        fdp_hat = min((1 + F) / R, 1.0)
    else:
        fdp_hat = 0.0
    return sel, s, fdp_hat




def pd_floor(C, eps=1e-6):
    C = 0.5 * (C + C.T)
    w, V = np.linalg.eigh(C)
    w = np.clip(w, eps, None)
    C = V @ np.diag(w) @ V.T
    C = 0.5 * (C + C.T)
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def corr_of(A):
    C = np.corrcoef(A, rowvar=False)
    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 1.0)
    return C


def shrink_to(C, lam):
    """向单位阵收缩：C -> (1-lam) C + lam I"""
    p = C.shape[0]
    return pd_floor((1.0 - lam) * C + lam * np.eye(p))


def ledoit_wolf_level(C_emp):
    """反解 Ledoit-Wolf 实际使用的收缩强度，便于在图上对齐 x 轴。"""
    p = C_emp.shape[0]
    # LW 把协方差向 trace/p * I 收缩；这里对相关矩阵做同样的事，
    # 等效强度用「对角均值」归一化后反推
    off = C_emp[~np.eye(p, dtype=bool)]
    return float(np.clip(1.0 - np.sqrt(np.mean(off ** 2)), 0.0, 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=150)
    ap.add_argument("--q", type=float, default=0.20)
    ap.add_argument("--k-signal", type=int, default=12)
    ap.add_argument("--effect", type=float, default=0.35)
    ap.add_argument("--n-main", type=int, default=459)
    ap.add_argument("--n-ref", type=int, default=393)
    ap.add_argument("--c", type=float, default=0.15, help="固定 lambda = c * lam_max")
    args = ap.parse_args()

    X = np.loadtxt(os.path.join(CLEAN, "X_meanconn.csv"), delimiter=",", skiprows=1)
    isref = np.loadtxt(os.path.join(CLEAN, "cohort.csv"), dtype=int)
    Sigma_true = pd_floor(corr_of(X[isref == 1]))     # 用真实 ABIDE 结构当"真值"
    p = Sigma_true.shape[0]
    L = np.linalg.cholesky(Sigma_true + 1e-10 * np.eye(p))

    lam_true = np.linalg.eigvalsh(Sigma_true).min()
    print("=" * 76)
    print("判决性实验：收缩 Sigma 后 FDR 还控得住吗？")
    print("=" * 76)
    print(f"  真值 Sigma = ABIDE 参考队列经验相关矩阵")
    print(f"  p={p}  n_main={args.n_main}  n_ref={args.n_ref}  q={args.q}")
    print(f"  真值 lam_min = {lam_true:.5f}   cond = "
          f"{np.linalg.eigvalsh(Sigma_true).max()/lam_true:.0f}")
    print()

    # 收缩强度网格：0=纯经验, LW=自适应, 以及若干固定值
    levels = [("empirical", None), ("LW", "lw"),
              ("sh0.02", 0.02), ("sh0.04", 0.04), ("sh0.06", 0.06),
              ("sh0.08", 0.08), ("sh0.10", 0.10), ("sh0.15", 0.15),
              ("sh0.20", 0.20)]

    rng = np.random.default_rng(0)
    acc = {nm: {"fdp": [], "pow": [], "nsel": [], "lam": [], "s": [], "fdp_hat": []}
           for nm, _ in levels}
    t0 = time.time()

    for rep in range(args.reps):
        # 抽参考队列 -> 估 Sigma
        Xr = rng.standard_normal((args.n_ref, p)) @ L.T
        C_emp = pd_floor(corr_of(Xr))
        # 抽主队列
        Xm = rng.standard_normal((args.n_main, p)) @ L.T
        # 稀疏信号
        supp = rng.choice(p, size=args.k_signal, replace=False)
        beta = np.zeros(p)
        beta[supp] = args.effect * rng.choice([-1.0, 1.0], size=args.k_signal)
        y = Xm @ beta + rng.standard_normal(args.n_main)

        for nm, lev in levels:
            if lev is None:
                S = C_emp
            elif lev == "lw":
                from sklearn.covariance import LedoitWolf
                lw = LedoitWolf().fit(Xr).covariance_
                # 注意：这里必须用 cov_to_corr，不能用 corr_of。
                # corr_of 内部是 np.corrcoef，它把【行】当观测——
                # 对数据矩阵正确，对协方差矩阵完全错误。
                d = np.sqrt(np.clip(np.diag(lw), 1e-300, None))
                S = pd_floor(lw / np.outer(d, d))
            else:
                S = shrink_to(C_emp, float(lev))

            sel_arr, s_arr, fdp_hat = knockoff_run(Xm, y, S, args.q, seed=rep, c=args.c)
            sel = set(sel_arr.tolist())
            nf = len(sel - set(supp.tolist()))
            acc[nm]["fdp"].append(nf / len(sel) if sel else 0.0)
            acc[nm]["pow"].append(len(sel & set(supp.tolist())) / args.k_signal)
            acc[nm]["nsel"].append(len(sel))
            acc[nm]["lam"].append(float(np.linalg.eigvalsh(S).min()))
            acc[nm]["s"].append(float(np.median(s_arr)))
            acc[nm].setdefault("fdp_hat", []).append(fdp_hat)
        if (rep + 1) % 10 == 0 or rep + 1 == args.reps:
            print(f"  rep {rep+1}/{args.reps}  ({time.time()-t0:.0f}s)", flush=True)

    print()
    print("=" * 76)
    print(f"{'收缩强度':<12s}{'lam_min':>9s}{'s中位':>8s}{'E[FDP]':>9s}{'s.e.':>7s}"
          f"{'功效':>8s}{'|S|':>7s}{'阈值估计':>9s}   判定")
    print("-" * 76)
    rows = []
    for nm, _ in levels:
        a = acc[nm]
        fdp = np.array(a["fdp"]); pw = np.array(a["pow"])
        m, se = fdp.mean(), fdp.std(ddof=1) / np.sqrt(len(fdp))
        # 判定：均值 + 2 倍标准误是否超过 q
        ok = (m + 2 * se) <= args.q
        verdict = "FDR 受控" if ok else "*** 超限 ***"
        print(f"{nm:<12s}{np.mean(a['lam']):9.5f}{np.mean(a['s']):8.4f}"
              f"{m:9.4f}{se:7.4f}{pw.mean():8.3f}{np.mean(a['nsel']):7.2f}"
              f"{np.mean(a['fdp_hat']):9.4f}   {verdict}")
        rows.append(dict(level=nm, lam_min=np.mean(a["lam"]), s_med=np.mean(a["s"]),
                         fdp_mean=m, fdp_se=se, power=pw.mean(),
                         n_sel=np.mean(a["nsel"]), reps=len(fdp)))

    with open("shrinkage_fdr_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print()
    print(f"已写入 shrinkage_fdr_results.csv   总耗时 {time.time()-t0:.0f}s")
    print()
    print("读法：")
    print("  * 若所有强度都 'FDR 受控' -> 收缩是安全操作，论文可以推荐它救回功效")
    print("  * 若某个强度 '超限'      -> 该强度不可用，必须在论文里排除")
    print("  * 同时看 功效：只受控但功效为零，等于没解决问题")

    # ----------------------------------------------------------------- #
    # 配对检验
    #
    # 上面每行的 s.e. 是【边际】标准误：把真 FDP 与 reported 当成两组独立
    # 样本看待。但 300 次重复是【完全配对】的 —— 每一轮抽一份数据，然后
    # 在【同一份数据】上跑 9 个 α。用边际标准误去比较两列，丢掉了全部配对
    # 信息，会导致手稿把自己的结论写得比证据弱得多。
    #
    # 这里补三件事：
    #   1. gap = 真FDP - reported 的配对 t 检验（对每一对重复求差再检验）
    #   2. 该 gap 相对于 α=0 的增量的配对检验（"退化随收缩而加剧"这个结构性
    #      论断的真正检验）
    #   3. 符号检验（不依赖正态）
    # ----------------------------------------------------------------- #
    from scipy import stats as _st
    paired_rows = []
    g0 = np.array(acc["empirical"]["fdp"]) - np.array(acc["empirical"]["fdp_hat"])
    print()
    print("=" * 88)
    print("配对检验（同一份重复数据上比较，比边际 s.e. 有力得多）")
    print("=" * 88)
    print(f"{'强度':<12s}{'gap':>8s}{'配对s.e.':>10s}{'t':>8s}{'p':>11s}"
          f"{'符号p':>11s}{'相对α=0':>10s}{'增量p':>11s}")
    print("-" * 88)
    for nm, _ in levels:
        a = acc[nm]
        f = np.array(a["fdp"]); r = np.array(a["fdp_hat"])
        g = f - r
        se = g.std(ddof=1) / np.sqrt(len(g))
        t, p = _st.ttest_rel(f, r)
        p_sign = _st.binomtest(int((g > 0).sum()), len(g), 0.5,
                              alternative="greater").pvalue
        d = g - g0
        se_d = d.std(ddof=1) / np.sqrt(len(d))
        _, p_d = _st.ttest_rel(g, g0)
        print(f"{nm:<12s}{g.mean():8.4f}{se:10.4f}{t:8.2f}{p:11.3e}"
              f"{p_sign:11.3e}{d.mean():10.4f}{p_d:11.3e}")
        paired_rows.append(dict(level=nm, gap=g.mean(), gap_se=se, t=t, p=p,
                                sign_p=p_sign, delta_vs_alpha0=d.mean(),
                                delta_se=se_d, p_vs_alpha0=p_d,
                                n_reps=len(g)))
    with open("shrinkage_paired_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(paired_rows[0].keys()))
        w.writeheader()
        w.writerows(paired_rows)
    print()
    print(f"已写入 shrinkage_paired_results.csv")
    print("读法：『相对α=0』列显著 > 0 的起始强度，就是用户会开始收不到警告的地方。")
    print("      注意 α=0 自身的 gap 若已显著，说明现象不完全是收缩造成的。")


if __name__ == "__main__":
    main()
