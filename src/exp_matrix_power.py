"""
exp_matrix_power.py -- 矩阵幂族：一条连续谱线取代四种手工定义

动机（来自「分数阶」的直觉，但落在正确的空间里）：

    逐 ROI（逐列）的变换【无法】改变共线性 —— Pearson 相关只对仿射变换不变，
    非线性逐列变换只能靠扭曲线性结构来改变它，实测有效秩几乎不动。

    真正能改变共线性的是【混合各列】的变换。最自然的一维族是矩阵幂：

        X_t = X @ Sigma^{(t-1)/2}      =>      corr(X_t) = Sigma^t

    这是恒等式：Sigma^{(t-1)/2} Sigma Sigma^{(t-1)/2} = Sigma^t。

    t = +1  原样（边际相关）          t = 0   白化（完全去相关）
    t = -1  精度矩阵（类偏相关）

实测（ABIDE I AAL116 meanconn）有效秩随 t 的走向：
    t=+1 -> 2.56,  t=+0.5 -> 34.5,  t=0 -> 113.0,  t=-1 -> 64.7

也就是说，我们此前当作四个独立"定义"的东西
（meanconn / gsr / pcresid / partial）很可能是同一条一维谱线上的采样点。

【这个实验的真正价值】它把两条看似无关的结论统一起来：
    * 收缩 (1-a)Sigma + aI 通过把谱拉平来提高有效秩
    * 白化 Sigma^0 把谱完全拉平
  两者是同一条轴上的不同走法。而代价也一样：白化放大低方差方向，
  也就是放大噪声（t=0 时放大 1/sqrt(lam_min) 倍）。

本脚本在 t 网格上扫描，报告：
    (i)   变换后特征的有效秩与 lam_min
    (ii)  固定注入信号（在【原始 ROI 空间】定义）下的 per-signal power 与 FDP
    (iii) 真实响应上的 P(非空)
    (iv)  套索收敛率

注意注入信号在原始空间定义（y = X beta + eps，beta 固定），然后只在 X_t 上
跑 knockoff。这样问的是「给定 ROI 空间里的固定信号，变换能否帮助找回它」，
而不是「给每个 t 单独配一个合适强度的信号」——后者会混淆比较。

Run:  python exp_matrix_power.py --reps 200 --workers 14
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.refcohort import (C_FIXED, DEF_DATASETS_FOR_DIAG, Q_DEFAULT,          # noqa: E402
                           corr_of, effective_rank, knockoff_run, ledoit_sigma,
                           pd_floor, prepare, provenance, spectral_report,
                           split_xy, write_csv)

T_GRID = (1.0, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6, -0.8, -1.0)


def matrix_power_transform(X, Sigma, t, floor=1e-6):
    """X_t = X Sigma^{(t-1)/2}，使得 corr(X_t) 约等于 Sigma^t。"""
    w, V = np.linalg.eigh(0.5 * (Sigma + Sigma.T))
    w = np.clip(w, floor, None)
    half = V @ np.diag(w ** ((t - 1.0) / 2.0)) @ V.T
    return X @ half


_W = {}


def _init_w(Xm, ym, Xr, c, seed):
    global _W
    _W = dict(Xm=Xm, ym=ym, Xr=Xr, c=c, seed=seed)


def _one(job):
    """(t, rep) -> 正对照结果；用同一份 beta 保证跨 t 可比。"""
    t, r = job
    w = _W
    Xm, Xr = w["Xm"], w["Xr"]
    # Sigma 从参考队列估（用户手上真实有的），再同时施加到两个队列
    S0 = ledoit_sigma(Xr)
    Xm_t = matrix_power_transform(Xm, S0, t)
    Xr_t = matrix_power_transform(Xr, S0, t)
    S_t = ledoit_sigma(Xr_t)

    rng = np.random.default_rng(w["seed"] * 100000 + r)
    p = Xm.shape[1]
    supp = rng.choice(p, 12, replace=False)
    beta = np.zeros(p)
    beta[supp] = 0.85 * rng.choice([-1.0, 1.0], 12)
    # 信号在【原始】空间生成
    mu, sd = Xm.mean(0), Xm.std(0, ddof=1)
    sd[sd == 0] = 1.0
    Xs = (Xm - mu) / sd
    y = Xs @ beta + rng.standard_normal(len(Xm))

    res = knockoff_run(Xm_t, y, S_t, q=Q_DEFAULT,
                       seed=w["seed"] * 1000 + r, c=w["c"])
    sel = set(res["sel"].tolist())
    tgt = set(supp.tolist())
    return {"t": t, "power": len(sel & tgt) / 12.0,
            "fdp": (len(sel - tgt) / len(sel)) if sel else 0.0,
            "nsel": len(sel), "conv": float(res["converged"])}


def _real(job):
    t, r = job
    w = _W
    S0 = ledoit_sigma(w["Xr"])
    Xm_t = matrix_power_transform(w["Xm"], S0, t)
    Xr_t = matrix_power_transform(w["Xr"], S0, t)
    S_t = ledoit_sigma(Xr_t)
    res = knockoff_run(Xm_t, w["ym"], S_t, q=Q_DEFAULT,
                       seed=w["seed"] * 100000 + r, c=w["c"])
    return {"t": t, "nsel": len(res["sel"]), "ratio": res["ratio_best"],
            "conv": float(res["converged"])}



def _null(job):
    """全局零假设：y 与 X 完全独立（随机标签）。
    此时每一个变量都是零假设，任何被选中的 ROI【按构造】都是假阳性，
    因此真实 FDP = 1 当 |S|>0，= 0 当 |S|=0。不需要任何 ground truth。
    """
    t, r = job
    w = _W
    S0 = ledoit_sigma(w["Xr"])
    Xm_t = matrix_power_transform(w["Xm"], S0, t)
    Xr_t = matrix_power_transform(w["Xr"], S0, t)
    S_t = ledoit_sigma(Xr_t)
    rng = np.random.default_rng(w["seed"] * 100000 + r)
    # 与 X 独立的随机响应
    y = rng.standard_normal(len(w["Xm"]))
    res = knockoff_run(Xm_t, y, S_t, q=Q_DEFAULT,
                       seed=w["seed"] * 1000 + r, c=w["c"])
    return {"t": t, "nsel": len(res["sel"]), "ratio": res["ratio_best"],
            "conv": float(res["converged"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--c", type=float, default=C_FIXED)
    ap.add_argument("--data", default="feat2_abide1_aal")
    ap.add_argument("--fname", default="X_meanconn.csv")
    ap.add_argument("--outdir", default="results_matrixpow")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(args.outdir, exist_ok=True)

    X, names, man, ism, ms = prepare(args.data, fname=args.fname)
    Xm, ym, Xr = split_xy(X, man, ism)
    S0 = ledoit_sigma(Xr)
    print(f"{args.data}/{args.fname}  p={X.shape[1]}  n_main={len(Xm)}"
          f"  n_ref={len(Xr)}  c={args.c}\n")

    # (i) 几何
    print("=== (i) 变换后的几何 ===")
    geo = {}
    for t in T_GRID:
        Xr_t = matrix_power_transform(Xr, S0, t)
        rep = spectral_report(pd_floor(corr_of(Xr_t)))
        geo[t] = rep
        print(f"  t={t:5.2f}  有效秩={rep['eff_rank']:7.2f}"
              f"  PC1={rep['pc1_share']:.4f}  lam_min={rep['lam_min']:.5f}")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    from collections import defaultdict
    out = {}
    for tag, fn, label in (("ctrl", _one, "(ii) 正对照：原始空间固定信号 → 恢复能力"),
                           ("real", _real, "(iii) 真实响应"),
                           ("null", _null, "(iv) 全局零假设：y 与 X 独立 → 任何发现都是假的")):
        print()
        print("=" * 78)
        print(f"{label}   ({args.reps} 次重复)")
        print("=" * 78)
        acc = defaultdict(list)
        jobs = [(t, r) for t in T_GRID for r in range(args.reps)]
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_init_w,
                                 initargs=(Xm, ym, Xr, args.c,
                                           args.seed)) as ex:
            futs = [ex.submit(fn, j) for j in jobs]
            for i, f in enumerate(as_completed(futs), 1):
                rr = f.result()
                acc[rr["t"]].append(rr)
                if i % 200 == 0:
                    print(f"    {i}/{len(jobs)}  ({time.time()-t0:.0f}s)",
                          flush=True)
        rows = []
        for t in T_GRID:
            g = acc[t]
            if not g:
                continue
            row = {"t": t, "eff_rank": geo[t]["eff_rank"],
                   "pc1_share": geo[t]["pc1_share"],
                   "lam_min": geo[t]["lam_min"], "reps": len(g),
                   "frac_converged": float(np.mean([x["conv"] for x in g]))}
            if tag == "ctrl":
                row["power"] = float(np.mean([x["power"] for x in g]))
                row["fdp"] = float(np.mean([x["fdp"] for x in g]))
                row["nsel_mean"] = float(np.mean([x["nsel"] for x in g]))
            elif tag == "real":
                row["p_nonempty"] = float(np.mean(
                    [1.0 if x["nsel"] > 0 else 0.0 for x in g]))
                row["nsel_median"] = float(np.median([x["nsel"] for x in g]))
                row["ratio_median"] = float(np.median([x["ratio"] for x in g]))
            else:
                # 全局零假设：|S|>0 的抽样里 FDP 恒为 1
                row["p_nonempty"] = float(np.mean(
                    [1.0 if x["nsel"] > 0 else 0.0 for x in g]))
                row["nsel_median"] = float(np.median([x["nsel"] for x in g]))
                row["nsel_max"] = int(np.max([x["nsel"] for x in g]))
                row["true_fdp"] = row["p_nonempty"] and 1.0 or 0.0
                row["ratio_median"] = float(np.median([x["ratio"] for x in g]))
            rows.append(row)
            cells = "  ".join(f"{k}={row[k]:.3f}" for k in
                              (("power", "fdp") if tag == "ctrl"
                               else ("p_nonempty", "nsel_median",
                                     "ratio_median")))
            print(f"  t={t:5.2f}  有效秩={row['eff_rank']:7.2f}  {cells}"
                  f"  conv={row['frac_converged']:.2f}")
        out[tag] = write_csv(
            os.path.join(args.outdir,
                         "matrixpow_%s.csv" % {"ctrl": "control",
                                               "real": "real",
                                               "null": "globalnull"}[tag]), rows)

    provenance(args.outdir, "exp_matrix_power", args)
    print(f"\n-> {out}\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
