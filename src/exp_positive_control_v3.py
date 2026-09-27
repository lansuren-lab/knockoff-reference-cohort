"""
exp_positive_control_v3.py -- E3: 重做正对照（论文的决定性实验）

旧正对照（positive_control_v2.py）有两个致命缺陷：

  (1) 【按边际 |t| 标定】注入的支撑集固定为 Precentral_L/R, Frontal_Sup_L
      （即全脑 VIF 最高的三个 ROI）。实测 a=0.4 时边际 |t|=22.3，但
      【条件】|t| 只有 2.36 —— 注入的信号几乎完全落在其余 110 列的张成
      空间里。knockoff 检验的是条件关联，所以三个臂都没拿到能检出的信号，
      实验对"参考队列是不是瓶颈"零信息量。

  (2) 【第三个臂不是 oracle】旧脚本里 "true" 是主队列自己的样本相关矩阵，
      同样是 n=459 的有噪估计，只是不做收缩。三个臂其实是同类估计器的
      三种口味，所以三列数字不可解释。

本脚本的设计：

    条件信号强度 ∈ {1, 2, 3, 4, 6}，按【条件 |t|】标定
    支撑集      ∈ {fixed_highVIF, random3, random12}
    Sigma 臂    ∈ {ref_100, ref_393, ref_all, in_sample, main_corr}
    重复        >= 200

关于「oracle」：在**真实协变量**上做注入时并不存在 oracle —— 任何 Σ 都是
估计。诚实的 oracle 臂属于模拟实验（Σ 由构造已知，见 exp_boundary_sweep.py
与 exp_cohort_shift.py）。这里的 main_corr 如实标注为"不收缩的样本内估计"，
不叫 oracle。

同时报告达成的【边际】|t|，让读者直接看到 VIF 造成的膨胀倍数。

Run:  python exp_positive_control_v3.py --reps 200
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.refcohort import (C_FIXED, Q_DEFAULT, conditional_t_fast,          # noqa: E402
                           conditional_t_all, corr_of, knockoff_run,
                           ledoit_sigma, marginal_t, pd_floor, prepare,
                           provenance, split_xy, vif, write_csv)

COND_TARGETS = (1.0, 2.0, 3.0, 4.0, 6.0)
# 参考队列规模的几个臂。注意 n_ref=393 时 "393" 与 "all" 是同一件事
# （子样本大小等于全集），所以这里用 100/200 与全集区分开。
N_REF_ARMS = (100, None)            # None = 全部参考队列


def calibrate_amplitude(Xs, design, supp, target, noise, iters=50):
    """找注入幅度 a 使注入 ROI 的平均【条件】|t| 达到 target。

    design 是 conditional_t_all 返回的冻结设计，(A'A)^-1 只算一次，
    每次迭代只有一次矩阵-向量乘。旧脚本按边际 |t| 标定，是错的。
    """
    p = Xs.shape[1]

    def cond_t(a):
        beta = np.zeros(p)
        beta[list(supp)] = a
        return float(np.mean(conditional_t_fast(Xs @ beta + noise, design)[list(supp)]))

    lo, hi = 0.0, 1.0
    for _ in range(20):
        if cond_t(hi) >= target:
            break
        hi *= 2
        if hi > 1e8:
            return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if cond_t(mid) >= target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _init_worker(Xs, Xm, design, arms, seed, p, fixed_supp, c):
    """把只读的大对象放进 worker 全局，避免逐个任务 pickle。"""
    global _W
    _W = dict(Xs=Xs, Xm=Xm, design=design, arms=arms, seed=seed, p=p,
              fixed_supp=fixed_supp, c=c)


def _one_rep(job):
    """一个 (支撑集, 目标条件|t|, 重复编号) 的全部工作。"""
    supp_name, target, r = job
    w = _W
    Xs, Xm, p = w["Xs"], w["Xm"], w["p"]
    rr = np.random.default_rng(w["seed"] * 100000 + r)
    if supp_name == "fixed_highVIF":
        supp = w["fixed_supp"]
    elif supp_name == "random3":
        supp = tuple(sorted(rr.choice(p, 3, replace=False).tolist()))
    else:
        supp = tuple(sorted(rr.choice(p, 12, replace=False).tolist()))
    noise = rr.standard_normal(len(Xs))
    a = calibrate_amplitude(Xs, w["design"], supp, target, noise)
    beta = np.zeros(p)
    beta[list(supp)] = a
    y = Xs @ beta + noise
    t_all, _ = conditional_t_all(Xs, y)
    supp_set = set(supp)
    out = {"support": supp_name, "target_cond_t": target, "rep": r,
           "achieved_marg_t": float(np.mean(
               [marginal_t(Xs, y, j) for j in supp])),
           "achieved_cond_t": float(np.mean(t_all[list(supp)]))}
    for tag, S in w["arms"].items():
        res = knockoff_run(Xm, y, S, q=Q_DEFAULT,
                           seed=w["seed"] * 1000 + r, c=w["c"])
        sel = set(res["sel"].tolist())
        out[f"det_{tag}"] = 1.0 if (sel & supp_set) else 0.0
        out[f"fdp_{tag}"] = len(sel - supp_set) / len(sel) if sel else 0.0
        out[f"conv_{tag}"] = float(res["converged"])
        # per-signal power：命中数 / 注入个数。det 是"至少命中一个"，与 k
        # 有关（k=3 与 k=12 不可比），跨支撑集比较必须用这一列。
        out[f"pow_{tag}"] = len(sel & supp_set) / len(supp)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--data", default="feat2_abide1_aal")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--supports", default="fixed_highVIF,random3,random12")
    ap.add_argument("--conds", default="1.0,2.0,3.0,4.0,6.0")
    ap.add_argument("--c", type=float, default=0.05,
                    help="固定-lambda 的 c；论文用 0.15，但实测 0.15 对强信号过罚")
    args = ap.parse_args()
    t0 = time.time()

    X, names, man, is_main, ms = prepare(args.data)
    Xm, ym, Xr = split_xy(X, man, is_main)
    p = X.shape[1]

    v = vif(Xm)
    fixed_supp = tuple(np.argsort(-v)[:3].tolist())
    print(f"数据 {args.data}  p={p}  n_main={len(Xm)}  n_ref={len(Xr)}")
    print(f"  最高 VIF 三 ROI: {[names[j] for j in fixed_supp]}  "
          f"VIF={np.round(v[list(fixed_supp)], 1).tolist()}  "
          f"(全脑中位 {np.median(v):.1f})")

    mu, sd = Xm.mean(0), Xm.std(0, ddof=1)
    sd[sd == 0] = 1.0
    Xs = (Xm - mu) / sd
    _, design = conditional_t_all(Xs, ym)

    rng = np.random.default_rng(args.seed)
    arms = {"in_sample": ledoit_sigma(Xm),
            "main_corr": pd_floor(corr_of(Xm))}
    for n in N_REF_ARMS:
        if n is None:
            arms["ref_all"] = ledoit_sigma(Xr)
        elif n < len(Xr):
            arms[f"ref_{n}"] = ledoit_sigma(
                Xr[rng.choice(len(Xr), n, replace=False)])
        else:
            arms[f"ref_{n}"] = ledoit_sigma(Xr)
    print(f"  Sigma 臂: {sorted(arms)}    workers={args.workers}\n")

    _sup = tuple(args.supports.split(","))
    _cond = tuple(float(x) for x in args.conds.split(","))
    jobs = [(s, t, r) for s in _sup for t in _cond for r in range(args.reps)]

    from concurrent.futures import ProcessPoolExecutor, as_completed
    results = []
    with ProcessPoolExecutor(
            max_workers=args.workers, initializer=_init_worker,
            initargs=(Xs, Xm, design, arms, args.seed, p, fixed_supp, args.c)) as ex:
        futs = [ex.submit(_one_rep, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            results.append(f.result())
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)}  ({time.time()-t0:.0f}s)", flush=True)

    # 聚合成每个 (支撑集, 目标) 一行
    rows = []
    for supp_name in _sup:
        for target in _cond:
            grp = [r for r in results if r["support"] == supp_name
                   and r["target_cond_t"] == target]
            if not grp:
                continue
            row = {"support": supp_name, "c": args.c, "k": len(fixed_supp) if
                   supp_name == "fixed_highVIF" else int(supp_name[-1]),
                   "target_cond_t": target,
                   "achieved_marg_t": float(np.mean(
                       [g["achieved_marg_t"] for g in grp])),
                   "achieved_cond_t": float(np.mean(
                       [g["achieved_cond_t"] for g in grp])),
                   "n_reps": len(grp)}
            for tag in arms:
                row[f"det_{tag}"] = float(np.mean([g[f"det_{tag}"] for g in grp]))
                row[f"fdp_{tag}"] = float(np.mean([g[f"fdp_{tag}"] for g in grp]))
                row[f"conv_{tag}"] = float(np.mean([g[f"conv_{tag}"] for g in grp]))
                row[f"pow_{tag}"] = float(np.mean([g[f"pow_{tag}"] for g in grp]))
            rows.append(row)
            print(f"  {supp_name:14s} cond|t|={target:4.1f}  " + "  ".join(
                f"{k}={row[f'det_{k}']:.2f}/{row[f'fdp_{k}']:.2f}" for k in arms))

    rows.sort(key=lambda r: (r["support"], r["target_cond_t"]))
    path = write_csv(os.path.join(args.outdir, "positive_control_v3.csv"), rows)
    prov = provenance(args.outdir, "exp_positive_control_v3", args,
                      inputs=[f"{args.data}/X_meanconn.csv"])
    print(f"\n-> {path}\n-> {prov}\n总耗时 {time.time()-t0:.0f}s")

    print("\n=== VIF 膨胀：边际 vs 条件 |t| ===")
    print(f"{'support':15s}{'target':>8s}{'marginal':>11s}{'conditional':>13s}{'inflation':>11s}")
    for r in rows:
        print(f"{r['support']:15s}{r['target_cond_t']:8.1f}"
              f"{r['achieved_marg_t']:11.2f}{r['achieved_cond_t']:13.2f}"
              f"{r['achieved_marg_t']/max(r['achieved_cond_t'],1e-9):11.1f}x")


if __name__ == "__main__":
    main()
