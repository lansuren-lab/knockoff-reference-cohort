"""
exp_boundary_sweep.py -- E2: 功效的边界在哪里（论文的核心新图）

论文旧版只证明了"边界内失败"，从未证明"边界外成功"，于是无法排除
"是我们的实现不行"这一解释。本脚本用一个参数同时扫过退化与良态两端，
把散落的阴性结果收成一条曲线。

三个族放在同一组坐标轴上，管线完全固定
（参考队列 Ledoit-Wolf 估计 -> equicorrelated 构造 -> 固定-λ LCD -> q=0.20）：

  族 A  Sigma(t) = (1-t) * Sigma_ABIDE + t * I,  t 从 0 到 1
  族 B  AR(1): Sigma_ij = rho^|i-j|,  rho 从 0.02 到 0.8
  族 C  真实的 5 数据集 x 4 特征定义（E1 的产物），落在同一坐标轴上

【关于 c —— 必须同时满足两个条件】实测（同一份数据，只改 c）：

    c=0.15  过罚：12 个强信号 0/12 找回，|S|=0；但收敛良好（中位 2800 步）
    c=0.03  不过罚：12/12 找回；但套索在 13%-47% 的重复里【不收敛】
            （中位 95000 步，撞上 max_iter 上限）—— 结果不可信
    c=0.05  两者兼顾：12/12 找回，且 0% 不收敛（中位 40000 步，2.5 倍余量）

所以本脚本默认 c=0.05。每个配置都会输出 frac_converged 一列，不收敛的
结果应当被丢弃而不是被解读。这一条本身就是论文 §solver 想说的话，现在
从一个"观测到的现象"变成了一条可操作的选参规则。

并行：按 (族, 参数, 重复) 展开任务；单线程下 c=0.05 每个重复约 2 秒。

Run:  python exp_boundary_sweep.py --reps 200 --workers 12 --c 0.05
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.refcohort import (DEF_DATASETS_FOR_DIAG, Q_DEFAULT, corr_of,          # noqa: E402
                           knockoff_run, ledoit_sigma, pd_floor, prepare,
                           provenance, spectral_report, split_xy, write_csv)

T_GRID = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 1.0)
RHO_GRID = (0.02, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.70, 0.80)
DEFS = ("meanconn", "gsr", "pcresid", "partial")


def base_abide():
    """族 A 的起点：ABIDE I AAL116 参考队列经验相关矩阵。"""
    X, names, man, ism, ms = prepare("feat2_abide1_aal")
    _, _, Xr = split_xy(X, man, ism)
    return pd_floor(corr_of(Xr))


def ar1(p, rho):
    i = np.arange(p)
    return pd_floor(rho ** np.abs(i[:, None] - i[None, :]))


# --------------------------------------------------------------------------- #
# 并行 worker
# --------------------------------------------------------------------------- #
def _init_worker(SIGMAS, n_ref, n_main, effect, k, c, seed):
    global _W
    _W = dict(SIGMAS=SIGMAS, n_ref=n_ref, n_main=n_main, effect=effect, k=k,
              c=c, seed=seed)


def _one_rep(job):
    """(参数键, 重复编号) -> 一次模拟的结果。

    两队列从【同一个】Sigma 抽 —— 这一族刻意只隔离"条件数"一个因素；
    队列间分布漂移由 exp_cohort_shift.py 单独处理。
    """
    key, r = job
    w = _W
    S = w["SIGMAS"][key]
    p = S.shape[0]
    L = np.linalg.cholesky(S + 1e-10 * np.eye(p))
    rng = np.random.default_rng(w["seed"] * 100000 + r)
    Xr = rng.standard_normal((w["n_ref"], p)) @ L.T
    Xm = rng.standard_normal((w["n_main"], p)) @ L.T
    idx = rng.choice(p, w["k"], replace=False)
    beta = np.zeros(p)
    beta[idx] = w["effect"] * rng.choice([-1.0, 1.0], w["k"])
    y = Xm @ beta + rng.standard_normal(w["n_main"])
    Shat = ledoit_sigma(Xr)
    res = knockoff_run(Xm, y, Shat, q=Q_DEFAULT, seed=w["seed"] * 1000 + r,
                       c=w["c"])
    sel = set(res["sel"].tolist())
    tgt = set(idx.tolist())
    return {"param_key": key, "rep": r, "nsel": len(sel),
            "fdp": (len(sel - tgt) / len(sel)) if sel else 0.0,
            "power": len(sel & tgt) / w["k"],
            "converged": float(res["converged"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--n-ref", type=int, default=393)
    ap.add_argument("--n-main", type=int, default=459)
    ap.add_argument("--effect", type=float, default=0.85)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--c", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fams", default="A,B,C")
    args = ap.parse_args()
    t0 = time.time()
    fams = set(args.fams.split(","))

    print(f"管线: 参考队列 LW Sigma-hat (n_ref={args.n_ref}) -> equicorrelated"
          f" -> 固定-λ LCD c={args.c} -> q={Q_DEFAULT}")
    print(f"      n_main={args.n_main}  effect={args.effect}  k={args.k}  "
          f"reps={args.reps}  workers={args.workers}\n")

    # 父进程先把所有 Sigma 建好（族 C 要读文件，不能在每个 rep 里重读）
    sigmas, meta = {}, {}
    if "A" in fams:
        S0 = base_abide()
        for t in T_GRID:
            key = f"A|{t}"
            sigmas[key] = pd_floor((1 - t) * S0 + t * np.eye(S0.shape[0]))
            meta[key] = {"family": "A_blend_to_I", "param": t,
                         "p": S0.shape[0], **spectral_report(sigmas[key])}
    if "B" in fams:
        for rho in RHO_GRID:
            key = f"B|{rho}"
            sigmas[key] = ar1(113, rho)
            meta[key] = {"family": "B_AR1", "param": rho, "p": 113,
                         **spectral_report(sigmas[key])}
    if "C" in fams:
        for ds, d in DEF_DATASETS_FOR_DIAG.items():
            for deff in DEFS:
                try:
                    X, names, man, ism, ms = prepare(d, fname=f"X_{deff}.csv")
                except Exception:
                    continue
                Xm, ym, Xr = split_xy(X, man, ism)
                if Xr.shape[0] < 50 or Xm.shape[0] < 50:
                    continue
                # 与族 A/B 一致：把真实结构当真值 Sigma，重新抽两个队列
                key = f"C|{ds}|{deff}"
                sigmas[key] = pd_floor(corr_of(Xr))
                meta[key] = {"family": "C_real_features", "param": np.nan,
                             "p": X.shape[1], "dataset": ds, "feature": deff,
                             **spectral_report(sigmas[key])}
    print(f"共 {len(sigmas)} 个 Sigma 配置\n")

    jobs = [(key, r) for key in sigmas for r in range(args.reps)]
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from collections import defaultdict
    acc = defaultdict(list)
    with ProcessPoolExecutor(
            max_workers=args.workers, initializer=_init_worker,
            initargs=(sigmas, args.n_ref, args.n_main, args.effect, args.k,
                      args.c, args.seed)) as ex:
        futs = [ex.submit(_one_rep, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            rr = f.result()
            acc[rr["param_key"]].append(rr)
            if i % 200 == 0:
                print(f"  {i}/{len(jobs)}  ({time.time()-t0:.0f}s)", flush=True)

    rows = []
    for key in sigmas:
        grp = acc[key]
        if not grp:
            continue
        rows.append({**meta[key], "c": args.c, "reps": len(grp),
                     "power": float(np.mean([g["power"] for g in grp])),
                     "fdp": float(np.mean([g["fdp"] for g in grp])),
                     "frac_converged": float(np.mean(
                         [g["converged"] for g in grp])),
                     "nsel": float(np.mean([g["nsel"] for g in grp]))})
    order = {"A_blend_to_I": 0, "B_AR1": 1, "C_real_features": 2}
    rows.sort(key=lambda r: (order.get(r["family"], 9), r["param"])
              if not np.isnan(r["param"]) else (order.get(r["family"], 9), -1))

    for r in rows:
        tag = (f"{r['family']:16s} {r.get('dataset',''):20s}"
               f"{r.get('feature',''):9s} param={r['param']:.2f}"
               if r["family"] != "C_real_features" else
               f"{r['family']:16s} {r['dataset']:20s}{r['feature']:9s}")
        print(f"  {tag} effrank={r['eff_rank']:7.2f} lam_min={r['lam_min']:.4f}"
              f"  power={r['power']:.3f}  FDP={r['fdp']:.3f}  |S|={r['nsel']:.1f}")

    path = write_csv(os.path.join(args.outdir, "boundary_sweep.csv"), rows)
    prov = provenance(args.outdir, "exp_boundary_sweep", args)
    print(f"\n-> {path}\n-> {prov}\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
