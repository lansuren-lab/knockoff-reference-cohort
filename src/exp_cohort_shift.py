"""
exp_cohort_shift.py -- E4: 队列间分布漂移（论文真正的研究对象）

这是论文旧版最大的设计漏洞。

手稿的模拟里两个队列从【同一个】Σ 抽样（validate_shrinkage_fdr.py:160,163；
ndesign_curve.py:62-64），所以模拟只测了"估计误差"，也就是最良性的那种不匹配。
但论文的题目与主题恰恰是"参考队列与主队列【不同】"——不同受试者、不同扫描仪、
不同站点。这个不匹配在旧设计里从未被构造过。

后果是可以直接验证的：用 ABIDE 的 Σ 做模拟，power 约 0.5-0.7；而同一套管线
在真实数据上返回零发现。模拟复现不出真实失败 —— 这个 gap 就是论文真正的
发现，但旧设计测不到它。

本脚本构造受控的漂移：

    X_ref  ~ N(0, Sigma)
    X_main ~ N(0, Sigma_delta)   ,  Sigma_delta = pd_floor((1-delta) Sigma + delta P)

P 是稀疏随机扰动（随机选若干 ROI 对，改动其相关）。扫 delta，报告 power / FDP，
并给出与真实数据不匹配程度的对照刻度。

关键一点：我们还测量【真实 ABIDE 主/参考队列之间的实际不匹配度】，
然后找出与之相当的 delta —— 这样"模拟里的失败"就能与"真实数据的失败"对上号。

Run:  python exp_cohort_shift.py --reps 200
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.refcohort import (C_FIXED, Q_DEFAULT, corr_of, knockoff_run,        # noqa: E402
                           ledoit_sigma, pd_floor, prepare, provenance,
                           split_xy, write_csv)

DELTA_GRID = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60, 0.80)


def mismatch(A, B):
    """两个相关矩阵的不匹配度：非对角元的 RMS 差。"""
    p = A.shape[0]
    off = ~np.eye(p, dtype=bool)
    D = (A - B)[off]
    return float(np.sqrt(np.mean(D ** 2)))


def sparse_perturbation(p, rng, n_pairs):
    """稀疏随机相关结构扰动 P（对称、对角为 1）。"""
    P = np.eye(p)
    for _ in range(n_pairs):
        i, j = rng.choice(p, 2, replace=False)
        v = rng.uniform(-0.9, 0.9)
        P[i, j] = P[j, i] = v
    return pd_floor(P)


def _init_worker(S_true, Lr, Lm_of, n_ref, n_main, effect, k, c, seed):
    global _W
    _W = dict(S_true=S_true, Lr=Lr, Lm_of=Lm_of, n_ref=n_ref, n_main=n_main,
              effect=effect, k=k, c=c, seed=seed)


def _one_rep(job):
    delta, r = job
    w = _W
    p = w["S_true"].shape[0]
    Lm = w["Lm_of"][delta]
    rng = np.random.default_rng(w["seed"] * 100000 + r)
    Xr = rng.standard_normal((w["n_ref"], p)) @ w["Lr"].T
    Xm = rng.standard_normal((w["n_main"], p)) @ Lm.T
    idx = rng.choice(p, w["k"], replace=False)
    beta = np.zeros(p)
    beta[idx] = w["effect"] * rng.choice([-1.0, 1.0], w["k"])
    y = Xm @ beta + rng.standard_normal(w["n_main"])
    Shat = ledoit_sigma(Xr)                  # 参考队列只知道 Sigma
    res = knockoff_run(Xm, y, Shat, q=Q_DEFAULT, seed=w["seed"] * 1000 + r,
                       c=w["c"])
    sel = set(res["sel"].tolist())
    tgt = set(idx.tolist())
    return {"delta": delta, "nsel": len(sel),
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
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--c", type=float, default=C_FIXED)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    t0 = time.time()

    X, names, man, is_main, ms = prepare("feat2_abide1_aal")
    Xm_real, ym, Xr_real = split_xy(X, man, is_main)
    S_true = pd_floor(corr_of(Xr_real))
    real_mismatch = mismatch(pd_floor(corr_of(Xm_real)), S_true)
    print(f"真实 ABIDE I 主/参考队列之间的不匹配度 = {real_mismatch:.4f}")
    print(f"  (参考队列经验相关 vs 主队列经验相关，非对角 RMS 差)")
    print(f"  -> 下面把模拟里的 delta 对到这个刻度上，"
          f"看哪个 delta 与之相当\n")

    p = S_true.shape[0]
    rng_p = np.random.default_rng(12345)
    Lr = np.linalg.cholesky(S_true + 1e-10 * np.eye(p))
    Lm_of, mm_of = {}, {}
    for delta in DELTA_GRID:
        S_main = pd_floor((1 - delta) * S_true
                          + delta * sparse_perturbation(p, rng_p, 40))
        Lm_of[delta] = np.linalg.cholesky(S_main + 1e-10 * np.eye(p))
        mm_of[delta] = mismatch(S_true, S_main)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    jobs = [(d, r) for d in DELTA_GRID for r in range(args.reps)]
    acc = {d: [] for d in DELTA_GRID}
    with ProcessPoolExecutor(
            max_workers=args.workers, initializer=_init_worker,
            initargs=(S_true, Lr, Lm_of, args.n_ref, args.n_main, args.effect,
                      args.k, args.c, args.seed)) as ex:
        futs = [ex.submit(_one_rep, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            rr = f.result()
            acc[rr["delta"]].append(rr)
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)}  ({time.time()-t0:.0f}s)", flush=True)

    rows = []
    for delta in DELTA_GRID:
        grp = acc[delta]
        if not grp:
            continue
        rows.append({"delta": delta, "mismatch": mm_of[delta],
                     "fdp": float(np.mean([g["fdp"] for g in grp])),
                     "power": float(np.mean([g["power"] for g in grp])),
                     "nsel": float(np.mean([g["nsel"] for g in grp])),
                     "frac_converged": float(np.mean(
                         [g["converged"] for g in grp])),
                     "c": args.c, "reps": len(grp),
                     "real_mismatch_reference": real_mismatch,
                     "is_closest_to_real": False})
        print(f"  delta={delta:4.2f}  不匹配={mm_of[delta]:.4f}  "
              f"power={rows[-1]['power']:.3f}  FDP={rows[-1]['fdp']:.3f}  "
              f"|S|={rows[-1]['nsel']:.1f}   ({time.time()-t0:.0f}s)", flush=True)

    # 标出与真实不匹配度最接近的一行
    if rows:
        j = int(np.argmin([abs(r["mismatch"] - real_mismatch) for r in rows]))
        rows[j]["is_closest_to_real"] = True
        print(f"\n  与真实不匹配度最接近的是 delta={rows[j]['delta']}"
              f"（不匹配 {rows[j]['mismatch']:.4f} vs 真实 {real_mismatch:.4f}）"
              f" -> power={rows[j]['power']:.3f}, FDP={rows[j]['fdp']:.3f}")

    path = write_csv(os.path.join(args.outdir, "cohort_shift.csv"), rows)
    prov = provenance(args.outdir, "exp_cohort_shift", args,
                      inputs=["feat2_abide1_aal/X_meanconn.csv"])
    print(f"\n-> {path}\n-> {prov}\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
