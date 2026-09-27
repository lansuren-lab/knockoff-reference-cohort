"""
exp_constructions.py -- 第1项：构造对照（等相关 vs SDP vs ASDP vs MVR vs ME）

为什么要做：我们全篇用的是等相关构造 s_j = 2*lam_min(Σ̂) ∧ 1。文献里有两条
直接针对它的警告，我们此前都没处理：

  * Candès et al. 2018 §3.4.2 原话：
      "λ_min(Σ) tends to be extremely small as p gets large. The result is
       that constructing equicorrelated knockoffs in high dimensions, while
       fairly computationally easy, will result in very low power."
  * 同文 §7 真实数据：把构造从等相关换成 ASDP，s 的中位数从 0.08 升到 0.57。
  * Askari et al. (FANOK) 在 HCP fMRI 上报告：等相关 knockoff
      "did not result in any discoveries"，而低秩 SDP 找到 34 个区域。

如果这两条在我们的数据上成立，那么"在 ABIDE 上什么都找不到"很可能**至少
一部分是构造选择造成的**，而不是数据或参考队列的问题。这是必须排除的
替代解释。

本脚本输出三组量：
  A. 各构造的 s 与 knockoff 相关性 Corr(X_j, X̃_j) = 1 - s_j
     —— 这是"knockoff 到底和原变量有多像"的直接度量
  B. 真实响应上各构造的发现情况（200 次抽样）
  C. 正对照（条件 |t|=6, k=12）上各构造的 per-signal power 与 FDP

Run:  python exp_constructions.py --draws 200 --workers 14
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.refcohort import (C_FIXED, DEF_DATASETS_FOR_DIAG, Q_DEFAULT,          # noqa: E402
                           corr_of, conditional_t_all, knockoff_run,
                           ledoit_sigma, pd_floor, prepare, provenance,
                           split_xy, write_csv)
from knockoff_adni.knockoffs import solve_s_diag                              # noqa: E402
from exp_positive_control_v3 import calibrate_amplitude                       # noqa: E402

METHODS = ("equicorrelated", "identity", "sdp", "asdp", "mvr", "me")


# --------------------------------------------------------------------------- #
# 并行 worker —— 必须定义在模块级：Windows 用 spawn 启动子进程，
# 定义在 main() 内部的局部函数无法被 pickle（会报
# "Can't get local object 'main.<locals>._init_w'"）。
# --------------------------------------------------------------------------- #
_W = {}


def _init_w(Xm, ym, S_full, S_BY_M, c, seed):
    global _W
    _W = dict(Xm=Xm, ym=ym, S_full=S_full, S_BY_M=S_BY_M, c=c, seed=seed)


def _real_draw(job):
    m, k = job
    w = _W
    res = knockoff_run(w["Xm"], w["ym"], w["S_full"], q=Q_DEFAULT,
                       seed=w["seed"] * 100000 + k, c=w["c"],
                       s_override=w["S_BY_M"][m])
    return {"method": m, "nsel": len(res["sel"]),
            "ratio": res["ratio_best"], "conv": float(res["converged"])}


def _ctrl_draw(job):
    m, r = job
    w = _W
    Xm = w["Xm"]
    rr = np.random.default_rng(w["seed"] * 100000 + r)
    supp = tuple(sorted(rr.choice(Xm.shape[1], 12, replace=False).tolist()))
    noise = rr.standard_normal(len(Xm))
    sd = Xm.std(0, ddof=1)
    sd[sd == 0] = 1.0
    Xs = (Xm - Xm.mean(0)) / sd
    _, des = conditional_t_all(Xs, w["ym"])
    a = calibrate_amplitude(Xs, des, supp, 6.0, noise)
    beta = np.zeros(Xm.shape[1])
    beta[list(supp)] = a
    y = Xs @ beta + noise
    res = knockoff_run(Xm, y, w["S_full"], q=Q_DEFAULT,
                       seed=w["seed"] * 1000 + r, c=w["c"],
                       s_override=w["S_BY_M"][m])
    sel = set(res["sel"].tolist())
    return {"method": m, "power": len(sel & set(supp)) / 12.0,
            "fdp": (len(sel - set(supp)) / len(sel)) if sel else 0.0,
            "nsel": len(sel), "conv": float(res["converged"])}


def method_s(S, method, n_ref=None):
    """算出一个构造的 s 向量；失败返回 None。"""
    try:
        kw = {}
        if method in ("mvr_robust", "me_robust"):
            kw["n_ref"] = n_ref
        return np.asarray(solve_s_diag(S, method=method, **kw), float)
    except Exception as e:
        print(f"    [skip] {method}: {type(e).__name__}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--c", type=float, default=C_FIXED)
    ap.add_argument("--outdir", default="results_construct")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(args.outdir, exist_ok=True)

    # ---------------- A. s 与 knockoff 相关性 ----------------
    print("=" * 78)
    print("A. 各构造的 s 与 knockoff 相关性  (Corr(X_j, X~_j) = 1 - s_j)")
    print("=" * 78)
    rows_a = []
    for ds, d in DEF_DATASETS_FOR_DIAG.items():
        for deff in ("meanconn", "partial"):
            try:
                X, names, man, ism, ms = prepare(d, fname=f"X_{deff}.csv")
            except Exception:
                continue
            Xm, ym, Xr = split_xy(X, man, ism)
            if Xr.shape[0] < 50:
                continue
            S = ledoit_sigma(Xr)
            for m in METHODS:
                s = method_s(S, m, n_ref=Xr.shape[0])
                if s is None:
                    continue
                rows_a.append({
                    "dataset": ds, "feature": deff, "method": m,
                    "p": X.shape[1],
                    "s_median": float(np.median(s)), "s_min": float(s.min()),
                    "s_max": float(s.max()),
                    "corr_median": float(1 - np.median(s)),
                    "corr_min": float(1 - s.max()),
                    "lam_min": float(np.linalg.eigvalsh(S).min())})
                print(f"  {ds:20s} {deff:9s} {m:15s} s中位={np.median(s):.4f}"
                      f"  Corr={1-np.median(s):.4f}", flush=True)
    write_csv(os.path.join(args.outdir, "construction_s.csv"), rows_a)

    # ---------------- B/C. 真实响应 + 正对照 ----------------
    X, names, man, ism, ms = prepare("feat2_abide1_aal")
    Xm, ym, Xr = split_xy(X, man, ism)
    p = X.shape[1]
    S = ledoit_sigma(Xr)
    mu, sd = Xm.mean(0), Xm.std(0, ddof=1)
    sd[sd == 0] = 1.0
    Xs = (Xm - mu) / sd
    _, design = conditional_t_all(Xs, ym)

    # 预先定好每个构造的 s（正对照里每个 rep 复用，避免重复解 SDP）
    S_BY_M = {m: method_s(S, m, n_ref=Xr.shape[0]) for m in METHODS}
    S_BY_M = {k: v for k, v in S_BY_M.items() if v is not None}

    from concurrent.futures import ProcessPoolExecutor, as_completed
    from collections import defaultdict

    for tag, fn, jobs, keys in (
            ("B", _real_draw,
             [(m, k) for m in S_BY_M for k in range(args.draws)],
             ("nsel", "ratio")),
            ("C", _ctrl_draw,
             [(m, r) for m in S_BY_M for r in range(args.draws)],
             ("power", "fdp", "nsel"))):
        label = ("B. 真实响应上的发现情况" if tag == "B"
                 else "C. 正对照（条件|t|=6, k=12）")
        print()
        print("=" * 78)
        print(f"{label}   ({args.draws} 次重复)")
        print("=" * 78)
        acc = defaultdict(list)
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_init_w,
                                 initargs=(Xm, ym, S, S_BY_M, args.c,
                                           args.seed)) as ex:
            futs = [ex.submit(fn, j) for j in jobs]
            for i, f in enumerate(as_completed(futs), 1):
                rr = f.result()
                acc[rr["method"]].append(rr)
                if i % 200 == 0:
                    print(f"    {i}/{len(jobs)}  ({time.time()-t0:.0f}s)",
                          flush=True)
        rows = []
        for m in S_BY_M:
            g = acc[m]
            if not g:
                continue
            row = {"method": m, "p": p, "n_main": int(Xm.shape[0]),
                   "s_median": float(np.median(S_BY_M[m])),
                   "corr_median": float(1 - np.median(S_BY_M[m])),
                   "reps": len(g),
                   "frac_converged": float(np.mean([x["conv"] for x in g]))}
            if tag == "B":
                row["p_nonempty"] = float(np.mean(
                    [1.0 if x["nsel"] > 0 else 0.0 for x in g]))
                row["nsel_median"] = float(np.median([x["nsel"] for x in g]))
                row["nsel_max"] = int(np.max([x["nsel"] for x in g]))
                row["ratio_median"] = float(np.median([x["ratio"] for x in g]))
            else:
                row["power"] = float(np.mean([x["power"] for x in g]))
                row["fdp"] = float(np.mean([x["fdp"] for x in g]))
                row["nsel_mean"] = float(np.mean([x["nsel"] for x in g]))
            rows.append(row)
            cells = "  ".join(f"{k}={row[k]:.3f}" for k in keys if k in row)
            print(f"  {m:15s} s中位={row['s_median']:.4f}  {cells}")
        write_csv(os.path.join(
            args.outdir, f"construction_{'real' if tag=='B' else 'control'}.csv"),
            rows)

    provenance(args.outdir, "exp_constructions", args)
    print(f"\n-> {args.outdir}/\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
